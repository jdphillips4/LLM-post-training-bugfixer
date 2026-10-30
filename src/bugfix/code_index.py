import ast
import hashlib
import threading
from pathlib import Path

from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from bugfix import config
from bugfix.edits import is_test_path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist", "docs", "examples"}


def build_nodes(repo: Path) -> list[TextNode]:
    """One node per function (plus a module header), so chunks follow code structure."""
    nodes = []
    for path in sorted(repo.rglob("*.py")):
        rel_path = path.relative_to(repo)
        rel = rel_path.as_posix()
        if any(p in SKIP_DIRS for p in rel_path.parts) or is_test_path(rel):
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        lines = src.splitlines()
        chunks = [("<module>", "\n".join(lines[:40]))]
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append((node.name, "\n".join(lines[node.lineno - 1: node.end_lineno])))
        except SyntaxError:
            pass
        for symbol, text in chunks:
            if text.strip():
                nodes.append(TextNode(text=f"# {rel}::{symbol}\n{text[:1500]}",
                                      metadata={"file_path": rel, "symbol": symbol}))
    return nodes


class CodeIndex:
    """Index is built once per repo from the CLEAN source and cached on disk.
    It is used only to locate files; file contents shown to the model come from
    the (buggy) workspace, so no label leaks through retrieval."""

    def __init__(self, cache_dir: str = ".cache/index", embed_model: str | None = None):
        self.cache_dir = Path(cache_dir)
        self.embed = HuggingFaceEmbedding(model_name=embed_model or config.EMBED_MODEL)
        Settings.embed_model = self.embed
        self._indexes: dict[str, VectorStoreIndex] = {}
        self._lock = threading.Lock()

    def _get(self, repo: Path) -> VectorStoreIndex:
        key = hashlib.md5(str(Path(repo).resolve()).encode()).hexdigest()[:12]
        if key not in self._indexes:
            persist = self.cache_dir / key
            if persist.exists():
                ctx = StorageContext.from_defaults(persist_dir=str(persist))
                idx = load_index_from_storage(ctx, embed_model=self.embed)
            else:
                idx = VectorStoreIndex(build_nodes(Path(repo)), embed_model=self.embed)
                idx.storage_context.persist(persist_dir=str(persist))
            self._indexes[key] = idx
        return self._indexes[key]

    def locate(self, repo: Path, query: str, k_chunks: int = 8, k_files: int = 3) -> list[str]:
        with self._lock:
            nodes = self._get(repo).as_retriever(similarity_top_k=k_chunks).retrieve(query[:2000])
        best: dict[str, float] = {}
        for n in nodes:
            fp = n.metadata["file_path"]
            best[fp] = max(best.get(fp, 0.0), n.score or 0.0)
        return [f for f, _ in sorted(best.items(), key=lambda kv: -kv[1])][:k_files]