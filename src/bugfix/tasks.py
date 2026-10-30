import argparse
import hashlib
import io
import json
import random
import tokenize
from pathlib import Path

from bugfix.edits import format_edit, is_test_path
from bugfix.sandbox import cleanup, make_workspace, run_tests

SWAPS = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==",
         "+": "-", "-": "+", "and": "or", "or": "and", "True": "False", "False": "True"}
SKIP_TOP = {"docs", "doc", "examples", "benchmarks", "scripts", "build"}
REPORT_CHARS = 2000


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows -> {path}")


def load_tasks(path, split=None):
    return [t for t in read_jsonl(path) if split is None or t["split"] == split]


def split_of(repo: str, file: str) -> str:
    """Split by file so the same file never appears in train and test."""
    h = int(hashlib.md5(f"{repo}:{file}".encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 80 else "test"


def candidate_mutations(source: str):
    """Yield (row, col, old_line, new_line) using real tokens, so strings/comments are safe."""
    lines = io.StringIO(source).readlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    for i, tok in enumerate(toks):
        if tok.type not in (tokenize.OP, tokenize.NAME) or tok.string not in SWAPS:
            continue
        if tok.start[0] != tok.end[0]:
            continue
        if tok.string in ("+", "-"):  # binary operators only
            prev = toks[i - 1]
            if not (prev.type in (tokenize.NAME, tokenize.NUMBER) or prev.string in (")", "]")):
                continue
        (row, col), (_, ecol) = tok.start, tok.end
        old = lines[row - 1]
        new = old[:col] + SWAPS[tok.string] + old[ecol:]
        if new != old:
            yield row, col, old, new


def generate_tasks(repos, out_path, max_per_file=4, seed=0):
    rng, tasks = random.Random(seed), []
    for repo in map(Path, repos):
        ws = make_workspace(repo)
        try:
            clean = run_tests(ws)
            if not clean.passed:
                print(f"skip {repo}: test suite does not pass cleanly")
                continue
            for src in sorted(ws.rglob("*.py")):
                rel = src.relative_to(ws).as_posix()
                if (is_test_path(rel) or rel == "setup.py"
                        or src.relative_to(ws).parts[0] in SKIP_TOP):
                    continue
                original = src.read_text(encoding="utf-8")
                cands = list(candidate_mutations(original))
                rng.shuffle(cands)
                kept = 0
                for row, col, old, new in cands:
                    if kept >= max_per_file:
                        break
                    lines = io.StringIO(original).readlines()
                    lines[row - 1] = new
                    src.write_text("".join(lines), encoding="utf-8")
                    quick = run_tests(ws, timeout=60, fast=True)
                    if quick.passed or quick.timed_out:
                        continue
                    full = run_tests(ws, timeout=120)
                    if full.passed or full.timed_out:
                        continue
                    report = ("The test suite is failing after a recent change. Find and fix the bug "
                              "in the source code; do not modify the tests.\n\nPytest output:\n"
                              + full.tail(REPORT_CHARS))
                    tasks.append({
                        "id": f"{repo.name}:{rel}:{row}:{col}", "repo": str(repo), "file": rel,
                        "line": row, "original_line": old, "mutated_line": new,
                        "n_total": clean.n_passed, "failed_ids": sorted(full.failed_ids),
                        "bug_report": report, "split": split_of(str(repo), rel)})
                    kept += 1
                src.write_text(original, encoding="utf-8")
        finally:
            cleanup(ws)
        print(f"{repo}: {len(tasks)} tasks so far")
    write_jsonl(out_path, tasks)


def materialize(task) -> Path:
    """Fresh workspace with the bug injected."""
    ws = make_workspace(Path(task["repo"]))
    path = ws / task["file"]
    lines = io.StringIO(path.read_text(encoding="utf-8")).readlines()
    assert lines[task["line"] - 1] == task["original_line"], "task/source mismatch"
    lines[task["line"] - 1] = task["mutated_line"]
    path.write_text("".join(lines), encoding="utf-8")
    return ws


def gold_edit(task, ws: Path):
    """Reference fix as an edit block, with just enough context to be unique."""
    text = (ws / task["file"]).read_text(encoding="utf-8")
    lines, i = text.split("\n"), task["line"] - 1
    fixed = task["original_line"].rstrip("\n")
    for ctx in range(6):
        lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
        search = "\n".join(lines[lo:hi])
        if text.count(search) == 1:
            replace = "\n".join(lines[lo:i] + [fixed] + lines[i + 1:hi])
            return format_edit(task["file"], search, replace)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--out", default="data/tasks.jsonl")
    ap.add_argument("--max-per-file", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate_tasks(a.repos, a.out, a.max_per_file, a.seed)