import re
import time
from pathlib import Path

from bugfix.edits import apply_edits, is_test_path, parse_edits, revert
from bugfix.llm import chat
from bugfix.sandbox import fully_passes, run_tests

MAX_CONTEXT_CHARS = 12_000
MAX_REPORT_CHARS = 2_000
TRACE_RE = re.compile(r"([\w./-]+\.py):\d+")

# Changing this string changes the prompt distribution: regenerate data/prompts.jsonl.
SYSTEM_PROMPT = """You are an expert Python engineer fixing a bug in a repository.
You get a bug report (usually failing test output) and the most relevant source files.
Fix the bug by editing source files only. Never edit tests.

Reply with at most 3 sentences of reasoning, then one or more edit blocks in exactly this format:

path/to/file.py
<<<<<<< SEARCH
exact lines to replace, copied verbatim from the file
=======
replacement lines
>>>>>>> REPLACE

Rules:
- SEARCH text must match the file exactly (including indentation) and appear exactly once; add surrounding lines if needed.
- Make the smallest change that fixes the root cause.
- Do not wrap edit blocks in Markdown code fences."""


class BugFixAgent:
    def __init__(self, llm, index=None, max_attempts: int = 3, max_files: int = 3):
        self.llm, self.index = llm, index
        self.max_attempts, self.max_files = max_attempts, max_files

    def _context_files(self, ws: Path, repo_root, report: str) -> list[str]:
        files: list[str] = []
        for m in TRACE_RE.findall(report):  # files named in tracebacks first
            p = m.lstrip("./")
            if (ws / p).is_file() and not is_test_path(p) and p not in files:
                files.append(p)
        if self.index is not None and repo_root is not None:  # LlamaIndex retrieval
            for p in self.index.locate(Path(repo_root), report, k_files=self.max_files):
                if p not in files and (ws / p).is_file():
                    files.append(p)
        if not files:  # fallback: any source file
            files = [p.relative_to(ws).as_posix() for p in sorted(ws.rglob("*.py"))
                     if not is_test_path(p.relative_to(ws).as_posix())]
        return files[: self.max_files]

    def build_messages(self, ws: Path, bug_report: str, repo_root=None):
        files = self._context_files(ws, repo_root, bug_report)
        budget, parts = MAX_CONTEXT_CHARS, []
        for rel in files:
            if budget <= 0:
                break
            text = (ws / rel).read_text(encoding="utf-8", errors="ignore")
            if len(text) > budget:
                text = text[:budget] + "\n# ... [truncated]"
            budget -= len(text)
            parts.append(f"=== FILE: {rel} ===\n{text}")
        user = (f"## Bug report\n{bug_report[-MAX_REPORT_CHARS:]}\n\n"
                f"## Source files\n" + "\n\n".join(parts) + "\n\nFix the bug.")
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}], files

    def solve(self, ws: Path, bug_report: str, repo_root=None, gold_file=None,
              n_total=None, append_test_output: bool = False) -> dict:
        t0 = time.perf_counter()
        base = run_tests(ws)
        if base.passed:
            return {"success": True, "note": "Tests already pass.", "attempts_log": [],
                    "messages": [], "diff": "", "regression": False, "context_files": [],
                    "gold_in_context": None, "prompt_tokens": 0, "completion_tokens": 0,
                    "latency_s": 0.0, "attempts": 0}
        n_total = n_total or (base.n_passed + base.n_failed + base.n_error)
        if append_test_output:
            bug_report += f"\n\nTest output:\n{base.tail(MAX_REPORT_CHARS)}"

        messages, files = self.build_messages(ws, bug_report, repo_root)
        log, diff, success, regression = [], "", False, False

        for attempt in range(1, self.max_attempts + 1):
            res = chat(self.llm, messages)
            messages.append({"role": "assistant", "content": res.text})
            entry = {"valid": False, "applied": False, "passed": False, "regression": False,
                     "diff_lines": 0, "latency_s": res.latency_s,
                     "prompt_tokens": res.prompt_tokens,
                     "completion_tokens": res.completion_tokens}
            feedback = None
            edits = parse_edits(res.text)
            if not edits:
                feedback = "I found no valid SEARCH/REPLACE edit blocks. Follow the required format exactly."
            else:
                entry["valid"] = True
                applied = apply_edits(ws, edits)
                if not applied.ok:
                    feedback = f"Your edits could not be applied: {applied.message}"
                else:
                    entry["applied"], entry["diff_lines"] = True, applied.diff_lines
                    diff = applied.diff
                    tests = run_tests(ws)
                    entry["passed"] = fully_passes(tests, n_total)
                    entry["regression"] = bool(tests.failed_ids - base.failed_ids)
                    regression = entry["regression"]
                    if not entry["passed"]:
                        revert(ws, applied.originals)
                        feedback = f"Your edit applied but tests still fail:\n{tests.tail(1500)}"
            log.append(entry)
            if entry["passed"]:
                success = True
                break
            if attempt < self.max_attempts:
                messages.append({"role": "user", "content":
                                 feedback + "\n\nYour changes were reverted; the files are exactly as shown above. Try a different fix."})

        return {
            "success": success, "attempts": len(log), "attempts_log": log,
            "messages": messages, "diff": diff, "regression": regression,
            "context_files": files,
            "gold_in_context": (gold_file in files) if gold_file else None,
            "prompt_tokens": sum(e["prompt_tokens"] for e in log),
            "completion_tokens": sum(e["completion_tokens"] for e in log),
            "latency_s": time.perf_counter() - t0,
        }