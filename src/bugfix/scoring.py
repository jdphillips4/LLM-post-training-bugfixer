from bugfix.edits import apply_edits, parse_edits
from bugfix.sandbox import cleanup, fully_passes, run_tests
from bugfix.tasks import materialize


def score_completion(task: dict, completion: str, timeout: int = 120) -> dict:
    """Run one single-shot completion against a task in a fresh workspace."""
    s = {"valid": False, "applied": False, "passed": False, "regression": False, "diff_lines": 0}
    edits = parse_edits(completion)
    if not edits:
        return s
    s["valid"] = True
    ws = materialize(task)
    try:
        res = apply_edits(ws, edits)
        if not res.ok:
            return s
        s["applied"], s["diff_lines"] = True, res.diff_lines
        tests = run_tests(ws, timeout=timeout)
        s["passed"] = fully_passes(tests, task["n_total"])
        s["regression"] = bool(tests.failed_ids - set(task["failed_ids"]))
        return s
    finally:
        cleanup(ws)


def reward_from_score(s: dict) -> float:
    """RL reward: format < applies < fails tests < passes (minimal diffs preferred)."""
    if not s["valid"]:
        return 0.0
    if not s["applied"]:
        return 0.1
    if not s["passed"]:
        return 0.2 - (0.05 if s["regression"] else 0.0)
    return max(0.5, 1.0 - 0.02 * max(0, s["diff_lines"] - 6))