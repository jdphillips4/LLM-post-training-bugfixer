import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

IGNORE = shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                                "*.pyc", ".tox", "build", "dist", "*.egg-info")
FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)


@dataclass
class RunResult:
    passed: bool
    returncode: int
    output: str
    seconds: float
    timed_out: bool = False
    n_passed: int = 0
    n_failed: int = 0
    n_error: int = 0
    failed_ids: frozenset = frozenset()

    def tail(self, n: int = 2000) -> str:
        return self.output[-n:]


def make_workspace(repo: Path) -> Path:
    root = Path(tempfile.mkdtemp(prefix="bugfix_"))
    ws = root / Path(repo).name
    shutil.copytree(repo, ws, ignore=IGNORE)
    return ws


def cleanup(ws: Path) -> None:
    shutil.rmtree(ws.parent, ignore_errors=True)


def _count(pattern: str, out: str) -> int:
    m = re.findall(pattern, out)
    return int(m[-1]) if m else 0


def run_tests(ws: Path, timeout: int = 120, fast: bool = False) -> RunResult:
    cmd = [sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=short",
           "--disable-warnings", "-p", "no:cacheprovider"]
    if fast:
        cmd.append("-x")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ws, capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = (p.stdout or "") + (p.stderr or "")
        return RunResult(
            p.returncode == 0, p.returncode, out, time.perf_counter() - t0, False,
            _count(r"(\d+) passed", out), _count(r"(\d+) failed", out),
            _count(r"(\d+) errors?\b", out), frozenset(FAILED_RE.findall(out)))
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return RunResult(False, -1, out + "\n[timeout]", time.perf_counter() - t0, True)


def fully_passes(res: RunResult, n_total: int) -> bool:
    """Guards against edits that make pytest exit 0 without running the tests."""
    return res.passed and res.n_passed >= n_total