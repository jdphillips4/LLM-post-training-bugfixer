import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

EDIT_RE = re.compile(
    r"^(?P<path>[^\n]+)\n<{7} SEARCH\n(?P<search>.*?)\n={7}\n(?P<replace>.*?)>{7} REPLACE",
    re.DOTALL | re.MULTILINE,
)


@dataclass
class Edit:
    path: str
    search: str
    replace: str


@dataclass
class ApplyResult:
    ok: bool
    message: str = ""
    diff: str = ""
    diff_lines: int = 0
    originals: dict = field(default_factory=dict)


def parse_edits(text: str) -> list[Edit]:
    edits = []
    for m in EDIT_RE.finditer(text):
        path = m.group("path").strip().strip("`*").removeprefix("./")
        replace = m.group("replace")
        replace = replace[:-1] if replace.endswith("\n") else replace
        edits.append(Edit(path, m.group("search"), replace))
    return edits


def is_test_path(path: str) -> bool:
    p = Path(path)
    return (
        any(part in {"tests", "test", "testing"} for part in p.parts[:-1])
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or p.name == "conftest.py"
    )


def apply_edits(root: Path, edits: list[Edit]) -> ApplyResult:
    """All-or-nothing. Test files are protected in code, not just in the prompt."""
    root = root.resolve()
    originals, updated = {}, {}
    for e in edits:
        if is_test_path(e.path):
            return ApplyResult(False, f"Editing test files is not allowed ({e.path}).")
        target = (root / e.path).resolve()
        if root not in target.parents or not target.is_file():
            return ApplyResult(False, f"File not found: {e.path}")
        if e.path not in originals:
            try:
                originals[e.path] = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return ApplyResult(False, f"Cannot read {e.path}")
            updated[e.path] = originals[e.path]
        n = updated[e.path].count(e.search)
        if n == 0:
            return ApplyResult(False, f"SEARCH block not found in {e.path}; it must match the file exactly.")
        if n > 1:
            return ApplyResult(False, f"SEARCH block matches {n} places in {e.path}; include more surrounding lines.")
        updated[e.path] = updated[e.path].replace(e.search, e.replace, 1)

    diff = ""
    for path, new in updated.items():
        diff += "".join(difflib.unified_diff(
            originals[path].splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}"))
        (root / path).write_text(new, encoding="utf-8")
    n_changed = sum(1 for l in diff.splitlines()
                    if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return ApplyResult(True, "ok", diff, n_changed, originals)


def revert(root: Path, originals: dict) -> None:
    for path, text in originals.items():
        (root / path).write_text(text, encoding="utf-8")


def format_edit(path: str, search: str, replace: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"