#!/usr/bin/env python3
"""Public-safety scan (ADR-076, ADR-078): keep personal info out of tracked files.

git_security_check.py / gitleaks catch *secret values*. This module catches
*personal-information expressions* that gitleaks cannot: e-mail addresses,
phone numbers, account names inside local paths, and any term listed in the
local-only deny-list (docs/public-safety-denylist.txt, gitignored, one term per
line). It only looks at ADDED lines (staged diff, unpushed commits) so old
history never blocks new work. The whole tracked tree is scanned only on
request (`--public-release`).

A line that is a deliberate false positive can carry the marker
``public-safety: allow`` (with a reason). Matched text is never printed in full.

Used by git_security_check.py (commit / push / Cursor hook). Pure functions
take the diff text, so they are unit-testable without git.
"""

from __future__ import annotations

import datetime
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DENYLIST_PATH = REPO_ROOT / "docs" / "public-safety-denylist.txt"
ALLOW_MARKER = "public-safety: allow"

# Whole-tree scan limits (public-release mode)
MAX_TREE_FILE_BYTES = 2 * 1024 * 1024

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@"
    r"([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
)
EMAIL_ALLOWED_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "users.noreply.github.com",
    "github.com",  # ssh remotes such as git@github.com:owner/repo
    "localhost",
}
EMAIL_ALLOWED_SUFFIXES = (".invalid", ".test", ".example", ".localhost")
EMAIL_ALLOWED_LOCALPARTS = {"git", "noreply", "no-reply"}

PHONE_RE = re.compile(r"(?<![0-9])01[016789][- ]?[0-9]{3,4}[- ]?[0-9]{4}(?![0-9])")

# Korean Resident Registration Number (RRN): YYMMDD-GXXXXXX
# Gender digit 1/2 = 1900s, 3/4 = 2000s. Boundary check prevents matching long digit runs.
RRN_RE = re.compile(
    r"(?<![0-9])([0-9]{2})([0-9]{2})([0-9]{2})[- ]?([1-4][0-9]{6})(?![0-9])"
)


def _is_valid_rrn_date(yy: int, mm: int, dd: int, gender: int) -> bool:
    """Validate that YY-MM-DD corresponds to a valid calendar date for the century."""
    century = 1900 if gender in (1, 2) else 2000
    year = century + yy
    try:
        datetime.date(year, mm, dd)
        return True
    except ValueError:
        return False


def _verify_rrn_checksum(digits13: str) -> bool:
    """Verify Korean RRN checksum for 13 digits.

    Weights: [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5].
    check = (11 - (sum % 11)) % 10.
    Post-Oct-2020 numbers lack checksum, so failure is NOT a disqualifier,
    only distinguishes 'rrn:high-confidence' from 'rrn'.
    """
    if len(digits13) != 13 or not digits13.isdigit():
        return False
    weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    s = sum(int(d) * w for d, w in zip(digits13[:12], weights))
    check = (11 - (s % 11)) % 10
    return check == int(digits13[12])


def _rrn_hits(content: str) -> Iterator[Tuple[str, str]]:
    """Yield (matched_text, kind) for valid Korean Resident Registration Number patterns."""
    for m in RRN_RE.finditer(content):
        yy = int(m.group(1))
        mm = int(m.group(2))
        dd = int(m.group(3))
        back7 = m.group(4)
        gender = int(back7[0])
        if not _is_valid_rrn_date(yy, mm, dd, gender):
            continue
        digits13 = f"{m.group(1)}{m.group(2)}{m.group(3)}{back7}"
        if _verify_rrn_checksum(digits13):
            yield m.group(0), "rrn:high-confidence"
        else:
            yield m.group(0), "rrn"

WIN_USER_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'`<>|*?)(\[\]{}%$]+)"
)
# Must start a path token: not preceded by a drive colon (already covered by the
# Windows pattern) nor by a name character (relative paths such as src/home/foo).
# WSL form /mnt/c/Users/<name> is covered by the optional prefix.
POSIX_USER_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.:-])(?:/mnt/[a-z])?/(?:home|Users)/([A-Za-z0-9_.-]+)"
)
PATH_PLACEHOLDER_NAMES = {
    "user", "users", "username", "yourname", "your-name", "name", "me", "you",
    "public", "default", "all", "runner", "vscode", "node", "app", "ubuntu",
    "ci", "build", "example", "someone", "foo", "bar",
}

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    masked: str

    def render(self) -> str:
        return f"public-safety [{self.kind}] {self.path}:{self.line}: {self.masked}"


def mask(text: str) -> str:
    """Show only enough to locate the hit; never the full value."""
    t = text.strip()
    if len(t) <= 3:
        return "***"
    return f"{t[:2]}***({len(t)} chars)"


def load_denylist(path: Optional[Path] = None) -> List[str]:
    p = Path(path) if path is not None else DENYLIST_PATH
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return []
    terms = []
    for line in raw.splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            terms.append(t.lower())
    return terms


def _email_hits(content: str) -> Iterator[str]:
    for m in EMAIL_RE.finditer(content):
        local, domain = m.group(1).lower(), m.group(2).lower()
        if local in EMAIL_ALLOWED_LOCALPARTS or local.startswith(("noreply", "no-reply")):
            continue
        if domain in EMAIL_ALLOWED_DOMAINS or domain.endswith(EMAIL_ALLOWED_SUFFIXES):
            continue
        yield m.group(0)


def _path_hits(content: str) -> Iterator[str]:
    for rx in (WIN_USER_PATH_RE, POSIX_USER_PATH_RE):
        for m in rx.finditer(content):
            name = m.group(1).strip().lower()
            if not name or name in PATH_PLACEHOLDER_NAMES:
                continue
            yield m.group(0)


def scan_line(
    path: str, lineno: int, content: str, denylist: Sequence[str] = ()
) -> List[Finding]:
    """Findings for one added line. A line carrying the allow marker is skipped."""
    if ALLOW_MARKER in content:
        return []
    out: List[Finding] = []
    for hit in _email_hits(content):
        out.append(Finding(path, lineno, "email", mask(hit)))
    for m in PHONE_RE.finditer(content):
        out.append(Finding(path, lineno, "phone", mask(m.group(0))))
    for hit, kind in _rrn_hits(content):
        out.append(Finding(path, lineno, kind, mask(hit)))
    for hit in _path_hits(content):
        out.append(Finding(path, lineno, "account-path", mask(hit)))
    if denylist:
        low = content.lower()
        for idx, term in enumerate(denylist, start=1):
            if term in low:
                out.append(Finding(path, lineno, "denylist", f"term #{idx}"))
    return out


def parse_added_lines(diff_text: str) -> Iterator[Tuple[str, int, str]]:
    """Yield (path, new_line_number, content) for every '+' line of a unified diff.

    Works for `git diff` and `git log -p` output with -U0. Deleted files
    (+++ /dev/null) and binary notices are skipped.
    """
    path: Optional[str] = None
    lineno = 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path = None
            lineno = 0
            continue
        if raw.startswith("+++ "):
            target = raw[4:].split("\t")[0].strip()
            if target == "/dev/null":
                path = None
            else:
                path = target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("--- ") or raw.startswith("Binary files"):
            continue
        m = HUNK_RE.match(raw)
        if m:
            lineno = int(m.group(1))
            continue
        if path is None:
            continue
        if raw.startswith("+"):
            yield path, lineno, raw[1:]
            lineno += 1
        elif raw.startswith(" "):
            lineno += 1


def scan_diff_text(
    diff_text: str, denylist: Optional[Sequence[str]] = None
) -> List[Finding]:
    deny = load_denylist() if denylist is None else list(denylist)
    findings: List[Finding] = []
    for path, lineno, content in parse_added_lines(diff_text):
        findings.extend(scan_line(path, lineno, content, deny))
    return findings


def _git(args: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def scan_staged(
    cwd: Optional[Path] = None, denylist: Optional[Sequence[str]] = None
) -> List[Finding]:
    """Added lines in the index (what the next commit would add)."""
    root = Path(cwd) if cwd is not None else REPO_ROOT
    diff = _git(
        ["diff", "--cached", "-U0", "--no-color", "--diff-filter=ACMR"], root
    )
    return scan_diff_text(diff, denylist)


def scan_range(
    log_opts: str, cwd: Optional[Path] = None, denylist: Optional[Sequence[str]] = None
) -> List[Finding]:
    """Added lines in commits selected by `log_opts` (e.g. 'base..HEAD').

    Uses `git log -p`, so a line added and later removed inside the range is
    still reported — pushing publishes every commit, not just the final tree.
    """
    root = Path(cwd) if cwd is not None else REPO_ROOT
    diff = _git(
        ["log", "-p", "-U0", "--no-color", "--format=", *shlex.split(log_opts)], root
    )
    return scan_diff_text(diff, denylist)


def scan_tracked_tree(
    cwd: Optional[Path] = None, denylist: Optional[Sequence[str]] = None
) -> List[Finding]:
    """Whole-content scan of every tracked text file (public-release mode)."""
    root = Path(cwd) if cwd is not None else REPO_ROOT
    deny = load_denylist() if denylist is None else list(denylist)
    names = [n for n in _git(["ls-files"], root).splitlines() if n.strip()]
    findings: List[Finding] = []
    for name in names:
        fp = root / name
        try:
            if fp.stat().st_size > MAX_TREE_FILE_BYTES:
                continue
            data = fp.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, content in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(name, lineno, content, deny))
    return findings


HINT = (
    "public-safety: replace with a general expression or [REDACTED] and re-stage. "
    "Only for a deliberate false positive, add 'public-safety: allow' plus a reason "
    "on that line."
)


def format_findings(findings: Iterable[Finding], limit: int = 25) -> List[str]:
    items = list(findings)
    if not items:
        return []
    lines = [f.render() for f in items[:limit]]
    if len(items) > limit:
        lines.append(f"public-safety: (+{len(items) - limit} more findings)")
    lines.append(HINT)
    return lines


def dedupe(findings: Iterable[Finding]) -> List[Finding]:
    seen: Set[Tuple[str, int, str, str]] = set()
    out: List[Finding] = []
    for f in findings:
        key = (f.path, f.line, f.kind, f.masked)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
