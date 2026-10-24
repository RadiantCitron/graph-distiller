"""Unit: scripts/security/public_safety_scan.py + its wiring in git_security_check.py (ADR-078).

No network, no LLM. Real `git` is used only inside a temp repo. Fake personal data is
assembled in code (never written as literals) because this very file is scanned when it
is committed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

HOOKS_PRIMARY = Path(__file__).resolve().parents[2] / "scripts" / "security"
HOOKS_LEGACY = Path(__file__).resolve().parents[2] / ".cursor" / "hooks"
HOOKS = HOOKS_PRIMARY if HOOKS_PRIMARY.exists() else HOOKS_LEGACY


def _load(name: str):
    target = (HOOKS_PRIMARY / f"{name}.py") if (HOOKS_PRIMARY / f"{name}.py").exists() else (HOOKS_LEGACY / f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pss = _load("public_safety_scan")
gsc = _load("git_security_check")

# --- fake personal data, assembled so no literal hit exists in this file -------------
AT = "@"
EMAIL = "alice" + AT + "corp-" + "mail.io"
PHONE = "010" + "-" + "2345" + "-" + "6789"
RRN_VERIFIED = "900101" + "-" + "1234568"
RRN_UNVERIFIED = "021231" + "-" + "3456780"
WIN_PATH = "C:" + "/Users/" + "alice" + "/proj/file.txt"
POSIX_PATH = "/home/" + "alice" + "/proj"
TERM = "Project" + "Nightingale"


def _kinds(findings):
    return sorted(f.kind for f in findings)


# --- single-line detection ------------------------------------------------------------
def test_email_detected_and_allowed_forms_are_not() -> None:
    assert _kinds(pss.scan_line("a.md", 1, f"contact {EMAIL} now")) == ["email"]
    for ok in (
        "noreply" + AT + "users.noreply.github.com",
        "someone" + AT + "example.com",
        "git" + AT + "github.com:owner/repo.git",
        "pkg" + AT + "1.2.3",
        "dev" + AT + "host.invalid",
    ):
        assert pss.scan_line("a.md", 1, f"x {ok} y") == [], ok


def test_phone_detected_digit_runs_are_not() -> None:
    assert _kinds(pss.scan_line("a.md", 1, f"call {PHONE}")) == ["phone"]
    assert pss.scan_line("a.md", 1, "id 20261001123456 ok") == []
    assert pss.scan_line("a.md", 1, "no phone: 123-4567") == []


def test_rrn_detected_with_checksum_high_confidence() -> None:
    assert _kinds(pss.scan_line("a.md", 1, f"rrn {RRN_VERIFIED}")) == ["rrn:high-confidence"]
    compact_verified = "900101" + "1234568"
    assert _kinds(pss.scan_line("a.md", 1, f"rrn {compact_verified}")) == ["rrn:high-confidence"]


def test_rrn_detected_without_checksum_reported_as_rrn() -> None:
    assert _kinds(pss.scan_line("a.md", 1, f"rrn {RRN_UNVERIFIED}")) == ["rrn"]


def test_rrn_invalid_calendar_dates_and_genders_are_not_detected() -> None:
    for invalid in (
        "901301" + "-1234567",  # month 13
        "900001" + "-1234567",  # month 00
        "900132" + "-1234567",  # day 32
        "900431" + "-1234567",  # day 31 in April
        "010229" + "-3234567",  # 2001 Feb 29 (non-leap year)
        "900101" + "-5234567",  # gender 5
        "900101" + "-0234567",  # gender 0
    ):
        assert pss.scan_line("a.md", 1, f"bad {invalid} x") == [], invalid

    # 2000 Feb 29 is a leap year -> should be detected
    leap_ok = "000229" + "-3234567"
    kinds = _kinds(pss.scan_line("a.md", 1, f"ok {leap_ok}"))
    assert len(kinds) == 1 and kinds[0].startswith("rrn")


def test_rrn_digit_runs_embedded_are_not_detected() -> None:
    embedded = "99" + RRN_VERIFIED + "88"
    assert pss.scan_line("a.md", 1, f"id {embedded}") == []


def test_rrn_masked_and_allow_marker() -> None:
    findings = pss.scan_line("a.md", 1, f"val: {RRN_VERIFIED}")
    assert len(findings) == 1
    assert RRN_VERIFIED not in findings[0].render()
    assert findings[0].masked.startswith("90***")
    # Allow marker
    assert pss.scan_line("a.md", 1, f"val: {RRN_VERIFIED} # public-safety: allow test") == []


def test_account_path_detected_placeholders_are_not() -> None:
    assert _kinds(pss.scan_line("a.md", 1, f"see {WIN_PATH}")) == ["account-path"]
    assert _kinds(pss.scan_line("a.md", 1, f"see {POSIX_PATH}")) == ["account-path"]
    wsl = "/mnt/" + "c" + "/Users/" + "alice" + "/x"
    assert _kinds(pss.scan_line("a.md", 1, f"see {wsl}")) == ["account-path"]
    for ok in (
        "src/" + "home/" + "alice" + "/page.tsx",  # relative path, not an account path
        "C:" + "/Users/" + "<user>" + "/x",
        "C:" + "/Users/" + "Public" + "/x",
        "/home/" + "runner" + "/work",
        "/home/" + "user" + "/x",
    ):
        assert pss.scan_line("a.md", 1, ok) == [], ok


def test_denylist_matches_case_insensitively_and_hides_the_term() -> None:
    deny = [TERM.lower()]
    found = pss.scan_line("a.md", 3, f"about {TERM.upper()} here", deny)
    assert _kinds(found) == ["denylist"]
    assert TERM.lower() not in found[0].render().lower()  # term is never echoed
    assert pss.scan_line("a.md", 3, "nothing here", deny) == []


def test_allow_marker_skips_the_line() -> None:
    line = f"{EMAIL}  # {pss.ALLOW_MARKER}: test fixture"
    assert pss.scan_line("a.md", 1, line) == []


def test_findings_are_masked() -> None:
    f = pss.scan_line("a.md", 1, f"mail {EMAIL}")[0]
    assert EMAIL not in f.render()
    assert "email" in f.render() and "a.md:1" in f.render()


def test_load_denylist_ignores_blank_and_comments_and_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "deny.txt"
    p.write_text(f"# note\n\n{TERM}\n  second term  \n", encoding="utf-8")
    assert pss.load_denylist(p) == [TERM.lower(), "second term"]
    assert pss.load_denylist(tmp_path / "missing.txt") == []


# --- diff parsing ---------------------------------------------------------------------
DIFF = "\n".join(
    [
        "diff --git a/docs/x.md b/docs/x.md",
        "--- a/docs/x.md",
        "+++ b/docs/x.md",
        "@@ -1,2 +10,3 @@",
        "-removed line",
        "+first added",
        "+second added",
        "diff --git a/gone.md b/gone.md",
        "--- a/gone.md",
        "+++ /dev/null",
        "@@ -1 +0,0 @@",
        "-bye",
        "diff --git a/src/y.py b/src/y.py",
        "--- a/src/y.py",
        "+++ b/src/y.py",
        "@@ -5 +7 @@",
        "+third added",
        "Binary files a/img.png and b/img.png differ",
    ]
)


def test_parse_added_lines_tracks_path_and_line_numbers() -> None:
    got = list(pss.parse_added_lines(DIFF))
    assert got == [
        ("docs/x.md", 10, "first added"),
        ("docs/x.md", 11, "second added"),
        ("src/y.py", 7, "third added"),
    ]


def test_scan_diff_text_reports_only_added_lines() -> None:
    diff = DIFF.replace("second added", f"mail {EMAIL}").replace("-removed line", f"-old {EMAIL}")
    f = pss.scan_diff_text(diff, [])
    assert [(x.path, x.line, x.kind) for x in f] == [("docs/x.md", 11, "email")]


def test_format_findings_adds_one_hint_and_caps() -> None:
    items = [pss.Finding("a", i, "email", "al***") for i in range(30)]
    out = pss.format_findings(items, limit=5)
    assert len(out) == 5 + 2  # 5 rows + "+25 more" + hint
    assert out[-1] == pss.HINT
    assert pss.format_findings([]) == []


# --- real git in a temp repo ----------------------------------------------------------
def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "user.email", "t" + AT + "example.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "base.txt").write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_scan_staged_sees_only_what_is_staged(repo: Path) -> None:
    (repo / "bad.txt").write_text(f"mail {EMAIL}\n", encoding="utf-8")
    (repo / "unstaged.txt").write_text(f"phone {PHONE}\n", encoding="utf-8")
    _git(repo, "add", "bad.txt")
    found = pss.scan_staged(repo, [])
    assert [(f.path, f.kind) for f in found] == [("bad.txt", "email")]


def test_scan_range_reports_lines_added_then_removed(repo: Path) -> None:
    (repo / "t.txt").write_text(f"mail {EMAIL}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "adds")
    (repo / "t.txt").write_text("fixed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "removes")
    found = pss.scan_range("HEAD~2..HEAD", repo, [])
    assert [(f.path, f.kind) for f in found] == [("t.txt", "email")]
    assert pss.scan_range("HEAD..HEAD", repo, []) == []


def test_scan_tracked_tree_scans_content_and_skips_binary(repo: Path) -> None:
    (repo / "old.md").write_text(f"x\n{TERM}\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01" + EMAIL.encode())
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "more")
    found = pss.scan_tracked_tree(repo, [TERM.lower()])
    assert [(f.path, f.line, f.kind) for f in found] == [("old.md", 2, "denylist")]


# --- wiring into git_security_check.run_checks ----------------------------------------
def test_run_checks_blocks_staged_personal_info_and_allows_clean(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gsc, "REPO_ROOT", repo)
    monkeypatch.setattr(pss, "DENYLIST_PATH", repo / "none.txt")
    (repo / "ok.md").write_text("all good\n", encoding="utf-8")
    _git(repo, "add", "ok.md")
    assert gsc.run_checks(staged=True, tracked=False, public_safety_staged=True) == []

    (repo / "bad.md").write_text(f"call {PHONE}\n", encoding="utf-8")
    _git(repo, "add", "bad.md")
    problems = gsc.run_checks(staged=True, tracked=False, public_safety_staged=True)
    assert any("public-safety [phone] bad.md:1" in p for p in problems)
    assert PHONE not in " ".join(problems)


def test_run_checks_scans_unpushed_range_only_when_asked(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gsc, "REPO_ROOT", repo)
    monkeypatch.setattr(pss, "DENYLIST_PATH", repo / "none.txt")
    (repo / "t.md").write_text(f"mail {EMAIL}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "adds")
    stdin = f"refs/heads/main {'a' * 40} refs/heads/main {'0' * 40}\n"
    # zero remote sha -> "<local> --not --remotes"; build the real local sha instead
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    base = subprocess.run(
        ["git", "rev-parse", "HEAD~1"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    stdin = f"refs/heads/main {head} refs/heads/main {base}\n"
    hit = gsc.run_checks(
        staged=False, tracked=True, push_stdin=stdin, public_safety_unpushed=True
    )
    assert any("public-safety [email] t.md:1" in p for p in hit)
    off = gsc.run_checks(staged=False, tracked=True, push_stdin=stdin)
    assert not any("public-safety" in p for p in off)


def test_public_release_main_reports_and_fails(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(gsc, "REPO_ROOT", repo)
    monkeypatch.setattr(pss, "DENYLIST_PATH", repo / "none.txt")
    assert gsc.public_release_main() == 0
    (repo / "leak.md").write_text(f"{EMAIL}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "leak")
    assert gsc.public_release_main() == 1
    err = capsys.readouterr().err
    assert "1 findings" in err and "email=1" in err and EMAIL not in err
