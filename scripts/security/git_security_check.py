#!/usr/bin/env python3
"""Pre-commit / pre-push security gate for GraphDistiller.

CLI:
  python scripts/security/git_security_check.py --staged
  python scripts/security/git_security_check.py --tracked
  python scripts/security/git_security_check.py --tracked --push-stdin
  python scripts/security/git_security_check.py --staged --allow-large
      # after explicit user approval for files in [warn, hard) only

Hook (stdin JSON from Cursor beforeShellExecution):
  Reads { "command": "..." }; if git commit|push|gh pr create, runs checks.
  Commit runs gitleaks on the index. Push/PR runs gitleaks on unpushed commits.
  Prints { "permission": "allow"|"deny", ... } to stdout.

Policy:
  ADR-027 — secrets / .env only for path bans; artifacts allowed.
  ADR-028 — file size gates (any path):
    >= hard max (default 50 MiB): always block
    >= warn (default 10 MiB) and < hard: block unless --allow-large / GIT_ALLOW_LARGE=1
  ADR-068 — gitleaks on commit (staged) and push (unpushed commits). Missing binary blocks.
  ADR-072 — repo-root `.gitleaks.toml` (extend defaults + OpenRouter/DB/cookie rules).
  ADR-078 — public-safety scan (public_safety_scan.py): e-mail, phone, account-name
    paths and local deny-list terms in ADDED lines. Commit scans the index, push scans
    unpushed commits. `--public-release` scans the whole tracked tree (report mode for
    the cleanup before a public release).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# public_safety_scan.py sits next to this script; make the import work whether this
# file is run as a script, as a Cursor hook, or loaded by path from a test.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_safety_scan as pss  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults (MiB) — override with GIT_WARN_FILE_BYTES / GIT_HARD_MAX_FILE_BYTES
DEFAULT_WARN_BYTES = 10 * 1024 * 1024
DEFAULT_HARD_BYTES = 50 * 1024 * 1024

FORBIDDEN_PATH_EXACT = {
    ".env",
}

ALLOWED_KEEP = {
    ".env.example",
}

SECRET_PATTERNS = [
    # Env-style exact name only (no re.I) — avoids Settings field
    # openrouter_api_key=_env_get(...) false positives.
    re.compile(r"OPENROUTER_API_KEY\s*=\s*(?!\S*\.\.\.)\S{8,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    # Negative lookbehind: do not match inside identifiers like openrouter_api_key=.
    re.compile(
        r"(?i)(?<![A-Za-z0-9_])(api[_-]?key|secret|token|password|passwd|passphrase)\s*[:=]\s*"
        r"['\"]?(?!\S*\.\.\.)[A-Za-z0-9_\-]{8,}"
    ),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)(aws_secret_access_key|private_key)\s*[:=]\s*\S{8,}"),
]


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip()


def _parse_positive_int(raw: Optional[str], default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip().replace("_", ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _dotenv_get(key: str) -> Optional[str]:
    """Read a single key from repo .env without loading secrets into the process broadly."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        if k.strip() == key:
            return v.strip().strip("'").strip('"')
    return None


def size_limits() -> Tuple[int, int]:
    """Return (warn_bytes, hard_bytes). Env wins over .env file over defaults."""
    warn = _parse_positive_int(
        os.environ.get("GIT_WARN_FILE_BYTES") or _dotenv_get("GIT_WARN_FILE_BYTES"),
        DEFAULT_WARN_BYTES,
    )
    hard = _parse_positive_int(
        os.environ.get("GIT_HARD_MAX_FILE_BYTES")
        or _dotenv_get("GIT_HARD_MAX_FILE_BYTES"),
        DEFAULT_HARD_BYTES,
    )
    if warn > hard:
        warn = hard
    return warn, hard


def allow_large_approved(argv: Optional[Sequence[str]] = None) -> bool:
    argv = list(argv or sys.argv[1:])
    if "--allow-large" in argv:
        return True
    flag = (os.environ.get("GIT_ALLOW_LARGE") or "").strip().lower()
    return flag in {"1", "true", "yes", "y"}


def _is_allowed_keep(path: str) -> bool:
    n = _norm(path)
    return n in {_norm(p) for p in ALLOWED_KEEP}


def _is_forbidden_path(path: str) -> bool:
    n = _norm(path)
    if _is_allowed_keep(n):
        return False
    if n in {_norm(p) for p in FORBIDDEN_PATH_EXACT}:
        return True
    if n.startswith(".env.") and n != ".env.example":
        return True
    if n == ".env" or n.endswith("/.env"):
        return True
    return False


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def staged_paths() -> List[str]:
    out = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def tracked_paths() -> List[str]:
    out = _git(["ls-files"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def file_size_bytes(path: str) -> Optional[int]:
    """Best-effort size for a repo-relative path (working tree, else staged blob)."""
    n = _norm(path)
    wt = REPO_ROOT / n
    sizes: List[int] = []
    try:
        if wt.is_file():
            sizes.append(int(wt.stat().st_size))
    except OSError:
        pass
    cat = _git(["cat-file", "-s", f":{n}"]).strip()
    if cat.isdigit():
        sizes.append(int(cat))
    if not sizes:
        return None
    return max(sizes)


def _fmt_mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB"


def scan_file_sizes(
    paths: Iterable[str],
    *,
    prefix: str,
    allow_large: bool,
) -> List[str]:
    warn_b, hard_b = size_limits()
    problems: List[str] = []
    for path in paths:
        size = file_size_bytes(path)
        if size is None:
            continue
        if size >= hard_b:
            problems.append(
                f"{prefix} file too large (hard max {_fmt_mib(hard_b)}): "
                f"{path} ({_fmt_mib(size)}) — never commit; unstage"
            )
        elif size >= warn_b and not allow_large:
            problems.append(
                f"{prefix} file needs user approval "
                f"(>={_fmt_mib(warn_b)}, <{_fmt_mib(hard_b)}): "
                f"{path} ({_fmt_mib(size)}) — ask user, then "
                f"--allow-large or GIT_ALLOW_LARGE=1"
            )
    return problems


def scan_staged_blobs(paths: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for path in paths:
        if _is_forbidden_path(path):
            continue
        blob = _git(["show", f":{path}"])
        if not blob:
            continue
        sample = blob[:200_000]
        for pat in SECRET_PATTERNS:
            if pat.search(sample):
                hits.append(f"{path}: matched secret pattern {pat.pattern!r}")
                break
    return hits


def _find_gitleaks() -> Optional[str]:
    found = shutil.which("gitleaks")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    if not local:
        return None
    root = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not root.is_dir():
        return None
    matches = sorted(root.glob("Gitleaks.Gitleaks_*/gitleaks.exe"))
    return str(matches[-1]) if matches else None


def _summarize_gitleaks(text: str) -> str:
    """Rule and path only. Never echo a Secret/Match line."""
    rules: List[str] = []
    files: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Secret:") or line.startswith("Match:"):
            continue
        if line.startswith("RuleID:"):
            rules.append(line.split(":", 1)[1].strip())
        elif line.startswith("File:"):
            files.append(line.split(":", 1)[1].strip())
    if not rules and not files:
        return "gitleaks: findings (details redacted)"
    pairs: List[str] = []
    for i in range(max(len(rules), len(files))):
        rule = rules[i] if i < len(rules) else "?"
        path = files[i] if i < len(files) else "?"
        pairs.append(f"{path} ({rule})")
    shown = "; ".join(pairs[:8])
    extra = f" (+{len(pairs) - 8} more)" if len(pairs) > 8 else ""
    return f"gitleaks: secret pattern in {shown}{extra}"


def _gitleaks_config_args() -> List[str]:
    """Prefer repo `.gitleaks.toml` when present (ADR-072)."""
    cfg = REPO_ROOT / ".gitleaks.toml"
    if cfg.is_file():
        return ["--config", str(cfg)]
    return []


def _gitleaks_problems(mode_args: Sequence[str]) -> List[str]:
    """Scan one range. mode_args is --staged or --log-opts + range. Never a full-history scan."""
    if not mode_args:
        return ["gitleaks: no scan range (blocked)"]
    exe = _find_gitleaks()
    if not exe:
        return ["gitleaks: not on PATH (commit/push blocked)"]
    cmd = [
        exe,
        "git",
        "--no-banner",
        "--redact",
        "--no-color",
        "--timeout",
        "180",
        *_gitleaks_config_args(),
        *mode_args,
        str(REPO_ROOT),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=200,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return ["gitleaks: timed out"]
    except OSError as err:
        return [f"gitleaks: failed to start ({err})"]
    if proc.returncode == 0:
        return []
    return [_summarize_gitleaks(proc.stdout + "\n" + proc.stderr)]


def _is_zero_sha(sha: str) -> bool:
    return bool(sha) and set(sha) == {"0"}


def log_opts_from_push_lines(text: str) -> List[str]:
    """pre-push stdin: local_ref local_sha remote_ref remote_sha."""
    opts: List[str] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        _local_ref, local_sha, _remote_ref, remote_sha = parts[:4]
        if _is_zero_sha(local_sha):
            continue
        if _is_zero_sha(remote_sha):
            opts.append(f"{local_sha} --not --remotes")
        else:
            opts.append(f"{remote_sha}..{local_sha}")
    return opts


def unpushed_log_opts() -> Optional[str]:
    """Commits that are not on the upstream yet. None means nothing new to scan."""
    upstream = _git(["rev-parse", "--verify", "@{upstream}"]).strip()
    base = upstream
    if not base:
        for cand in ("origin/HEAD", "origin/main", "origin/master"):
            if _git(["rev-parse", "--verify", cand]).strip():
                base = cand
                break
    if base:
        count = _git(["rev-list", "--count", f"{base}..HEAD"]).strip()
        if count in {"", "0"}:
            return None
        return f"{base}..HEAD"
    count = _git(["rev-list", "--count", "HEAD", "--not", "--remotes"]).strip()
    if count in {"", "0"}:
        return None
    return "HEAD --not --remotes"


def run_checks(
    *,
    staged: bool,
    tracked: bool,
    allow_large: bool = False,
    gitleaks_staged: bool = False,
    gitleaks_unpushed: bool = False,
    push_stdin: Optional[str] = None,
    public_safety_staged: bool = False,
    public_safety_unpushed: bool = False,
) -> List[str]:
    problems: List[str] = []
    if staged:
        paths = staged_paths()
        bad = [p for p in paths if _is_forbidden_path(p)]
        for p in bad:
            problems.append(f"staged forbidden path: {p}")
        problems.extend(scan_staged_blobs(paths))
        problems.extend(
            scan_file_sizes(paths, prefix="staged", allow_large=allow_large)
        )
    if tracked:
        paths = tracked_paths()
        bad = [p for p in paths if _is_forbidden_path(p)]
        for p in bad:
            problems.append(f"tracked forbidden path: {p}")
        # Push gate: hard-max always; warn-band requires allow_large too
        problems.extend(
            scan_file_sizes(paths, prefix="tracked", allow_large=allow_large)
        )
    if gitleaks_staged:
        problems.extend(_gitleaks_problems(["--staged"]))
    ranges: List[str] = []
    if gitleaks_unpushed or public_safety_unpushed:
        if push_stdin is not None and push_stdin.strip():
            ranges = log_opts_from_push_lines(push_stdin)
        else:
            one = unpushed_log_opts()
            ranges = [one] if one else []
    if gitleaks_unpushed:
        for opts in ranges:
            problems.extend(_gitleaks_problems(["--log-opts", opts]))
    # ADR-078: personal-information expressions in ADDED lines (not secret values).
    ps_findings: List["pss.Finding"] = []
    if public_safety_staged:
        ps_findings.extend(pss.scan_staged(REPO_ROOT))
    if public_safety_unpushed:
        for opts in ranges:
            ps_findings.extend(pss.scan_range(opts, REPO_ROOT))
    problems.extend(pss.format_findings(pss.dedupe(ps_findings)))
    return problems


def _looks_like_git_publish(command: str) -> bool:
    c = command.lower()
    if re.search(r"\bgit\s+commit\b", c):
        return True
    if re.search(r"\bgit\s+push\b", c):
        return True
    if re.search(r"\bgh\s+pr\s+create\b", c):
        return True
    return False


def hook_main(payload: dict) -> dict:
    command = str(payload.get("command") or "")
    if not _looks_like_git_publish(command):
        return {"permission": "allow"}

    # Hook never auto-approves large files; agent must get user OK then CLI --allow-large.
    # gitleaks: commit scans the index; push/PR scans commits not on the remote yet.
    lowered = command.lower()
    is_commit = re.search(r"\bgit\s+commit\b", lowered) is not None
    is_push = (
        re.search(r"\bgit\s+push\b", lowered) is not None
        or re.search(r"\bgh\s+pr\s+create\b", lowered) is not None
    )
    if not is_commit and not is_push:
        is_commit = True
        is_push = True
    problems = run_checks(
        staged=True,
        tracked=True,
        allow_large=False,
        gitleaks_staged=is_commit,
        gitleaks_unpushed=is_push,
        # 커밋·푸시 훅에서는 공개 부적합 표현 스캔을 끈다.
        # 공개 전환 시에는 `--public-release`를 수동 실행한다.
        public_safety_staged=False,
        public_safety_unpushed=False,
    )
    if not problems:
        return {"permission": "allow"}

    detail = "; ".join(problems[:12])
    if len(problems) > 12:
        detail += f" (+{len(problems) - 12} more)"
    msg = (
        "Git security gate blocked commit/push. "
        "Fix secrets/.env or personal-information expressions (public-safety), "
        "or handle large files "
        "(>=50MiB forever blocked; >=10MiB needs user approval + --allow-large). "
        f"Issues: {detail}"
    )
    return {
        "permission": "deny",
        "user_message": msg,
        "agent_message": msg,
    }


def public_release_main() -> int:
    """Whole tracked-tree public-safety scan (report mode before a public release)."""
    findings = pss.dedupe(pss.scan_tracked_tree(REPO_ROOT))
    if not findings:
        print("public-release scan OK (no public-safety findings in tracked files)")
        return 0
    by_kind: dict = {}
    by_file: dict = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
        by_file[f.path] = by_file.get(f.path, 0) + 1
    print(f"PUBLIC-RELEASE SCAN: {len(findings)} findings", file=sys.stderr)
    print(
        "  by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())),
        file=sys.stderr,
    )
    top = sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    print("  top files: " + ", ".join(f"{n}({c})" for n, c in top), file=sys.stderr)
    for line in pss.format_findings(findings, limit=60):
        print(f"  - {line}", file=sys.stderr)
    return 1


def cli_main(argv: List[str]) -> int:
    if "--public-release" in argv:
        return public_release_main()
    staged = "--staged" in argv
    tracked = "--tracked" in argv
    if "--staged" not in argv and "--tracked" not in argv:
        staged = True
        tracked = False
    allow_large = allow_large_approved(argv)
    push_stdin: Optional[str] = None
    if "--push-stdin" in argv:
        try:
            push_stdin = "" if sys.stdin.isatty() else sys.stdin.read()
        except OSError:
            push_stdin = ""
    problems = run_checks(
        staged=staged,
        tracked=tracked or False,
        allow_large=allow_large,
        gitleaks_staged=staged,
        gitleaks_unpushed=tracked or False,
        push_stdin=push_stdin,
        # 커밋·푸시 훅에서는 끈다 (전수 스캔은 --public-release로 수동 실행).
        public_safety_staged=False,
        public_safety_unpushed=False,
    )
    if problems:
        print("SECURITY CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        warn_b, hard_b = size_limits()
        print(
            f"(limits: warn>={_fmt_mib(warn_b)}, hard>={_fmt_mib(hard_b)}; "
            f"allow_large={allow_large})",
            file=sys.stderr,
        )
        return 1
    warn_b, hard_b = size_limits()
    print(
        f"security check OK "
        f"(warn>={_fmt_mib(warn_b)}, hard>={_fmt_mib(hard_b)}, "
        f"allow_large={allow_large})"
    )
    return 0


def main() -> int:
    if any(
        a
        in (
            "--staged",
            "--tracked",
            "--allow-large",
            "--push-stdin",
            "--public-release",
            "--help",
            "-h",
        )
        for a in sys.argv[1:]
    ):
        if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
            print(
                "Usage: git_security_check.py [--staged] [--tracked] "
                "[--push-stdin] [--allow-large] [--public-release]\n"
                "Env: GIT_WARN_FILE_BYTES GIT_HARD_MAX_FILE_BYTES GIT_ALLOW_LARGE\n"
                f"Defaults: warn={DEFAULT_WARN_BYTES} hard={DEFAULT_HARD_BYTES}\n"
                "gitleaks: --staged scans the index; --tracked scans unpushed commits.\n"
                "--push-stdin reads pre-push ref lines from stdin.\n"
                "public-safety (ADR-078): --staged/--tracked also scan ADDED lines for "
                "e-mail/phone/account paths/deny-list terms; --public-release scans the "
                "whole tracked tree (report mode)."
            )
            return 0
        return cli_main(sys.argv)

    raw = ""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
    except OSError:
        raw = ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    result = hook_main(payload)
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    return 0 if result.get("permission") == "allow" else 2


if __name__ == "__main__":
    raise SystemExit(main())
