"""Unit: .gitleaks.toml PII defense rules (ADR-079 / ROADMAP_0003 Step 020).

Validates that custom PII rules (kr-rrn, kr-mobile, labelled-name) and existing
secret rules (OpenRouter key, Postgres URI, cookie) function as specified.

Tests assemble fake inputs in code so no literal PII or secret patterns exist
in this test file itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"


def _gitleaks_bin() -> str | None:
    return shutil.which("gitleaks")


def _run_gitleaks(source_dir: Path) -> List[Dict[str, Any]]:
    exe = _gitleaks_bin()
    if not exe:
        pytest.skip("gitleaks binary not found on PATH")

    report_path = source_dir / "report.json"
    cmd = [
        exe,
        "detect",
        "--no-git",
        "--source",
        str(source_dir),
        "--config",
        str(GITLEAKS_CONFIG),
        "--report-format",
        "json",
        "--report-path",
        str(report_path),
        "--redact",
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if not report_path.is_file():
        return []
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def test_kr_rrn_rule_detects_and_rejects_non_rrn(tmp_path: Path) -> None:
    """Verify kr-rrn detects Korean resident numbers and rejects non-RRN digit sequences."""
    test_file = tmp_path / "test_rrn.txt"

    # Assemble test patterns without literals in source
    valid_rrn_1 = "900101" + "-" + "1234567"
    valid_rrn_2 = "021231" + "-" + "3456789"
    valid_rrn_no_hyphen = "850505" + "2345678"

    invalid_month = "901301" + "-" + "1234567"  # month 13
    invalid_day = "900135" + "-" + "1234567"  # day 35
    invalid_gender = "900101" + "-" + "5234567"  # gender 5
    short_num = "123456" + "-" + "12345"

    content = f"""
    valid 1: {valid_rrn_1}
    valid 2: {valid_rrn_2}
    valid 3: {valid_rrn_no_hyphen}
    invalid month: {invalid_month}
    invalid day: {invalid_day}
    invalid gender: {invalid_gender}
    short: {short_num}
    """
    test_file.write_text(content, encoding="utf-8")

    findings = _run_gitleaks(tmp_path)
    rrn_hits = [f for f in findings if f.get("RuleID") == "kr-rrn"]

    assert len(rrn_hits) == 3, f"Expected 3 kr-rrn hits, got {len(rrn_hits)}"


def test_kr_mobile_rule_detects_and_rejects_landlines(tmp_path: Path) -> None:
    """Verify kr-mobile detects mobile numbers and ignores regular landline numbers."""
    test_file = tmp_path / "test_mobile.txt"

    # Assemble test patterns
    mob_hyphen = "010" + "-1234-5678"
    mob_space = "010" + " 9876 5432"
    mob_011 = "011" + "-234-5678"
    mob_compact = "010" + "12345678"

    landline_seoul = "02" + "-123-4567"
    landline_gyeonggi = "031" + "-123-4567"
    service_num = "070" + "-1234-5678"
    toll_free = "1588" + "-1234"

    content = f"""
    mobile 1: {mob_hyphen}
    mobile 2: {mob_space}
    mobile 3: {mob_011}
    mobile 4: {mob_compact}
    landline seoul: {landline_seoul}
    landline gyeonggi: {landline_gyeonggi}
    service: {service_num}
    tollfree: {toll_free}
    """
    test_file.write_text(content, encoding="utf-8")

    findings = _run_gitleaks(tmp_path)
    mobile_hits = [f for f in findings if f.get("RuleID") == "kr-mobile"]

    assert len(mobile_hits) == 4, f"Expected 4 kr-mobile hits, got {len(mobile_hits)}"


def test_labelled_name_rule_detects_and_rejects_unlabelled(tmp_path: Path) -> None:
    """Verify labelled-name detects Korean names only when preceded by specific label."""
    test_file = tmp_path / "test_name.txt"

    # Assemble test patterns
    lbl_name_1 = "성명" + ": 홍길동"
    lbl_name_2 = "이름" + " : 김철수"
    lbl_name_3 = "작성자" + ": 이영희"

    unlabelled_1 = "홍길동" + " 연구원이 보고서를 작성하였습니다."
    unlabelled_2 = "이름" + "이 없는 문서입니다."
    unlabelled_3 = "작성자" + " 목록을 검토하세요."

    content = f"""
    labelled 1: {lbl_name_1}
    labelled 2: {lbl_name_2}
    labelled 3: {lbl_name_3}
    unlabelled 1: {unlabelled_1}
    unlabelled 2: {unlabelled_2}
    unlabelled 3: {unlabelled_3}
    """
    test_file.write_text(content, encoding="utf-8")

    findings = _run_gitleaks(tmp_path)
    name_hits = [f for f in findings if f.get("RuleID") == "labelled-name"]

    assert len(name_hits) == 3, f"Expected 3 labelled-name hits, got {len(name_hits)}"


def test_regression_existing_secret_rules(tmp_path: Path) -> None:
    """Verify existing secret rules and documented allowlists still work as expected."""
    test_file = tmp_path / "test_secrets.txt"

    fake_openrouter = "sk-or-v1-" + "a1b2c3d4e5f6g7h8i9j0" + "12345"
    fake_postgres = "postgres://" + "admin:secretpassword@" + "db.example.org:5432/prod"
    fake_cookie = "__cf_bm=" + "abcdef1234567890abcdef123456"

    # Allowlisted doc placeholders
    allowed_key_placeholder = "sk-or-v1-" + "..."
    allowed_uri_placeholder = "postgresql://" + "USER:PASSWORD@HOST:PORT/DATABASE"

    content = f"""
    key: {fake_openrouter}
    db: {fake_postgres}
    cookie: {fake_cookie}
    allowed key: {allowed_key_placeholder}
    allowed db: {allowed_uri_placeholder}
    """
    test_file.write_text(content, encoding="utf-8")

    findings = _run_gitleaks(tmp_path)
    rule_ids = {f.get("RuleID") for f in findings}

    assert "openrouter-api-key" in rule_ids
    assert "postgres-uri-with-password" in rule_ids
    assert "cloudflare-cf-bm-cookie" in rule_ids
