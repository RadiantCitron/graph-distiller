#!/usr/bin/env python3
"""install_to_repo.py — 원클릭 Git 보안 및 에이전트 거버넌스 전파 도구.

다른 프로젝트 저장소에 Git 보안 체계(gitleaks, PII 정밀 스캐너, Git Core Hooks)와
에이전트 거버넌스 규칙(AGENTS.md, docs/plans/)을 안전하게 이식합니다.

원칙:
  - 대상 저장소의 소스코드(src/, app/, lib/ 등)는 단 1바이트도 건드리지 않습니다.
  - 보안 파일, 훅 스크립트, 거버넌스 문서만 정밀하게 배포합니다.

사용법:
  python scripts/security/install_to_repo.py "D:/path/to/target-repo"
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]

SECURITY_FILES = [
    ("scripts/security/git_security_check.py", "scripts/security/git_security_check.py"),
    ("scripts/security/public_safety_scan.py", "scripts/security/public_safety_scan.py"),
    (".githooks/pre-commit", ".githooks/pre-commit"),
    (".githooks/pre-push", ".githooks/pre-push"),
    (".gitleaks.toml", ".gitleaks.toml"),
    ("docs/design/git-security-architecture.md", "docs/design/git-security-architecture.md"),
]

GOVERNANCE_SNIPPET = """
## 에이전트 거버넌스 및 자산화 규칙 (공통 강제)

- **비판적 페어 프로그래밍 & 오버엔지니어링 방지 (Critique First)**:
  - 에이전트는 사용자의 지시를 맹목적으로 추종해 코드를 불리지 않는다 (Yes-man 금지).
  - 사용자의 요구라도 YAGNI(오버엔지니어링)·주객전도(Bikeshedding)·복잡성 과잉이 발생하지 않도록
    **항상 균형 잡힌 엔지니어링 시각을 제시하고 직언하여 오버엔지니어링을 사전에 방지한다.**
  - 특히 「보안을 위해 외부 LLM을 부르는 것」은 그 호출 자체가 유출(보안 파산)이므로 원천 금지한다.
- **계획 문서의 저장소 내 자산화 (Preserve Decision Context)**:
  - 에이전트가 작성한 계획 문서(brain/ 등)는 단순 임시 데이터가 아니라 분석과 의사결정 맥락이 담긴 귀중한 지적 자산이다.
  - 사용자가 다른 PC나 작업 환경으로 이동하더라도 맥락이 끊기지 않도록, 툴 임시 캐시에 방치하지 않고 저장소 내에 영구 보존한다:
    - 세션별/리팩토링/기능 계획 문서: `docs/plans/` (날짜별 디렉토리 등)
"""


def install_to_repo(target_dir: Path | str) -> int:
    target = Path(target_dir).resolve()
    print(f"=== Git 보안 및 에이전트 거버넌스 전파 시작 ===")
    print(f"소스 저장소: {SOURCE_ROOT}")
    print(f"대상 저장소: {target}\n")

    if not target.exists() or not target.is_dir():
        print(f"[오류] 대상 디렉터리가 존재하지 않습니다: {target}", file=sys.stderr)
        return 1

    # 1. 보안 및 아키텍처 파일 복사
    print("[1/4] 보안 인프라 및 가이드 문서 복사 중...")
    for src_rel, dst_rel in SECURITY_FILES:
        src_path = SOURCE_ROOT / src_rel
        dst_path = target / dst_rel
        if not src_path.exists():
            print(f"  - [경고] 소스 파일 없음: {src_rel}")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"  + 복사 완료: {dst_rel}")

    # 2. docs/plans 디렉터리 보장
    print("\n[2/4] 계획서 자체저장 서랍(docs/plans/) 생성 중...")
    plans_dir = target / "docs" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = plans_dir / ".gitkeep"
    if not any(plans_dir.iterdir()):
        gitkeep.touch()
    print(f"  + 디렉터리 준비 완료: docs/plans/")

    # 3. AGENTS.md 점검 및 거버넌스 지침 주입
    print("\n[3/4] AGENTS.md 점검 및 거버넌스 지침 확인 중...")
    agents_file = target / "AGENTS.md"
    if agents_file.exists():
        content = agents_file.read_text(encoding="utf-8")
        if "비판적 페어 프로그래밍" not in content and "Critique First" not in content:
            print("  + AGENTS.md에 비판적 직언 및 계획문서 자산화 규칙 추가")
            with open(agents_file, "a", encoding="utf-8") as f:
                f.write(GOVERNANCE_SNIPPET)
        else:
            print("  - AGENTS.md에 이미 거버넌스 지침이 포함되어 있습니다 (유지).")
    else:
        print("  + AGENTS.md 신규 생성 (거버넌스 지침 포함)")
        agents_file.write_text("# AGENTS.md\n" + GOVERNANCE_SNIPPET, encoding="utf-8")

    # 4. Git Core Hooks 안내
    print("\n[4/4] Git Core Hooks 활성화 안내:")
    print("  대상 저장소에서 아래 명령어를 실행하여 Git 표준 훅을 활성화하십시오:")
    print(f"  git -C \"{target}\" config core.hooksPath .githooks")

    print("\n=== 전파 완료: 소스코드는 일체 수정되지 않았습니다. ===")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/security/install_to_repo.py <대상_저장소_경로>")
        sys.exit(1)

    sys.exit(install_to_repo(sys.argv[1]))
