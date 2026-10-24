# scripts/security/ — Git 보안 및 에이전트 거버넌스 도구 가이드

이 폴더는 저장소의 **Git 커밋/푸시 보안**과 **AI 에이전트 거버넌스(지침 전파)**를 담당하는 중립적 도구 모음입니다.

---

## 1. 이 폴더의 파일들이 하는 일

| 파일 | 역할 | 특징 |
|---|---|---|
| [`git_security_check.py`](./git_security_check.py) | **Git 보안 메인 게이트** | `git commit` / `git push` 직전에 자동 실행되어 API 키, 비밀번호, .env, 10MB+ 대용량 파일 커밋을 기계적으로 차단 |
| [`public_safety_scan.py`](./public_safety_scan.py) | **한국형 개인정보 정밀 스캐너** | 주민등록번호(2020년 10월 임의번호 개편 체계 반영), 국내 휴대전화번호, 계정명 경로 탐지 |
| [`install_to_repo.py`](./install_to_repo.py) | **원클릭 다른 레포 전파 도구** | 새 프로젝트나 다른 레포에 이 보안 및 거버넌스 체계를 명령어 한 줄로 안전하게 이식 |

---

## 2. 다른 프로젝트에 원클릭으로 전파하기 (`install_to_repo.py`)

새로 만든 프로젝트나 기존의 다른 레포지토리(예: `D:/my-other-project`)에도 동일한 보안 및 에이전트 지침을 적용하고 싶다면 터미널에서 다음 명령어를 실행하면 됩니다:

```bash
python scripts/security/install_to_repo.py "D:/path/to/target-repo"
```

### 🔒 소스코드 무간섭 보장 (Safe Guarantee)
- 대상 저장소의 소스코드(`src/`, `app/`, `lib/` 등 비즈니스 로직)는 **단 1바이트도 건드리지 않습니다.**
- 오직 보안 스크립트, `.githooks/`, `.gitleaks.toml`, 정본 가이드 문서, `AGENTS.md`의 행동 규칙, 그리고 계획서 보관 서랍(`docs/plans/`)만 안전하게 복사·세팅합니다.

### 전파 후 대상 저장소에서 할 일 (단 1줄)
대상 저장소 터미널에서 Git 표준 훅을 활성화합니다:
```bash
git config core.hooksPath .githooks
```

---

## 3. 핵심 보안 철학 요약

1. **저장소 보안 = Git 보안 = 공개용 프로젝트 보안**:
   개발 초기 단계에서부터 Git 커밋/푸시 시점에 기계적으로 차단해야만, 오픈소스 환경에서 단 1건의 시크릿/개인정보 유출 사고도 발생하지 않습니다.
2. **LLM 보안 역설 (LLM 호출 = 보안 파산)**:
   보안을 검사하겠다고 소스코드를 외부 LLM API로 보내는 순간 그 자체가 데이터 유출입니다. 보안은 100% 로컬의 결정적 도구(gitleaks, 정규식)와 로컬 사람의 육안 검수로만 완결합니다.
3. **계획 문서의 저장소 내 자산화**:
   에이전트가 만든 계획 문서는 PC 임시 캐시(`brain/` 등)에 방치하지 않고 `docs/plans/`에 보존하여, 작업 PC나 환경을 바꾸어도 과거의 의사결정 맥락을 완벽히 이어받을 수 있게 합니다.

상세 아키텍처는 [`docs/design/git-security-architecture.md`](../../docs/design/git-security-architecture.md) 정본 문서를 참고하십시오.
