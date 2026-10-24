# AGENTS.md

AI 코딩 에이전트(Cursor, Claude Code, Copilot, Antigravity 등) 및 오픈소스 기여자가 이 저장소를 다룰 때 따르는 작업 지침이다.

---

## 1. 프로젝트 개요

`graph-distiller`는 파편화된 비정형 메모와 문서(`.txt`, `.md`, `.docx`, `.json`)를 GraphRAG 스타일로 분석·추출하고, PageRank 기반 계층 압축(RepoMap)을 통해 컨텍스트를 최적화하여 사람이 읽는 구조화된 마크다운 책(**Living Codex**)을 자동 편찬하는 **비동기 배치 컴파일러**입니다.

- **핵심 명세**: [`SPEC.md`](./SPEC.md)
- **보안 아키텍처**: [`docs/design/git-security-architecture.md`](./docs/design/git-security-architecture.md)
- **목표 산출물**: `output/codex/` (단일 책 형태의 목차 및 장별 마크다운)

### 핵심 설계 철학

1. **무프레임워크 상태 머신 (Zero Black-Box Orchestration)**:
   블랙박스형 에이전트 프레임워크에 종속되지 않고, 순수 파이썬 루프 기반의 상태 머신(`SimpleLoop`: Planner → RepoMap → Synthesizer → Verifier)으로 제어 흐름과 예외를 100% 결정적으로 다룹니다.
2. **PageRank 컨텍스트 다이어트 (Context Diet)**:
   Google의 PageRank 링크 분석 알고리즘으로 추출된 지식 그래프의 노드 중요도를 계산하고, Aider의 RepoMap 컨텍스트 압축 원리를 적용하여 LLM 컨텍스트 윈도우에 핵심 지식만 계층형 트리로 주입합니다.
3. **엄격한 원문 인용 검증 (Citation Grounding)**:
   생성된 본문의 모든 주장은 원본 청크 식별자(`[DOC_ID: xxx]`)를 인용해야 하며, Verifier가 환각(존재하지 않는 ID 인용, 인용 밀도 미달)을 기계적으로 검증하여 통과하지 못하면 재집필 또는 독약 격리(Poison Pill) 처리합니다.
4. **사전 비용 게이트 (Pre-flight Cost Gate)**:
   Phase 실행 전 예상 토큰과 비용을 산출하여 사용자 확인을 거치며, 무비용 드라이런(`--check-config`, `--explain`)을 지원합니다.

---

## 2. 파이프라인 아키텍처

4개 Phase가 하나의 `GraphDistillerState` (TypedDict, `src/graph_distiller/schema.py`)를 순서대로 변형합니다. 각 Phase는 무상태(Stateless) 순수 함수 형태로 동작합니다.

| Phase | 모듈 | 책임 | 주요 처리 |
|---|---|---|---|
| **Phase 1: Chunking** | `ingestion/` | 파일 수집 및 의미론적 분할 | 파일 로드 → `SemanticChunker`(percentile=80) → 1,000토큰 초과 청크 2차 분할 |
| **Phase 2: Extraction** | `graph/extraction.py` | 엔티티/관계 추출 및 정규화 | `Semaphore` + 지수 백오프 병렬 LLM 추출 → DLQ 에러 격리 → 2차 정규화(이표기 병합 및 가중치 합산) → 체크포인트 저장 |
| **Phase 3: Graph Build** | `graph/build.py` | 그래프 구성 및 중요도 산출 | NetworkX 그래프 빌드 → 고립 노드(Orphan) 제거 → Leiden/Louvain 커뮤니티 분할(목차 생성) → PageRank 중요도 계산 |
| **Phase 4: Synthesis** | `agents/` | 계층 지식 압축 및 검증 집필 | 상태 머신 순환: Planner(목차/제목 선정) → RepoMap(PageRank 상위 컨텍스트 트리 생성) → Synthesizer(마크다운 집필) → Verifier(인용 검증, PASS/FAIL 라우팅) |

### 재시도 및 독약 격리 (Poison Pill)

- 특정 챕터가 Verifier 검증에서 연속 N회(기본 3회) 실패하면, 파이프라인 전체를 중단시키지 않고 `poisoned_chapters`에 등록하여 영구 스킵한 뒤 다음 챕터로 진행합니다.
- 오류 로그는 `output/errors.log`에 기록됩니다.

---

## 3. 아키텍처 경계 및 구현 규칙

- **비용 계산 격리 (`cost.py`)**:
  - `src/graph_distiller/cost.py`는 순수 계산 모듈로 유지합니다. `print`, `input`, 로깅, 파일/네트워크 I/O를 포함하지 않으며 항상 결과 딕셔너리를 반환합니다.
  - 요율 조회는 `pricing.py`에서 담당합니다.
- **Phase 책임 격리**:
  - Phase 1(청킹), Phase 2(추출), Phase 3(그래프), Phase 4(집필)의 책임을 섞지 않습니다.
  - 전역 상태는 `GraphDistillerState` 계약을 따릅니다.
- **테스트 격리**:
  - 테스트 코드는 `tests/`에만 위치하며, 테스트를 통과시키기 위해 운영 로직의 보안이나 비즈니스 규칙을 약화시키지 않습니다.
- **시크릿 노출 금지**:
  - API 키(`OPENROUTER_API_KEY`)나 비밀번호는 콘솔 출력이나 로그에 절대 노출하지 않습니다.

---

## 4. 실행 및 개발 커맨드

### 설치 (Windows / Linux / macOS)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

`.env` 파일에 `OPENROUTER_API_KEY`를 설정합니다. 컴파일러는 모델명을 CLI 인자(`--extraction-model`, `--synthesis-model`)로 지정받습니다.

### 파이프라인 실행

```bash
# 로컬 기본 메모 컴파일 (전체 파이프라인)
python cli_rag_compiler.py --extraction-model openai/gpt-4o-mini --synthesis-model google/gemini-2.5-flash-lite --source local -y

# 워크스페이스 지정 실행
python cli_rag_compiler.py --extraction-model openai/gpt-4o-mini --synthesis-model google/gemini-2.5-flash-lite --workspace path/to/workspace -y

# 무비용 사전 점검 및 파이프라인 분석 ($0, LLM 미호출)
python cli_rag_compiler.py --check-config
python cli_rag_compiler.py --explain
```

### 테스트 실행 (Fail-Fast 피라미드)

Unit(무비용) → Integration(무비용, Mock LLM) 순서로 검증합니다:

```bash
# 단위 테스트
pytest tests/unit/ -v

# 통합 테스트
pytest tests/integration/ -v

# 전체 테스트 스위트
pytest
```

---

## 5. 저장소 보안 및 Git Pre-commit 가이드

본 프로젝트는 오픈소스 배포 환경에서의 비밀값 및 개인정보 유출을 방지하기 위해 2단계 로컬 보안 검사 체계를 강제합니다.

### Git Core Hooks 활성화

저장소를 클론한 후 최초 1회 실행합니다:

```bash
git config core.hooksPath .githooks
```

이후 `git commit` 시 `pre-commit` 훅이, `git push` 시 `pre-push` 훅이 자동 실행되어 다음을 차단합니다:
- `.env` 및 시크릿(API 키, 토큰, DB 패스워드) 커밋 차단 (`gitleaks`)
- 한국형 개인정보(주민등록번호, 전화번호) 정밀 탐지 (`public_safety_scan.py`)
- 대용량 파일 커밋 차단 (10MB 이상 경고, 50MB 이상 하드 차단)

### 수동 보안 검사 실행

```bash
# 스테이징된 변경사항 검사 (커밋 전)
python scripts/security/git_security_check.py --staged

# 푸시 대상 커밋 검사 (푸시 전)
python scripts/security/git_security_check.py --tracked

# 전체 저장소 공개 배포 적합성 검사
python scripts/security/git_security_check.py --public-release
```
