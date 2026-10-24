# graph-distiller SPEC

**Project:** `graph-distiller` (자율형 지식 편찬 엔진)

본 문서는 설계 Level 1 → Level 3.5를 순서대로 통합한 단일 명세서다.

- [Level 1 — 개념 설계](#프로덕션-레벨-개념-설계-명세서-level-1)
- [Level 2 — 논리/컨테이너 설계](#프로덕션-레벨-논리컨테이너-설계-명세서-level-2)
- [Level 3 — 상세 설계](#프로덕션-레벨-상세-설계-명세서-level-3)
- [Level 3.5 — Physical I/O Contracts](#프로덕션-레벨-상세-설계-명세서-보완-level-35-physical-io-contracts)

---

# 프로덕션 레벨 개념 설계 명세서 (Level 1)

**Project:** `graph-distiller` (자율형 지식 편찬 엔진)  
**Target Audience:** 비즈니스 이해관계자, 프로덕트 오너(PO), 리드 엔지니어

## 1. 문제 정의 (Problem Statement & The "Why")

현재 사내 RDB에 적재된 방대한 지식 데이터는 파편화되어 있어, 기존의 단순 벡터 검색(Vector RAG)으로는 전체적인 맥락(Context)과 지식 간의 위계(Hierarchy)를 파악할 수 없다.

따라서, **파편화된 데이터를 의미론적(Semantic)으로 분해하고 그래프(Graph) 형태로 조립한 뒤, 그래프 PageRank 중심성 분석과 RepoMap 스타일의 컨텍스트 압축 기술을 활용하여 사람이 읽을 수 있는 한 권의 구조화된 '책(Living Codex)'으로 자동 편찬 및 영구 갱신하는 자율형 백그라운드 엔진**을 구축한다.

## 2. 비목표 (Non-Goals: 절대 구현하지 않는 것)

경계가 모호한 아키텍처는 반드시 실패한다. 본 시스템은 다음 사항을 **절대 지원하지 않는다 (MUST NOT).**

1. **실시간 대화형 챗봇 (Real-time Chatbot):** 이 시스템은 사용자의 질문에 1~2초 내로 답하는 챗봇이 아니다. 수십만 토큰을 읽고 책을 집필하는 **비동기 배치(Batch) 컴파일러**다.
2. **인간 개입형 저작 도구 (Human-in-the-loop Editor):** 사용자가 중간에 개입하여 책의 목차나 내용을 수동으로 편집하는 UI/UX는 제공하지 않는다. 모든 갱신은 RDB 원본 데이터의 수정과 에러 로그 피드백 루프를 통해서만 이루어진다.
3. **순수 Vector DB 의존:** 텍스트 전체를 Vector DB에 넣고 코사인 유사도로만 검색하는 기존 RAG 방식은 배제한다. 의미적 검색은 보조 수단이며, 주력은 GraphDB 기반의 구조적 참조(Reference) 탐색이다.

## 3. 시스템 컨텍스트 다이어그램 (System Context)

시스템이 외부 세계(Actors)와 어떻게 상호작용하는지 보여주는 최상위 물리적 조감도.

```text
                      [External Actors & Data Sources]
 ┌────────────────┐                                      ┌─────────────────┐
 │ Enterprise RDB │                                      │ Admin / Log DB  │
 │ (파편화된 원본)│                                      │ (피드백 및 에러)│
 └───────┬────────┘                                      └────────┬────────┘
         │ (1) Raw Data Ingestion                                 │ (4) Error Logs
         ▼                                                        ▼
 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
 ┃                       [ graph-distiller Engine ]                               ┃
 ┃                                                                         ┃
 ┃  - Semantic Chunking: 임베딩 기반 의미 분해                             ┃
 ┃  - Graph Construction: Entity Extraction & Community Detection (목차)  ┃
 ┃  - PageRank Context Compression: 중심성 기반 토큰 예산 최적화 (RepoMap) ┃
 ┃  - Stateless State Machine: 순수 Python 무상태 상태 머신 오케스트레이션 ┃
 ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                  │ (2) Synthesis & Compilation
                                  ▼
                         ┌──────────────────┐
                         │ The Living Codex │
                         │ (구조화된 마크다운)│
                         │ - Chapters       │
                         │ - Sections       │
                         │ - Citations      │
                         └──────────────────┘
                                  │ (3) Read
                                  ▼
                         [ End Users / Readers ]
```

## 4. 유비쿼터스 용어 사전 (Ubiquitous Language)

도메인 용어와 인프라 용어의 혼용을 막기 위해 아래의 정의를 강제한다.

- **Chunk (청크):** 의미 단위(Semantic)로 쪼개진 500~1,000 토큰 길이의 원본 텍스트 조각.
- **Entity (엔티티 / 노드):** 청크에서 LLM이 추출한 도메인 특화 고유명사 (예: `AUTH_SERVER`).
- **Community (군집 / 챕터):** GraphDB 내에서 촘촘하게 연결된 엔티티들의 집합. 이 군집이 곧 책의 '챕터(Chapter)'가 된다.
- **RepoMap (레포맵):** 특정 챕터를 집필할 때 PageRank 중심성이 높은 핵심 엔티티와 원본 청크만 선별하여 만든 계층형 텍스트 지식 요약 지도.
- **Citation (인용/출처):** 책의 본문에 삽입되는 원본 데이터의 물리적 주소 (예: `[DOC_ID: 1042]`).

## 5. 암묵적 가정 및 규모 (Hidden Assumptions & Scale)

시스템 바닥에 깔린 물리적 한계와 가정을 선제적으로 폭로한다.

- **비용(Cost)의 트레이드오프:** GraphRAG의 엔티티 추출(Phase 2)은 일반 RAG 대비 막대한 LLM API 비용(토큰 소모)을 발생시킨다. 이는 '고품질의 구조화된 지식'을 얻기 위해 지불하기로 합의된 대가다.
- **최종적 일관성 (Eventual Consistency):** RDB의 데이터가 수정되더라도 '책(Codex)'에 즉시 반영되지 않는다. 야간 배치(Nightly Batch) 또는 특정 트리거에 의해 재컴파일(Re-compilation)이 완료된 후에만 동기화된다.
- **목표 규모 (Target Scale):**
  - 초기 Ingestion 대상: RDB 내 텍스트 레코드 약 100,000건.
  - 생성될 책의 규모: 50~100개의 챕터, 총 300~500 페이지 분량의 마크다운.
  - 허용 가능한 컴파일 시간(RTO): 전체 재컴파일 시 최대 4~6시간 (비동기 워커 기준).

---

# 프로덕션 레벨 논리/컨테이너 설계 명세서 (Level 2)

**Project:** `graph-distiller` (자율형 지식 편찬 엔진)  
**Target Audience:** 백엔드 엔지니어, 데이터 파이프라인 설계자, AI 에이전트 (Cursor / Claude)

## 1. 전체 데이터 파이프라인 아키텍처 (Data Pipeline Architecture)

시스템은 크게 '데이터 전처리 및 그래프 구축(Phase 1~3)'과 '에이전트 기반 집필(Phase 4)'의 두 가지 거대한 비동기 배치(Batch) 스트림으로 나뉜다.

```text
[Source]                  [ Phase 1 & 2: Ingestion & Extraction ]                 [ Phase 3: Graph Build ]
 ┌──────┐       ┌───────────────────────────────────────────────────────┐       ┌─────────────────────────┐
 │      │       │ 1. Semantic Chunker                                   │       │ In-Memory Graph / Neo4j │
 │ RDB  │ ────► │  - RDB 텍스트를 500~1000 토큰 단위 의미망으로 분할    │ ────► │  - Node & Edge 적재     │
 │      │       │                                                       │       │  - Leiden Algorithm     │
 │      │       │ 2. LLM Extraction Worker (Async Batch)                │       │    (커뮤니티/목차 탐지) │
 │      │       │  - 청크에서 Entity(개념)와 Reference(관계) 추출       │       │  - PageRank 연산        │
 └──────┘       └───────────────────────────────────────────────────────┘       └──────────┬──────────────┘
                                                                                           │
                                ┌──────────────────────────────────────────────────────────┘
                                │ (TOC 및 Graph 메타데이터 전달)
                                ▼
[ Phase 4: Synthesis Orchestration (Stateless State Machine) ]
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Global State: { chunks, nodes, edges, toc, repo_map, drafts, errors }                           │
 │                                                                                                 │
 │  [START] ──► ┌────────────────┐       ┌──────────────────┐       ┌──────────────────┐           │
 │              │ Planner Agent  │ ────► │ RepoMap Agent    │ ────► │ Synthesizer Agent│           │
 │              │ (집필 챕터 할당)│       │ (PageRank 압축)  │       │ (마크다운 집필)  │           │
 │              └────────────────┘       └──────────────────┘       └────────┬─────────┘           │
 │                      ▲                                                    │                     │
 │                      │ (Handoff / Retry)                                  ▼                     │
 │              ┌───────┴────────┐                                  ┌──────────────────┐           │
 │              │ Admin / Log DB │ ◄──── [에러 로깅] ────────────── │ Verifier Agent   │ ──► [END] │
 │              │ (피드백 루프)  │                                  │ (출처/환각 검증) │           │
 │              └────────────────┘                                  └──────────────────┘           │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Synthesis 상태 머신 토폴로지 (State Machine Topology)

Phase 4의 오케스트레이터는 철저히 **무상태(Stateless)**로 동작한다. 각 노드(Agent)는 외부 DB를 직접 찌르지 않으며, 오직 전달받은 State 객체만 읽고 변형하여 다음 노드로 넘긴다.

### Planner Agent (계획자)

- **Input:** `State["toc"]` (Phase 3에서 생성된 커뮤니티/목차 정보).
- **Action:** 아직 집필되지 않은 챕터(커뮤니티)를 식별하고, 이번 사이클에서 집필할 타깃 챕터 ID를 결정한다.

### RepoMap Agent (컨텍스트 압축기)

- **Input:** 타깃 챕터 ID, `State["nodes"]`, `State["edges"]`, `State["chunks"]`.
- **Action:** 타깃 챕터 내 노드들을 PageRank 순으로 정렬하고, 상위 노드와 연결된 원본 청크 텍스트를 묶어 계층형 지식 맵(`State["repo_map"]`)으로 렌더링한다.

### Synthesizer Agent (집필자)

- **Input:** `State["repo_map"]`.
- **Action:** 압축된 컨텍스트를 바탕으로 실제 사람이 읽을 수 있는 마크다운 텍스트를 생성한다. 반드시 원본 문서 ID를 인용(`[DOC_ID: xxx]`)해야 한다. 결과를 `State["drafts"]`에 추가한다.

### Verifier Agent (검증자 및 라우터)

- **Input:** `State["drafts"]`.
- **Action:** 생성된 초안에 환각(Hallucination)이 없는지, 인용된 문서 ID가 실제 `State["chunks"]`에 존재하는지 검증한다.

### Conditional Edge (조건부 라우팅)

- **PASS:** 모든 챕터가 완성되었으면 END로 라우팅, 남은 챕터가 있으면 Planner로 라우팅.
- **FAIL:** 에러 사유를 `State["errors"]`에 기록하고 Planner로 라우팅하여 해당 챕터 재집필을 지시한다.

## 3. 컴포넌트 간 통신 및 I/O 명세 (Communication & I/O)

분산 시스템의 물리적 한계를 고려하여 통신 방식을 엄격히 통제한다.

### RDB → Phase 1 (Ingestion)

실시간 CDC(Change Data Capture)를 쓰지 않는다. 매일 자정(Nightly)에 변경된 레코드만 긁어오는 Pull 기반의 Batch Job으로 동작한다.

### Phase 2 (LLM Extraction) → LLM API

- **병목 구간 (Bottleneck):** 수만 개의 청크를 LLM API로 쏠 때 Rate Limit(HTTP 429)이 무조건 발생한다.
- **해결책:** 비동기 큐(예: Celery, Redis Queue, 또는 Python asyncio + Semaphore)를 사용하여 동시 요청 수를 20개 이하로 제한하고, 지수 백오프(Exponential Backoff)를 강제한다.

### Phase 3 (Graph Build) → Phase 4 (Synthesis Orchestrator)

그래프 연산(Leiden, PageRank)은 I/O 바운드가 아닌 CPU 바운드 작업이다. Python의 networkx를 사용할 경우 메모리 내(In-memory)에서 처리 후, 그 결과(State)를 Phase 4 오케스트레이터의 진입점으로 넘긴다.

## 4. 데이터 정합성 및 장애 격리 (Data Consistency & Blast Radius)

### 체크포인트(Checkpointing) 강제

Phase 1~3은 실행 시간이 매우 길다(수 시간 소요). 도중에 서버가 죽으면 처음부터 다시 LLM API 비용을 태워야 한다.

- **대응:** Phase 2가 끝난 직후 추출된 Node와 Edge 데이터는 반드시 로컬 디스크(JSON/Parquet)나 임시 DB에 **체크포인트로 저장(Persist)**되어야 한다 (MUST).

### Dead Letter Queue (DLQ) 처리

Phase 2에서 LLM이 JSON 스키마(Structured Output)를 어기고 이상한 텍스트를 뱉어 파싱 에러가 나는 청크가 반드시 발생한다.

- **대응:** 파싱에 실패한 청크는 전체 파이프라인을 중단시키지 말고, DLQ(에러 로그 DB)에 문서 ID와 함께 기록한 뒤 무시(Skip)하고 다음 청크로 넘어간다. (부분적 데이터 유실을 허용하여 전체 가용성을 확보하는 트레이드오프).

---

# 프로덕션 레벨 상세 설계 명세서 (Level 3)

**Project:** `graph-distiller` (자율형 지식 편찬 엔진)  
**Target Audience:** 구현 담당 시니어 엔지니어 및 AI 코딩 에이전트 (Cursor / Claude)

## 1. 전역 상태 (Global State) 스키마

파이프라인의 모든 노드와 단계는 무상태(Stateless) 순수 함수로 동작해야 하며, 아래 정의된 `GraphDistillerState`만을 입력받고 변형하여 반환해야 한다.

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# --- Sub-schemas ---
class ChunkMeta(BaseModel):
    chunk_id: str
    source_doc_id: str
    content: str
    token_count: int

class GraphNode(BaseModel):
    id: str = Field(description="엔티티의 고유 이름 (대문자 정규화)")
    type: str = Field(description="CONCEPT, PERSON, SYSTEM 중 택 1")
    description: str = Field(description="이 엔티티에 대한 1~2줄 요약")
    source_chunk_ids: List[str] = Field(description="이 엔티티가 발견된 청크 ID 목록")

class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str = Field(description="두 엔티티 간의 관계 설명")
    weight: float = Field(default=1.0)

class ChapterDraft(BaseModel):
    chapter_id: int
    title: str
    content: str
    cited_doc_ids: List[str]

# --- Main State (TypedDict) ---
class GraphDistillerState(TypedDict):
    raw_documents: List[Dict[str, Any]]  # 소스에서 인입된 원본 데이터
    chunks: List[ChunkMeta]              # Phase 1 결과물
    nodes: List[GraphNode]               # Phase 2 결과물
    edges: List[GraphEdge]               # Phase 2 결과물
    toc: Dict[int, List[str]]            # Phase 3 결과물 (Community ID -> Node IDs)
    repo_map: str                        # Phase 4 입력용 (PageRank 기반 압축 지식 맵)
    drafts: List[ChapterDraft]           # Phase 4 결과물
    errors: List[str]                    # 에러 및 재시도 로그
```

## 2. Phase 1: Ingestion & Semantic Chunking

파편화된 소스 텍스트를 의미 단위로 절단한다.

- **사용 모듈:** `langchain_experimental.text_splitter.SemanticChunker`
- **임베딩 모델:** `text-embedding-3-small` (비용 효율성 목적)

### 제약 조건 (MUST)

- 분할 기준(Breakpoint)은 percentile 방식을 사용하며, 임계값은 80으로 설정한다.
- 청크가 생성된 후, 각 청크의 토큰 수를 계산하여 1,000 토큰을 초과하는 청크는 `RecursiveCharacterTextSplitter`로 강제 2차 분할해야 한다. (LLM 엔티티 추출 시 Context Window 초과 방지).
- 모든 청크는 `ChunkMeta` 스키마에 맞게 매핑되어 State에 저장된다.

## 3. Phase 2: Graph Extraction (LLM Structured Output)

청크에서 엔티티와 관계를 추출한다. (MS GraphRAG의 핵심 로직)

- **사용 모델 (OpenRouter slug, `.env` 필수):** `EXTRACTION_MODEL=openai/gpt-4o-mini`

### 출력 강제 (Structured Output)

LLM 호출 시 `response_format`에 위 1항의 `GraphNode`, `GraphEdge` 리스트를 포함하는 Pydantic 모델을 주입하여 JSON 출력을 강제한다.

### 프롬프트 제약 (MUST)

- "일반 명사(예: 시스템, 회사, 사용자, 데이터)는 절대 노드로 추출하지 마라. 도메인 특화 고유명사만 추출하라." (Super-node 방지)
- "추출된 노드 ID는 반드시 공백을 언더스코어(`_`)로 치환하고 대문자로 정규화하라."

## 4. Phase 3: Hierarchy & PageRank (NetworkX)

추출된 노드와 엣지를 메모리 상의 그래프로 빌드하고, 목차(Hierarchy)와 중요도(PageRank)를 연산한다.

### 논리적 흐름도 (Algorithm Flow)

1. **Graph Build:** `State["nodes"]`와 `State["edges"]`를 순회하며 `networkx.Graph()` 객체를 생성한다.
2. **Community Detection (목차 생성):**
   - `cdlib` 라이브러리의 Leiden 알고리즘(또는 networkx의 `louvain_communities`)을 실행하여 노드들을 N개의 커뮤니티(군집)로 분할한다.
   - 각 커뮤니티가 책의 **'챕터(Chapter)'**가 된다. `State["toc"]`에 `{커뮤니티ID: [노드 리스트]}` 형태로 저장한다.
3. **PageRank (중요도 산출):**
   - 전체 그래프를 대상으로 `nx.pagerank(weight='weight')`를 실행하여 각 노드의 점수를 계산하고, 노드 메타데이터에 `pagerank_score`를 업데이트한다.

## 5. Phase 4: Synthesis (PageRank 기반 RepoMap 지식 압축 및 집필)

PageRank 알고리즘으로 지식 그래프의 중요도를 산출하고, Aider의 RepoMap 컨텍스트 압축 기법을 벤치마킹하여 LLM이 책을 쓸 때 필요한 핵심 컨텍스트만 계층형으로 압축하여 제공한다.

### RepoMap 생성 로직 (Pure Function)

1. 특정 챕터(커뮤니티)를 집필할 때, 해당 커뮤니티 내의 노드들을 PageRank 점수 순으로 정렬한다.
2. 상위 20%의 핵심 노드만 추려낸다.
3. 핵심 노드와 연결된 원본 텍스트(`ChunkMeta.content`)를 가져온다.
4. 아래와 같은 ASCII 트리 형태의 문자열(`repo_map`)로 렌더링한다.

```text
[Chapter: 1]
├── Core Concept: AUTH_SERVER (PageRank: 0.08)
│   └── Source [DOC_ID: 402]: "인증 서버는 OAuth2.0을 사용하여..."
├── Core Concept: JWT_TOKEN (PageRank: 0.05)
│   └── Source [DOC_ID: 115]: "토큰 만료 시간은 15분이며..."
```

### 집필 에이전트 (Synthesizer Agent)

- **사용 모델 (OpenRouter slug, `.env` 필수):** `SYNTHESIS_MODEL=google/gemini-2.5-flash-lite`
- **지시어:** "제공된 RepoMap을 바탕으로 마크다운 형식의 챕터를 집필하라. 본문 작성 시 반드시 `[DOC_ID: 402]` 형태로 출처를 인용(Citation)하라."
- **설정 강제:** `EXTRACTION_MODEL` / `SYNTHESIS_MODEL`이 `.env`에 없으면 프로세스는 로그를 남기고 즉시 종료한다 (암묵 기본값 폴백 금지).

## 6. 장애 격리 및 붕괴 시나리오 (Blast Radius & Resilience)

AI 에이전트가 코드를 작성할 때 반드시 아래의 방어 로직을 포함하도록 강제한다.

### LLM Rate Limit 붕괴 (HTTP 429)

Phase 2(Graph Extraction)는 수천 개의 청크를 병렬로 LLM에 던진다.

- **구현 강제:** `asyncio.gather`를 날것으로 쓰지 마라. 반드시 `tenacity` 라이브러리를 사용하여 `wait_exponential(multiplier=1, min=4, max=60)` 백오프 로직과 동시성 제한(Semaphore, 최대 10~20)을 걸어야 한다.

### 고립된 노드 (Disconnected Components)

Phase 3에서 어떤 엣지도 가지지 못한 고립된 노드(Islands)가 발생할 수 있다.

- **구현 강제:** 커뮤니티 탐지 알고리즘 실행 전, `nx.degree(G) == 0`인 노드는 그래프에서 제거하고 `State["errors"]`에 `"Orphan Node Dropped: {node_id}"`로 로깅하라.

### 토큰 예산 초과 (Token Budget Overflow)

Phase 4에서 RepoMap 문자열이 100,000 토큰을 넘어갈 수 있다.

- **구현 강제:** `tiktoken`을 사용하여 RepoMap 렌더링 중 누적 토큰을 계산하고, 80,000 토큰 도달 시 하위 PageRank 노드의 텍스트 병합을 즉시 중단(Truncate)하는 `break` 문을 삽입하라.

---

# 프로덕션 레벨 상세 설계 명세서 보완 (Level 3.5: Physical I/O Contracts)

**Project:** `graph-distiller` (자율형 지식 편찬 엔진)  
**Target Audience:** AI 코딩 에이전트 (Cursor / Claude)

본 문서는 Level 1~3 명세에서 누락된 물리적 경계(DB 스키마, 파일 I/O, 예외 처리 정책)를 확정하는 계약서다. 모든 네이밍은 `graph_distiller`으로 통일한다.

## 1. Source Data 계약 (Input Contract for PoC)

초기 PoC 단계에서는 RDB 연동을 배제하고, 로컬 디렉터리에 있는 수천 개의 단편적 메모(`.txt` / `.md` / `.docx` / `.json`) 파일들을 원본 데이터로 사용한다. 향후 RDB로의 전환을 고려하여 식별자(ID) 체계를 추상화한다.

- **데이터 소스 경로:** 프로젝트 루트 하위의 `/data/raw_memos/` 디렉터리.
- **파일 포맷:** `.txt`, `.md`, `.docx`, 또는 `.json` (각 파일이 하나의 단편적 지식/메모를 담고 있다고 가정).
- **로더 분기 (`ingestion/ingestion.py`):**
  - `.txt` / `.md` → UTF-8 평문 로드
  - `.docx` → `langchain_community.document_loaders.Docx2txtLoader` (실패 시 `UnstructuredWordDocumentLoader` 폴백)
  - `.json` → JSON 파싱 후 키-값 쌍을 `key: value` 텍스트로 Flatten하여 청커에 전달
- **식별자(ID) 매핑 규칙:**
  - 파일의 이름(확장자 제외)을 해당 문서의 고유 ID로 사용한다.
  - 예: `/data/raw_memos/memo_20231012.txt` → `source_doc_id: "memo_20231012"`
  - 예: `/data/raw_memos/memo_policy.docx` → `source_doc_id: "memo_policy"`
  - 예: `/data/raw_memos/memo_config.json` → `source_doc_id: "memo_config"`
- **스키마 변경 (Agnostic ID):**
  - 기존 명세의 `source_rdb_id`라는 명칭을 데이터 소스에 종속되지 않는 **`source_doc_id`**로 전면 수정한다.
  - 향후 Synthesizer Agent가 책을 집필할 때 남기는 출처 인용 포맷도 `[RDB_ID: xxx]`에서 **`[DOC_ID: xxx]`**로 변경한다.
- **Ingestion 동작 방식:**
  - `ingestion.py`는 `/data/raw_memos/` 내의 지원 확장자 파일을 순회·로드한 뒤, `SemanticChunker`로 넘긴다.

## 2. The Living Codex 출력 계약 (Output Contract)

Synthesizer Agent가 뱉어낼 최종 산출물(책)의 물리적 디렉터리 구조와 포맷을 정의한다.

- **출력 경로:** 프로젝트 루트 하위의 `/output/codex/` 디렉터리.
- **파일 네이밍 및 구조:**

```text
/output/codex/
├── README.md (전체 목차 및 메타데이터)
├── chapter_01_authentication_system.md
├── chapter_02_database_architecture.md
└── ...
```

## 3. TOC 제목 생성 및 Verifier 판정 규칙 (Agent Logic)

### TOC → Title 생성 규칙 (Planner Agent)

Phase 3에서 생성된 커뮤니티(군집)는 단순한 숫자 ID(`comm_1`)를 갖는다.

Planner Agent는 해당 커뮤니티에 속한 핵심 노드(Entity) 이름 상위 5개를 LLM에 던져, **"이 개념들을 포괄하는 사람이 읽기 좋은 챕터 제목(예: 데이터베이스 아키텍처)을 5단어 이내로 생성하라"**고 지시해야 한다.

### Verifier Agent 판정 규칙 (PASS/FAIL)

**PASS 조건:**

- 본문에 `[DOC_ID: xxx]` 형태의 인용이 최소 1개 이상 존재할 것.
- 인용된 ID가 `State["chunks"]`의 `source_doc_id` 목록에 실제로 존재할 것.

**FAIL 조건 (환각 판정):**

- 존재하지 않는 문서 ID를 인용했거나, 프롬프트 지시를 무시하고 마크다운 포맷이 깨진 경우.

## 4. 상태 머신 제약 (Retry & Poison Pill)

무한 루프를 방지하기 위해 `GraphDistillerState`에 재시도 카운터를 추가하고 상한선을 둔다.

- **State 스키마 추가:** `GraphDistillerState`에 `retry_counts: Dict[int, int]` (챕터 ID별 재시도 횟수) 추가.
- **Retry 상한 (Max Retries):** 최대 3회.
- **Poison Chapter 처리:** 특정 챕터가 3회 연속 Verifier를 통과하지 못하면(FAIL), 해당 챕터는 **Poison Pill(독약)**로 간주한다.
- **조치:** 해당 챕터 집필을 영구 포기(Skip)하고, Admin/Log DB(또는 `errors.log` 파일)에 `[FATAL] Chapter {id} failed after 3 retries. Reason: {error_msg}`를 기록한 뒤 다음 챕터로 넘어간다.

## 5. 패키지 구조 및 엔트리포인트 (Project Scaffolding)

Cursor는 아래의 디렉터리 레이아웃을 준수하여 코드를 스캐폴딩해야 한다.

```text
graph_distiller/
├── src/
│   ├── graph_distiller/
│   │   ├── __init__.py
│   │   ├── config.py          # 환경변수 (LLM API Key, DB URI 등)
│   │   ├── schema.py          # Pydantic 모델 및 GraphDistillerState 정의
│   │   ├── ingestion/         # Phase 1: Semantic Chunking
│   │   ├── graph/             # Phase 2 & 3: LLM Extraction & NetworkX (Leiden, PageRank)
│   │   └── agents/            # Phase 4: State Machine Nodes (Planner, RepoMap, Synthesizer, Verifier)
│   └── main.py                # CLI 엔트리포인트 (예: python main.py compile --full)
├── output/codex/              # 산출물 디렉터리
├── requirements.txt
└── .env.example
```

## 6. 용어 및 수치 정정

- **프로젝트 및 상태 객체 명칭:** `scribegraph` → `graph_distiller`, `ScribeGraphState` → `GraphDistillerState`로 전면 교체.
- **인용/ID 명칭:** `source_rdb_id` / `[RDB_ID: xxx]` → `source_doc_id` / `[DOC_ID: xxx]`.
- **목표 규모 수치 명확화:** 챕터 수 50~100개, 총 페이지 수 300~500장 분량.
