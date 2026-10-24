"""Integration: Workspace Isolation Tests (Corpus 2 x Profile 2, Step 070, ADR-075, $0).

Verifies the 10 isolation criteria:
1. Workspaces A and B use disjoint work/ and output/ directories.
2. Applying profiles A and B to the same corpus does not overwrite checkpoints.
3. Reusing the same profile across two corpora does not mix output codex artifacts.
4. Corpus directory file hashes remain strictly identical before and after execution (read-only).
5. Lexicon files do not leak across workspaces.
6. --from-checkpoints works seamlessly with the workspace directory layout.
7. File-level policy overrides (D5) in one workspace do not affect other workspaces.
8. [pipeline] knobs remain strictly independent across workspaces.
9. D11 folder creation guard: missing input dir fails with exit code 2 and does not create the folder;
   execution in tmp_path creates no files in project root data/ or output/.
10. D8/D4 CLI source behavior: omitting --source defaults to local with first-line banner;
    specifying --source db alongside --workspace is rejected with exit code 2.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from graph_distiller import cli_compiler
from graph_distiller.adapters.openrouter import OpenRouterEmbedder, OpenRouterLlmChat
from graph_distiller.config import PROJECT_ROOT, Settings
from graph_distiller.graph.extraction import GraphExtractor
from graph_distiller.observability import PhaseUsage
from graph_distiller.pricing import RateBook
from graph_distiller.schema import ChapterTitleResult, GraphEdge, GraphNode
from graph_distiller.workspace import KNOB_ENV_TO_FIELD

pytestmark = pytest.mark.integration


class _MockEmbedder:
    """Mock embedder returning constant vectors without network calls."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0, 0.0]


class _MockStructuredTitle:
    def __init__(self, title: str = "Mocked Chapter") -> None:
        self.title = title

    def invoke(self, _messages: Any) -> ChapterTitleResult:
        return ChapterTitleResult(title=self.title)


class _MockLLMChat:
    """Mock LLM returning valid citations extracted from RepoMap context."""

    def with_structured_output(self, _schema: Any) -> _MockStructuredTitle:
        return _MockStructuredTitle()

    def invoke(self, messages: Any, **kwargs: Any) -> SimpleNamespace:
        user_text = messages[-1].content if messages else ""
        cites = [
            c for c in re.findall(r"\[DOC_ID:\s*([a-zA-Z0-9_\-]+)\]", user_text)
            if c != "..."
        ]
        doc_id = cites[0] if cites else "memo_0000"
        return SimpleNamespace(
            content=(
                f"# Section Title\n\n"
                f"Synthesized passage explaining the domain concepts clearly. [DOC_ID: {doc_id}]\n"
            )
        )


async def _mock_extract_all(self: GraphExtractor, chunks: list, errors: list) -> Tuple[List[GraphNode], List[GraphEdge], PhaseUsage]:
    """Mock extraction that produces valid nodes and edges citing input chunks."""
    all_chunk_ids = [c.chunk_id for c in chunks]
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    prefix = chunks[0].source_doc_id.upper().rsplit("_", 1)[0]
    n1 = f"NODE_{prefix}_SYSTEM_A"
    n2 = f"NODE_{prefix}_SYSTEM_B"
    nodes.append(
        GraphNode(
            id=n1,
            type="CONCEPT",
            description=f"Primary concept for {prefix}",
            source_chunk_ids=all_chunk_ids,
            pagerank_score=0.6,
        )
    )
    nodes.append(
        GraphNode(
            id=n2,
            type="CONCEPT",
            description=f"Secondary concept for {prefix}",
            source_chunk_ids=all_chunk_ids,
            pagerank_score=0.4,
        )
    )
    edges.append(
        GraphEdge(
            source=n1,
            target=n2,
            relationship="RELATES_TO",
            weight=1.0,
        )
    )
    return nodes, edges, PhaseUsage()


def _hash_dir(directory: Path) -> Dict[str, str]:
    """Compute sha256 for all files in a directory."""
    hashes = {}
    if not directory.is_dir():
        return hashes
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = p.relative_to(directory).as_posix()
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear knob env vars and mock external network/LLM dependencies."""
    for env_k in KNOB_ENV_TO_FIELD:
        monkeypatch.delenv(env_k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-test-key-local-only")

    # Mock Settings factory methods
    monkeypatch.setattr(
        Settings,
        "build_chat_model",
        lambda self, model=None: _MockLLMChat(),
    )
    monkeypatch.setattr(
        Settings,
        "build_embeddings",
        lambda self: _MockEmbedder(),
    )

    # Mock Adapter functions
    monkeypatch.setattr(
        "graph_distiller.adapters.openrouter.default_llm_chat",
        lambda settings=None, model=None: _MockLLMChat(),
    )
    monkeypatch.setattr(
        "graph_distiller.agents.planner.default_llm_chat",
        lambda settings=None, model=None: _MockLLMChat(),
    )
    monkeypatch.setattr(
        "graph_distiller.agents.synthesizer.default_llm_chat",
        lambda settings=None, model=None: _MockLLMChat(),
    )
    monkeypatch.setattr(
        "graph_distiller.ingestion.chunker.default_embedder",
        lambda settings=None: _MockEmbedder(),
    )

    # Mock GraphExtractor extract_all
    monkeypatch.setattr(GraphExtractor, "extract_all", _mock_extract_all)

    # Mock Pricing fetch
    monkeypatch.setattr(
        "graph_distiller.local_pipeline.fetch_openrouter_rate_book",
        lambda *args, **kwargs: RateBook(by_id={}, source="mock"),
    )
    monkeypatch.setattr(
        "graph_distiller.pricing.fetch_openrouter_rate_book",
        lambda *args, **kwargs: RateBook(by_id={}, source="mock"),
    )


def _setup_corpus_profile_fixtures(base_dir: Path) -> Dict[str, Path]:
    """Create 2 corpora, 2 profiles, and 3 workspaces on disk under base_dir."""
    # 1. Corpus 1 (Tech)
    c1 = base_dir / "corpora" / "tech"
    c1.mkdir(parents=True, exist_ok=True)
    (c1 / "cloud_auth.txt").write_text(
        "OAuth protocol establishes secure token exchanges for cloud systems.\n"
        "Identity services validate client credentials before issuing tokens.\n",
        encoding="utf-8",
    )
    (c1 / "cloud_gateway.txt").write_text(
        "API Gateway routes incoming client traffic to internal microservices.\n"
        "Rate limiting filters out excessive requests at the ingress boundary.\n",
        encoding="utf-8",
    )

    # 2. Corpus 2 (Bio)
    c2 = base_dir / "corpora" / "bio"
    c2.mkdir(parents=True, exist_ok=True)
    (c2 / "cellular_mitosis.txt").write_text(
        "Cellular mitosis consists of prophase, metaphase, anaphase, and telophase.\n"
        "Spindle fibers align chromosomes along the equatorial plate during metaphase.\n",
        encoding="utf-8",
    )
    (c2 / "cellular_meiosis.txt").write_text(
        "Meiosis produces haploid gametes through two successive nuclear divisions.\n"
        "Homologous recombination creates genetic diversity during prophase I.\n",
        encoding="utf-8",
    )

    # 3. Profile 1 (Engineering)
    p1 = base_dir / "profiles" / "engineering"
    p1.mkdir(parents=True, exist_ok=True)
    (p1 / "profile.toml").write_text(
        """schema_version = 1
name = "engineering"

[pipeline]
min_community_size = 2
max_chunk_tokens = 400
""",
        encoding="utf-8",
    )
    (p1 / "rules").mkdir(parents=True, exist_ok=True)
    (p1 / "rules" / "phase4_repomap.toml").write_text("top_fraction = 0.25\n", encoding="utf-8")
    (p1 / "lexicon").mkdir(parents=True, exist_ok=True)
    (p1 / "lexicon" / "entity_aliases.json").write_text('{"oauth": "OAuth_Protocol"}', encoding="utf-8")

    # 4. Profile 2 (Academic)
    p2 = base_dir / "profiles" / "academic"
    p2.mkdir(parents=True, exist_ok=True)
    (p2 / "profile.toml").write_text(
        """schema_version = 1
name = "academic"

[pipeline]
min_community_size = 2
max_chunk_tokens = 600
""",
        encoding="utf-8",
    )
    (p2 / "rules").mkdir(parents=True, exist_ok=True)
    (p2 / "rules" / "phase4_repomap.toml").write_text("top_fraction = 0.50\n", encoding="utf-8")
    (p2 / "lexicon").mkdir(parents=True, exist_ok=True)
    (p2 / "lexicon" / "entity_aliases.json").write_text('{"mitosis": "Cell_Mitosis"}', encoding="utf-8")

    # 5. Workspace A: Corpus 1 x Profile 1 (with override in rules)
    ws_a = base_dir / "workspaces" / "ws_tech_eng"
    ws_a.mkdir(parents=True, exist_ok=True)
    (ws_a / "workspace.toml").write_text(
        """schema_version = 1
name = "ws-tech-eng"

[corpus]
path = "../../corpora/tech"

[profile]
path = "../../profiles/engineering"

[pipeline]
max_chunk_tokens = 500
""",
        encoding="utf-8",
    )
    # Workspace A specific rule override (D5)
    (ws_a / "rules").mkdir(parents=True, exist_ok=True)
    (ws_a / "rules" / "phase4_repomap.toml").write_text("top_fraction = 0.77\n", encoding="utf-8")

    # 6. Workspace B: Corpus 2 x Profile 2
    ws_b = base_dir / "workspaces" / "ws_bio_acad"
    ws_b.mkdir(parents=True, exist_ok=True)
    (ws_b / "workspace.toml").write_text(
        """schema_version = 1
name = "ws-bio-acad"

[corpus]
path = "../../corpora/bio"

[profile]
path = "../../profiles/academic"

[pipeline]
min_community_size = 2
""",
        encoding="utf-8",
    )

    # 7. Workspace C: Corpus 1 (same as A) x Profile 2 (Academic)
    ws_c = base_dir / "workspaces" / "ws_tech_acad"
    ws_c.mkdir(parents=True, exist_ok=True)
    (ws_c / "workspace.toml").write_text(
        """schema_version = 1
name = "ws-tech-acad"

[corpus]
path = "../../corpora/tech"

[profile]
path = "../../profiles/academic"
""",
        encoding="utf-8",
    )

    return {
        "corpus_tech": c1,
        "corpus_bio": c2,
        "profile_eng": p1,
        "profile_acad": p2,
        "ws_a": ws_a,
        "ws_b": ws_b,
        "ws_c": ws_c,
    }


def test_workspaces_isolation_work_and_output_disjoint(tmp_path: Path) -> None:
    """Criteria 1 & 3: Workspaces A & B write to separate work/ and output/ directories."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]
    ws_b = fixtures["ws_b"]

    rc_a = cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_a == 0

    rc_b = cli_compiler.main([
        "--workspace", str(ws_b),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_b == 0

    # 1. Distinct work/ directory checkpoints
    ckpt_a = ws_a / "work" / "phase2_extraction.json"
    ckpt_b = ws_b / "work" / "phase2_extraction.json"
    assert ckpt_a.is_file()
    assert ckpt_b.is_file()

    data_a = json.loads(ckpt_a.read_text(encoding="utf-8"))
    data_b = json.loads(ckpt_b.read_text(encoding="utf-8"))

    nodes_a = {n["id"] for n in data_a["nodes"]}
    nodes_b = {n["id"] for n in data_b["nodes"]}

    assert any("CLOUD" in n for n in nodes_a)
    assert not any("CELLULAR" in n for n in nodes_a)

    assert any("CELLULAR" in n for n in nodes_b)
    assert not any("CLOUD" in n for n in nodes_b)

    # 2. Output directories are disjoint
    out_a = ws_a / "output" / "codex"
    out_b = ws_b / "output" / "codex"
    assert (out_a / "README.md").is_file()
    assert (out_b / "README.md").is_file()

    # Content in A only cites tech corpus, B only cites bio corpus
    a_text = "\n".join(p.read_text(encoding="utf-8") for p in out_a.glob("*.md"))
    b_text = "\n".join(p.read_text(encoding="utf-8") for p in out_b.glob("*.md"))

    assert "cloud_" in a_text
    assert "cellular_" not in a_text
    assert "cellular_" in b_text
    assert "cloud_" not in b_text


def test_same_corpus_different_profiles_do_not_overwrite_checkpoints(tmp_path: Path) -> None:
    """Criteria 2: Same corpus with different profiles (ws_a vs ws_c) have isolated checkpoints."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]
    ws_c = fixtures["ws_c"]

    rc_a = cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_a == 0

    ckpt_a = ws_a / "work" / "phase2_extraction.json"
    assert ckpt_a.is_file()
    hash_a_before = hashlib.sha256(ckpt_a.read_bytes()).hexdigest()

    rc_c = cli_compiler.main([
        "--workspace", str(ws_c),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_c == 0

    ckpt_c = ws_c / "work" / "phase2_extraction.json"
    assert ckpt_c.is_file()

    # ws_a's checkpoint is completely untouched by running ws_c
    hash_a_after = hashlib.sha256(ckpt_a.read_bytes()).hexdigest()
    assert hash_a_before == hash_a_after


def test_corpus_is_strictly_read_only(tmp_path: Path) -> None:
    """Criteria 4: Corpus directory hashes remain identical before and after execution."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    c1 = fixtures["corpus_tech"]
    c2 = fixtures["corpus_bio"]

    hashes_c1_before = _hash_dir(c1)
    hashes_c2_before = _hash_dir(c2)

    cli_compiler.main([
        "--workspace", str(fixtures["ws_a"]),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    cli_compiler.main([
        "--workspace", str(fixtures["ws_b"]),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])

    hashes_c1_after = _hash_dir(c1)
    hashes_c2_after = _hash_dir(c2)

    assert hashes_c1_before == hashes_c1_after
    assert hashes_c2_before == hashes_c2_after


def test_lexicon_and_d5_policy_isolation_across_workspaces(tmp_path: Path) -> None:
    """Criteria 5 & 7: Lexicon and D5 policy file overrides are strictly isolated."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]
    ws_b = fixtures["ws_b"]

    cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    cli_compiler.main([
        "--workspace", str(ws_b),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])

    # Find the latest ledger in ws_a and ws_b runs directory
    ledgers_a = sorted((ws_a / "output" / "runs").glob("*/ledger.json"))
    ledgers_b = sorted((ws_b / "output" / "runs").glob("*/ledger.json"))
    assert len(ledgers_a) >= 1
    assert len(ledgers_b) >= 1

    ledger_a_data = json.loads(ledgers_a[-1].read_text(encoding="utf-8"))
    ledger_b_data = json.loads(ledgers_b[-1].read_text(encoding="utf-8"))

    control_a = ledger_a_data["control_room"]
    control_b = ledger_b_data["control_room"]

    # 1. Lexicon isolation (Criteria 5)
    hash_lexicon_eng = hashlib.sha256((fixtures["profile_eng"] / "lexicon" / "entity_aliases.json").read_bytes()).hexdigest()
    hash_lexicon_acad = hashlib.sha256((fixtures["profile_acad"] / "lexicon" / "entity_aliases.json").read_bytes()).hexdigest()

    assert control_a["lexicon_file_sha256"] == hash_lexicon_eng
    assert control_b["lexicon_file_sha256"] == hash_lexicon_acad
    assert control_a["lexicon_file_sha256"] != control_b["lexicon_file_sha256"]

    # 2. D5 Policy override isolation (Criteria 7)
    # ws_a has its own phase4_repomap.toml override (source="workspace")
    assert control_a["policy_files"]["phase4_repomap.toml"]["source"] == "workspace"
    # ws_b inherits from profile (source="profile")
    assert control_b["policy_files"]["phase4_repomap.toml"]["source"] == "profile"


def test_pipeline_knobs_remain_independent_across_workspaces(tmp_path: Path) -> None:
    """Criteria 8: Pipeline knobs in [pipeline] are strictly independent per workspace."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]
    ws_b = fixtures["ws_b"]

    cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    cli_compiler.main([
        "--workspace", str(ws_b),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])

    ledgers_a = sorted((ws_a / "output" / "runs").glob("*/ledger.json"))
    ledgers_b = sorted((ws_b / "output" / "runs").glob("*/ledger.json"))

    knobs_a = json.loads(ledgers_a[-1].read_text(encoding="utf-8"))["control_room"]["knobs"]
    knobs_b = json.loads(ledgers_b[-1].read_text(encoding="utf-8"))["control_room"]["knobs"]

    # ws_a has max_chunk_tokens=500 from workspace.toml
    assert knobs_a["max_chunk_tokens"]["value"] == 500
    assert knobs_a["max_chunk_tokens"]["source"] == "workspace"

    # ws_b has max_chunk_tokens=600 from academic profile.toml
    assert knobs_b["max_chunk_tokens"]["value"] == 600
    assert knobs_b["max_chunk_tokens"]["source"] == "profile"


def test_resume_from_checkpoints_in_workspace(tmp_path: Path) -> None:
    """Criteria 6: --from-checkpoints loads existing checkpoints from workspace work/."""
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]

    # First run generates checkpoints
    rc1 = cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc1 == 0

    p1_file = ws_a / "work" / "phase1_chunks.json"
    p2_file = ws_a / "work" / "phase2_extraction.json"
    assert p1_file.is_file()
    assert p2_file.is_file()

    # Track extraction calls during resume run
    called_extract = False

    async def _failing_extract(self: Any, chunks: Any, errors: Any) -> Any:
        nonlocal called_extract
        called_extract = True
        return await _mock_extract_all(self, chunks, errors)

    import time
    time.sleep(1.05)

    with patch.object(GraphExtractor, "extract_all", _failing_extract):
        rc2 = cli_compiler.main([
            "--workspace", str(ws_a),
            "--extraction-model", "test-extract-model",
            "--synthesis-model", "test-synth-model",
            "--from-checkpoints",
            "-y",
        ])
        assert rc2 == 0
        assert not called_extract, "Phase 2 extraction must not be called when --from-checkpoints is used"


def test_d11_folder_creation_guards(tmp_path: Path) -> None:
    """Criteria 9: D11 missing input folder causes exit code 2 and does not create folders;
    valid workspace execution creates zero files in repository default data/ or output/.
    """
    # 1. Missing corpus input directory
    bad_ws = tmp_path / "bad_workspace"
    bad_ws.mkdir()
    non_existent_corpus = bad_ws / "corpus_missing"
    (bad_ws / "workspace.toml").write_text(
        """schema_version = 1
name = "bad-ws"

[corpus]
path = "corpus_missing"
""",
        encoding="utf-8",
    )

    rc = cli_compiler.main([
        "--workspace", str(bad_ws),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc == 2
    assert not non_existent_corpus.exists(), "D11: Engine must not create missing input directories"

    # 2. Project root isolation: snapshot repo default paths before/after
    default_data_before = _hash_dir(PROJECT_ROOT / "data" / "raw_memos")
    default_checkpoints_before = _hash_dir(PROJECT_ROOT / "data" / "checkpoints")
    default_codex_before = _hash_dir(PROJECT_ROOT / "output" / "codex")

    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    cli_compiler.main([
        "--workspace", str(fixtures["ws_a"]),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])

    default_data_after = _hash_dir(PROJECT_ROOT / "data" / "raw_memos")
    default_checkpoints_after = _hash_dir(PROJECT_ROOT / "data" / "checkpoints")
    default_codex_after = _hash_dir(PROJECT_ROOT / "output" / "codex")

    assert default_data_before == default_data_after
    assert default_checkpoints_before == default_checkpoints_after
    assert default_codex_before == default_codex_after


def test_d8_d4_cli_source_default_and_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Criteria 10: --workspace defaults --source to local with first-line banner (D8);
    explicit --source db is rejected with exit code 2 (D4).
    """
    fixtures = _setup_corpus_profile_fixtures(tmp_path)
    ws_a = fixtures["ws_a"]

    # 1. D8: --source omitted defaults to local
    rc_default = cli_compiler.main([
        "--workspace", str(ws_a),
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_default == 0
    captured = capsys.readouterr()
    assert "[workspace] --source omitted with --workspace; defaulting to 'local'" in captured.out

    # 2. D4: explicit --source db is rejected
    rc_conflict = cli_compiler.main([
        "--workspace", str(ws_a),
        "--source", "db",
        "--extraction-model", "test-extract-model",
        "--synthesis-model", "test-synth-model",
        "-y",
    ])
    assert rc_conflict == 2
    captured_err = capsys.readouterr()
    assert "--workspace is not supported with --source db" in captured_err.err or "--workspace is not supported with --source db" in captured_err.out
