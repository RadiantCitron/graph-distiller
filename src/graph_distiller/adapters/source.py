"""SourcePull default Adapters — file memos and read-only ragdb pull."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from graph_distiller.ports import SourcePull


class FileSourcePull:
    """
    Default SourcePull Adapter (file PoC).

    Invariant: read-only on the memo tree — never WRITE origin.
    """

    def load_documents(self, root: Path) -> List[Dict[str, Any]]:
        from graph_distiller.ingestion.ingestion import load_raw_memos

        return load_raw_memos(root)


class RdbSourcePull:
    """
    RAG DB SourcePull Adapter (SELECT only — never WRITE public.*).

    load_documents is the file-tree Protocol method and is not used for
    ragdb. Use preview_counts / load_sample_payload / load_full_payload.
    """

    def load_documents(self, root: Path) -> List[Dict[str, Any]]:
        raise RuntimeError(
            "RdbSourcePull does not read a file tree "
            f"(got {root}). Use --source db with --sample or --full-db."
        )

    def preview_counts(self, uri: str) -> Dict[str, Any]:
        from graph_distiller.ingestion.rdb import connect, counts_as_dict, fetch_counts

        conn = connect(uri)
        try:
            return counts_as_dict(fetch_counts(conn))
        finally:
            conn.close()

    def load_sample_payload(self, uri: str, n_jo: int, m_qna: int):
        from graph_distiller.ingestion.rdb import connect, load_sample

        conn = connect(uri)
        try:
            return load_sample(conn, n_jo, m_qna)
        finally:
            conn.close()

    def load_full_payload(self, uri: str):
        from graph_distiller.ingestion.rdb import connect, load_full

        conn = connect(uri)
        try:
            return load_full(conn)
        finally:
            conn.close()


def default_source_pull() -> SourcePull:
    return FileSourcePull()


def rag_source_pull() -> RdbSourcePull:
    return RdbSourcePull()
