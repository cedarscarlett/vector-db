"""PostgreSQL + pgvector persistence layer for code chunk storage and search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
from psycopg import sql

import psycopg
from psycopg.rows import dict_row

from codesem.config.settings import get_settings


# ============================================================
# Data Models
# ============================================================


@dataclass(frozen=True)
class CodeChunkRecord:
    file_path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    embedding: Sequence[float]
    embedding_model: str
    embedding_dimensions: Optional[int]


@dataclass(frozen=True)
class SearchHit:
    file_path: str
    start_line: int
    end_line: int
    content: str
    similarity: float


# ============================================================
# Repository
# ============================================================


class VectorRepository:
    """
    PostgreSQL + pgvector persistence layer.

    Responsibilities:
    - Insert chunks (bulk)
    - Delete stale chunks
    - Check existing hashes
    - Perform cosine similarity search
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.database_url:
            raise ValueError("DATABASE_URL must be set.")

        self._conn = psycopg.connect(settings.database_url, row_factory=dict_row)

    # ---------------------------------------------------------
    # Context manager support
    # ---------------------------------------------------------

    def __enter__(self) -> "VectorRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------------------------------------------------------
    # Insert
    # ---------------------------------------------------------

    def insert_chunks(self, chunks: Iterable[CodeChunkRecord]) -> int:
        """
        Bulk insert code chunks.

        Returns number of rows inserted.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        # Build single multi-row INSERT for accurate rowcount.
        # Safe: SQL structure is static; all chunk data is passed via %s params.
        values = []
        params = []
        for c in chunk_list:
            values.append(
                "(%s,%s,%s,%s,%s,%s,%s,%s)"
            )
            params.extend(
                [
                    c.file_path,
                    c.start_line,
                    c.end_line,
                    c.content,
                    c.content_hash,
                    list(c.embedding),
                    c.embedding_model,
                    c.embedding_dimensions,
                ]
            )

        # TODO: Batch inserts (e.g. 500 rows) to avoid PostgreSQL parameter limits
        # on very large repos (~32767 params/query; 8 params per row here).
        query = f"""
            insert into code_chunks (
                file_path,
                start_line,
                end_line,
                content,
                content_hash,
                embedding,
                embedding_model,
                embedding_dimensions
            )
            values {",".join(values)}
            on conflict do nothing
        """

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            inserted = cur.rowcount or 0

        self._conn.commit()
        return inserted

    # ---------------------------------------------------------
    # Hash lookup
    # ---------------------------------------------------------

    def existing_hashes(
        self, file_path: str, hashes: Iterable[str]
    ) -> set[str]:
        """
        Return subset of hashes that already exist in DB
        for the given file_path.
        """
        hash_list = list(hashes)
        if not hash_list:
            return set()

        with self._conn.cursor() as cur:
            cur.execute(
                """
                select content_hash
                from code_chunks
                where file_path = %s
                and content_hash = any(%s)
                """,
                (file_path, hash_list),
            )
            rows = cur.fetchall()

        return {row["content_hash"] for row in rows}

    # ---------------------------------------------------------
    # Delete stale
    # ---------------------------------------------------------

    def delete_chunks_not_in_paths(self, valid_paths: Iterable[str]) -> int:
        """
        Delete chunks whose file_path is not in the provided set.

        Returns number of rows deleted.
        """
        path_list = list(valid_paths)

        with self._conn.cursor() as cur:
            if not path_list:
                # Explicitly delete all if no valid paths remain
                cur.execute("delete from code_chunks")
            else:
                cur.execute(
                    """
                    delete from code_chunks
                    where file_path <> all(%s)
                    """,
                    (path_list,),
                )
            deleted = cur.rowcount

        self._conn.commit()
        return deleted

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[SearchHit]:
        """
        Perform cosine similarity search using pgvector.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                with q as (
                    select %s::vector as v
                )
                select
                    c.file_path,
                    c.start_line,
                    c.end_line,
                    c.content,
                    1 - (c.embedding <=> q.v) as similarity
                from code_chunks c, q
                order by c.embedding <=> q.v
                limit %s
                """,
                (list(query_embedding), top_k),
            )
            rows = cur.fetchall()

        return [
            SearchHit(
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
                similarity=row["similarity"],
            )
            for row in rows
        ]

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("select count(*) as c from code_chunks")
            row = cur.fetchone()
        return int(row["c"])

    def close(self) -> None:
        self._conn.close()
