"""
Index a repository by scanning, chunking, embedding new
content, and syncing chunks with the vector database.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List

from codesem.config.settings import get_settings
from codesem.embeddings.openai_embedder import OpenAIEmbedder
from codesem.indexing.chunker import CodeChunker
from codesem.indexing.file_scanner import scan_repository
from codesem.utils.hashing import hash_file
from codesem.storage.vector_repository import (
    CodeChunkRecord,
    VectorRepository,
)


# ============================================================
# Indexing
# ============================================================


def index_repo(
    repo_path: str | Path,
    include_extensions: Iterable[str] | None = None,
    exclude_default_dirs: bool | None = None,
    chunk_token_size: int | None = None,
    chunk_token_overlap: int | None = None,
    max_file_bytes: int | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    delete_stale: bool | None = None,
) -> Dict:
    """
    Index a repository into the vector database.

    Returns a dict matching the CLI IndexResult TypedDict contract.
    """
    settings = get_settings()
    start_time = time.time()

    repo_path = Path(repo_path).resolve()
    files = scan_repository(
        repo_path,
        include_extensions=include_extensions,
        exclude_default_dirs=exclude_default_dirs,
        max_file_bytes=max_file_bytes,
    )

    embedder = OpenAIEmbedder(
        model=embedding_model,
        dimensions=embedding_dimensions,
    )
    chunker = CodeChunker(
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
    )

    files_scanned = 0
    files_indexed = 0
    chunks_total = 0
    chunks_inserted = 0
    chunks_skipped_unchanged = 0
    chunks_deleted_stale = 0

    valid_paths: List[str] = []

    with VectorRepository() as repo:
        for file_path in files:
            files_scanned += 1
            valid_paths.append(str(file_path))

            # Intentionally compute file hash as readability guard:
            # ensures file can be read before chunking.
            try:
                _ = hash_file(file_path)
            except Exception:
                continue

            chunks = chunker.chunk_file(file_path)
            if not chunks:
                continue

            files_indexed += 1
            chunks_total += len(chunks)

            # NOTE:
            # existing_hashes is called per-file to preserve correctness.
            # Different files may contain identical content; scoping by file_path
            # prevents cross-file deduplication errors.
            content_hashes = [c.content_hash for c in chunks]
            existing = repo.existing_hashes(str(file_path), content_hashes)

            new_chunks = [
                c for c in chunks
                if c.content_hash not in existing
            ]

            chunks_skipped_unchanged += len(chunks) - len(new_chunks)

            if not new_chunks:
                continue

            texts = [c.content for c in new_chunks]
            embeddings = embedder.embed_texts(texts)

            records: List[CodeChunkRecord] = []
            for chunk, embedding in zip(new_chunks, embeddings):
                records.append(
                    CodeChunkRecord(
                        file_path=str(file_path),
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                        embedding=embedding,
                        embedding_model=embedding_model
                        or settings.embedding_model,
                        embedding_dimensions=embedding_dimensions
                        if embedding_dimensions is not None
                        else settings.embedding_dimensions,
                    )
                )

            inserted = repo.insert_chunks(records)
            chunks_inserted += inserted

        delete_flag = (
            settings.delete_stale_on_reindex
            if delete_stale is None
            else delete_stale
        )

        if delete_flag:
            chunks_deleted_stale = repo.delete_chunks_not_in_paths(valid_paths)

    elapsed = time.time() - start_time

    return {
        "files_scanned": files_scanned,
        "files_indexed": files_indexed,
        "chunks_total": chunks_total,
        "chunks_inserted": chunks_inserted,
        "chunks_skipped_unchanged": chunks_skipped_unchanged,
        "chunks_deleted_stale": chunks_deleted_stale,
        "embedding_model": embedding_model
        or settings.embedding_model,
        "embedding_dimensions": embedding_dimensions
        if embedding_dimensions is not None
        else settings.embedding_dimensions,
        "elapsed_sec": elapsed,
    }


# Backwards-compatible alias if older code imports index_repository
index_repository = index_repo


# ============================================================
# Cost Estimation
# ============================================================


def estimate_index_cost(
    repo_path: str | Path,
    include_extensions: Iterable[str] | None = None,
    exclude_default_dirs: bool | None = None,
    chunk_token_size: int | None = None,
    chunk_token_overlap: int | None = None,
    max_file_bytes: int | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
) -> Dict:
    """
    Estimate embedding cost without writing to the database.

    Strategy:
    - Scan files
    - Chunk files
    - Count total tokens across chunks
    - Estimate OpenAI embedding cost using model pricing

    NOTE:
    This is a rough estimate and assumes text-embedding-3-small pricing.
    """
    settings = get_settings()
    repo_path = Path(repo_path).resolve()

    files = scan_repository(
        repo_path,
        include_extensions=include_extensions,
        exclude_default_dirs=exclude_default_dirs,
        max_file_bytes=max_file_bytes,
    )
    chunker = CodeChunker(
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
    )

    total_tokens = 0
    total_chunks = 0

    for file_path in files:
        try:
            chunks = chunker.chunk_file(file_path)
        except Exception:
            continue

        total_chunks += len(chunks)

        for chunk in chunks:
            # Re-tokenize chunk content using same encoding as chunker
            tokens = chunker.encoding.encode(chunk.content)
            total_tokens += len(tokens)

    # Pricing assumption: text-embedding-3-small = $0.00002 per 1K tokens
    # Adjust here if model changes.
    cost_per_1k = 0.00002
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    return {
        "files_scanned": len(files),
        "chunks_total": total_chunks,
        "estimated_tokens_total": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "embedding_model": embedding_model
        or settings.embedding_model,
        "embedding_dimensions": embedding_dimensions
        if embedding_dimensions is not None
        else settings.embedding_dimensions,
    }
