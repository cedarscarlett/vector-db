"""Tests for content hashing and CodeChunkRecord construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from codesem.indexing.hashing import hash_file, hash_text
from codesem.storage.vector_repository import CodeChunkRecord


# ============================================================
# Hashing Tests
# ============================================================


def test_hash_text_is_deterministic():
    content = "def foo():\n    return 42"

    h1 = hash_text(content)
    h2 = hash_text(content)

    assert h1 == h2


def test_hash_text_changes_with_content():
    content1 = "print('hello')"
    content2 = "print('hello world')"

    h1 = hash_text(content1)
    h2 = hash_text(content2)

    assert h1 != h2


def test_hash_file_changes_when_file_changes(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text("print('hello')", encoding="utf-8")
    hash1 = hash_file(file_path)

    file_path.write_text("print('hello world')", encoding="utf-8")
    hash2 = hash_file(file_path)

    assert hash1 != hash2


def test_hash_file_missing_raises(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.py"

    with pytest.raises(FileNotFoundError):
        hash_file(missing_file)


# ============================================================
# CodeChunkRecord sanity checks
# ============================================================


def test_code_chunk_record_creation():
    record = CodeChunkRecord(
        file_path="example.py",
        start_line=1,
        end_line=5,
        content="print('hello')",
        content_hash=hash_text("print('hello')"),
        embedding=[0.1, 0.2, 0.3],
        embedding_model="text-embedding-3-small",
        embedding_dimensions=3,
    )

    assert record.file_path == "example.py"
    assert record.start_line == 1
    assert record.end_line == 5
    assert isinstance(record.embedding, list)
