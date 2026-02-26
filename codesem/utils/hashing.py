"""SHA-256 content and file hashing for chunk deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path


# ============================================================
# Content Hashing
# ============================================================


def hash_text(text: str) -> str:
    """
    Compute a stable SHA-256 hash of a text string.

    Used for:
    - Chunk-level deduplication
    - Skipping re-embedding unchanged content

    Returns:
        Hex digest string.
    """
    if not isinstance(text, str):
        raise TypeError("hash_text expects a string")

    hasher = hashlib.sha256()
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def hash_file(path: str | Path) -> str:
    """
    Compute SHA-256 hash of a file's raw bytes.

    Used for:
    - Detecting file-level changes
    - Re-index skip logic

    Returns:
        Hex digest string.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    hasher = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
