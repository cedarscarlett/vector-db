"""
Chunk source code into deterministic, token-aware
overlapping segments with line tracking and stable content hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import tiktoken

from codesem.config.settings import get_settings
from codesem.utils.hashing import hash_text


# ============================================================
# Data Model
# ============================================================


@dataclass(frozen=True)
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str


# ============================================================
# Chunker
# ============================================================


class CodeChunker:
    """
    Token-based sliding window chunker for source files.

    Design goals:
    - Deterministic chunk boundaries
    - Token-aware sizing (model-compatible)
    - Line-range tracking
    - Overlap support
    - Stable content hashing per chunk

    Notes:
    - Token counting uses tiktoken with cl100k_base encoding.
    - Line mapping is approximate: token windows are mapped back
      to line numbers via cumulative token counts per line.
    """

    def __init__(
        self,
        chunk_token_size: int | None = None,
        chunk_token_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        settings = get_settings()

        self.chunk_size = (
            chunk_token_size
            if chunk_token_size is not None
            else settings.chunk_token_size
        )
        self.chunk_overlap = (
            chunk_token_overlap
            if chunk_token_overlap is not None
            else settings.chunk_token_overlap
        )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_token_overlap must be smaller than chunk_token_size"
            )

        self.encoding = tiktoken.get_encoding(encoding_name)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def chunk_file(self, file_path: str | Path) -> List[Chunk]:
        """
        Chunk a file from disk into token-based segments.

        Returns:
            List[Chunk]
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        return self.chunk_text(text, file_path=str(file_path))

    def chunk_text(self, text: str, file_path: str) -> List[Chunk]:
        """
        Chunk raw text into token windows.

        This is the core logic used by chunk_file.
        """
        if not text:
            return []

        lines = text.splitlines()
        if not lines:
            return []

        # Encode full text
        full_tokens = self.encoding.encode(text)
        total_tokens = len(full_tokens)

        if total_tokens == 0:
            return []

        # Precompute cumulative token counts per line
        line_token_counts: List[int] = []
        for line in lines:
            # Add newline to preserve realistic tokenization
            encoded = self.encoding.encode(line + "\n")
            line_token_counts.append(len(encoded))

        cumulative: List[int] = []
        running = 0
        for count in line_token_counts:
            running += count
            cumulative.append(running)

        chunks: List[Chunk] = []

        stride = self.chunk_size - self.chunk_overlap
        start_token = 0

        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_size, total_tokens)

            token_slice = full_tokens[start_token:end_token]
            chunk_text = self.encoding.decode(token_slice)

            start_line = self._token_to_line(start_token, cumulative)
            end_line = self._token_to_line(end_token - 1, cumulative)

            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_text,
                    content_hash=hash_text(chunk_text),
                )
            )

            if end_token == total_tokens:
                break

            start_token += stride

        return chunks

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _token_to_line(token_index: int, cumulative: List[int]) -> int:
        """
        Map a token index to a 1-based line number using cumulative
        token counts per line.
        """
        for i, token_count in enumerate(cumulative):
            if token_index < token_count:
                return i + 1

        return len(cumulative)
