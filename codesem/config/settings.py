# codesem/config/settings.py

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


# ============================================================
# Settings
# ------------------------------------------------------------
# Centralized configuration loader for CodeSem.
#
# - Reads environment variables
# - Applies sane defaults
# - Performs light validation
# - Exposes a cached Settings object
#
# All other modules should import `get_settings()` rather than
# reading os.environ directly.
# ============================================================


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        warnings.warn(
            f"Invalid integer value '{value}' for environment variable. "
            f"Falling back to default={default}.",
            RuntimeWarning,
        )
        return default


def _parse_csv(value: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    parsed = tuple(
        v.strip().lstrip(".").lower() for v in value.split(",") if v.strip()
    )
    return parsed or default


@dataclass(frozen=True)
class Settings:
    # =============================
    # Core Environment
    # =============================
    database_url: Optional[str]
    openai_api_key: Optional[str]

    # =============================
    # Embeddings
    # =============================
    embedding_model: str
    embedding_dimensions: Optional[int]

    # =============================
    # Chunking
    # =============================
    chunk_token_size: int
    chunk_token_overlap: int
    max_file_bytes: int

    # =============================
    # Indexing Behavior
    # =============================
    include_extensions: Tuple[str, ...]
    exclude_default_dirs: bool
    delete_stale_on_reindex: bool

    # =============================
    # Retrieval
    # =============================
    top_k_default: int
    hybrid_weight_vector: float
    hybrid_weight_keyword: float

    # =============================
    # Debug / Logging
    # =============================
    debug: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings from environment variables.

    This function should be called once at application start
    or lazily within modules that need configuration.
    """

    # -----------------------------
    # Required-ish (validated elsewhere)
    # -----------------------------
    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # -----------------------------
    # Embeddings
    # -----------------------------
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    embedding_dimensions_env = os.getenv("EMBEDDING_DIMENSIONS")
    embedding_dimensions: Optional[int] = None
    if embedding_dimensions_env is not None:
        try:
            embedding_dimensions = int(embedding_dimensions_env)
        except ValueError:
            warnings.warn(
                f"Invalid EMBEDDING_DIMENSIONS='{embedding_dimensions_env}'. "
                "Falling back to provider default (None).",
                RuntimeWarning,
            )

    # -----------------------------
    # Chunking
    # -----------------------------
    chunk_token_size = _parse_int(os.getenv("CHUNK_TOKEN_SIZE"), 400)
    chunk_token_overlap = _parse_int(os.getenv("CHUNK_TOKEN_OVERLAP"), 50)

    # 5 MB default max file size
    max_file_mb = _parse_int(os.getenv("MAX_FILE_MB"), 5)
    max_file_bytes = max_file_mb * 1024 * 1024

    # -----------------------------
    # Indexing behavior
    # -----------------------------
    include_extensions = _parse_csv(
        os.getenv("INCLUDE_EXTENSIONS"),
        ("py", "ts", "js", "cs", "go"),
    )

    exclude_default_dirs = _parse_bool(
        os.getenv("EXCLUDE_DEFAULT_DIRS"),
        True,
    )

    delete_stale_on_reindex = _parse_bool(
        os.getenv("DELETE_STALE_ON_REINDEX"),
        True,
    )

    # -----------------------------
    # Retrieval
    # -----------------------------
    top_k_default = _parse_int(os.getenv("TOP_K_DEFAULT"), 5)

    # Hybrid weighting
    # Defaults must sum to 1.0 for clarity.
    hybrid_weight_vector = float(os.getenv("HYBRID_WEIGHT_VECTOR", "0.7"))
    hybrid_weight_keyword = float(os.getenv("HYBRID_WEIGHT_KEYWORD", "0.3"))

    # -----------------------------
    # Debug
    # -----------------------------
    # NOTE:
    # CLI --debug flag should take precedence over CODESEM_DEBUG env var.
    # This module only reads env var; CLI layer must override explicitly if needed.
    debug = _parse_bool(os.getenv("CODESEM_DEBUG"), False)

    # -----------------------------
    # Validation
    # -----------------------------
    if chunk_token_overlap >= chunk_token_size:
        raise ValueError(
            "CHUNK_TOKEN_OVERLAP must be smaller than CHUNK_TOKEN_SIZE."
        )

    total_weight = hybrid_weight_vector + hybrid_weight_keyword

    if total_weight <= 0:
        raise ValueError(
            "Hybrid weights must sum to a positive value."
        )

    # Enforce that weights approximately sum to 1.0
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            "HYBRID_WEIGHT_VECTOR and HYBRID_WEIGHT_KEYWORD must sum to 1.0."
        )

    return Settings(
        database_url=database_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        chunk_token_size=chunk_token_size,
        chunk_token_overlap=chunk_token_overlap,
        max_file_bytes=max_file_bytes,
        include_extensions=include_extensions,
        exclude_default_dirs=exclude_default_dirs,
        delete_stale_on_reindex=delete_stale_on_reindex,
        top_k_default=top_k_default,
        hybrid_weight_vector=hybrid_weight_vector,
        hybrid_weight_keyword=hybrid_weight_keyword,
        debug=debug,
    )
