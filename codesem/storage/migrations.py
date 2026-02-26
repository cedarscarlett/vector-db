# codesem/storage/migrations.py

"""Database migration utilities for CodeSem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import psycopg

from codesem.config.settings import get_settings


SCHEMA_FILENAME = "schema.sql"


def _get_schema_path() -> Path:
    """
    Resolve path to schema.sql relative to this file.
    """
    return Path(__file__).parent / SCHEMA_FILENAME


def run_migrations(database_url: Optional[str] = None) -> None:
    """
    Execute schema.sql against the configured database.

    This function:
    - Connects to DATABASE_URL
    - Reads schema.sql
    - Executes it in a single transaction
    - Commits on success

    Raises:
        ValueError: if DATABASE_URL is not configured
        FileNotFoundError: if schema.sql cannot be located
        psycopg.Error: for database-level failures
    """
    settings = get_settings()
    db_url = database_url or settings.database_url

    if not db_url:
        raise ValueError("DATABASE_URL must be set to run migrations.")

    schema_path = _get_schema_path()

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def reset_db(database_url: Optional[str] = None) -> None:
    """
    Drop and recreate all CodeSem tables.

    WARNING:
    This will delete all indexed chunks.

    Intended for:
    - Local development
    - Test environments
    - Benchmark resets

    It does NOT drop the entire database —
    only the `code_chunks` table and related indexes.
    """
    settings = get_settings()
    db_url = database_url or settings.database_url

    if not db_url:
        raise ValueError("DATABASE_URL must be set to reset database.")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("drop table if exists code_chunks cascade;")
        conn.commit()

    # Re-run schema to recreate
    run_migrations(database_url=db_url)
