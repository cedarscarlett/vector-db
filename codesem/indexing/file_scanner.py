"""
Recursively scan a repository and return index-eligible
file paths based on extension, size, and directory filters.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Set

from codesem.config.settings import get_settings


# ============================================================
# File Scanner
# ============================================================


_DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
}


def _should_exclude_dir(dirname: str, exclude_default: bool) -> bool:
    if not exclude_default:
        return False
    return dirname in _DEFAULT_EXCLUDED_DIRS


def scan_repository(
    repo_path: str | Path,
    include_extensions: Iterable[str] | None = None,
    exclude_default_dirs: bool | None = None,
    max_file_bytes: int | None = None,
) -> List[Path]:
    """
    Recursively scan a repository and return a list of file paths
    eligible for indexing.

    Filtering rules:
    - Only include files whose extension matches include_extensions
    - Skip directories in the default exclusion list (if enabled)
    - Skip files larger than max_file_bytes (if provided)

    Returns:
        List of absolute Path objects.
    """
    settings = get_settings()

    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    extensions: Set[str] = set(
        (include_extensions or settings.include_extensions)
    )
    extensions = {ext.lower().lstrip(".") for ext in extensions}

    exclude_dirs = (
        settings.exclude_default_dirs
        if exclude_default_dirs is None
        else exclude_default_dirs
    )

    max_bytes = (
        settings.max_file_bytes
        if max_file_bytes is None
        else max_file_bytes
    )

    matched_files: List[Path] = []

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)

        # Modify dirs in-place to prevent walking excluded directories
        dirs[:] = [
            d for d in dirs
            if not _should_exclude_dir(d, exclude_dirs)
        ]

        for filename in files:
            file_path = root_path / filename

            # Extension filter
            ext = file_path.suffix.lower().lstrip(".")
            if extensions and ext not in extensions:
                continue

            # Size filter
            try:
                if max_bytes is not None:
                    size = file_path.stat().st_size
                    if size > max_bytes:
                        continue
            except OSError:
                continue

            matched_files.append(file_path)

    return matched_files
