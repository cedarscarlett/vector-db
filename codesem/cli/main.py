# codesem/cli/main.py
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, List, Dict, cast

import typer

app = typer.Typer(
    name="codesem",
    add_completion=False,
    no_args_is_help=True,
    help="CodeSem — semantic (natural-language) search over a codebase using pgvector.",
)


# ============================================================
# Backend Return Contracts (TypedDicts)
# ------------------------------------------------------------
# These define the interface boundary between CLI and backend.
# Backend modules (index_repo, estimate_index_cost, search_code,
# run_benchmark) MUST return dicts conforming to these shapes.
# ============================================================


class IndexResult(TypedDict):
    files_scanned: int
    chunks_total: int
    chunks_inserted: int
    chunks_skipped_unchanged: int
    chunks_deleted_stale: int
    embedding_model: str
    embedding_dimensions: Optional[int]


class CostEstimateResult(TypedDict):
    files_scanned: int
    chunks_total: int
    estimated_tokens_total: int
    embedding_model: str
    embedding_dimensions: Optional[int]
    estimated_cost_usd: float


class SearchResultItem(TypedDict):
    file_path: str
    start_line: int
    end_line: int
    score: float
    preview: Optional[str]
    content: Optional[str]


class BenchmarkResult(TypedDict, total=False):
    per_query: List[Dict[str, Any]]
    overall: Dict[str, Any]


# -----------------------------
# Shared helpers
# -----------------------------
def _eprint(msg: str) -> None:
    typer.echo(msg, err=True)


def _require_env(var_name: str) -> str:
    val = os.getenv(var_name)
    if not val:
        raise typer.BadParameter(
            f"Missing required environment variable: {var_name}. "
            f"Set it (or create a .env and load it in your shell)."
        )
    return val


def _path_exists(p: Path) -> Path:
    if not p.exists():
        raise typer.BadParameter(f"Path does not exist: {p}")
    return p


def _to_json(obj: Any) -> str:
    def default(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, indent=2, default=default)


@dataclass(frozen=True)
class CLIIndexSummary:
    repo_path: str
    files_scanned: int
    chunks_total: int
    chunks_inserted: int
    chunks_skipped_unchanged: int
    chunks_deleted_stale: int
    embedding_model: str
    embedding_dimensions: Optional[int]
    elapsed_sec: float


@dataclass(frozen=True)
class CLICostEstimate:
    repo_path: str
    files_scanned: int
    chunks_total: int
    estimated_tokens_total: int
    embedding_model: str
    embedding_dimensions: Optional[int]
    estimated_cost_usd: float


@dataclass(frozen=True)
class CLISearchHit:
    rank: int
    file_path: str
    start_line: int
    end_line: int
    score: float
    preview: str


# -----------------------------
# Commands
# -----------------------------
@app.command("index")
def cmd_index(
    repo_path: Path = typer.Argument(..., help="Path to the repository root."),
    embedding_model: str = typer.Option(
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "--embedding-model",
        help="Embedding model name.",
    ),
    embedding_dimensions: Optional[int] = typer.Option(
        None,
        "--dimensions",
        min=32,
        help="Optional reduced embedding dimensions (e.g. 512). If omitted, provider default is used.",
    ),
    chunk_token_size: int = typer.Option(
        int(os.getenv("CHUNK_TOKEN_SIZE", "400")),
        "--chunk-tokens",
        min=64,
        help="Chunk size in tokens.",
    ),
    chunk_token_overlap: int = typer.Option(
        int(os.getenv("CHUNK_TOKEN_OVERLAP", "50")),
        "--chunk-overlap",
        min=0,
        help="Token overlap between adjacent chunks.",
    ),
    max_file_mb: int = typer.Option(
        5,
        "--max-file-mb",
        min=1,
        help="Skip files larger than this size (MB).",
    ),
    delete_stale: bool = typer.Option(
        True,
        "--delete-stale/--keep-stale",
        help="On re-index, delete chunks for files no longer present.",
    ),
    exclude_default: bool = typer.Option(
        True,
        "--exclude-default/--no-exclude-default",
        help="Exclude common directories (.git, node_modules, venv, dist, build, etc.).",
    ),
    include_ext: list[str] = typer.Option(
        ["py", "ts", "js", "cs", "go"],
        "--ext",
        help="File extensions to include (repeatable). Example: --ext py --ext ts",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (do not suppress tracebacks).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """
    Index a repository into the vector store.

    Notes:
    - Requires DATABASE_URL and OPENAI_API_KEY (or provider key) env vars.
    - Skips unchanged chunks using content hashing.
    - Optionally deletes stale chunks for removed files.
    """
    repo_path = _path_exists(repo_path).resolve()

    # Ensure required env is present up front (better UX than crashing later).
    _require_env("DATABASE_URL")
    _require_env("OPENAI_API_KEY")

    t0 = time.time()

    try:
        # Lazy imports so this file can exist before the rest of the modules.
        from codesem.indexing.indexer import index_repo  # type: ignore
    except Exception as e:
        _eprint("Indexing module not available yet (codesem.indexing.indexer.index_repo).")
        _eprint(f"Import error: {e}")
        raise typer.Exit(code=2)

    try:
        result = index_repo(
            repo_path=str(repo_path),
            include_extensions=[x.lstrip(".").lower() for x in include_ext],
            exclude_default=exclude_default,
            chunk_token_size=chunk_token_size,
            chunk_token_overlap=chunk_token_overlap,
            max_file_bytes=max_file_mb * 1024 * 1024,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            delete_stale=delete_stale,
        )
        elapsed = time.time() - t0

        # Enforce strict return contract (must be dict-like)
        if not isinstance(result, dict):
            raise TypeError("index_repo must return dict with required fields.")

        result = cast(IndexResult, result)

        summary = CLIIndexSummary(
            repo_path=str(repo_path),
            files_scanned=int(result["files_scanned"]),
            chunks_total=int(result["chunks_total"]),
            chunks_inserted=int(result["chunks_inserted"]),
            chunks_skipped_unchanged=int(result["chunks_skipped_unchanged"]),
            chunks_deleted_stale=int(result["chunks_deleted_stale"]),
            embedding_model=str(result.get("embedding_model", embedding_model)),
            embedding_dimensions=result.get("embedding_dimensions", embedding_dimensions),
            elapsed_sec=elapsed,
        )

        if json_out:
            typer.echo(_to_json(asdict(summary)))
        else:
            typer.echo(f"Repo: {summary.repo_path}")
            typer.echo(f"Files scanned: {summary.files_scanned}")
            typer.echo(f"Chunks total: {summary.chunks_total}")
            typer.echo(f"Inserted: {summary.chunks_inserted}")
            typer.echo(f"Skipped (unchanged): {summary.chunks_skipped_unchanged}")
            typer.echo(f"Deleted (stale): {summary.chunks_deleted_stale}")
            typer.echo(f"Embedding model: {summary.embedding_model}")
            if summary.embedding_dimensions is not None:
                typer.echo(f"Embedding dimensions: {summary.embedding_dimensions}")
            typer.echo(f"Time: {summary.elapsed_sec:.2f}s")

    except typer.Exit:
        raise
    except Exception as e:
        if debug:
            raise
        _eprint(f"Indexing failed: {e}")
        raise typer.Exit(code=1)


@app.command("cost-estimate")
def cmd_cost_estimate(
    repo_path: Path = typer.Argument(..., help="Path to the repository root."),
    embedding_model: str = typer.Option(
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "--embedding-model",
        help="Embedding model name.",
    ),
    embedding_dimensions: Optional[int] = typer.Option(
        None,
        "--dimensions",
        min=32,
        help="Optional reduced embedding dimensions (e.g. 512).",
    ),
    chunk_token_size: int = typer.Option(
        int(os.getenv("CHUNK_TOKEN_SIZE", "400")),
        "--chunk-tokens",
        min=64,
        help="Chunk size in tokens.",
    ),
    chunk_token_overlap: int = typer.Option(
        int(os.getenv("CHUNK_TOKEN_OVERLAP", "50")),
        "--chunk-overlap",
        min=0,
        help="Token overlap between adjacent chunks.",
    ),
    max_file_mb: int = typer.Option(
        5,
        "--max-file-mb",
        min=1,
        help="Skip files larger than this size (MB).",
    ),
    exclude_default: bool = typer.Option(
        True,
        "--exclude-default/--no-exclude-default",
        help="Exclude common directories (.git, node_modules, venv, dist, build, etc.).",
    ),
    include_ext: list[str] = typer.Option(
        ["py", "ts", "js", "cs", "go"],
        "--ext",
        help="File extensions to include (repeatable). Example: --ext py --ext ts",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (do not suppress tracebacks).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """
    Estimate indexing cost (tokens + $) without writing to the DB.

    Intended as a production-thinking signal for clients.
    """
    repo_path = _path_exists(repo_path).resolve()

    # OPENAI_API_KEY not required unless implementation needs it

    try:
        from codesem.indexing.indexer import estimate_index_cost  # type: ignore
    except Exception as e:
        _eprint("Cost estimate module not available yet (codesem.indexing.indexer.estimate_index_cost).")
        _eprint(f"Import error: {e}")
        raise typer.Exit(code=2)

    try:
        result = estimate_index_cost(
            repo_path=str(repo_path),
            include_extensions=[x.lstrip(".").lower() for x in include_ext],
            exclude_default=exclude_default,
            chunk_token_size=chunk_token_size,
            chunk_token_overlap=chunk_token_overlap,
            max_file_bytes=max_file_mb * 1024 * 1024,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

        if not isinstance(result, dict):
            raise TypeError("estimate_index_cost must return dict with required fields.")

        result = cast(CostEstimateResult, result)

        est = CLICostEstimate(
            repo_path=str(repo_path),
            files_scanned=int(result["files_scanned"]),
            chunks_total=int(result["chunks_total"]),
            estimated_tokens_total=int(result["estimated_tokens_total"]),
            embedding_model=str(result.get("embedding_model", embedding_model)),
            embedding_dimensions=result.get("embedding_dimensions", embedding_dimensions),
            estimated_cost_usd=float(result["estimated_cost_usd"]),
        )

        if json_out:
            typer.echo(_to_json(asdict(est)))
        else:
            typer.echo(f"Repo: {est.repo_path}")
            typer.echo(f"Files scanned: {est.files_scanned}")
            typer.echo(f"Chunks (estimated): {est.chunks_total}")
            typer.echo(f"Tokens (estimated): {est.estimated_tokens_total:,}")
            typer.echo(f"Embedding model: {est.embedding_model}")
            if est.embedding_dimensions is not None:
                typer.echo(f"Embedding dimensions: {est.embedding_dimensions}")
            typer.echo(f"Estimated embedding cost: ${est.estimated_cost_usd:.4f}")

    except Exception as e:
        if debug:
            raise
        _eprint(f"Cost estimate failed: {e}")
        raise typer.Exit(code=1)


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural language query."),
    k: int = typer.Option(int(os.getenv("TOP_K_DEFAULT", "5")), "--k", min=1, help="Number of results to return."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid ranking (vector + keyword)."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (do not suppress tracebacks).",
    ),
) -> None:
    """
    Search indexed code semantically.
    """
    _require_env("DATABASE_URL")
    _require_env("OPENAI_API_KEY")

    if not query.strip():
        raise typer.BadParameter("Query must be non-empty.")

    try:
        from codesem.retrieval.search_service import search_code  # type: ignore
    except Exception as e:
        _eprint("Search module not available yet (codesem.retrieval.search_service.search_code).")
        _eprint(f"Import error: {e}")
        raise typer.Exit(code=2)

    try:
        t0 = time.time()
        results = search_code(query=query, top_k=k, hybrid=hybrid)
        elapsed = time.time() - t0

        hits: list[CLISearchHit] = []
        for i, r in enumerate(results, start=1):
            if not isinstance(r, dict):
                raise TypeError("search_code must return list[SearchResultItem].")

            r = cast(SearchResultItem, r)

            file_path = r["file_path"]
            start_line = int(r["start_line"])
            end_line = int(r["end_line"])
            score = float(r["score"])
            preview = r.get("preview") or r.get("content", "")
            hits.append(
                CLISearchHit(
                    rank=i,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    score=score,
                    preview=str(preview),
                )
            )

        if json_out:
            typer.echo(
                _to_json(
                    {
                        "query": query,
                        "k": k,
                        "hybrid": hybrid,
                        "elapsed_sec": elapsed,
                        "results": [asdict(h) for h in hits],
                    }
                )
            )
            return

        typer.echo(f'Query: "{query}"')
        typer.echo(f"Mode: {'hybrid' if hybrid else 'vector'} | k={k} | {elapsed:.2f}s")
        typer.echo()

        if not hits:
            typer.echo("No results.")
            return

        for h in hits:
            typer.echo(f"{h.rank}. {h.file_path} (lines {h.start_line}-{h.end_line})  score={h.score:.4f}")
            typer.echo("-" * 70)
            # Keep preview readable (don’t spam terminal).
            preview = h.preview.strip("\n")
            if len(preview) > 1200:
                preview = preview[:1200].rstrip() + "\n…"
            typer.echo(preview)
            typer.echo()

    except Exception as e:
        if debug:
            raise
        _eprint(f"Search failed: {e}")
        raise typer.Exit(code=1)


@app.command("benchmark")
def cmd_benchmark(
    repo_path: Optional[Path] = typer.Option(
        None,
        "--repo",
        help="Optional repo path (used if benchmark runner needs to run grep baseline).",
    ),
    queries_path: Path = typer.Option(
        Path("benchmarks/queries.json"),
        "--queries",
        help="Path to benchmark queries JSON.",
    ),
    k: int = typer.Option(5, "--k", min=1, help="Top-k for recall@k."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (do not suppress tracebacks).",
    ),
) -> None:
    """
    Run benchmark comparing semantic recall@k vs grep baseline.
    """
    _require_env("DATABASE_URL")
    _require_env("OPENAI_API_KEY")

    if repo_path is not None:
        repo_path = _path_exists(repo_path).resolve()

    if not queries_path.exists():
        raise typer.BadParameter(f"Queries file not found: {queries_path}")

    try:
        from codesem.benchmarking.benchmark_runner import run_benchmark  # type: ignore
    except Exception as e:
        _eprint("Benchmark module not available yet (codesem.benchmarking.benchmark_runner.run_benchmark).")
        _eprint(f"Import error: {e}")
        raise typer.Exit(code=2)

    try:
        result = run_benchmark(
            queries_path=str(queries_path),
            repo_path=str(repo_path) if repo_path else None,
            top_k=k,
        )

        result = cast(BenchmarkResult, result)

        if json_out:
            typer.echo(_to_json(result))
        else:
            # Expect result to contain per-query and overall summaries.
            overall = result.get("overall", {})
            per_query = result.get("per_query", [])

            for q in per_query:
                typer.echo(f'Query: {q.get("query")}')
                typer.echo(f'  Vector recall@{k}: {q.get("vector_recall_at_k")}')
                typer.echo(f'  Grep hit: {q.get("grep_hit")}')
                typer.echo()

            typer.echo("Overall:")
            typer.echo(f'  Vector avg recall@{k}: {overall.get("vector_avg_recall_at_k")}')
            typer.echo(f'  Grep hit rate: {overall.get("grep_hit_rate")}')
            if overall.get("elapsed_sec") is not None:
                typer.echo(f'  Time: {overall.get("elapsed_sec"):.2f}s')

    except Exception as e:
        if debug:
            raise
        _eprint(f"Benchmark failed: {e}")
        raise typer.Exit(code=1)


@app.command("reset")
def cmd_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm destructive reset."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (do not suppress tracebacks).",
    ),
) -> None:
    """
    Delete all indexed chunks (destructive).
    """
    _require_env("DATABASE_URL")

    if not yes:
        _eprint("Refusing to reset without confirmation. Re-run with --yes.")
        raise typer.Exit(code=2)

    try:
        from codesem.storage.migrations import reset_db  # type: ignore
    except Exception as e:
        _eprint("Reset module not available yet (codesem.storage.migrations.reset_db).")
        _eprint(f"Import error: {e}")
        raise typer.Exit(code=2)

    try:
        reset_db()
        typer.echo("OK: database reset (all chunks deleted).")
    except Exception as e:
        if debug:
            raise
        _eprint(f"Reset failed: {e}")
        raise typer.Exit(code=1)


def main() -> None:
    """
    Console entrypoint for `codesem`.
    """
    try:
        app()
    except KeyboardInterrupt:
        _eprint("Interrupted.")
        raise typer.Exit(code=130)


if __name__ == "__main__":
    main()
