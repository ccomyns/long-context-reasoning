"""Async orchestration: 3-phase pipeline."""

import asyncio
import logging
from typing import Any

from pipeline.config import MIN_REPO_SIZE_KB, MIN_CODE_TOKENS, MAX_CONCURRENT_CLONES
from pipeline.db import get_client, fetch_all_repos, fetch_processed_repo_ids, upsert_result
from pipeline.github_api import fetch_all_metadata
from pipeline.clone import ensure_clone_dir, shallow_clone, cleanup_clone
from pipeline.token_counter import count_tokens_in_repo
from pipeline.dependency_scorer import score_dependencies
from pipeline.logging_config import setup_logging

logger = logging.getLogger("pipeline.main")


async def run_pipeline(limit: int | None = None) -> None:
    """Run the full 3-phase pipeline.

    Args:
        limit: If set, only process this many repos (for testing).
    """
    setup_logging()
    client = get_client()

    # --- Phase 0: Load data and check resumability ---
    logger.info("Loading repos from Supabase...")
    all_repos = fetch_all_repos(client)
    if limit:
        all_repos = all_repos[:limit]

    processed_ids = fetch_processed_repo_ids(client)
    unprocessed = [r for r in all_repos if r["id"] not in processed_ids]
    logger.info(f"{len(unprocessed)} repos to process ({len(processed_ids)} already done)")

    if not unprocessed:
        logger.info("All repos already processed. Nothing to do.")
        return

    # --- Phase 1: GitHub API pre-filter ---
    logger.info("Phase 1: Fetching GitHub metadata...")
    metadata = await fetch_all_metadata(unprocessed)
    logger.info(f"Got metadata for {len(metadata)} repos")

    # Filter by size
    candidates = []
    for repo in unprocessed:
        full_name = repo.get("full_name") or repo.get("url", "").replace(
            "https://github.com/", ""
        )
        if not full_name:
            continue

        meta = metadata.get(full_name)
        if meta is None:
            # Record as failed
            _record_skipped(client, repo, full_name, "github_api_failed")
            continue

        if meta["size_kb"] < MIN_REPO_SIZE_KB:
            _record_skipped(
                client, repo, full_name, "skipped",
                error_msg=f"size_kb={meta['size_kb']} < {MIN_REPO_SIZE_KB}",
                meta=meta,
            )
            continue

        candidates.append((repo, full_name, meta))

    logger.info(
        f"Phase 1 complete: {len(candidates)} candidates "
        f"(filtered {len(unprocessed) - len(candidates)} repos)"
    )

    # --- Phase 2 & 3: Clone, count tokens, score dependencies ---
    logger.info("Phase 2/3: Cloning, counting tokens, scoring dependencies...")
    ensure_clone_dir()
    clone_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLONES)

    tasks = [
        _process_repo(client, repo, full_name, meta, clone_semaphore)
        for repo, full_name, meta in candidates
    ]

    completed = 0
    passed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result:
            passed += 1
        if completed % 50 == 0 or completed == len(tasks):
            logger.info(
                f"Progress: {completed}/{len(tasks)} processed, {passed} passed token threshold"
            )

    logger.info(f"Pipeline complete. {passed}/{len(tasks)} repos passed all filters.")


async def _process_repo(
    client: Any,
    repo: dict[str, Any],
    full_name: str,
    meta: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> bool:
    """Clone, count tokens, score dependencies for a single repo.

    Returns True if the repo passed all filters and was recorded.
    """
    clone_path = None
    try:
        clone_path = await shallow_clone(
            full_name, branch=meta.get("default_branch", "main"), semaphore=semaphore
        )
        if not clone_path:
            _record_skipped(
                client, repo, full_name, "clone_failed",
                error_msg="shallow clone failed", meta=meta,
            )
            return False

        # Count tokens
        token_result = count_tokens_in_repo(clone_path)

        if token_result.total_tokens < MIN_CODE_TOKENS:
            _record_skipped(
                client, repo, full_name, "skipped",
                error_msg=f"tokens={token_result.total_tokens} < {MIN_CODE_TOKENS}",
                meta=meta,
            )
            return False

        # Score dependencies
        dep_result = score_dependencies(clone_path)

        # Write result
        record = {
            "repo_id": repo["id"],
            "full_name": full_name,
            "url": repo.get("url", f"https://github.com/{full_name}"),
            "github_size_kb": meta["size_kb"],
            "primary_language": meta.get("language"),
            "default_branch": meta.get("default_branch"),
            "total_code_tokens": token_result.total_tokens,
            "total_code_files": token_result.total_files,
            "total_code_bytes": token_result.total_bytes,
            "tokens_by_extension": token_result.tokens_by_extension,
            "dependency_score": round(dep_result.dependency_score, 4),
            "internal_import_count": dep_result.internal_import_count,
            "unique_internal_imports": dep_result.unique_internal_imports,
            "graph_density": round(dep_result.graph_density, 6) if dep_result.graph_density else None,
            "avg_in_degree": round(dep_result.avg_in_degree, 4) if dep_result.avg_in_degree else None,
            "max_in_degree": dep_result.max_in_degree,
            "num_connected_components": dep_result.num_connected_components,
            "largest_component_fraction": (
                round(dep_result.largest_component_fraction, 4)
                if dep_result.largest_component_fraction
                else None
            ),
            "conference": repo.get("conference"),
            "year": repo.get("year"),
            "paper_title": repo.get("paper_title"),
            "processing_status": "completed",
        }
        upsert_result(client, record)
        logger.info(
            f"[PASS] {full_name}: {token_result.total_tokens} tokens, "
            f"dep_score={dep_result.dependency_score:.3f}"
        )
        return True

    except Exception as e:
        logger.error(f"Error processing {full_name}: {e}", exc_info=True)
        _record_skipped(
            client, repo, full_name, "error",
            error_msg=str(e)[:500], meta=meta,
        )
        return False
    finally:
        if clone_path:
            cleanup_clone(clone_path)


def _record_skipped(
    client: Any,
    repo: dict[str, Any],
    full_name: str,
    status: str,
    error_msg: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Record a skipped/failed repo so it isn't reprocessed."""
    record = {
        "repo_id": repo["id"],
        "full_name": full_name,
        "url": repo.get("url", f"https://github.com/{full_name}"),
        "github_size_kb": meta.get("size_kb", 0) if meta else 0,
        "primary_language": meta.get("language") if meta else None,
        "default_branch": meta.get("default_branch") if meta else None,
        "total_code_tokens": 0,
        "total_code_files": 0,
        "total_code_bytes": 0,
        "dependency_score": 0.0,
        "internal_import_count": 0,
        "unique_internal_imports": 0,
        "conference": repo.get("conference"),
        "year": repo.get("year"),
        "paper_title": repo.get("paper_title"),
        "processing_status": status,
        "error_message": error_msg,
    }
    try:
        upsert_result(client, record)
    except Exception as e:
        logger.error(f"Failed to record skip for {full_name}: {e}")
