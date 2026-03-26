"""Supabase read/write helpers with pagination."""

import logging
from typing import Any

from supabase import create_client, Client

from pipeline.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_PAGE_SIZE

logger = logging.getLogger("pipeline.db")


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_repos(client: Client) -> list[dict[str, Any]]:
    """Fetch all rows from github_repositories with pagination."""
    all_rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = (
            client.table("github_repositories")
            .select("*")
            .range(offset, offset + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        logger.info(f"Fetched {len(all_rows)} repos so far...")
        if len(rows) < SUPABASE_PAGE_SIZE:
            break
        offset += SUPABASE_PAGE_SIZE

    logger.info(f"Total repos fetched: {len(all_rows)}")
    return all_rows


def fetch_processed_repo_ids(client: Client) -> set[str]:
    """Get repo_ids already in long_context_repositories (for resumability)."""
    processed = set()
    offset = 0

    while True:
        response = (
            client.table("long_context_repositories")
            .select("repo_id")
            .range(offset, offset + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        for row in rows:
            processed.add(row["repo_id"])
        if len(rows) < SUPABASE_PAGE_SIZE:
            break
        offset += SUPABASE_PAGE_SIZE

    logger.info(f"Already processed: {len(processed)} repos")
    return processed


def upsert_result(client: Client, record: dict[str, Any]) -> None:
    """Upsert a single result into long_context_repositories."""
    client.table("long_context_repositories").upsert(
        record, on_conflict="repo_id"
    ).execute()
