"""GitHub API helpers with rate limiting."""

import asyncio
import logging
from typing import Any

import httpx

from pipeline.config import GITHUB_TOKEN, GITHUB_API_BASE, GITHUB_RATE_LIMIT_BUFFER

logger = logging.getLogger("pipeline.github_api")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


async def fetch_repo_metadata(
    client: httpx.AsyncClient,
    full_name: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch repo metadata from GitHub API. Returns None on failure."""
    async with semaphore:
        url = f"{GITHUB_API_BASE}/repos/{full_name}"
        try:
            response = await client.get(url, headers=HEADERS)

            # Handle rate limiting
            remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
            if remaining < GITHUB_RATE_LIMIT_BUFFER:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                import time
                sleep_duration = max(reset_time - int(time.time()) + 1, 1)
                logger.warning(
                    f"Rate limit low ({remaining} remaining), sleeping {sleep_duration}s"
                )
                await asyncio.sleep(sleep_duration)

            if response.status_code == 404:
                logger.warning(f"Repo not found: {full_name}")
                return None
            if response.status_code == 403:
                # Rate limited or access denied
                logger.warning(f"Access denied for {full_name}: {response.status_code}")
                return None

            response.raise_for_status()
            data = response.json()

            return {
                "full_name": full_name,
                "size_kb": data.get("size", 0),
                "language": data.get("language"),
                "default_branch": data.get("default_branch", "main"),
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {full_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {full_name}: {e}")
            return None


async def fetch_all_metadata(
    repos: list[dict[str, Any]],
    max_concurrent: int = 20,
) -> dict[str, dict[str, Any]]:
    """Fetch metadata for all repos concurrently. Returns dict keyed by full_name."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for repo in repos:
            full_name = repo.get("full_name") or repo.get("url", "").replace(
                "https://github.com/", ""
            )
            if not full_name:
                continue
            tasks.append((full_name, fetch_repo_metadata(client, full_name, semaphore)))

        for i in range(0, len(tasks), 100):
            batch = tasks[i : i + 100]
            batch_results = await asyncio.gather(
                *[t[1] for t in batch], return_exceptions=True
            )
            for (name, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Exception for {name}: {result}")
                elif result is not None:
                    results[name] = result

            logger.info(
                f"GitHub API progress: {min(i + 100, len(tasks))}/{len(tasks)}"
            )

    return results
