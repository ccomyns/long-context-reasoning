"""Shallow clone management."""

import asyncio
import logging
import os
import shutil

from pipeline.config import CLONE_BASE_DIR

logger = logging.getLogger("pipeline.clone")


def ensure_clone_dir() -> None:
    os.makedirs(CLONE_BASE_DIR, exist_ok=True)


async def shallow_clone(
    full_name: str,
    branch: str = "main",
    semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Shallow clone a repo. Returns the clone path or None on failure."""

    async def _clone() -> str | None:
        safe_name = full_name.replace("/", "_")
        clone_path = os.path.join(CLONE_BASE_DIR, safe_name)

        # Clean up any leftover from previous run
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)

        url = f"https://github.com/{full_name}.git"
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, url, clone_path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            if proc.returncode != 0:
                # Try without specifying branch (some repos have non-standard defaults)
                cmd_fallback = [
                    "git", "clone", "--depth", "1", url, clone_path,
                ]
                if os.path.exists(clone_path):
                    shutil.rmtree(clone_path)
                proc2 = await asyncio.create_subprocess_exec(
                    *cmd_fallback,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr2 = await asyncio.wait_for(proc2.communicate(), timeout=120)
                if proc2.returncode != 0:
                    logger.error(
                        f"Clone failed for {full_name}: {stderr2.decode(errors='replace')}"
                    )
                    return None

            logger.debug(f"Cloned {full_name}")
            return clone_path

        except asyncio.TimeoutError:
            logger.error(f"Clone timed out for {full_name}")
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)
            return None
        except Exception as e:
            logger.error(f"Clone error for {full_name}: {e}")
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)
            return None

    if semaphore:
        async with semaphore:
            return await _clone()
    return await _clone()


def cleanup_clone(clone_path: str) -> None:
    """Remove a cloned repo directory."""
    try:
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)
    except Exception as e:
        logger.warning(f"Failed to clean up {clone_path}: {e}")
