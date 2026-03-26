"""Walk repo files and count tokens with tiktoken."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from pipeline.config import (
    SOURCE_EXTENSIONS,
    SKIP_EXTENSIONS,
    SKIP_DIRS,
    MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger("pipeline.token_counter")

# Initialize encoder once
_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class TokenCountResult:
    total_tokens: int = 0
    total_files: int = 0
    total_bytes: int = 0
    tokens_by_extension: dict[str, int] = field(default_factory=dict)


def count_tokens_in_repo(repo_path: str) -> TokenCountResult:
    """Walk a cloned repo and count tokens in source code files."""
    result = TokenCountResult()

    for root, dirs, files in os.walk(repo_path):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()

            # Skip non-source and binary files
            if ext in SKIP_EXTENSIONS:
                continue
            if ext not in SOURCE_EXTENSIONS:
                continue

            filepath = os.path.join(root, fname)

            try:
                file_size = os.path.getsize(filepath)
                if file_size > MAX_FILE_SIZE_BYTES:
                    continue
                if file_size == 0:
                    continue

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                tokens = len(_encoder.encode(content, disallowed_special=()))
                result.total_tokens += tokens
                result.total_files += 1
                result.total_bytes += file_size
                result.tokens_by_extension[ext] = (
                    result.tokens_by_extension.get(ext, 0) + tokens
                )

            except (OSError, UnicodeDecodeError) as e:
                logger.debug(f"Skipping {filepath}: {e}")
                continue

    return result
