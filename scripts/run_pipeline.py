"""Entry point: python -m scripts.run_pipeline"""

import argparse
import asyncio
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.main import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the long-context filtering pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of repos to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(limit=args.limit))


if __name__ == "__main__":
    main()
