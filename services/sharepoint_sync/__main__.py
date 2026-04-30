from __future__ import annotations

import argparse
import asyncio

from .main import run_loop


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="services.sharepoint_sync")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync pass and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse()
    asyncio.run(run_loop(once=args.once))


if __name__ == "__main__":
    main()
