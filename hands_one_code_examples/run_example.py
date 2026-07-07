from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one hands_one_code_examples script by file path."
    )
    parser.add_argument(
        "path",
        help=(
            "Relative path to a Python file, for example "
            "hands_one_code_examples/7_2_multi_agent.py"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
