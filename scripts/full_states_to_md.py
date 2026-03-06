#!/usr/bin/env python3
"""
Convert full_states_log JSON files to Markdown.

Use as a standalone script:
  python scripts/full_states_to_md.py path/to/full_states_log_2024-05-10.json
  python scripts/full_states_to_md.py eval_results/NVDA/TradingAgentsStrategy_logs/*.json

Or import and call:
  from tradingagents.graph.log_utils import full_states_json_to_md
  full_states_json_to_md("path/to/file.json")
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tradingagents.graph.log_utils import full_states_json_to_md


def main():
    parser = argparse.ArgumentParser(
        description="Convert full_states_log JSON files to Markdown."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="JSON file(s) to convert (e.g. full_states_log_2024-05-10.json)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as input file)",
    )
    args = parser.parse_args()

    for json_file in args.json_files:
        p = Path(json_file)
        if not p.exists():
            print(f"Skip (not found): {p}", file=sys.stderr)
            continue
        if not p.suffix.lower() == ".json":
            print(f"Skip (not .json): {p}", file=sys.stderr)
            continue
        try:
            md_path = full_states_json_to_md(
                p,
                md_path=Path(args.output_dir) / p.with_suffix(".md").name if args.output_dir else None,
            )
            print(f"Wrote: {md_path}")
        except Exception as e:
            print(f"Error converting {p}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
