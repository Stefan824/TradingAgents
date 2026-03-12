#!/usr/bin/env python3
"""CLI: Compare agent pilot results with forecasting baselines on pilot windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parents[2]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

deep_root = pkg_root.parents[1] / "deep-trading"
if deep_root.exists() and str(deep_root) not in sys.path:
    sys.path.insert(0, str(deep_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare agent pilot metrics with forecasting models on pilot windows",
    )
    parser.add_argument(
        "--agent-dir", required=True,
        help="Path to agent output dir (e.g. outputs/20260311T231326Z)",
    )
    parser.add_argument(
        "--deep-artifacts", default=str(deep_root / "artifacts"),
        help="Path to deep-trading artifacts directory",
    )
    parser.add_argument(
        "--run-id", default="full_universe_2015_2025_20260309",
        help="Deep-trading run ID",
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Symbol folder in artifacts",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: same as --agent-dir)",
    )
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    if not agent_dir.exists():
        print(f"ERROR: Agent dir not found: {agent_dir}")
        sys.exit(1)

    deep_artifacts = Path(args.deep_artifacts)
    if not deep_artifacts.exists():
        print(f"ERROR: Deep-trading artifacts not found: {deep_artifacts}")
        sys.exit(1)

    from agent_experiment.experiment.compare import run_compare

    md_path = run_compare(
        agent_dir=agent_dir,
        deep_artifacts_dir=deep_artifacts,
        run_id=args.run_id,
        symbol=args.symbol,
        output_path=Path(args.output) if args.output else None,
    )

    print(f"Comparison written to: {md_path}")


if __name__ == "__main__":
    main()
