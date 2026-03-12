#!/usr/bin/env python3
"""CLI: Evaluate agent pilot signals against deep-trading metrics."""

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
        description="Evaluate agent pilot signals with deep-trading metrics",
    )
    parser.add_argument(
        "--signals", required=True,
        help="Path to signals.csv from the pilot run",
    )
    parser.add_argument(
        "--data", default=str(deep_root / "data" / "BTCUSDT_1h.parquet"),
        help="Path to BTCUSDT_1h.parquet",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for evaluation artifacts (defaults to same dir as signals)",
    )
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()

    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"ERROR: signals file not found: {signals_path}")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: OHLCV data not found: {data_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else signals_path.parent

    from agent_experiment.experiment.evaluate import evaluate_pilot

    result = evaluate_pilot(
        signals_path=signals_path,
        data_path=data_path,
        output_dir=output_dir,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - agent_metrics.json")
    print(f"  - agent_metrics.csv")
    print(f"  - agent_backtest.csv")
    print(f"  - summary_metrics.md")


if __name__ == "__main__":
    main()
