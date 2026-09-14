#!/usr/bin/env python3
"""Aggregate per-seed comparison metrics without mixing survey/label levels."""

import argparse
import re
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.comparison_root).expanduser().resolve()
    rows = []
    for path in sorted(root.glob("seed_*/**/overall_metrics.csv")):
        match = re.search(r"seed_(\d+)", str(path))
        if not match:
            continue
        level = path.parent.name
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(match.group(1)))
        frame.insert(1, "level", level)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No seed_*/**/overall_metrics.csv under {root}")
    combined = pd.concat(rows, ignore_index=True)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "all_seed_metrics.csv", index=False)
    numeric = [column for column in combined.select_dtypes(include="number").columns if column != "seed"]
    summary = combined.groupby(["level", "method"])[numeric].agg(["mean", "std", "min", "max"])
    summary.to_csv(output / "metric_summary_mean_std.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
