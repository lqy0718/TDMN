#!/usr/bin/env python3
"""Audit raw CRTS objects before expensive smoothing/augmentation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-observations", type=int, default=95)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    metadata_path = root / "data/SSS_Per_Tab.dat"
    data_root = root / "data/cartlinDR2/original_data/type"
    if not metadata_path.is_file() or not data_root.is_dir():
        raise FileNotFoundError(
            f"Expected {metadata_path} and {data_root}; use the server project root."
        )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(
        metadata_path,
        header=None,
        skiprows=3,
        sep=r"\s+",
        usecols=range(9),
        names=["SSS_ID", "ID", "RA", "Dec", "Period", "V_CSS", "Npts", "V_amp", "Type"],
    ).set_index("ID", drop=False)

    rows = []
    for type_dir in [data_root / str(value) for value in range(1, 11)] + [data_root / "12"]:
        for path in sorted(type_dir.glob("*.dat")):
            if "_" in path.stem:
                continue
            try:
                object_id = int(path.stem)
                frame = pd.read_csv(path, sep=r"\s+", names=["mjd", "mag", "err"])
                finite = np.isfinite(frame[["mjd", "mag", "err"]]).all(axis=1)
                times = frame.loc[finite, "mjd"].to_numpy()
                meta = metadata.loc[object_id] if object_id in metadata.index else None
                label = int(meta["Type"]) if meta is not None else int(type_dir.name)
                label = 11 if label == 12 else label
                period = float(meta["Period"]) if meta is not None else np.nan
                span = float(np.ptp(times)) if len(times) else np.nan
                rows.append(
                    {
                        "object_id": object_id,
                        "class_id": label,
                        "source_path": str(path.resolve()),
                        "nobs_raw": int(len(frame)),
                        "nobs_finite": int(finite.sum()),
                        "duplicate_times": int(pd.Series(times).duplicated().sum()),
                        "time_span_days": span,
                        "period_days": period,
                        "cycles_covered": span / period if period > 0 else np.nan,
                        "raw_eligible": bool(finite.sum() >= args.min_observations),
                        "read_error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "object_id": path.stem,
                        "class_id": type_dir.name,
                        "source_path": str(path.resolve()),
                        "raw_eligible": False,
                        "read_error": repr(exc),
                    }
                )

    objects = pd.DataFrame(rows)
    objects.to_csv(output / "raw_object_audit.csv", index=False)
    summary = (
        objects.groupby("class_id", dropna=False)
        .agg(
            objects=("object_id", "size"),
            raw_eligible=("raw_eligible", "sum"),
            median_nobs=("nobs_finite", "median"),
            median_time_span_days=("time_span_days", "median"),
            median_cycles_covered=("cycles_covered", "median"),
            read_errors=("read_error", lambda values: int((values != "").sum())),
        )
        .reset_index()
    )
    summary.to_csv(output / "raw_class_summary.csv", index=False)
    report = {
        "project_root": str(root),
        "minimum_observations": args.min_observations,
        "objects_found": int(len(objects)),
        "raw_eligible": int(objects["raw_eligible"].sum()),
        "read_errors": int((objects["read_error"] != "").sum()),
        "class_summary": summary.to_dict(orient="records"),
        "note": "Final eligibility is determined after SuperSmoother/outlier rejection by experiments/crts/1_prepare_crts_tdmn.py.",
    }
    (output / "raw_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
