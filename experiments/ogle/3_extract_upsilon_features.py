#!/usr/bin/env python3
"""Extract the official 16 UPSILoN features from a frozen split manifest."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

SURVEY = "ogle"
FEATURE_NAMES = [
    "amplitude", "hl_amp_ratio", "kurtosis", "period", "phase_cusum",
    "phase_eta", "phi21", "phi31", "quartile31", "r21", "r31",
    "shapiro_w", "skewness", "slope_per10", "slope_per90", "stetson_k",
]


def extract_one(task):
    row, feature_names, use_sigma_clipping = task
    object_id = str(row["object_id"])
    try:
        import upsilon

        frame = pd.read_csv(row["source_path"], sep=r"\s+", names=["time", "mag", "err"])
        values = frame[["time", "mag", "err"]].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        nobs_input = len(values)
        if len(values) < 10:
            raise ValueError(f"only {len(values)} finite observations")
        if use_sigma_clipping:
            date, mag, err = upsilon.utils.sigma_clipping(
                values[:, 0], values[:, 1], values[:, 2], threshold=3, iteration=1
            )
            values = np.column_stack((date, mag, err))
        if len(values) < 10:
            raise ValueError(f"only {len(values)} observations after UPSILoN sigma clipping")
        extractor = upsilon.ExtractFeatures(values[:, 0], values[:, 1], values[:, 2], n_threads=1)
        extractor.run()
        extracted = extractor.get_features()
        missing = [name for name in feature_names if name not in extracted]
        if missing:
            raise KeyError(f"missing official UPSILoN features: {missing}")
        result = {
            "object_id": object_id,
            "split": str(row["split"]),
            "y": int(row["class_index"]),
            "source_path": str(row["source_path"]),
            "nobs_input": int(nobs_input),
            "nobs_after_sigma_clip": int(len(values)),
            "extraction_error": "",
        }
        for name in feature_names:
            result[name] = float(extracted[name])
        return result
    except Exception as exc:
        result = {
            "object_id": object_id,
            "split": str(row.get("split", "")),
            "y": int(row.get("class_index", -1)),
            "source_path": str(row.get("source_path", "")),
            "nobs_input": 0,
            "nobs_after_sigma_clip": 0,
            "extraction_error": repr(exc),
        }
        for name in feature_names:
            result[name] = np.nan
        return result


def atomic_write(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data/processed/ogle_tdmn.manifest.csv"),
        help="Real-object manifest emitted by the matching TDMN data preparation",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/processed/ogle_upsilon_16_features.csv"),
    )
    parser.add_argument("--errors-output", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-sigma-clipping", action="store_true")
    args = parser.parse_args()

    feature_names = FEATURE_NAMES
    manifest = pd.read_csv(args.manifest, dtype={"object_id": str})
    required = {"object_id", "split", "class_index", "source_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest needs columns {sorted(required)}")
    if "origin" in manifest:
        manifest = manifest[manifest["origin"] == "real"].copy()
    if manifest["object_id"].duplicated().any():
        raise ValueError("Real-object manifest contains duplicate object_id values")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val, and test")

    existing = pd.DataFrame()
    output_path = Path(args.output)
    if args.resume and output_path.is_file():
        existing = pd.read_csv(output_path, dtype={"object_id": str})
        completed = set(existing["object_id"].astype(str))
        manifest = manifest[~manifest["object_id"].astype(str).isin(completed)]

    rows = existing.to_dict(orient="records")
    tasks = [
        (row, feature_names, not args.no_sigma_clipping)
        for row in manifest.to_dict(orient="records")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(extract_one, tasks, chunksize=1), start=1):
            rows.append(result)
            if index % args.checkpoint_every == 0:
                atomic_write(pd.DataFrame(rows), output_path)
                print(f"checkpoint: {len(rows)} objects", flush=True)

    frame = pd.DataFrame(rows)
    atomic_write(frame, output_path)
    errors_path = Path(args.errors_output) if args.errors_output else output_path.with_suffix(".errors.csv")
    atomic_write(frame[frame["extraction_error"] != ""], errors_path)
    successful = frame[frame["extraction_error"] == ""]
    report = {
        "survey": SURVEY,
        "objects_in_manifest": int(len(frame)),
        "successful": int(len(successful)),
        "failed": int((frame["extraction_error"] != "").sum()),
        "feature_names": feature_names,
        "upsilon_period_source": "estimated internally by the official ExtractFeatures implementation",
        "upsilon_sigma_clipping": not args.no_sigma_clipping,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
