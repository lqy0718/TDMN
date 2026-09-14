#!/usr/bin/env python3
"""Validate CRTS or OGLE TDMN HDF5 files against the shared experiment contract."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE.parent / "configs" / "experiment_contract.json"


def decode(values):
    return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])


def main(default_survey=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", choices=("crts", "ogle"), default=default_survey, required=default_survey is None)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    number_classes = len(contract["surveys"][args.survey]["class_names"])
    errors, warnings, counts, cohorts = [], [], {}, {}
    core = ("X_seq_interp", "X_mask", "X_img_interp", "X_feat", "y", "y_onehot", "is_gen", "object_id")
    with h5py.File(args.h5, "r") as handle:
        for split in ("train", "val", "test"):
            missing = [f"{name}_{split}" for name in core if f"{name}_{split}" not in handle]
            if missing:
                errors.append(f"{split}: missing {missing}")
                continue
            lengths = {name: len(handle[f"{name}_{split}"]) for name in core}
            if len(set(lengths.values())) != 1:
                errors.append(f"{split}: inconsistent lengths {lengths}")
                continue
            ids = decode(handle[f"object_id_{split}"][:])
            labels = handle[f"y_{split}"][:].astype(int).reshape(-1)
            generated = handle[f"is_gen_{split}"][:].astype(bool)
            if len(np.unique(ids)) != len(ids):
                errors.append(f"{split}: object_id is not unique")
            if np.any((labels < 0) | (labels >= number_classes)):
                errors.append(f"{split}: labels outside [0, {number_classes - 1}]")
            class_counts = np.bincount(labels, minlength=number_classes)
            if np.any(class_counts == 0):
                errors.append(f"{split}: missing classes {np.flatnonzero(class_counts == 0).tolist()}")
            if split != "train" and np.any(generated):
                errors.append(f"{split}: generated rows in held-out split")
            parent_key = f"parent_id_{split}"
            cohort_ids = decode(handle[parent_key][:]) if parent_key in handle else ids
            cohorts[split] = set(cohort_ids.tolist())
            counts[split] = {
                "rows": int(len(ids)),
                "unique_objects_or_parents": int(len(cohorts[split])),
                "generated": int(generated.sum()),
                "class_counts": class_counts.tolist(),
            }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if left in cohorts and right in cohorts:
                overlap = cohorts[left] & cohorts[right]
                if overlap:
                    errors.append(f"object leakage {left}/{right}: {sorted(overlap)[:10]}")
    report = {
        "survey": args.survey,
        "h5": str(Path(args.h5).resolve()),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
