#!/usr/bin/env python3
"""Train a same-split random forest using the official UPSILoN feature vector."""

import argparse
import itertools
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]

SURVEY = "ogle"
FEATURE_NAMES = [
    "amplitude", "hl_amp_ratio", "kurtosis", "period", "phase_cusum",
    "phase_eta", "phi21", "phi31", "quartile31", "r21", "r31",
    "shapiro_w", "skewness", "slope_per10", "slope_per90", "stetson_k",
]
CLASS_NAMES = ["ACEP_F", "ACEP_1O", "CEP_F", "CEP_1O", "CEP_1O2O", "DSCT_SINGLEMODE", "DSCT_MULTIMODE", "ECL_C", "ECL_NC", "RRLYR_RRAB", "RRLYR_RRC", "RRLYR_RRD", "T2CEP_BLHER", "T2CEP_RVTAU", "T2CEP_WVIR"]
MAJOR_CLASS_NAMES = ["ACEP", "CEP", "DSCT", "ECL", "RRLYR", "T2CEP"]
SUB_TO_MAJOR = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]


def expected_calibration_error(y_true, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs((predicted[selected] == y_true[selected]).mean() - confidence[selected].mean())
    return float(value)


def save_evaluation(y_true, y_pred, probabilities, class_names, output, prefix):
    labels = np.arange(len(class_names))
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
    f1_values = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    onehot = np.eye(len(class_names))[y_true]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "gmean_recall": float(np.exp(np.mean(np.log(np.clip(recall, 1e-12, 1.0))))),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "ece_15bin": expected_calibration_error(y_true, probabilities),
        "n_test": int(len(y_true)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0),
    }
    pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "support": cm.sum(axis=1),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1_values,
        }
    ).to_csv(output / f"per_class_{prefix}.csv", index=False)
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output / f"confusion_counts_{prefix}.csv")
    pd.DataFrame(normalized, index=class_names, columns=class_names).to_csv(output / f"confusion_normalized_{prefix}.csv")
    for matrix, suffix, value_format in ((cm, "counts", "d"), (normalized, "normalized", ".2f")):
        size = max(8, len(class_names) * 0.72)
        fig, ax = plt.subplots(figsize=(size, size))
        shown = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if suffix == "normalized" else None)
        ax.set_xticks(labels, class_names, rotation=45, ha="right")
        ax.set_yticks(labels, class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"UPSILoN-feature RF: {prefix} ({suffix})")
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in labels:
            for j in labels:
                value = format(int(matrix[i, j]), value_format) if suffix == "counts" else format(matrix[i, j], value_format)
                ax.text(j, i, value, ha="center", va="center", fontsize=6, color="white" if matrix[i, j] > threshold else "black")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output / f"confusion_{suffix}_{prefix}.png", dpi=300)
        plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        default=str(REPO_ROOT / "data/processed/ogle_upsilon_16_features.csv"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-grid", action="store_true", help="One RF candidate for pipeline smoke tests only")
    args = parser.parse_args()

    feature_names = FEATURE_NAMES
    class_names = CLASS_NAMES
    labels = np.arange(len(class_names))
    frame = pd.read_csv(args.features_csv, dtype={"object_id": str})
    required = {"object_id", "split", "y", "extraction_error", *feature_names}
    if not required.issubset(frame.columns):
        raise ValueError(f"Feature CSV missing: {sorted(required - set(frame.columns))}")
    failed = frame["extraction_error"].fillna("") != ""
    if failed.any():
        raise ValueError(f"Feature extraction failed for {int(failed.sum())} objects; do not silently change the test cohort")
    if frame["object_id"].duplicated().any():
        raise ValueError("object_id must be unique")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("split must contain exactly train, val, and test")

    x = frame[feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=int)
    split = frame["split"].to_numpy()
    train, val, test = split == "train", split == "val", split == "test"
    for name, mask in (("train", train), ("val", val), ("test", test)):
        missing = set(labels) - set(np.unique(y[mask]))
        if missing:
            raise ValueError(f"{name} is missing labels: {sorted(missing)}")

    grid_records = []
    parameter_grid = (
        [(100, None, 1, "sqrt")]
        if args.quick_grid
        else itertools.product((300, 600), (None, 20, 40), (1, 2), ("sqrt", 10))
    )
    for n_estimators, max_depth, min_samples_leaf, max_features in parameter_grid:
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=args.seed,
            ),
        )
        pipeline.fit(x[train], y[train])
        prediction = pipeline.predict(x[val])
        grid_records.append(
            {
                "validation_macro_f1": float(f1_score(y[val], prediction, average="macro", zero_division=0)),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
        )
    grid_records.sort(key=lambda row: row["validation_macro_f1"], reverse=True)
    selected = grid_records[0]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=selected["n_estimators"],
            max_depth=selected["max_depth"],
            min_samples_leaf=selected["min_samples_leaf"],
            max_features=selected["max_features"],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        ),
    )
    # Match the TDMN protocol: train rows fit the estimator, validation selects it,
    # and the held-out test set is accessed once after selection is frozen.
    model.fit(x[train], y[train])
    prediction = model.predict(x[test])
    probability_raw = model.predict_proba(x[test])
    probability = np.zeros((int(test.sum()), len(labels)), dtype=float)
    forest = model.named_steps["randomforestclassifier"]
    probability[:, forest.classes_.astype(int)] = probability_raw

    output = Path(
        args.output_dir or REPO_ROOT / "results/ogle" / f"upsilon_rf_seed{args.seed}"
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation_prefix = "15class" if SURVEY == "ogle" else "11class"
    metrics = save_evaluation(y[test], prediction, probability, class_names, output, evaluation_prefix)
    metrics = {
        "method": "UPSILoN-feature RF",
        "survey": SURVEY,
        "selection_metric": "validation_macro_f1",
        "selected_parameters": selected,
        "feature_names": feature_names,
        "seed": args.seed,
        **metrics,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if MAJOR_CLASS_NAMES:
        y_major = np.array([SUB_TO_MAJOR[value] for value in y[test]], dtype=int)
        prediction_major = np.array([SUB_TO_MAJOR[value] for value in prediction], dtype=int)
        probability_major = np.zeros((len(y_major), len(MAJOR_CLASS_NAMES)), dtype=float)
        for subclass, major in enumerate(SUB_TO_MAJOR):
            probability_major[:, major] += probability[:, subclass]
        metrics_major = save_evaluation(
            y_major, prediction_major, probability_major, MAJOR_CLASS_NAMES, output, "6class"
        )
        (output / "metrics_major_6class.json").write_text(
            json.dumps(metrics_major, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (output / "validation_grid.json").write_text(json.dumps(grid_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "test_predictions.npz",
        object_id=frame.loc[test, "object_id"].astype(str).to_numpy(),
        y_true=y[test],
        y_pred=prediction,
        probabilities=probability,
    )
    pd.DataFrame({"feature": feature_names, "importance": forest.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
