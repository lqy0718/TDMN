#!/usr/bin/env python3
"""Compare two methods on identical test objects and emit full metric/CM artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE.parent / "configs" / "experiment_contract.json"


def decode(values):
    return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])


def load_predictions(path):
    data = np.load(path, allow_pickle=False)
    required = {"object_id", "y_true", "y_pred", "probabilities"}
    if not required.issubset(data.files):
        raise ValueError(f"{path} is missing {sorted(required - set(data.files))}")
    return {
        "object_id": decode(data["object_id"]),
        "y_true": data["y_true"].astype(int).reshape(-1),
        "y_pred": data["y_pred"].astype(int).reshape(-1),
        "probabilities": data["probabilities"].astype(float),
    }


def align(reference, candidate):
    if len(np.unique(reference["object_id"])) != len(reference["object_id"]):
        raise ValueError("Reference object_id is not unique")
    if len(np.unique(candidate["object_id"])) != len(candidate["object_id"]):
        raise ValueError("Candidate object_id is not unique")
    if set(reference["object_id"]) != set(candidate["object_id"]):
        only_reference = sorted(set(reference["object_id"]) - set(candidate["object_id"]))[:10]
        only_candidate = sorted(set(candidate["object_id"]) - set(reference["object_id"]))[:10]
        raise ValueError(f"Test cohorts differ; only A={only_reference}, only B={only_candidate}")
    lookup = {object_id: index for index, object_id in enumerate(candidate["object_id"])}
    order = np.array([lookup[object_id] for object_id in reference["object_id"]])
    aligned = {key: value[order] for key, value in candidate.items()}
    if not np.array_equal(reference["y_true"], aligned["y_true"]):
        raise ValueError("Ground-truth labels disagree after object-id alignment")
    return aligned


def expected_calibration_error(y_true, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def evaluate(y_true, y_pred, probabilities, class_names):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    gmean_recall = float(np.exp(np.mean(np.log(np.clip(recall, 1e-12, 1.0)))))
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    onehot = np.eye(len(class_names))[y_true]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "gmean_recall": gmean_recall,
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "ece_15bin": expected_calibration_error(y_true, probabilities),
        "n_test": int(len(y_true)),
    }
    per_class = pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "support": cm.sum(axis=1),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        }
    )
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    return metrics, per_class, cm, report


def plot_cm(cm, class_names, title, path, normalize):
    shown = cm.astype(float)
    if normalize:
        shown = np.divide(shown, shown.sum(axis=1, keepdims=True), out=np.zeros_like(shown), where=shown.sum(axis=1, keepdims=True) > 0)
    size = max(8, len(class_names) * 0.72)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if shown[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def bootstrap_difference(y_true, pred_a, pred_b, iterations, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    metrics = {
        "macro_f1": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
    }
    values = {name: np.empty(iterations) for name in metrics}
    for index in range(iterations):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for name, function in metrics.items():
            values[name][index] = function(y_true[sample], pred_a[sample]) - function(y_true[sample], pred_b[sample])
    return {
        name: {
            "mean_a_minus_b": float(samples.mean()),
            "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
            "two_sided_p": float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))),
        }
        for name, samples in values.items()
    }


def aggregate(prediction, mapping, number_major):
    mapped_true = np.array([mapping[value] for value in prediction["y_true"]], dtype=int)
    mapped_pred = np.array([mapping[value] for value in prediction["y_pred"]], dtype=int)
    probabilities = np.zeros((len(mapped_true), number_major), dtype=float)
    for subclass, major in enumerate(mapping):
        probabilities[:, major] += prediction["probabilities"][:, subclass]
    return {**prediction, "y_true": mapped_true, "y_pred": mapped_pred, "probabilities": probabilities}


def compare_level(data_a, data_b, class_names, name_a, name_b, output, iterations, seed, level):
    output.mkdir(parents=True, exist_ok=True)
    result = {"level": level, "method_a": name_a, "method_b": name_b}
    metric_rows = []
    for name, data in ((name_a, data_a), (name_b, data_b)):
        metrics, per_class, cm, report = evaluate(data["y_true"], data["y_pred"], data["probabilities"], class_names)
        result[name] = {"metrics": metrics, "classification_report": report}
        metric_rows.append({"method": name, **metrics})
        per_class.insert(0, "method", name)
        per_class.to_csv(output / f"per_class_{name}.csv", index=False)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output / f"confusion_counts_{name}.csv")
        normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)
        pd.DataFrame(normalized, index=class_names, columns=class_names).to_csv(output / f"confusion_normalized_{name}.csv")
        plot_cm(cm, class_names, f"{name}: confusion matrix (counts)", output / f"confusion_counts_{name}.png", False)
        plot_cm(cm, class_names, f"{name}: confusion matrix (row-normalized)", output / f"confusion_normalized_{name}.png", True)
    result["paired_bootstrap"] = bootstrap_difference(data_a["y_true"], data_a["y_pred"], data_b["y_pred"], iterations, seed)
    pd.DataFrame(metric_rows).to_csv(output / "overall_metrics.csv", index=False)
    (output / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(default_survey=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", choices=("crts", "ogle"), default=default_survey, required=default_survey is None)
    parser.add_argument("--predictions-a", required=True)
    parser.add_argument("--predictions-b", required=True)
    parser.add_argument("--name-a", default="TDMN")
    parser.add_argument("--name-b", default="UPSILoN-feature_RF")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    survey = contract["surveys"][args.survey]
    data_a = load_predictions(args.predictions_a)
    data_b = align(data_a, load_predictions(args.predictions_b))
    if data_a["probabilities"].shape[1] != len(survey["class_names"]) or data_b["probabilities"].shape[1] != len(survey["class_names"]):
        raise ValueError("Probability columns do not match the survey class contract")
    output = Path(args.output_dir).expanduser().resolve()
    compare_level(data_a, data_b, survey["class_names"], args.name_a, args.name_b, output / "subclass_or_native", args.iterations, args.seed, "subclass_or_native")
    if args.survey == "ogle":
        mapping = survey["sub_to_major"]
        major_names = survey["major_class_names"]
        compare_level(aggregate(data_a, mapping, len(major_names)), aggregate(data_b, mapping, len(major_names)), major_names, args.name_a, args.name_b, output / "major_6class", args.iterations, args.seed, "major_6class")
    print(f"Comparison artifacts written to {output}")


if __name__ == "__main__":
    main()
