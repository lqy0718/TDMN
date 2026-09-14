"""Independently summarize V5-faithful Exp1-4 and existing Full TDMN runs."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 142, 242)
CLASS_NAMES = ["RRab", "RRc", "RRd", "Blazkho", "Ecl", "EA", "Rot", "LPV",
               "delta-Scuti", "ACep", "Cep-II"]
GROUPS = [
    ("exp1", "exp1_ce"),
    ("exp2", "exp2_augmentation_ce"),
    ("exp3", "exp3_proxy_contrastive"),
    ("exp4", "exp4_two_stage_cb"),
    ("full", None),
]
V5_HASHES = {
    "exp1": "ebe98c6390f8080a4b915ecf80019d2d133a4a6879d88302a418988857715da9",
    "exp2": "01cf3ab289cccfbcc847e5d99fe68663df50b58840ee0598472fed6c65d1d2dd",
    "exp3": "ea19e7cffbc33a421325420081baa1048f9fa0715c3fa4e2a241ce255e8fd774",
    "exp4": "37ccb2eec51444dd901da52903be724b27009f1cecd2cfd6ea5881be41f9c903",
}


def full_directory(seed):
    repo_root = ROOT.parents[1]
    return repo_root / "results/crts" / f"tdmn_seed{seed}"


def load_run(experiment, folder, seed):
    directory = (full_directory(seed) if experiment == "full"
                 else ROOT / folder / "results_v5" / f"{experiment}_seed{seed}")
    required = ["metrics.json", "test_predictions.npz", "model_selection.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {missing}")
    if experiment != "full":
        config = json.loads((directory / "run_config.json").read_text())
        complete = json.loads((directory / "COMPLETED.json").read_text())
        if (config.get("protocol") != "V5-faithful-review-fix-v1"
                or complete.get("protocol") != config["protocol"]
                or config.get("experiment") != experiment or config.get("seed") != seed
                or config.get("v5_source_sha256") != V5_HASHES[experiment]
                or complete.get("v5_source_sha256") != V5_HASHES[experiment]):
            raise ValueError(f"Wrong V5 protocol/configuration: {directory}")
        expected = ({"epochs": 200} if experiment in ("exp1", "exp2")
                    else {"epochs_stage1": 150, "epochs_stage2": 100})
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Wrong V5 epoch settings: {directory}")
    else:
        config = json.loads((directory / "run_config.json").read_text())
        expected = dict(seed=seed, batch_size=256, epochs_stage1=300, epochs_stage2=120,
                        anchor_weight=0.1, logit_coefficient=0.3, frequency_source="real")
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Wrong Full reference configuration: {directory}")
    selection = json.loads((directory / "model_selection.json").read_text())
    if selection.get("selection_metric") not in ("validation_loss", "validation_macro_f1"):
        raise ValueError(f"Test set was not isolated for selection: {directory}")
    with np.load(directory / "test_predictions.npz", allow_pickle=True) as values:
        ids = values["object_id"].astype(str).reshape(-1)
        truth = values["y_true"].astype(int).reshape(-1)
        pred = values["y_pred"].astype(int).reshape(-1)
        prob = values["probabilities"]
    if (len(ids) != len(set(ids)) or prob.shape != (len(ids), 11)
            or not np.array_equal(prob.argmax(axis=1), pred)
            or not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError(f"Invalid object predictions: {directory}")
    metrics = dict(
        accuracy=float(accuracy_score(truth, pred)),
        balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
        macro_precision=float(precision_score(truth, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(truth, pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(truth, pred, average="macro")),
        weighted_f1=float(f1_score(truth, pred, average="weighted")),
    )
    saved = json.loads((directory / "metrics.json").read_text())
    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
        stored = saved.get(key, saved.get("test_" + key))
        # Existing Full TDMN metrics store accuracy inside sklearn's
        # classification_report rather than as a top-level field.
        if stored is None and key == "accuracy":
            stored = saved.get("classification_report", {}).get("accuracy")
        if stored is None or not np.isclose(float(stored), metrics[key], atol=1e-8, rtol=0):
            raise ValueError(f"Saved/recomputed {key} mismatch: {directory}")
    return {"seed": seed, "directory": str(directory), "cohort": dict(zip(ids, truth)),
            "metrics": metrics}


def main():
    groups = []
    cohort = None
    for experiment, folder in GROUPS:
        runs = [load_run(experiment, folder, seed) for seed in SEEDS]
        for run in runs:
            if cohort is None:
                cohort = run["cohort"]
            if run["cohort"] != cohort:
                raise ValueError(f"Test IDs/labels differ: {run['directory']}")
            del run["cohort"]
        statistics = {}
        for key in runs[0]["metrics"]:
            values = [run["metrics"][key] for run in runs]
            statistics[key] = {"mean": float(np.mean(values)),
                               "std_ddof1": float(np.std(values, ddof=1))}
        groups.append({"experiment": experiment, "runs": runs, "statistics": statistics})
    output = ROOT / "summary_v5"
    output.mkdir(exist_ok=True)
    result = {"protocol": "V5-faithful Exp1-4; existing current Full TDMN",
              "seeds": list(SEEDS), "std": "sample standard deviation, ddof=1",
              "n_test": len(cohort), "groups": groups}
    (output / "ablation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# V5逻辑修正版消融汇总", "",
             "Exp1–4严格保留V5核心训练逻辑；仅修复独立验证、数据接口、Stage1最佳权重恢复及结果审计。", "",
             "| Experiment | Macro-F1 | Balanced accuracy | Accuracy |",
             "|---|---:|---:|---:|"]
    for group in groups:
        s = group["statistics"]
        lines.append(f"| {group['experiment']} | {s['macro_f1']['mean']:.4f} ± {s['macro_f1']['std_ddof1']:.4f} | "
                     f"{100*s['balanced_accuracy']['mean']:.2f} ± {100*s['balanced_accuracy']['std_ddof1']:.2f}% | "
                     f"{100*s['accuracy']['mean']:.2f} ± {100*s['accuracy']['std_ddof1']:.2f}% |")
    lines += ["", "## Per-seed audit", ""]
    for group in groups:
        for run in group["runs"]:
            m = run["metrics"]
            lines.append(f"- {group['experiment']} seed {run['seed']}: "
                         f"Macro-F1={m['macro_f1']:.6f}, BA={m['balanced_accuracy']:.6f}, "
                         f"accuracy={m['accuracy']:.6f}")
    (output / "ablation_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
