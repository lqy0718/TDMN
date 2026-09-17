# TDMN: Periodic Variable-Star Classification in Imbalanced Data

Official code accompanying the accepted manuscript **“Periodic Variable Star
Classification in Imbalanced Data: A Two-Stage Decoupled Approach with
Multimodal Contrastive Learning”**.

TDMN combines phase-folded sequences, three-channel morphology images, and
global features. Stage 1 learns multimodal representations with proxy
contrastive objectives and real-observation class prototypes. Stage 2 adjusts
the decision boundary using class-balanced, logit-adjusted, and cosine-focal
classification heads with sample-adaptive fusion.

## Repository contents

```text
configs/                  Experiment contract and run matrix
data/                     Data layout and download instructions (no raw data)
experiments/crts/         CRTS preparation, TDMN, and UPSILoN-feature RF
experiments/ogle/         OGLE-IV preparation, TDMN, and UPSILoN-feature RF
experiments/ablation/     CRTS Exp1–Exp4 ablation training and aggregation
tools/                    Dataset validation and object-matched comparison
tests/                    Static reproducibility checks
```

The repository contains source code only. Raw survey files, generated HDF5
files, extracted features, checkpoints, and predictions are deliberately not
tracked by Git; see [data/README.md](data/README.md).

## Environment

The original experiments used Python 3.10 on Linux with NVIDIA GPUs. Create a
dedicated environment for TDMN:

```bash
conda create -n tdmn python=3.10 -y
conda activate tdmn
pip install -r requirements.txt
```

UPSILoN 1.2.10 is an older package and is best installed in a separate
environment:

```bash
conda create -n tdmn-upsilon python=3.10 -y
conda activate tdmn-upsilon
pip install -r requirements-upsilon.txt
```

The released implementation uses the TensorFlow/Keras 2 Functional API and is
therefore constrained to TensorFlow 2.10–2.15 (`keras<3`). TensorFlow 2.16 and
later bundle Keras 3, whose symbolic-tensor rules are incompatible with several
model-construction expressions in the experiment code. Select the appropriate
TensorFlow/CUDA build for the local driver and CUDA runtime within this range.

## Data preparation

Download the public CRTS and/or OGLE-IV files and arrange them exactly as shown
in [data/README.md](data/README.md). Run all commands below from the repository
root.

### CRTS

```bash
python experiments/crts/1_prepare_crts_tdmn.py
python tools/validate_tdmn_dataset.py \
  --survey crts \
  --h5 data/processed/crts_tdmn.h5 \
  --output data/processed/crts_validation.json
```

The preparation step creates:

- `data/processed/crts_tdmn.h5`;
- `data/processed/crts_tdmn.manifest.csv`;
- a configuration JSON, exclusion table, and preparation log.

### OGLE-IV

```bash
python experiments/ogle/1_prepare_ogle_tdmn.py
python tools/validate_tdmn_dataset.py \
  --survey ogle \
  --h5 data/processed/ogle_tdmn.h5 \
  --output data/processed/ogle_validation.json
```

## Main TDMN experiments

One prepared HDF5 split is reused for all three training seeds. Do not rerun
the preparation script between seeds.

### CRTS (fixed 7:2:1 split)

```bash
python experiments/crts/2_train_crts_tdmn.py --seed 42
python experiments/crts/2_train_crts_tdmn.py --seed 142
python experiments/crts/2_train_crts_tdmn.py --seed 242
```

Results are written to `results/crts/tdmn_seed<seed>/`.

### OGLE-IV (fixed survey-specific split)

```bash
python experiments/ogle/2_train_ogle_tdmn.py --seed 42
python experiments/ogle/2_train_ogle_tdmn.py --seed 142
python experiments/ogle/2_train_ogle_tdmn.py --seed 242
```

Results are written to `results/ogle/tdmn_seed<seed>/`.

## UPSILoN-feature random-forest baseline

Feature extraction reads the TDMN manifest, so both pipelines use the same
real objects and split labels. Extraction is performed once per survey; the
three RF seeds reuse the resulting CSV.

```bash
# CRTS
python experiments/crts/3_extract_upsilon_features.py --workers 16 --resume
python experiments/crts/4_train_upsilon_rf.py --seed 42
python experiments/crts/4_train_upsilon_rf.py --seed 142
python experiments/crts/4_train_upsilon_rf.py --seed 242

# OGLE-IV
python experiments/ogle/3_extract_upsilon_features.py --workers 16 --resume
python experiments/ogle/4_train_upsilon_rf.py --seed 42
python experiments/ogle/4_train_upsilon_rf.py --seed 142
python experiments/ogle/4_train_upsilon_rf.py --seed 242
```

The scripts fail if feature extraction changes the held-out cohort. Extraction
errors must be investigated instead of silently dropping test objects.

## Object-matched comparison

Example for CRTS seed 42:

```bash
python tools/compare_predictions.py \
  --survey crts \
  --predictions-a results/crts/tdmn_seed42/test_predictions.npz \
  --predictions-b results/crts/upsilon_rf_seed42/test_predictions.npz \
  --name-a TDMN \
  --name-b UPSILoN-feature_RF \
  --output-dir results/comparisons/crts/seed_42 \
  --iterations 10000
```

Replace `42` with `142` and `242`, then aggregate:

```bash
python tools/aggregate_comparisons.py \
  --comparison-root results/comparisons/crts \
  --output-dir results/comparisons/crts/aggregate
```

Use `--survey ogle` and the corresponding OGLE paths for the 15-subclass and
6-superclass analysis. The comparison tool verifies object IDs and true labels
before computing metrics, confusion matrices, and paired class-stratified
bootstrap intervals.

## CRTS 6:1:3 split-sensitivity experiment

The same preparation and training implementations are reused with a separate
output file:

```bash
python experiments/crts/1_prepare_crts_tdmn.py \
  --split 6:1:3 \
  --output-path data/processed/crts_tdmn_split613.h5 \
  --cache-dir data/cache/crts_split613

python experiments/crts/2_train_crts_tdmn.py \
  --data data/processed/crts_tdmn_split613.h5 \
  --output-dir results/crts_split613/tdmn_seed42 --seed 42
```

Repeat only the training command for seeds 142 and 242, changing both the
output directory and `--seed`. These estimates must not be pooled with the
main 7:2:1 results because they use a different held-out cohort.

## CRTS ablation experiments

The ablation scripts reuse `data/processed/crts_tdmn.h5`. Each training script
runs seeds 42, 142, and 242 sequentially:

```bash
python experiments/ablation/0_check_data.py
python experiments/ablation/0_smoke_test.py
python experiments/ablation/exp1_ce/train.py
python experiments/ablation/exp2_augmentation_ce/train.py
python experiments/ablation/exp3_proxy_contrastive/train.py
python experiments/ablation/exp4_two_stage_cb/train.py
python experiments/ablation/5_summarize.py
```

The final TDMN row (Exp5) is read from the three main CRTS result directories.
Set `TDMN_CRTS_H5=/absolute/path/to/crts_tdmn.h5` only when the prepared file
is stored outside the repository.

## Reproducibility notes

- Seeds 42, 142, and 242 are repeated training runs on one fixed split.
- Model selection uses validation macro F1; the selected frozen estimator is
  evaluated once on the held-out test set.
- Generated samples are restricted to the training split.
- Prediction archives contain `object_id`, `y_true`, `y_pred`, and
  `probabilities`, allowing cohort-level audits.
- CRTS and OGLE-IV have different label spaces and are evaluated separately.

Run the lightweight repository checks with:

```bash
python -m unittest discover -s tests -v
```

## Citation



## Third-party software and data

The UPSILoN-feature baseline uses UPSILoN 1.2.10 and its 16-feature extractor:

> Kim, D.-W., & Bailer-Jones, C. A. L. (2016). A Package for the Automated
> Classification of Periodic Variable Stars. *Astronomy & Astrophysics*, 587,
> A18. https://doi.org/10.1051/0004-6361/201527188

Survey data remain subject to their providers' terms and citation policies.

## License

The source code is released under the [MIT License](LICENSE). Survey data and
third-party packages remain subject to their respective terms and licenses.
