"""V5-faithful CRTS ablation with review-safety fixes only."""
import os
import sys
import subprocess
from pathlib import Path

EXPERIMENT_ID = "exp2"
SEEDS = (42, 142, 242)
CHILD_SEED_ENV = "V5_ABLATION_ACTIVE_SEED"
if __name__ == "__main__" and CHILD_SEED_ENV not in os.environ:
    for _seed in SEEDS:
        _env = os.environ.copy()
        _env[CHILD_SEED_ENV] = str(_seed)
        _env["PYTHONUNBUFFERED"] = "1"
        print(f"\\n===== {EXPERIMENT_ID}: seed={_seed} =====", flush=True)
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve())], env=_env, check=True)
    print(f"{EXPERIMENT_ID}: all three seeds completed.", flush=True)
    raise SystemExit(0)

SEED = int(os.environ.get(CHILD_SEED_ENV, "42"))
if SEED not in SEEDS:
    raise ValueError(f"Unsupported seed: {SEED}")

import csv
import gc
import json
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, backend as K
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, log_loss)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH_H5 = Path(os.environ.get(
    "TDMN_CRTS_H5", REPO_ROOT / "data/processed/crts_tdmn.h5"
))
SAVE_BASE_DIR = str(EXPERIMENT_DIR / "results_v5" / f"{EXPERIMENT_ID}_seed{SEED}")
BATCH_SIZE = 256
EPOCHS = 200
RESIZE_TARGET = (128, 128)
V5_SOURCE_SHA256 = "01cf3ab289cccfbcc847e5d99fe68663df50b58840ee0598472fed6c65d1d2dd"
PROTOCOL = "V5-faithful-review-fix-v1"
class_names = ["RRab", "RRc", "RRd", "Blazkho", "Ecl", "EA", "Rot", "LPV", "delta-Scuti", "ACep", "Cep-II"]
NUM_CLASSES = len(class_names)

# TDMN backbone components.
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation="relu", use_bias=False)(se)
    se = layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])

def build_pure_di_modal_encoder(seq_shape, img_shape):
    reg = regularizers.l2(1.5e-4)
    seq_in = keras.Input(shape=seq_shape, name="seq_in")
    x_seq = layers.Conv1D(32, 5, padding="same", use_bias=False, kernel_regularizer=reg)(seq_in[:, :, 0:3])
    x_seq = layers.Activation("swish")(layers.BatchNormalization()(x_seq))
    x_seq = layers.SpatialDropout1D(0.2)(layers.Concatenate()([layers.Conv1D(32, 3, padding="same", activation="swish")(x_seq), layers.Conv1D(32, 7, padding="same", activation="swish")(x_seq)]))
    x_seq = layers.Add()([x_seq, layers.Dense(64)(seq_in[:, :, 3:6])])
    attn_mask = tf.cast(seq_in[:, :, 6], tf.bool)[:, tf.newaxis, :]
    for _ in range(2):
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + layers.MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.2)(x_seq, x_seq, attention_mask=attn_mask))
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + layers.Dense(64)(layers.Dropout(0.2)(layers.Dense(128, activation="swish")(x_seq))))

    mask_exp = tf.expand_dims(seq_in[:, :, 6], -1)
    seq_repr = layers.Dense(128, activation="swish", kernel_regularizer=reg)(layers.Concatenate()([tf.reduce_sum(x_seq * mask_exp, axis=1)/tf.maximum(tf.reduce_sum(mask_exp, axis=1), 1.0), tf.reduce_max(x_seq + (1.0 - mask_exp)*-1e4, axis=1)]))

    img_in = keras.Input(shape=img_shape, name="img_in")
    x_img = layers.MaxPooling2D(2)(layers.Activation("swish")(layers.BatchNormalization()(layers.Conv2D(32, 3, padding="same")(img_in))))
    for f, d in [(64, 0.05), (128, 0.05), (256, 0.1)]:
        x_img = layers.SpatialDropout2D(d)(squeeze_excite_block(layers.Activation("swish")(layers.BatchNormalization()(layers.SeparableConv2D(f, 3, padding="same")(x_img)))))
        if f != 256: x_img = layers.MaxPooling2D(2)(x_img)
    img_repr = layers.Dense(128, activation="swish", kernel_regularizer=reg)(layers.Concatenate()([layers.GlobalAveragePooling2D()(x_img), layers.GlobalMaxPooling2D()(x_img)]))
    return keras.Model(inputs={"seq_in": seq_in, "img_in": img_in}, outputs=[seq_repr, img_repr])

class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.cond(step < self.warmup_steps, lambda: self.initial_learning_rate * step / self.warmup_steps, lambda: self.initial_learning_rate * (self.alpha + (1.0 - self.alpha) * 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * ((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))))

def preprocess(seq, img, feat, mask, label):
    img = tf.image.resize(img, RESIZE_TARGET, method="bilinear")
    phase, raw_mag, smooth_mag, err = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
    is_valid = mask; raw_mag *= is_valid; smooth_mag *= is_valid; err *= is_valid
    max_phase = tf.reduce_max(phase * is_valid)
    norm_phase = tf.math.divide_no_nan(phase, max_phase + 1e-8)
    pi_val = 3.141592653589793
    seq_processed = tf.stack([raw_mag, smooth_mag, err, tf.math.sin(2.0 * pi_val * norm_phase)*is_valid, tf.math.cos(2.0 * pi_val * norm_phase)*is_valid, phase, is_valid], axis=-1)
    return {"seq_in": seq_processed, "img_in": img, "feat_in": feat}, label

# Exp2 uses class-conditional noise to preserve Rot amplitudes.
def augment_with_shield(inputs, label):
    seq = inputs["seq_in"]
    raw_mag, smooth_mag, err, phase_sin, phase_cos, phase, valid_mask = tf.unstack(seq, axis=-1)
    noise_scale = 0.02 * (1.0 - label[6]) # Exp2: 动态阻断测光噪声
    mag_noise = tf.random.normal(tf.shape(raw_mag), stddev=noise_scale) * valid_mask
    seq_aug = tf.stack([raw_mag + mag_noise, smooth_mag, err, phase_sin, phase_cos, phase, valid_mask], axis=-1)
    return {"seq_in": seq_aug, "img_in": inputs["img_in"], "feat_in": inputs["feat_in"]}, label

def create_dataset(indices, X_seq, X_img, X_feat, X_mask, y_oh, feat_dim, is_train=False):
    # 强制在 CPU 上构建 Dataset，保护宝贵的 GPU 显存！
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((
            X_seq[indices],
            X_img[indices],
            X_feat[indices][:, :feat_dim],
            X_mask[indices],
            y_oh[indices]
        ))

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        # Exp 1 用 augment_global_noise, Exp 2 用 augment_with_shield
        ds = ds.map(augment_with_shield, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(8192)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_exp2_model(seq_shape, img_shape, feat_dim):
    encoder = build_pure_di_modal_encoder(seq_shape, img_shape)
    seq_in, img_in = keras.Input(shape=seq_shape, name="seq_in"), keras.Input(shape=img_shape, name="img_in")
    feat_in = keras.Input(shape=(feat_dim,), name="feat_in")

    seq_repr, img_repr = encoder({"seq_in": seq_in, "img_in": img_in})
    feat_repr = layers.Activation("swish")(layers.BatchNormalization()(layers.Dense(64, kernel_regularizer=regularizers.l2(1.5e-4))(feat_in)))

    stacked = layers.SpatialDropout1D(0.15)(tf.stack([seq_repr, img_repr], axis=1))
    fusion = layers.Dropout(0.5)(layers.Activation("swish")(layers.BatchNormalization()(layers.Dense(128, kernel_regularizer=regularizers.l2(1.5e-4))(layers.Reshape((256,))(stacked)))))

    output = layers.Dense(NUM_CLASSES, activation="softmax", name="standard_head")(layers.Concatenate()([fusion, feat_repr]))
    return keras.Model(inputs={"seq_in": seq_in, "img_in": img_in, "feat_in": feat_in}, outputs=output)


def _load_current_review_data():
    if not DATA_PATH_H5.is_file():
        raise FileNotFoundError(f"Current CRTS HDF5 not found: {DATA_PATH_H5}")
    result = {}
    with h5py.File(DATA_PATH_H5, "r") as hf:
        for tag in ("train", "val", "test"):
            required = [f"X_seq_interp_{tag}", f"X_img_interp_{tag}", f"X_feat_{tag}",
                        f"X_mask_{tag}", f"y_{tag}", f"y_{tag}_onehot", f"is_gen_{tag}",
                        f"object_id_{tag}"]
            missing = [key for key in required if key not in hf]
            if missing:
                raise ValueError(f"Missing current review fields: {missing}")
            seq = np.nan_to_num(hf[f"X_seq_interp_{tag}"][:].astype(np.float32), nan=-10.0)
            img = hf[f"X_img_interp_{tag}"][:].astype(np.float32)
            feat = hf[f"X_feat_{tag}"][:].astype(np.float32)
            mask = hf[f"X_mask_{tag}"][:].astype(np.float32)
            y = hf[f"y_{tag}"][:].astype(np.int32).reshape(-1)
            y_oh = hf[f"y_{tag}_onehot"][:].astype(np.float32)
            is_gen = hf[f"is_gen_{tag}"][:].astype(np.float32)
            object_id = hf[f"object_id_{tag}"].asstr()[:].astype(str)
            n = len(y)
            if (seq.ndim != 3 or seq.shape[-1] not in (4, 5)
                    or mask.shape != seq.shape[:2] or img.shape[0] != n
                    or feat.shape[0] != n or y_oh.shape != (n, NUM_CLASSES)
                    or len(is_gen) != n or len(object_id) != n):
                raise ValueError(f"Incompatible {tag} shapes in {DATA_PATH_H5}")
            if not np.array_equal(y_oh.argmax(axis=1), y):
                raise ValueError(f"{tag} integer and one-hot labels disagree")
            result[tag] = dict(seq=seq, img=img, feat=feat, mask=mask, y=y,
                               y_oh=y_oh, is_gen=is_gen, object_id=object_id)
    if set(result["train"]["object_id"]) & set(result["test"]["object_id"]):
        raise ValueError("Train/test object-ID leakage")
    if set(result["val"]["object_id"]) & set(result["test"]["object_id"]):
        raise ValueError("Validation/test object-ID leakage")
    return result


def _prepare_output():
    output = Path(SAVE_BASE_DIR)
    completed = output / "COMPLETED.json"
    if completed.is_file():
        previous = json.loads(completed.read_text())
        if (previous.get("protocol") == PROTOCOL and previous.get("seed") == SEED
                and previous.get("v5_source_sha256") == V5_SOURCE_SHA256):
            print(f"[SKIP] completed: {output}")
            return None
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"{output} contains an incomplete or incompatible run. Rename it before rerunning.")
    output.mkdir(parents=True, exist_ok=True)
    stat = DATA_PATH_H5.stat()
    config = dict(protocol=PROTOCOL, experiment=EXPERIMENT_ID, seed=SEED,
                  data=str(DATA_PATH_H5), data_size_bytes=stat.st_size,
                  data_mtime_ns=stat.st_mtime_ns, batch_size=BATCH_SIZE,
                  v5_source_sha256=V5_SOURCE_SHA256,
                  corrections=["current HDF5 interface", "validation set for model selection",
                               "load best stage-1 checkpoint before stage 2",
                               "held-out test evaluated once", "three training seeds",
                               "result provenance and object-level predictions"])
    if EXPERIMENT_ID in ("exp1", "exp2"):
        config["epochs"] = EPOCHS
    else:
        config.update(epochs_stage1=EPOCHS_STAGE1, epochs_stage2=EPOCHS_STAGE2)
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return output, stat, config


def _finish_run(output, data_stat, config, y_true, probabilities, object_ids,
                selection, title, cmap):
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    if probabilities.shape != (len(y_true), NUM_CLASSES):
        raise ValueError("Invalid probability shape")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid final probabilities")
    cm = confusion_matrix(y_true, predictions, labels=np.arange(NUM_CLASSES))
    recalls = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(NUM_CLASSES, float), where=cm.sum(axis=1) > 0)
    report = classification_report(y_true, predictions, labels=np.arange(NUM_CLASSES),
                                   target_names=class_names, output_dict=True, zero_division=0)
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_ids = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float((bin_ids == k).mean()) * abs(float(correct[bin_ids == k].mean())
              - float(confidence[bin_ids == k].mean())) for k in range(15) if np.any(bin_ids == k))
    metrics = dict(selection, protocol=PROTOCOL, experiment=EXPERIMENT_ID, seed=SEED,
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        macro_precision=float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, predictions, average="macro")),
        weighted_f1=float(f1_score(y_true, predictions, average="weighted")),
        test_macro_f1=float(f1_score(y_true, predictions, average="macro")),
        test_balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        test_gmean_recall=float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0))))),
        log_loss=float(log_loss(y_true, probabilities, labels=np.arange(NUM_CLASSES))),
        multiclass_brier=float(np.mean(np.sum((probabilities - np.eye(NUM_CLASSES)[y_true]) ** 2, axis=1))),
        ece_15bin=float(ece), n_test=int(len(y_true)), classification_report=report)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False,
        default=lambda value: value.item()) + "\n")
    (output / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    np.savez_compressed(output / "test_predictions.npz", object_id=np.asarray(object_ids, dtype=str),
                        y_true=y_true, y_pred=predictions, probabilities=probabilities)
    for filename, matrix in (("confusion_counts_11class.csv", cm),
                             ("confusion_normalized_11class.csv", np.divide(cm, cm.sum(axis=1, keepdims=True),
                               out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0))):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true / predicted"] + class_names)
            writer.writerows([[name] + row.tolist() for name, row in zip(class_names, matrix)])
    with (output / "per_class_11class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in class_names:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, labels=np.arange(NUM_CLASSES),
        display_labels=class_names, normalize="true", cmap=cmap, ax=ax, xticks_rotation=45, values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix_test.png", dpi=300)
    plt.close(fig)
    print(classification_report(y_true, predictions, target_names=class_names, digits=4, zero_division=0))
    current = DATA_PATH_H5.stat()
    if current.st_size != data_stat.st_size or current.st_mtime_ns != data_stat.st_mtime_ns:
        raise RuntimeError("HDF5 changed during training; run not marked complete")
    completion = dict(protocol=PROTOCOL, experiment=EXPERIMENT_ID, seed=SEED,
                      v5_source_sha256=V5_SOURCE_SHA256)
    (output / "COMPLETED.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(f"Completed: {output}")


def run_exp2():
    K.clear_session()
    prepared = _prepare_output()
    if prepared is None:
        return
    output, data_stat, config = prepared
    data = _load_current_review_data()
    tr, val, te = data["train"], data["val"], data["test"]
    feat_dim = tr["feat"].shape[1]
    train_ds = create_dataset(np.arange(len(tr["y"])), tr["seq"], tr["img"], tr["feat"],
                              tr["mask"], tr["y_oh"], feat_dim, is_train=True)
    val_ds = create_dataset(np.arange(len(val["y"])), val["seq"], val["img"], val["feat"],
                            val["mask"], val["y_oh"], feat_dim, is_train=False)
    test_ds = create_dataset(np.arange(len(te["y"])), te["seq"], te["img"], te["feat"],
                             te["mask"], te["y_oh"], feat_dim, is_train=False)
    model = build_exp2_model((tr["seq"].shape[1], 7), (128, 128, 3), feat_dim)
    total_steps = EPOCHS * (len(tr["y"]) // BATCH_SIZE)
    schedule = CosineDecayWithWarmup(4e-4, int(total_steps * 0.1), total_steps, 1e-6)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=schedule,
                  weight_decay=1e-4, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = str(output / "gentle_aug_best.ckpt")
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
        callbacks=[keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True,
                   save_weights_only=True, monitor="val_loss"),
                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                   restore_best_weights=True),
                   keras.callbacks.CSVLogger(str(output / "history.csv"))],
        verbose=1)
    model.load_weights(checkpoint)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    selection = {"selection_metric": "validation_loss", "selected_model": "single_best",
                 "best_epoch": best_epoch,
                 "best_validation_loss": float(np.min(history.history["val_loss"]))}
    probabilities = model.predict(test_ds, verbose=1)
    _finish_run(output, data_stat, config, te["y"], probabilities, te["object_id"],
                selection, "V5 Exp 2, seed=" + str(SEED), "Greens")


if __name__ == "__main__":
    run_exp2()
