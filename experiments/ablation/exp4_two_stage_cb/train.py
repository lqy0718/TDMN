"""V5-faithful CRTS ablation with review-safety fixes only."""
import os
import sys
import subprocess
from pathlib import Path

EXPERIMENT_ID = "exp4"
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
EPOCHS_STAGE1 = 150
EPOCHS_STAGE2 = 100
RESIZE_TARGET = (128, 128)
V5_SOURCE_SHA256 = "37ccb2eec51444dd901da52903be724b27009f1cecd2cfd6ea5881be41f9c903"
PROTOCOL = "V5-faithful-review-fix-v1"
class_names = ["RRab", "RRc", "RRd", "Blazkho", "Ecl", "EA", "Rot", "LPV", "delta-Scuti", "ACep", "Cep-II"]
NUM_CLASSES = len(class_names)

def get_class_balanced_weights(y, beta=0.99):
    unique, counts = np.unique(y, return_counts=True)
    samples_per_cls = [dict(zip(unique, counts)).get(i, 0) for i in range(NUM_CLASSES)]
    weights = (1.0 - beta) / np.array(1.0 - np.power(beta, samples_per_cls) + 1e-9)
    return tf.constant(
        weights / np.sum(weights) * len(samples_per_cls), dtype=tf.float32
    )


# ==========================================
# 1. TDMN backbone components
# ==========================================
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
    mag_in = seq_in[:, :, 0:3]
    pos_in = seq_in[:, :, 3:6]
    valid_mask = seq_in[:, :, 6]
    attn_mask = tf.cast(valid_mask, tf.bool)[:, tf.newaxis, :]

    x_seq = layers.Conv1D(
        32, 5, padding="same", use_bias=False, kernel_regularizer=reg
    )(mag_in)
    x_seq = layers.BatchNormalization()(x_seq)
    x_seq = layers.Activation("swish")(x_seq)
    conv_3 = layers.Conv1D(32, 3, padding="same", activation="swish")(x_seq)
    conv_7 = layers.Conv1D(32, 7, padding="same", activation="swish")(x_seq)
    x_seq = layers.Concatenate()([conv_3, conv_7])
    x_seq = layers.SpatialDropout1D(0.2)(x_seq)

    pos_emb = layers.Dense(64)(pos_in)
    x_seq = layers.Add()([x_seq, pos_emb])

    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.2)(
            x_seq, x_seq, attention_mask=attn_mask
        )
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + attn)
        ffn = layers.Dense(128, activation="swish")(x_seq)
        ffn = layers.Dropout(0.2)(ffn)
        ffn = layers.Dense(64)(ffn)
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + ffn)

    valid_mask_exp = tf.expand_dims(valid_mask, -1)
    seq_sum = tf.reduce_sum(x_seq * valid_mask_exp, axis=1)
    seq_lens = tf.maximum(tf.reduce_sum(valid_mask_exp, axis=1), 1.0)
    seq_mean = seq_sum / seq_lens
    mask_for_max = (1.0 - valid_mask_exp) * -1e4
    seq_max = tf.reduce_max(x_seq + mask_for_max, axis=1)
    seq_pooled = layers.Concatenate()([seq_mean, seq_max])
    seq_repr = layers.Dense(
        128, activation="swish", kernel_regularizer=reg, name="seq_repr"
    )(seq_pooled)

    img_in = keras.Input(shape=img_shape, name="img_in")
    x_img = layers.Conv2D(32, 3, strides=1, padding="same")(img_in)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Activation("swish")(x_img)
    x_img = layers.MaxPooling2D(2)(x_img)
    for filters, do_rate in [(64, 0.05), (128, 0.05), (256, 0.1)]:
        x_img = layers.SeparableConv2D(filters, 3, padding="same")(x_img)
        x_img = layers.BatchNormalization()(x_img)
        x_img = layers.Activation("swish")(x_img)
        x_img = squeeze_excite_block(x_img, ratio=16)
        x_img = layers.SpatialDropout2D(do_rate)(x_img)
        if filters != 256:
            x_img = layers.MaxPooling2D(2)(x_img)

    img_repr = layers.Concatenate()(
        [layers.GlobalAveragePooling2D()(x_img), layers.GlobalMaxPooling2D()(x_img)]
    )
    img_repr = layers.Dense(
        128, activation="swish", kernel_regularizer=reg, name="img_repr"
    )(img_repr)

    return keras.Model(
        inputs={"seq_in": seq_in, "img_in": img_in},
        outputs=[seq_repr, img_repr],
        name="Encoder_V19_5",
    )


class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_fn = lambda: self.initial_learning_rate * step / self.warmup_steps

        def decay_fn():
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.initial_learning_rate * (
                self.alpha
                + (1.0 - self.alpha)
                * 0.5
                * (1.0 + tf.cos(tf.constant(math.pi) * progress))
            )

        return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)


# ==========================================
# 2. 数据处理与增强 (带 CPU 显存保护)
# ==========================================
def preprocess_bi_modal(seq, img, feat, mask, label, is_gen):
    img = tf.image.resize(img, RESIZE_TARGET, method="bilinear")
    phase, raw_mag, smooth_mag, err = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
    is_valid = mask
    raw_mag *= is_valid
    smooth_mag *= is_valid
    err *= is_valid

    max_phase = tf.reduce_max(phase * is_valid)
    norm_phase = tf.math.divide_no_nan(phase, max_phase + 1e-8)
    pi_val = 3.141592653589793
    phase_sin = tf.math.sin(2.0 * pi_val * norm_phase) * is_valid
    phase_cos = tf.math.cos(2.0 * pi_val * norm_phase) * is_valid

    seq_processed = tf.stack(
        [raw_mag, smooth_mag, err, phase_sin, phase_cos, phase, is_valid], axis=-1
    )
    return {
        "seq_in": seq_processed,
        "img_in": img,
        "feat_in": feat,
        "is_gen": is_gen,
    }, label


def augment_bi_modal(inputs, label):
    seq, img, is_gen = inputs["seq_in"], inputs["img_in"], inputs["is_gen"]
    raw_mag, smooth_mag, err, phase_sin, phase_cos, absolute_phase, valid_mask = (
        tf.unstack(seq, axis=-1)
    )
    noise_scale = 0.02 * (1.0 - label[6])  # Preserve Rot amplitudes.
    mag_noise = (
        tf.random.normal(tf.shape(raw_mag), mean=0.0, stddev=noise_scale) * valid_mask
    )
    seq_aug = tf.stack(
        [
            raw_mag + mag_noise,
            smooth_mag,
            err,
            phase_sin,
            phase_cos,
            absolute_phase,
            valid_mask,
        ],
        axis=-1,
    )
    return {
        "seq_in": seq_aug,
        "img_in": img,
        "feat_in": inputs["feat_in"],
        "is_gen": is_gen,
    }, label


def create_dataset_with_indices(
    indices, X_seq, X_img, X_feat, X_mask, y_oh, is_gen, feat_dim, is_train=False
):
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                X_seq[indices],
                X_img[indices],
                X_feat[indices][:, :feat_dim],
                X_mask[indices],
                y_oh[indices],
                is_gen[indices],
            )
        )

    def map_fn(s, i, f, m, y, ig):
        return preprocess_bi_modal(s, i, f, m, y, ig)

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.map(augment_bi_modal, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==========================================
# 3. Keras Subclass Model (修复了 Input Shape 注入)
# ==========================================
class DecoupledCBModel(keras.Model):
    def __init__(self, encoder, feat_dim, num_classes, cb_weights, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.cb_weights = cb_weights
        self.current_stage = 1

        # 🚨 [修复1]: Variable 指定 name
        self.temperature = tf.Variable(
            0.1, trainable=False, dtype=tf.float32, name="temperature"
        )

        # 🚨 [修复2]: 给每一个 Sequential 强制注入 keras.Input，确保类初始化时权重就建立！
        self.feat_encoder = keras.Sequential(
            [
                keras.Input(shape=(self.feat_dim,)),
                layers.Dense(64, kernel_regularizer=regularizers.l2(1.5e-4)),
                layers.BatchNormalization(),
                layers.Activation("swish"),
            ],
            name="feat_encoder",
        )

        self.proj_seq = keras.Sequential(
            [
                keras.Input(shape=(128,)),
                layers.Dense(256, use_bias=False),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(128, use_bias=False),
            ],
            name="proj_seq",
        )

        self.proj_img = keras.Sequential(
            [
                keras.Input(shape=(128,)),
                layers.Dense(256, use_bias=False),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(128, use_bias=False),
            ],
            name="proj_img",
        )

        self.proj_feat = keras.Sequential(
            [
                keras.Input(shape=(64,)),
                layers.Dense(32, use_bias=False),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(32, use_bias=False),
            ],
            name="proj_feat",
        )

        self.fusion_dense = keras.Sequential(
            [
                keras.Input(shape=(256,)),
                layers.Dense(128, kernel_regularizer=regularizers.l2(1.5e-4)),
                layers.BatchNormalization(),
                layers.Activation("swish"),
                layers.Dropout(0.5),
            ],
            name="fusion_dense",
        )

        self.classifier_cb = layers.Dense(
            num_classes, activation="softmax", name="cb_head"
        )

        # 🚨 [修复3]: 强制 Build Dense 层
        self.classifier_cb.build((None, 192))  # 128 (fusion) + 64 (feat) = 192

        # 🚨 [修复4]: add_weight 指定 name
        self.prototypes_modal = self.add_weight(
            shape=(num_classes, 128),
            initializer="zeros",
            trainable=False,
            name="class_prototypes_modal",
        )
        self.prototypes_feat = self.add_weight(
            shape=(num_classes, 32),
            initializer="zeros",
            trainable=False,
            name="class_prototypes_feat",
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

    def call(self, inputs, training=False):
        if self.current_stage == 1:
            seq_repr, img_repr = self.encoder(
                {"seq_in": inputs["seq_in"], "img_in": inputs["img_in"]},
                training=training,
            )
            feat_repr = self.feat_encoder(inputs["feat_in"], training=training)

            z_s = self.proj_seq(seq_repr, training=training)
            z_i = self.proj_img(img_repr, training=training)
            z_f = self.proj_feat(feat_repr, training=training)

            logits_m = (
                tf.matmul(
                    tf.math.l2_normalize((z_s + z_i) / 2.0, axis=1),
                    tf.math.l2_normalize(self.prototypes_modal, axis=1),
                    transpose_b=True,
                )
                / self.temperature
            )
            logits_f = (
                tf.matmul(
                    tf.math.l2_normalize(z_f, axis=1),
                    tf.math.l2_normalize(self.prototypes_feat, axis=1),
                    transpose_b=True,
                )
                / self.temperature
            )
            return (tf.nn.softmax(logits_m) + tf.nn.softmax(logits_f)) / 2.0
        else:
            # Stage 2: 冻结 Backbone
            seq_repr, img_repr = self.encoder(
                {"seq_in": inputs["seq_in"], "img_in": inputs["img_in"]}, training=False
            )
            feat_repr = self.feat_encoder(inputs["feat_in"], training=False)

            fusion = self.fusion_dense(
                layers.Reshape((256,))(
                    layers.SpatialDropout1D(0.15)(
                        tf.stack([seq_repr, img_repr], axis=1), training=training
                    )
                ),
                training=training,
            )
            return self.classifier_cb(
                layers.Concatenate()([fusion, feat_repr]), training=training
            )

    def calculate_proxy_loss(self, z, p, y_onehot):
        logits = (
            tf.matmul(
                tf.math.l2_normalize(z, axis=1),
                tf.math.l2_normalize(p, axis=1),
                transpose_b=True,
            )
            / self.temperature
        )
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            if self.current_stage == 1:
                seq_repr, img_repr = self.encoder(
                    {"seq_in": x["seq_in"], "img_in": x["img_in"]}, training=True
                )
                feat_repr = self.feat_encoder(x["feat_in"], training=True)
                z_s = self.proj_seq(seq_repr, training=True)
                z_i = self.proj_img(img_repr, training=True)
                z_f = self.proj_feat(feat_repr, training=True)

                loss = (
                    self.calculate_proxy_loss(z_s, self.prototypes_modal, y)
                    + self.calculate_proxy_loss(z_i, self.prototypes_modal, y)
                    + self.calculate_proxy_loss(z_f, self.prototypes_feat, y)
                ) / 3.0
            else:
                # Compute the class-balanced weighted loss.
                seq_repr, img_repr = self.encoder(
                    {"seq_in": x["seq_in"], "img_in": x["img_in"]}, training=False
                )
                feat_repr = self.feat_encoder(x["feat_in"], training=False)

                fusion = self.fusion_dense(
                    layers.Reshape((256,))(
                        layers.SpatialDropout1D(0.15)(
                            tf.stack([seq_repr, img_repr], axis=1), training=True
                        )
                    ),
                    training=True,
                )
                preds = self.classifier_cb(
                    layers.Concatenate()([fusion, feat_repr]), training=True
                )

                ce_loss = tf.keras.losses.categorical_crossentropy(y, preds)
                loss = tf.reduce_mean(
                    tf.reduce_sum(self.cb_weights * y, axis=1) * ce_loss
                )

        # 彻底解耦更新：由于上面 __init__ 中加上了 keras.Input，这里的层权重必然已经生成，不会报错
        train_vars = (
            self.trainable_variables
            if self.current_stage == 1
            else self.fusion_dense.trainable_variables
            + self.classifier_cb.trainable_variables
        )

        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        if self.current_stage == 1:
            z_avg_modal = (z_s + z_i) / 2.0
            real_mask = tf.expand_dims(1.0 - x["is_gen"], -1)
            y_real = y * real_mask
            counts = tf.transpose(tf.reduce_sum(y_real, axis=0, keepdims=True))
            valid_mask = counts > 0

            centers_m = tf.math.l2_normalize(
                tf.math.divide_no_nan(
                    tf.matmul(
                        y_real,
                        tf.math.l2_normalize(z_avg_modal, axis=1),
                        transpose_a=True,
                    ),
                    counts,
                ),
                axis=1,
            )
            self.prototypes_modal.assign(
                tf.where(
                    valid_mask,
                    tf.math.l2_normalize(
                        0.99 * self.prototypes_modal + 0.01 * centers_m, axis=1
                    ),
                    self.prototypes_modal,
                )
            )
            centers_f = tf.math.l2_normalize(
                tf.math.divide_no_nan(
                    tf.matmul(
                        y_real, tf.math.l2_normalize(z_f, axis=1), transpose_a=True
                    ),
                    counts,
                ),
                axis=1,
            )
            self.prototypes_feat.assign(
                tf.where(
                    valid_mask,
                    tf.math.l2_normalize(
                        0.99 * self.prototypes_feat + 0.01 * centers_f, axis=1
                    ),
                    self.prototypes_feat,
                )
            )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        seq_repr, img_repr = self.encoder(
            {"seq_in": x["seq_in"], "img_in": x["img_in"]}, training=False
        )
        feat_repr = self.feat_encoder(x["feat_in"], training=False)

        if self.current_stage == 1:
            loss = (
                self.calculate_proxy_loss(
                    self.proj_seq(seq_repr), self.prototypes_modal, y
                )
                + self.calculate_proxy_loss(
                    self.proj_img(img_repr), self.prototypes_modal, y
                )
                + self.calculate_proxy_loss(
                    self.proj_feat(feat_repr), self.prototypes_feat, y
                )
            ) / 3.0
        else:
            fusion = self.fusion_dense(
                layers.Reshape((256,))(tf.stack([seq_repr, img_repr], axis=1))
            )
            preds = self.classifier_cb(layers.Concatenate()([fusion, feat_repr]))
            loss = tf.reduce_mean(
                tf.reduce_sum(self.cb_weights * y, axis=1)
                * tf.keras.losses.categorical_crossentropy(y, preds)
            )

        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}


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


def run_exp4():
    K.clear_session()
    gc.collect()
    prepared = _prepare_output()
    if prepared is None:
        return
    output, data_stat, config = prepared
    data = _load_current_review_data()
    tr, val, te = data["train"], data["val"], data["test"]
    feat_dim = tr["feat"].shape[1]
    cb_weights = get_class_balanced_weights(tr["y"])
    train_ds = create_dataset_with_indices(np.arange(len(tr["y"])), tr["seq"], tr["img"],
        tr["feat"], tr["mask"], tr["y_oh"], tr["is_gen"], feat_dim, is_train=True)
    val_ds = create_dataset_with_indices(np.arange(len(val["y"])), val["seq"], val["img"],
        val["feat"], val["mask"], val["y_oh"], val["is_gen"], feat_dim, is_train=False)
    test_ds = create_dataset_with_indices(np.arange(len(te["y"])), te["seq"], te["img"],
        te["feat"], te["mask"], te["y_oh"], te["is_gen"], feat_dim, is_train=False)

    encoder = build_pure_di_modal_encoder((tr["seq"].shape[1], 7), (128, 128, 3))
    model = DecoupledCBModel(encoder, feat_dim, NUM_CLASSES, cb_weights)
    for dummy_x, _ in train_ds.take(1):
        model(dummy_x)

    # V5 prototype warm-start: all rows with is_gen == 0.
    sum_modal = tf.zeros((NUM_CLASSES, 128))
    sum_feat = tf.zeros((NUM_CLASSES, 32))
    class_counts = tf.zeros((NUM_CLASSES, 1))
    real_idx = [i for i in range(len(tr["y"])) if tr["is_gen"][i] == 0.0]
    warm_ds = create_dataset_with_indices(real_idx, tr["seq"], tr["img"], tr["feat"],
        tr["mask"], tr["y_oh"], tr["is_gen"], feat_dim, is_train=False)
    for x_batch, y_batch in warm_ds:
        seq_repr, img_repr = model.encoder(
            {"seq_in": x_batch["seq_in"], "img_in": x_batch["img_in"]}, training=False)
        feat_repr = model.feat_encoder(x_batch["feat_in"], training=False)
        z_modal = tf.math.l2_normalize(
            (model.proj_seq(seq_repr) + model.proj_img(img_repr)) / 2.0, axis=1)
        z_feat = tf.math.l2_normalize(model.proj_feat(feat_repr), axis=1)
        sum_modal += tf.matmul(y_batch, z_modal, transpose_a=True)
        sum_feat += tf.matmul(y_batch, z_feat, transpose_a=True)
        class_counts += tf.transpose(tf.reduce_sum(y_batch, axis=0, keepdims=True))
    model.prototypes_modal.assign(tf.where(class_counts > 0,
        tf.math.l2_normalize(tf.math.divide_no_nan(sum_modal, class_counts), axis=1),
        tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 128)), axis=1)))
    model.prototypes_feat.assign(tf.where(class_counts > 0,
        tf.math.l2_normalize(tf.math.divide_no_nan(sum_feat, class_counts), axis=1),
        tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 32)), axis=1)))

    # V5 Stage 1: 150 epochs, temperature 0.1, EMA 0.99, equal proxy losses.
    model.current_stage = 1
    total_steps_1 = EPOCHS_STAGE1 * (len(tr["y"]) // BATCH_SIZE)
    model.compile(optimizer=keras.optimizers.AdamW(
        learning_rate=CosineDecayWithWarmup(4e-4, int(total_steps_1 * 0.1),
                                             total_steps_1, 1e-6),
        weight_decay=1e-4, clipnorm=1.0))
    stage1_checkpoint = str(output / "stage1.ckpt")
    history1 = model.fit(train_ds, epochs=EPOCHS_STAGE1, validation_data=val_ds,
        callbacks=[keras.callbacks.ModelCheckpoint(stage1_checkpoint,
                   save_best_only=True, save_weights_only=True, monitor="val_loss"),
                   keras.callbacks.CSVLogger(str(output / "history_stage1.csv"))],
        verbose=1)
    # Correctness fix: V5 loaded this only when a checkpoint pre-existed.
    model.load_weights(stage1_checkpoint)

    # V5 Stage 2: retain its original E2E/freeze choice and its original loss.
    model.current_stage = 2
    model.encoder.trainable = False
    model.feat_encoder.trainable = False
    for layer in model.encoder.layers:
        layer.trainable = False
    for layer in model.feat_encoder.layers:
        layer.trainable = False
    total_steps_2 = EPOCHS_STAGE2 * (len(tr["y"]) // BATCH_SIZE)
    model.compile(optimizer=keras.optimizers.AdamW(
        learning_rate=CosineDecayWithWarmup(1e-4, int(total_steps_2 * 0.1),
                                             total_steps_2, 1e-5),
        weight_decay=1e-4, clipnorm=1.0))
    stage2_checkpoint = str(output / "stage2.ckpt")
    history2 = model.fit(train_ds, epochs=EPOCHS_STAGE2, validation_data=val_ds,
        callbacks=[keras.callbacks.ModelCheckpoint(stage2_checkpoint,
                   save_best_only=True, save_weights_only=True, monitor="val_loss"),
                   keras.callbacks.CSVLogger(str(output / "history_stage2.csv"))],
        verbose=1)
    model.load_weights(stage2_checkpoint)
    selection = {
        "selection_metric": "validation_loss",
        "stage1_best_epoch": int(np.argmin(history1.history["val_loss"]) + 1),
        "stage1_best_validation_loss": float(np.min(history1.history["val_loss"])),
        "stage2_best_epoch": int(np.argmin(history2.history["val_loss"]) + 1),
        "stage2_best_validation_loss": float(np.min(history2.history["val_loss"])),
        "selected_model": "stage2_single_best"
    }
    probabilities = model.predict(test_ds, verbose=1)
    _finish_run(output, data_stat, config, te["y"], probabilities, te["object_id"],
                selection, "V5 Exp 4, seed=" + str(SEED), "Oranges")


if __name__ == "__main__":
    run_exp4()
