"""Train and evaluate TDMN on the fixed CRTS split.

Model selection uses validation macro F1. The frozen selected estimator is
evaluated once on the held-out test set and object-level predictions are saved.
"""

import argparse
import json
import os
from pathlib import Path
import gc
import glob
import h5py
import math
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, backend as K
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =======================
# 1. 环境与硬件配置
# =======================
def auto_select_gpu():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,index",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        gpu_info = [
            (int(x.split(",")[0]), x.split(",")[1].strip())
            for x in result.strip().split("\n")
        ]
        gpu_info.sort(key=lambda x: x[0], reverse=True)
        if gpu_info:
            print(f"✅ Auto-selected GPU: {gpu_info[0][1]} (Free: {gpu_info[0][0]} MB)")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_info[0][1]
    except Exception:
        pass


auto_select_gpu()
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
np.random.seed(42)

# =======================
# 2. 全局参数设置
# =======================
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_H5 = str(REPO_ROOT / "data/processed/crts_tdmn.h5")

SAVE_BASE_DIR = None

BATCH_SIZE = 256
EPOCHS_STAGE1 = 300
EPOCHS_STAGE2 = 120
ANCHOR_WEIGHT = 0.1
LOGIT_COEFFICIENT = 0.3
FREQUENCY_SOURCE = "real"
SEED = 42
RESIZE_TARGET = (128, 128)

class_names = [
    "RRab",
    "RRc",
    "RRd",
    "Blazkho",
    "Ecl",
    "EA",
    "Rot",
    "LPV",
    "delta-Scuti",
    "ACep",
    "Cep-II",
]
NUM_CLASSES = len(class_names)


# =======================
# 3. 评估指标与数据预处理
# =======================
def calculate_per_class_metrics(cm, class_names):
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FP + FN)

    recall = TP / (TP + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    specificity = TN / (TN + FP + 1e-9)
    gmean_pr = np.sqrt(recall * precision)
    gmean_ss = np.sqrt(recall * specificity)

    print("\n" + "=" * 95)
    print(
        f"{'Class':<15} | {'Recall':<8} | {'Precision':<9} | {'Spec':<8} | {'G-Mean(PR)':<15} | {'G-Mean(SS)':<15}"
    )
    print("-" * 95)
    for i, cls in enumerate(class_names):
        print(
            f"{cls:<15} | {recall[i]:.4f}   | {precision[i]:.4f}    | {specificity[i]:.4f}   | {gmean_pr[i]:.4f}          | {gmean_ss[i]:.4f}"
        )
    print("=" * 95)
    return gmean_pr, gmean_ss


def get_class_balanced_weights(y, beta=0.99):
    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))
    samples_per_cls = [count_dict.get(i, 0) for i in range(NUM_CLASSES)]
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num + 1e-9)
    weights = weights / np.sum(weights) * len(samples_per_cls)
    return tf.constant(weights, dtype=tf.float32)


def augment_bi_modal(inputs, label):
    seq, img, is_gen = inputs["seq_in"], inputs["img_in"], inputs["is_gen"]

    # Do not roll sequence channels independently; preserve temporal alignment.

    raw_mag, smooth_mag, err, phase_sin, phase_cos, absolute_phase, valid_mask = (
        tf.unstack(seq, axis=-1)
    )

    # Class-conditional noise: preserve the weak modulation of Rot samples.
    noise_scale = 0.02 * (1.0 - label[6])
    mag_noise = (
        tf.random.normal(tf.shape(raw_mag), mean=0.0, stddev=noise_scale) * valid_mask
    )
    raw_mag = raw_mag + mag_noise

    seq_aug = tf.stack(
        [raw_mag, smooth_mag, err, phase_sin, phase_cos, absolute_phase, valid_mask],
        axis=-1,
    )

    return {
        "seq_in": seq_aug,
        "img_in": img,
        "feat_in": inputs["feat_in"],
        "is_gen": is_gen,
    }, label


def preprocess_bi_modal(seq, img, feat, mask, label, is_gen):
    img = tf.image.resize(img, RESIZE_TARGET, method="bilinear")

    # 获取数据，此时的 phase 是绝对天数 [0, period)
    phase, raw_mag, smooth_mag, err = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
    is_valid = mask
    raw_mag *= is_valid
    smooth_mag *= is_valid
    err *= is_valid

    # 【V19.5 绝对时间计算】
    # 动态获取序列中的最大天数(近似周期)以将相位规范化算 sin/cos，避免频率错误
    max_phase = tf.reduce_max(phase * is_valid)
    norm_phase = tf.math.divide_no_nan(phase, max_phase + 1e-8)

    pi_val = 3.141592653589793
    phase_sin = tf.math.sin(2.0 * pi_val * norm_phase) * is_valid
    phase_cos = tf.math.cos(2.0 * pi_val * norm_phase) * is_valid

    # 构建 7D 序列输入，同时保留相对循环特征(sin/cos) 和绝对时间特征(phase)
    seq_processed = tf.stack(
        [raw_mag, smooth_mag, err, phase_sin, phase_cos, phase, is_valid], axis=-1
    )

    return {
        "seq_in": seq_processed,
        "img_in": img,
        "feat_in": feat,
        "is_gen": is_gen,
    }, label


def create_dataset_with_indices(
    indices, X_seq, X_img, X_feat, X_mask, y_oh, is_gen, feat_dim, is_train=False
):
    seq_slice = X_seq[indices]
    img_slice = X_img[indices]
    feat_slice = X_feat[indices][:, :feat_dim]
    mask_slice = X_mask[indices]
    y_slice = y_oh[indices]
    gen_slice = is_gen[indices]

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (seq_slice, img_slice, feat_slice, mask_slice, y_slice, gen_slice)
        )

    def map_fn(s, i, f, m, y, ig):
        return preprocess_bi_modal(s, i, f, m, y, ig)

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.map(augment_bi_modal, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=8192, seed=42, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE, drop_remainder=is_train).prefetch(tf.data.AUTOTUNE)


# =======================
# 4. 模型架构组件
# =======================
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

    # 【V19.5 更新】：mag_in 取前三维，pos_in 取后续三维 [sin, cos, absolute_phase]
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

    # 极具破坏力的绝对位置注入！让 Transformer 知道这段序列有多长
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
    avg_pool = layers.GlobalAveragePooling2D()(x_img)
    max_pool = layers.GlobalMaxPooling2D()(x_img)
    img_repr = layers.Concatenate()([avg_pool, max_pool])
    img_repr = layers.Dense(
        128, activation="swish", kernel_regularizer=reg, name="img_repr"
    )(img_repr)

    return keras.Model(
        inputs={"seq_in": seq_in, "img_in": img_in},
        outputs=[seq_repr, img_repr],
        name="StrictDecoupledEncoder",
    )


class CosineClassifier(layers.Layer):
    def __init__(self, num_classes, scale=20.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
            name="cosine_weights",
        )

    def call(self, inputs):
        x_norm = tf.math.l2_normalize(inputs, axis=1)
        w_norm = tf.math.l2_normalize(self.w, axis=0)
        return self.scale * tf.matmul(x_norm, w_norm)


# =======================================================
# 5. 主模型类：三模态引航 & 专家路由门控 (MoE)
# =======================================================
class HeterogeneousEnsembleModel(keras.Model):
    def __init__(
        self,
        encoder,
        num_classes,
        class_weights,
        class_priors,
        feat_dim,
        projection_dim_modal=128,
        projection_dim_feat=32,
        label_smoothing=0.05,
        anchor_weight=0.1,
        logit_coefficient=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.class_priors = class_priors
        self.feat_dim = feat_dim
        self.anchor_weight = float(anchor_weight)
        self.logit_coefficient = float(logit_coefficient)
        self.current_stage = 1

        self.temperature = tf.Variable(
            0.2, trainable=False, dtype=tf.float32, name="temperature"
        )

        self.proj_seq = keras.Sequential(
            [
                layers.Dense(256, use_bias=False, input_shape=(128,)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(projection_dim_modal, use_bias=False),
            ]
        )
        self.proj_img = keras.Sequential(
            [
                layers.Dense(256, use_bias=False, input_shape=(128,)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(projection_dim_modal, use_bias=False),
            ]
        )

        self.feat_encoder = keras.Sequential(
            [
                layers.Dense(
                    64,
                    kernel_regularizer=regularizers.l2(1.5e-4),
                    input_shape=(self.feat_dim,),
                ),
                layers.BatchNormalization(),
                layers.Activation("swish"),
            ]
        )

        self.proj_feat = keras.Sequential(
            [
                layers.Dense(32, use_bias=False, input_shape=(64,)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(projection_dim_feat, use_bias=False),
            ]
        )

        self.fusion_dropout_1d = layers.SpatialDropout1D(0.15)
        self.fusion_dense = keras.Sequential(
            [
                layers.Dense(
                    128, kernel_regularizer=regularizers.l2(1.5e-4), input_shape=(256,)
                ),
                layers.BatchNormalization(),
                layers.Activation("swish"),
                layers.Dropout(0.5),
            ]
        )

        self.router = keras.Sequential(
            [
                layers.Dense(64, activation="swish", input_shape=(192,)),
                layers.Dense(3, activation="softmax", name="expert_router"),
            ]
        )

        self.classifier_cb = layers.Dense(
            num_classes, activation="linear", name="head_cb"
        )
        self.classifier_la = layers.Dense(
            num_classes, activation="linear", name="head_la"
        )
        self.classifier_cos = CosineClassifier(num_classes, name="head_cos")

        self.prototypes_modal = self.add_weight(
            shape=(num_classes, projection_dim_modal),
            initializer="zeros",
            trainable=False,
            name="class_prototypes_modal",
        )
        self.prototypes_feat = self.add_weight(
            shape=(num_classes, projection_dim_feat),
            initializer="zeros",
            trainable=False,
            name="class_prototypes_feat",
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.loss_seq_tracker = keras.metrics.Mean(name="con_loss_seq")
        self.loss_img_tracker = keras.metrics.Mean(name="con_loss_img")
        self.loss_feat_tracker = keras.metrics.Mean(name="con_loss_feat")

        self.fusion_dense.build((None, 256))
        self.feat_encoder.build((None, self.feat_dim))
        self.classifier_cb.build((None, 192))
        self.classifier_la.build((None, 192))
        self.classifier_cos.build((None, 192))

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.val_loss_tracker,
            self.loss_seq_tracker,
            self.loss_img_tracker,
            self.loss_feat_tracker,
        ]

    def _fuse_features(self, seq_repr, img_repr, training=False):
        stacked = tf.stack([seq_repr, img_repr], axis=1)
        stacked = self.fusion_dropout_1d(stacked, training=training)
        flat = layers.Reshape((256,))(stacked)
        return self.fusion_dense(flat, training=training)

    def calculate_proxy_loss(self, projections, prototypes, y_onehot):
        z = tf.math.l2_normalize(projections, axis=1)
        p = tf.math.l2_normalize(prototypes, axis=1)
        logits_con = tf.matmul(z, p, transpose_b=True) / self.temperature
        labels_idx = tf.argmax(y_onehot, axis=1)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels_idx, logits_con, from_logits=True
            )
        )

    def call(self, inputs, training=False):
        seq_repr, img_repr = self.encoder(
            {"seq_in": inputs["seq_in"], "img_in": inputs["img_in"]}, training=training
        )
        feat_repr = self.feat_encoder(inputs["feat_in"], training=training)

        if self.current_stage == 1:
            z_seq = self.proj_seq(seq_repr, training=training)
            z_img = self.proj_img(img_repr, training=training)
            z_feat = self.proj_feat(feat_repr, training=training)

            z_modal_norm = tf.math.l2_normalize((z_seq + z_img) / 2.0, axis=1)
            p_modal_norm = tf.math.l2_normalize(self.prototypes_modal, axis=1)
            logits_modal = (
                tf.matmul(z_modal_norm, p_modal_norm, transpose_b=True)
                / self.temperature
            )

            z_feat_norm = tf.math.l2_normalize(z_feat, axis=1)
            p_feat_norm = tf.math.l2_normalize(self.prototypes_feat, axis=1)
            logits_feat = (
                tf.matmul(z_feat_norm, p_feat_norm, transpose_b=True) / self.temperature
            )

            return (tf.nn.softmax(logits_modal) + tf.nn.softmax(logits_feat)) / 2.0
        else:
            fusion_features = self._fuse_features(seq_repr, img_repr, training=training)
            final_features = layers.Concatenate()([fusion_features, feat_repr])

            logits_cb = self.classifier_cb(final_features)
            # Logit adjustment is used only while fitting this head.  At
            # inference, the unadjusted scores implement the prior-corrected
            # decision rule described by Menon et al. (2021).
            logits_la = self.classifier_la(final_features)
            logits_cos = self.classifier_cos(final_features)

            prob_cb = tf.nn.softmax(logits_cb)
            prob_la = tf.nn.softmax(logits_la)
            prob_cos = tf.nn.softmax(logits_cos)
            probs = tf.stack([prob_cb, prob_la, prob_cos], axis=-1)

            routing_weights = self.router(final_features)
            routing_weights = tf.expand_dims(routing_weights, axis=1)

            final_best_prob = tf.reduce_sum(probs * routing_weights, axis=-1)
            return final_best_prob

    def calculate_losses(self, x, y_onehot, training=False):
        encoder_training_mode = (
            training if (self.current_stage == 1 or self.encoder.trainable) else False
        )
        seq_repr, img_repr = self.encoder(
            {"seq_in": x["seq_in"], "img_in": x["img_in"]},
            training=encoder_training_mode,
        )
        feat_repr = self.feat_encoder(x["feat_in"], training=training)

        if self.current_stage == 1:
            z_seq = self.proj_seq(seq_repr, training=training)
            z_img = self.proj_img(img_repr, training=training)
            z_feat = self.proj_feat(feat_repr, training=training)

            loss_seq = self.calculate_proxy_loss(z_seq, self.prototypes_modal, y_onehot)
            loss_img = self.calculate_proxy_loss(z_img, self.prototypes_modal, y_onehot)
            loss_feat = self.calculate_proxy_loss(
                z_feat, self.prototypes_feat, y_onehot
            )

            z_avg_modal = (z_seq + z_img) / 2.0

            if training:
                loss_seq_val = tf.stop_gradient(loss_seq)
                loss_img_val = tf.stop_gradient(loss_img)
                loss_feat_val = tf.stop_gradient(loss_feat)
                sum_loss = loss_seq_val + loss_img_val + loss_feat_val + 1e-9

                w_seq = loss_seq_val / sum_loss
                w_img = loss_img_val / sum_loss
                w_feat = loss_feat_val / sum_loss

                con_loss = (
                    tf.pow(w_seq, 0.5) * loss_seq
                    + tf.pow(w_img, 0.5) * loss_img
                    + tf.pow(w_feat, 0.5) * loss_feat
                )
            else:
                con_loss = (loss_seq + loss_img + loss_feat) / 3.0

            return con_loss, z_avg_modal, z_feat, loss_seq, loss_img, loss_feat

        else:
            fusion_features = self._fuse_features(seq_repr, img_repr, training=training)
            final_features = layers.Concatenate()([fusion_features, feat_repr])
            final_features_stopped = tf.stop_gradient(final_features)

            logits_la = self.classifier_la(final_features)
            logits_cb = self.classifier_cb(final_features_stopped)
            logits_cos = self.classifier_cos(final_features_stopped)

            y_true_smooth = y_onehot * (1.0 - self.label_smoothing) + (
                self.label_smoothing / self.num_classes
            )

            ce_cb = tf.keras.losses.categorical_crossentropy(
                y_true_smooth, logits_cb, from_logits=True
            )
            loss_cb = tf.reduce_mean(
                tf.reduce_sum(self.class_weights * y_onehot, axis=1) * ce_cb
            )

            adjusted_logits_la = logits_la + self.logit_coefficient * tf.math.log(
                self.class_priors + 1e-9
            )
            loss_la = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    y_true_smooth, adjusted_logits_la, from_logits=True
                )
            )

            probs_cos = tf.nn.softmax(logits_cos)
            focal_weight = tf.pow(
                1.0 - tf.reduce_sum(probs_cos * y_onehot, axis=-1), 2.0
            )
            ce_cos = tf.keras.losses.categorical_crossentropy(
                y_true_smooth, logits_cos, from_logits=True
            )
            loss_cos = tf.reduce_mean(focal_weight * ce_cos)

            # Keep the router objective consistent with inference: the LA
            # expert contributes its prior-corrected (unadjusted) posterior.
            prob_la = tf.nn.softmax(logits_la)
            prob_cb = tf.nn.softmax(logits_cb)
            probs = tf.stack([prob_cb, prob_la, probs_cos], axis=-1)

            routing_weights = self.router(final_features_stopped)
            routing_weights_exp = tf.expand_dims(routing_weights, axis=1)
            final_ensemble_prob = tf.reduce_sum(probs * routing_weights_exp, axis=-1)

            ce_router = tf.keras.losses.categorical_crossentropy(
                y_true_smooth, final_ensemble_prob, from_logits=False
            )
            loss_router = tf.reduce_mean(
                tf.reduce_sum(self.class_weights * y_onehot, axis=1) * ce_router
            )

            z_seq = self.proj_seq(seq_repr, training=training)
            z_img = self.proj_img(img_repr, training=training)
            z_feat = self.proj_feat(feat_repr, training=training)

            anchor_loss_seq = self.calculate_proxy_loss(
                z_seq, self.prototypes_modal, y_onehot
            )
            anchor_loss_img = self.calculate_proxy_loss(
                z_img, self.prototypes_modal, y_onehot
            )
            anchor_loss_feat = self.calculate_proxy_loss(
                z_feat, self.prototypes_feat, y_onehot
            )
            anchor_loss = (anchor_loss_seq + anchor_loss_img + anchor_loss_feat) / 3.0

            total_loss = (
                loss_cb
                + loss_la
                + loss_cos
                + loss_router
                + self.anchor_weight * anchor_loss
            )
            return total_loss, None, None, 0.0, 0.0, 0.0

    def train_step(self, data):
        x, y_onehot = data

        with tf.GradientTape() as tape:
            total_loss, z_avg_modal, z_feat, l_seq, l_img, l_feat = (
                self.calculate_losses(x, y_onehot, training=True)
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.current_stage == 1 and z_avg_modal is not None:
            real_mask = tf.expand_dims(1.0 - x["is_gen"], -1)
            y_onehot_real = y_onehot * real_mask

            counts = tf.transpose(tf.reduce_sum(y_onehot_real, axis=0, keepdims=True))
            valid_mask = counts > 0

            z_modal_norm = tf.math.l2_normalize(z_avg_modal, axis=1)
            sum_modal = tf.matmul(y_onehot_real, z_modal_norm, transpose_a=True)
            centers_modal = tf.math.l2_normalize(
                tf.math.divide_no_nan(sum_modal, counts), axis=1
            )
            new_p_modal = tf.math.l2_normalize(
                0.995 * self.prototypes_modal + 0.005 * centers_modal, axis=1
            )
            self.prototypes_modal.assign(
                tf.where(valid_mask, new_p_modal, self.prototypes_modal)
            )

            z_feat_norm = tf.math.l2_normalize(z_feat, axis=1)
            sum_feat = tf.matmul(y_onehot_real, z_feat_norm, transpose_a=True)
            centers_feat = tf.math.l2_normalize(
                tf.math.divide_no_nan(sum_feat, counts), axis=1
            )
            new_p_feat = tf.math.l2_normalize(
                0.995 * self.prototypes_feat + 0.005 * centers_feat, axis=1
            )
            self.prototypes_feat.assign(
                tf.where(valid_mask, new_p_feat, self.prototypes_feat)
            )

            self.loss_seq_tracker.update_state(l_seq)
            self.loss_img_tracker.update_state(l_img)
            self.loss_feat_tracker.update_state(l_feat)

        self.loss_tracker.update_state(total_loss)

        results = {"loss": self.loss_tracker.result()}
        if self.current_stage == 1:
            results["con_loss_seq"] = self.loss_seq_tracker.result()
            results["con_loss_img"] = self.loss_img_tracker.result()
            results["con_loss_feat"] = self.loss_feat_tracker.result()
        return results

    def test_step(self, data):
        x, y_onehot = data
        total_loss, _, _, l_seq, l_img, l_feat = self.calculate_losses(
            x, y_onehot, training=False
        )
        self.val_loss_tracker.update_state(total_loss)
        results = {"loss": self.val_loss_tracker.result()}

        if self.current_stage == 1:
            self.loss_seq_tracker.update_state(l_seq)
            self.loss_img_tracker.update_state(l_img)
            self.loss_feat_tracker.update_state(l_feat)
            results["con_loss_seq"] = self.loss_seq_tracker.result()
            results["con_loss_img"] = self.loss_img_tracker.result()
            results["con_loss_feat"] = self.loss_feat_tracker.result()

        return results


# =======================
# 6. 回调函数与调度器
# =======================
class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        def warmup_fn():
            return self.initial_learning_rate * step / self.warmup_steps

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


class TemperatureAnnealing(keras.callbacks.Callback):
    def __init__(
        self, initial_temp=0.2, final_temp=0.08, hold_epochs=40, decay_epochs=100
    ):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.hold_epochs = hold_epochs
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.hold_epochs:
            new_temp = self.initial_temp
        elif epoch < self.hold_epochs + self.decay_epochs:
            progress = (epoch - self.hold_epochs) / self.decay_epochs
            new_temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            new_temp = self.final_temp

        self.model.temperature.assign(new_temp)
        if epoch % 5 == 0:
            print(f"\n🌡️ [Temp Annealing] Epoch {epoch} - T: {new_temp:.4f}")


class Stage1LossEvaluation(keras.callbacks.Callback):
    def __init__(self, model_save_path, patience=25):
        super().__init__()
        self.model_save_path = model_save_path
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss", float("inf"))
        seq_loss = logs.get("val_con_loss_seq", 0.0)
        img_loss = logs.get("val_con_loss_img", 0.0)
        feat_loss = logs.get("val_con_loss_feat", 0.0)
        print(
            f" — val_loss: {current_loss:.4f} | Proxy(Seq): {seq_loss:.4f} | Proxy(Img): {img_loss:.4f} | Proxy(Feat): {feat_loss:.4f}"
        )

        if current_loss < self.best_loss:
            print(
                f"✅ Loss improved from {self.best_loss:.4f} to {current_loss:.4f}. Saving weights."
            )
            self.best_loss = current_loss
            self.wait = 0
            self.model.save_weights(self.model_save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(
                    f"\n⛔ Early stopping triggered after {self.patience} epochs without improvement."
                )


class F1Evaluation(keras.callbacks.Callback):
    def __init__(self, val_ds, y_val_true, model_save_path, patience=25):
        super().__init__()
        self.val_ds = val_ds
        self.y_val_true = y_val_true
        self.model_save_path = model_save_path
        self.patience = patience
        self.best_f1 = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        current_f1 = f1_score(self.y_val_true, y_pred, average="macro")
        if logs is not None:
            logs["val_macro_f1"] = current_f1

        print(
            f" — val_loss: {logs.get('loss', 0.0):.4f} — val_macro_f1: {current_f1:.4f}"
        )

        if current_f1 > self.best_f1:
            print(
                f"✅ Metric improved from {self.best_f1:.4f} to {current_f1:.4f}. Saving weights."
            )
            self.best_f1 = current_f1
            self.wait = 0
            self.model.save_weights(self.model_save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(
                    f"\n⛔ Early stopping triggered after {self.patience} epochs without improvement."
                )


class TopKAveragingCallback(keras.callbacks.Callback):
    def __init__(self, save_dir, k=3):
        super().__init__()
        self.save_dir = save_dir
        self.k = k
        self.top_k_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_f1 = logs.get("val_macro_f1")
        if current_f1 is None:
            return

        filepath = os.path.join(self.save_dir, f"topk_model_epoch_{epoch}.ckpt")

        if len(self.top_k_records) < self.k:
            self.model.save_weights(filepath)
            self.top_k_records.append((current_f1, filepath))
            self.top_k_records.sort(key=lambda x: x[0], reverse=True)
            print(
                f"🌟 [Top-{self.k} SWA] Epoch {epoch} added to Top-K list with F1: {current_f1:.4f}"
            )
        else:
            worst_f1_in_top_k = self.top_k_records[-1][0]
            if current_f1 > worst_f1_in_top_k:
                old_filepath = self.top_k_records[-1][1]
                for f in glob.glob(old_filepath + "*"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

                self.model.save_weights(filepath)
                self.top_k_records[-1] = (current_f1, filepath)
                self.top_k_records.sort(key=lambda x: x[0], reverse=True)
                print(
                    f"🌟 [Top-{self.k} SWA] Epoch {epoch} replaced worst in Top-K with F1: {current_f1:.4f}"
                )

    def get_averaged_weights(self, target_model):
        records = self.top_k_records
        if not records:
            files = glob.glob(
                os.path.join(self.save_dir, "topk_model_epoch_*.ckpt.index")
            )
            if not files:
                return None
            records = [(0.0, f.replace(".index", "")) for f in files]

        print(f"\n🔄 Calculating Top-{len(records)} SWA Weights from checkpoints...")
        averaged_weights = None

        for _, filepath in records:
            target_model.load_weights(filepath)
            current_weights = target_model.get_weights()
            if averaged_weights is None:
                averaged_weights = current_weights
            else:
                averaged_weights = [
                    avg_w + curr_w
                    for avg_w, curr_w in zip(averaged_weights, current_weights)
                ]

        averaged_weights = [w / len(records) for w in averaged_weights]
        return averaged_weights


# =======================
# 7. 主执行函数
# =======================
def run_experiment():
    K.clear_session()
    gc.collect()

    print(f"\n🚀 Loading CRTS Data (Including is_gen & Target Feat)...")
    with h5py.File(DATA_PATH_H5, "r") as hf:
        X_seq_tr, X_seq_val, X_seq_te = [
            np.nan_to_num(hf[f"X_seq_interp_{k}"][:].astype(np.float32), nan=-10.0)
            for k in ["train", "val", "test"]
        ]
        X_img_tr, X_img_val, X_img_te = [
            hf[f"X_img_interp_{k}"][:].astype(np.float32)
            for k in ["train", "val", "test"]
        ]
        X_feat_tr, X_feat_val, X_feat_te = [
            hf[f"X_feat_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]
        ]
        X_mask_tr, X_mask_val, X_mask_te = [
            hf[f"X_mask_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]
        ]
        y_tr, y_val, y_te = [
            hf[f"y_{k}"][:].astype(np.int32).squeeze() for k in ["train", "val", "test"]
        ]
        y_tr_oh, y_val_oh, y_te_oh = [
            hf[f"y_{k}_onehot"][:] for k in ["train", "val", "test"]
        ]

        is_gen_tr = hf["is_gen_train"][:].astype(np.float32)
        is_gen_val = hf["is_gen_val"][:].astype(np.float32)
        is_gen_te = hf["is_gen_test"][:].astype(np.float32)
        if "prototype_eligible_train" in hf:
            prototype_eligible_tr = hf["prototype_eligible_train"][:].astype(bool)
        else:
            prototype_eligible_tr = is_gen_tr == 0.0
            print(
                "WARNING: legacy HDF5 has no prototype_eligible_train; "
                "replacement duplicates cannot be separated from unique real objects."
            )
        object_id_test = (
            hf["object_id_test"][:].astype(str)
            if "object_id_test" in hf
            else np.array([str(i) for i in range(len(y_te))])
        )

    FEAT_DIM = X_feat_tr.shape[1]
    print(f"✅ Dynamic Feature Dimension Detected: {FEAT_DIM}")

    train_indices = np.arange(len(y_tr))
    if FREQUENCY_SOURCE == "real":
        frequency_labels = y_tr[prototype_eligible_tr]
    else:
        frequency_labels = y_tr
    if len(frequency_labels) == 0:
        raise RuntimeError("No training labels are available for class-frequency estimation.")
    class_weights = get_class_balanced_weights(frequency_labels)
    counts = np.bincount(frequency_labels, minlength=NUM_CLASSES).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise RuntimeError(f"Frequency source is missing training classes: {missing}")
    class_priors = tf.constant(counts / np.sum(counts), dtype=tf.float32)

    train_ds = create_dataset_with_indices(
        train_indices,
        X_seq_tr,
        X_img_tr,
        X_feat_tr,
        X_mask_tr,
        y_tr_oh,
        is_gen_tr,
        feat_dim=FEAT_DIM,
        is_train=True,
    )
    val_ds = create_dataset_with_indices(
        np.arange(len(y_val)),
        X_seq_val,
        X_img_val,
        X_feat_val,
        X_mask_val,
        y_val_oh,
        is_gen_val,
        feat_dim=FEAT_DIM,
        is_train=False,
    )
    test_ds = create_dataset_with_indices(
        np.arange(len(y_te)),
        X_seq_te,
        X_img_te,
        X_feat_te,
        X_mask_te,
        y_te_oh,
        is_gen_te,
        feat_dim=FEAT_DIM,
        is_train=False,
    )

    # V19.5 更新：网络输入通道由 6 扩展为 7（融入绝对天数特征）
    seq_shape, img_shape = (
        (X_seq_tr.shape[1], 7),
        (RESIZE_TARGET[0], RESIZE_TARGET[1], 3),
    )

    model = HeterogeneousEnsembleModel(
        build_pure_di_modal_encoder(seq_shape, img_shape),
        NUM_CLASSES,
        class_weights,
        class_priors,
        feat_dim=FEAT_DIM,
        anchor_weight=ANCHOR_WEIGHT,
        logit_coefficient=LOGIT_COEFFICIENT,
    )

    for dummy_x, dummy_y in train_ds.take(1):
        model(dummy_x)

    print("\n🔥 [Optimization] Performing Prototype Warm-start (Real Data ONLY!)...")
    model.current_stage = 1
    sum_modal = tf.zeros((NUM_CLASSES, 128))
    sum_feat = tf.zeros((NUM_CLASSES, 32))
    class_counts = tf.zeros((NUM_CLASSES, 1))

    real_train_indices = np.flatnonzero(prototype_eligible_tr).tolist()

    clean_warmup_ds = create_dataset_with_indices(
        real_train_indices,
        X_seq_tr,
        X_img_tr,
        X_feat_tr,
        X_mask_tr,
        y_tr_oh,
        is_gen_tr,
        feat_dim=FEAT_DIM,
        is_train=False,
    )
    for x_batch, y_batch in clean_warmup_ds:
        seq_repr, img_repr = model.encoder(
            {"seq_in": x_batch["seq_in"], "img_in": x_batch["img_in"]}, training=False
        )
        feat_repr = model.feat_encoder(x_batch["feat_in"], training=False)

        z_seq, z_img = model.proj_seq(seq_repr, training=False), model.proj_img(
            img_repr, training=False
        )
        z_feat = model.proj_feat(feat_repr, training=False)

        z_modal_norm = tf.math.l2_normalize((z_seq + z_img) / 2.0, axis=1)
        z_feat_norm = tf.math.l2_normalize(z_feat, axis=1)

        sum_modal += tf.matmul(y_batch, z_modal_norm, transpose_a=True)
        sum_feat += tf.matmul(y_batch, z_feat_norm, transpose_a=True)
        class_counts += tf.transpose(tf.reduce_sum(y_batch, axis=0, keepdims=True))

    valid_mask = class_counts > 0
    init_modal_protos = tf.math.l2_normalize(
        tf.math.divide_no_nan(sum_modal, class_counts), axis=1
    )
    init_feat_protos = tf.math.l2_normalize(
        tf.math.divide_no_nan(sum_feat, class_counts), axis=1
    )

    rand_modal = tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 128)), axis=1)
    rand_feat = tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 32)), axis=1)

    model.prototypes_modal.assign(tf.where(valid_mask, init_modal_protos, rand_modal))
    model.prototypes_feat.assign(tf.where(valid_mask, init_feat_protos, rand_feat))
    print("✅ Prototype Warm-start Completed.")

    # ==========================================
    # 🏁 STAGE 1: 三模态早期引航与对比对齐
    # ==========================================
    print("\n" + "=" * 50)
    print("🏁 STAGE 1: Contrastive Pretraining (Seq + Img + Feat)")
    print("=" * 50)

    total_steps_s1 = EPOCHS_STAGE1 * (len(train_indices) // BATCH_SIZE)
    lr_schedule_s1 = CosineDecayWithWarmup(
        initial_learning_rate=4e-4,
        warmup_steps=int(total_steps_s1 * 0.1),
        total_steps=total_steps_s1,
        alpha=1e-6,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule_s1, weight_decay=1e-4, clipnorm=1.0
        )
    )

    stage1_ckpt_path = os.path.join(SAVE_BASE_DIR, "stage1_tri_modal_best.ckpt")

    if os.path.exists(stage1_ckpt_path + ".index") or os.path.exists(stage1_ckpt_path):
        print(f"\n⏭️ [Stage 1] Skipping... Loading existing Tri-modal Stage 1 weights.")
        model.load_weights(stage1_ckpt_path)
    else:
        print(
            f"\n🚀 [Stage 1] Retraining Stage 1 to incorporate Feat early guidance..."
        )
        model.fit(
            train_ds,
            epochs=EPOCHS_STAGE1,
            validation_data=val_ds,
            callbacks=[
                TemperatureAnnealing(
                    initial_temp=0.2, final_temp=0.08, hold_epochs=20, decay_epochs=100
                ),
                Stage1LossEvaluation(stage1_ckpt_path, patience=30),
            ],
            verbose=1,
        )

    # ==============================================================================
    # Stage 2: representation adjustment and adaptive expert fusion.
    # ==============================================================================
    print("\n" + "=" * 50)
    print("🏁 STAGE 2: Unified Fine-tuning + MoE Routing + Max-Confidence")
    print("=" * 50)

    if os.path.exists(stage1_ckpt_path + ".index") or os.path.exists(stage1_ckpt_path):
        model.load_weights(stage1_ckpt_path)

    model.current_stage = 2
    model.encoder.trainable = True
    for layer in model.encoder.layers:
        if isinstance(
            layer,
            (layers.Conv1D, layers.Conv2D, layers.SeparableConv2D, layers.MaxPooling2D),
        ):
            layer.trainable = False

    total_steps_s2 = EPOCHS_STAGE2 * (len(train_indices) // BATCH_SIZE)
    lr_schedule_s2 = CosineDecayWithWarmup(
        initial_learning_rate=1e-4,
        warmup_steps=int(total_steps_s2 * 0.1),
        total_steps=total_steps_s2,
        alpha=1e-5,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule_s2, weight_decay=1e-4, clipnorm=1.0
        )
    )

    stage2_ckpt_path = os.path.join(SAVE_BASE_DIR, "stage2_merged_best.ckpt")
    topk_swa_cb = TopKAveragingCallback(save_dir=SAVE_BASE_DIR, k=3)

    model.fit(
        train_ds,
        epochs=EPOCHS_STAGE2,
        validation_data=val_ds,
        callbacks=[
            F1Evaluation(val_ds, y_val, stage2_ckpt_path, patience=30),
            topk_swa_cb,
        ],
        verbose=1,
    )

    # ==========================================
    # 🏁 FINAL EVALUATION
    # ==========================================
    print("\n[FINAL EVALUATION: Physics-Informed MoE Ensemble]")

    if os.path.exists(stage2_ckpt_path + ".index") or os.path.exists(stage2_ckpt_path):
        model.load_weights(stage2_ckpt_path)

    # Select between the ordinary checkpoint and the Top-K average on validation
    # data only. The test set is not read until this decision has been frozen.
    print("\nEvaluating Standard Best Model on validation data...")
    val_preds_best = model.predict(val_ds)
    val_y_pred_best = np.argmax(val_preds_best, axis=1)
    val_f1_best = f1_score(y_val, val_y_pred_best, average="macro")

    print("\nEvaluating Top-K averaged model on validation data...")
    swa_model = HeterogeneousEnsembleModel(
        build_pure_di_modal_encoder(seq_shape, img_shape),
        NUM_CLASSES,
        class_weights,
        class_priors,
        feat_dim=FEAT_DIM,
        anchor_weight=ANCHOR_WEIGHT,
        logit_coefficient=LOGIT_COEFFICIENT,
    )
    swa_model.current_stage = 2
    for dummy_x, dummy_y in train_ds.take(1):
        swa_model(dummy_x)

    avg_weights = topk_swa_cb.get_averaged_weights(swa_model)
    if avg_weights is not None:
        swa_model.set_weights(avg_weights)
        val_preds_swa = swa_model.predict(val_ds)
        val_y_pred_swa = np.argmax(val_preds_swa, axis=1)
        val_f1_swa = f1_score(y_val, val_y_pred_swa, average="macro")
    else:
        print("No valid Top-K checkpoints were found; selecting the standard checkpoint.")
        val_f1_swa = float("-inf")

    if val_f1_swa > val_f1_best:
        selected_model = swa_model
        selected_name = "top3_average"
        selected_val_f1 = val_f1_swa
    else:
        selected_model = model
        selected_name = "standard_best"
        selected_val_f1 = val_f1_best

    selection = {
        "selection_metric": "validation_macro_f1",
        "standard_best_validation_macro_f1": float(val_f1_best),
        "top3_average_validation_macro_f1": (
            None if not np.isfinite(val_f1_swa) else float(val_f1_swa)
        ),
        "selected_model": selected_name,
        "selected_validation_macro_f1": float(selected_val_f1),
    }
    with open(os.path.join(SAVE_BASE_DIR, "model_selection.json"), "w", encoding="utf-8") as fh:
        json.dump(selection, fh, ensure_ascii=False, indent=2)

    print(f"Selected {selected_name} using validation Macro-F1 only.")
    print("\nEvaluating the frozen estimator once on the held-out test set...")
    final_probabilities = selected_model.predict(test_ds)
    final_preds = np.argmax(final_probabilities, axis=1)

    cm = confusion_matrix(y_te, final_preds, labels=np.arange(NUM_CLASSES))
    per_class_recall = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=cm.sum(axis=1) > 0,
    )
    gmean_recall = float(
        np.exp(np.mean(np.log(np.clip(per_class_recall, 1e-12, 1.0))))
    )
    report = classification_report(
        y_te,
        final_preds,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        **selection,
        "test_macro_f1": float(f1_score(y_te, final_preds, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, final_preds)),
        "test_gmean_recall": gmean_recall,
        "test_support": int(len(y_te)),
        "frequency_source": FREQUENCY_SOURCE,
        "anchor_weight": float(ANCHOR_WEIGHT),
        "logit_coefficient": float(LOGIT_COEFFICIENT),
        "seed": int(SEED),
        "classification_report": report,
    }
    with open(os.path.join(SAVE_BASE_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(SAVE_BASE_DIR, "test_predictions.npz"),
        object_id=object_id_test,
        y_true=y_te,
        y_pred=final_preds,
        probabilities=final_probabilities,
    )

    print(classification_report(y_te, final_preds, target_names=class_names, digits=4))
    calculate_per_class_metrics(cm, class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, final_preds, normalize="true"),
        display_labels=class_names,
    ).plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format=".2f")
    plt.title(f"TDMN confusion matrix: {selected_name}")
    plt.savefig(os.path.join(SAVE_BASE_DIR, "confusion_matrix_test.png"), dpi=300)
    plt.close(fig)
    print("Review experiment completed; metrics and object-level predictions were saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate the accepted-paper TDMN. Model selection is made "
            "on validation Macro-F1; the held-out test set is evaluated once."
        )
    )
    parser.add_argument("--data", default=DATA_PATH_H5, help="Prepared CRTS HDF5 dataset")
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/crts/tdmn_seed<seed>")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs-stage1", type=int, default=EPOCHS_STAGE1)
    parser.add_argument("--epochs-stage2", type=int, default=EPOCHS_STAGE2)
    parser.add_argument("--anchor-weight", type=float, default=ANCHOR_WEIGHT)
    parser.add_argument("--logit-coefficient", type=float, default=LOGIT_COEFFICIENT)
    parser.add_argument(
        "--frequency-source",
        choices=("real", "all"),
        default=FREQUENCY_SOURCE,
        help=(
            "Use only prototype-eligible unique real objects (recommended) or all "
            "augmented training rows to estimate class weights/logit priors."
        ),
    )
    return parser.parse_args()


def configure_from_args(args):
    global DATA_PATH_H5, SAVE_BASE_DIR, SEED, BATCH_SIZE
    global EPOCHS_STAGE1, EPOCHS_STAGE2, ANCHOR_WEIGHT
    global LOGIT_COEFFICIENT, FREQUENCY_SOURCE

    DATA_PATH_H5 = str(Path(args.data).expanduser().resolve())
    output_dir = args.output_dir or REPO_ROOT / "results/crts" / f"tdmn_seed{args.seed}"
    SAVE_BASE_DIR = str(Path(output_dir).expanduser().resolve())
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    EPOCHS_STAGE1 = args.epochs_stage1
    EPOCHS_STAGE2 = args.epochs_stage2
    ANCHOR_WEIGHT = args.anchor_weight
    LOGIT_COEFFICIENT = args.logit_coefficient
    FREQUENCY_SOURCE = args.frequency_source

    if not Path(DATA_PATH_H5).is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH_H5}")
    if ANCHOR_WEIGHT < 0 or LOGIT_COEFFICIENT < 0:
        raise ValueError("Loss coefficients must be non-negative")
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        print(f"WARNING: deterministic TensorFlow operations unavailable: {exc}")

    config = vars(args).copy()
    config["data"] = DATA_PATH_H5
    config["output_dir"] = SAVE_BASE_DIR
    with open(os.path.join(SAVE_BASE_DIR, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    configure_from_args(parse_args())
    run_experiment()
