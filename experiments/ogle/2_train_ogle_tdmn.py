"""
================================================================================
【双模态+物理先验  V19.4】
核心护城河：
1. [绝对防御 NaN] log1p 压缩绝对时间 + MHA 阵前 LayerNorm，彻底斩断 Float16 下 Q*K 点积溢出崩溃！
2. [底层防线] Float32 空间安全池化；全空序列掩码前置苏醒防 Softmax 除零。
3. [跨卡守卫] All-Reduce 全局同步 Prototype，确保 4 卡聚类中心绝对一致。
4. [降维瓶颈] 1D CNN + Pooling 将 800 序列安全压缩至 400，滤除局部底噪并暴增 Transformer 算力。
5. [OGM 压制] 动态梯度调制 (提拔差生，压制学霸)，抑制过拟合，强迫多模态同步前行。
================================================================================
"""

import argparse
import json
import os
import logging
import warnings
from pathlib import Path

# Reduce verbose TensorFlow/XLA runtime logging.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Value in checkpoint could not be found.*")
warnings.filterwarnings("ignore", message=".*Gradients do not exist.*")

import gc
import glob
import h5py
import math
import subprocess
import numpy as np
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
# 1. 极致加速与选卡配置 (A100 专属)
# =======================
tf.config.optimizer.set_jit(True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def auto_select_best_gpus(num_gpus=4, min_memory_mb=30000):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        gpu_info = [
            (int(x.split(",")[0]), x.split(",")[1].strip())
            for x in result.strip().split("\n")
        ]
        valid_gpus = [g for g in gpu_info if g[0] >= min_memory_mb]
        valid_gpus.sort(key=lambda x: x[0], reverse=True)
        if len(valid_gpus) >= num_gpus:
            selected_gpus = valid_gpus[:num_gpus]
            selected_indices = ",".join([g[1] for g in selected_gpus])
            print(f"✅ Auto-selected {num_gpus} GPUs: {selected_indices}")
            return num_gpus
        else:
            print(f"⚠️ Warning: Only found {len(valid_gpus)} GPUs with >{min_memory_mb}MB free.")
            selected_indices = ",".join([g[1] for g in valid_gpus])
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_indices
            return max(1, len(valid_gpus))
    except Exception as e:
        print(f"GPU selection failed: {e}")
        return 1

NUM_AVAILABLE_GPUS = auto_select_best_gpus(num_gpus=4, min_memory_mb=30000)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
np.random.seed(42)

# =======================
# 2. 全局参数设置
# =======================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_H5 = str(REPO_ROOT / "data/processed/ogle_tdmn.h5")
SAVE_BASE_DIR = None

PER_REPLICA_BATCH_SIZE = 128
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * NUM_AVAILABLE_GPUS

EPOCHS_STAGE1 = 300
EPOCHS_STAGE2 = 150
SEED = 42
RESIZE_TARGET = (128, 128)

class_names = [
    "ACEP_F", "ACEP_1O", "CEP_F", "CEP_1O", "CEP_1O2O", "DSCT_SINGLEMODE",
    "DSCT_MULTIMODE", "ECL_C", "ECL_NC", "RRLYR_RRAB", "RRLYR_RRC", "RRLYR_RRD",
    "T2CEP_BLHER", "T2CEP_RVTAU", "T2CEP_WVIR",
]
NUM_CLASSES = len(class_names)

major_class_names = ["ACEP", "CEP", "DSCT", "ECL", "RRLYR", "T2CEP"]
sub_to_major_map = {
    0: 0, 1: 0,
    2: 1, 3: 1, 4: 1,
    5: 2, 6: 2,
    7: 3, 8: 3,
    9: 4, 10: 4, 11: 4,
    12: 5, 13: 5, 14: 5,
}

# =======================
# 3. 评估指标与数据预处理
# =======================
def calculate_per_class_metrics(cm, class_list):
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
    print(f"{'Class':<15} | {'Recall':<8} | {'Precision':<9} | {'Spec':<8} | {'G-Mean(PR)':<15} | {'G-Mean(SS)':<15}")
    print("-" * 95)
    for i, cls in enumerate(class_list):
        print(f"{cls:<15} | {recall[i]:.4f}   | {precision[i]:.4f}    | {specificity[i]:.4f}   | {gmean_pr[i]:.4f}          | {gmean_ss[i]:.4f}")
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

    absolute_phase, raw_mag, smooth_mag, err, valid_mask = tf.unstack(seq, axis=-1)
    raw_noise = tf.random.normal(tf.shape(raw_mag), mean=0.0, stddev=0.02) * valid_mask
    smooth_noise = tf.random.normal(tf.shape(smooth_mag), mean=0.0, stddev=0.005) * valid_mask
    seq_aug = tf.stack([absolute_phase, raw_mag + raw_noise, smooth_mag + smooth_noise, err, valid_mask], axis=-1)

    img_aug = tf.image.random_brightness(img, max_delta=0.05)
    img_aug = tf.image.random_contrast(img_aug, lower=0.9, upper=1.1)

    feat_noise = tf.random.normal(tf.shape(inputs["feat_in"]), mean=0.0, stddev=0.01)
    feat_aug = inputs["feat_in"] + feat_noise

    return {"seq_in": seq_aug, "img_in": img_aug, "feat_in": feat_aug, "is_gen": is_gen}, label

def preprocess_bi_modal(seq, img, feat, mask, label, is_gen):
    img = tf.image.resize(img, RESIZE_TARGET, method="bilinear")
    phase, raw_mag, smooth_mag, err = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
    is_valid = mask

    # 🛑 【底层防线】掩码强制苏醒：只要是空序列，强制令第0点有效，防 Softmax 0/0 NaN
    is_empty = 1.0 - tf.reduce_max(is_valid)
    is_valid = is_valid + tf.one_hot(0, tf.shape(is_valid)[0], dtype=is_valid.dtype) * is_empty

    phase = phase * is_valid
    raw_mag *= is_valid
    smooth_mag *= is_valid
    err *= is_valid

    seq_processed = tf.stack([phase, raw_mag, smooth_mag, err, is_valid], axis=-1)
    return {"seq_in": seq_processed, "img_in": img, "feat_in": feat, "is_gen": is_gen}, label

def create_dataset_with_indices(
    indices, X_seq, X_img, X_feat, X_mask, y_oh, is_gen, feat_dim, is_train=False, custom_batch_size=None
):
    seq_slice = X_seq[indices]
    img_slice = X_img[indices]
    feat_slice = X_feat[indices][:, :feat_dim]
    mask_slice = X_mask[indices]
    y_slice = y_oh[indices]
    gen_slice = is_gen[indices]

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((seq_slice, img_slice, feat_slice, mask_slice, y_slice, gen_slice))

    def map_fn(s, i, f, m, y, ig):
        return preprocess_bi_modal(s, i, f, m, y, ig)

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.map(augment_bi_modal, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=8192, seed=42, reshuffle_each_iteration=True)

    bs = custom_batch_size if custom_batch_size is not None else GLOBAL_BATCH_SIZE
    ds = ds.batch(bs, drop_remainder=is_train)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    return ds.prefetch(tf.data.AUTOTUNE)

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
    pos_in = seq_in[:, :, 0:1]
    mag_in = seq_in[:, :, 1:4]
    valid_mask = seq_in[:, :, 4]

    x_seq = layers.Conv1D(32, 5, padding="same", use_bias=False, kernel_regularizer=reg)(mag_in)
    x_seq = layers.BatchNormalization()(x_seq)
    x_seq = layers.Activation("swish")(x_seq)
    conv_3 = layers.Conv1D(32, 3, padding="same", activation="swish")(x_seq)
    conv_7 = layers.Conv1D(32, 7, padding="same", activation="swish")(x_seq)
    x_seq = layers.Concatenate()([conv_3, conv_7])
    x_seq = layers.SpatialDropout1D(0.2)(x_seq)

    x_seq = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(x_seq)
    pos_in_down = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(pos_in)

    valid_mask_exp = tf.expand_dims(valid_mask, -1)
    valid_mask_down_exp = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(valid_mask_exp)
    valid_mask_down = tf.squeeze(valid_mask_down_exp, axis=-1)

    attn_mask = tf.cast(valid_mask_down, tf.bool)[:, tf.newaxis, :]

    # Compress the absolute-time scale with log1p to improve attention stability.
    pos_in_log = tf.math.log1p(tf.maximum(pos_in_down, 0.0))
    pos_emb = layers.Dense(64, activation="swish", name="phase_dense")(pos_in_log)
    x_seq = layers.Add()([x_seq, pos_emb])

    # Normalize feature scales before multi-head attention.
    x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq)

    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.2)(
            x_seq, x_seq, attention_mask=attn_mask
        )
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + attn)
        ffn = layers.Dense(128, activation="swish")(x_seq)
        ffn = layers.Dropout(0.2)(ffn)
        ffn = layers.Dense(64)(ffn)
        x_seq = layers.LayerNormalization(epsilon=1e-6)(x_seq + ffn)

    # Float32 降维池化防 nan
    x_seq_f32 = tf.cast(x_seq, tf.float32)
    valid_mask_f32 = tf.cast(valid_mask_down_exp, tf.float32)

    seq_sum = tf.reduce_sum(x_seq_f32 * valid_mask_f32, axis=1)
    seq_lens = tf.maximum(tf.reduce_sum(valid_mask_f32, axis=1), 1.0)
    seq_mean = tf.cast(seq_sum / seq_lens, x_seq.dtype)

    mask_for_max = (1.0 - valid_mask_f32) * -1e4
    seq_max = tf.cast(tf.reduce_max(x_seq_f32 + mask_for_max, axis=1), x_seq.dtype)

    seq_pooled = layers.Concatenate()([seq_mean, seq_max])
    seq_repr = layers.Dense(128, activation="swish", kernel_regularizer=reg, name="seq_repr")(seq_pooled)

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
    img_repr = layers.Dense(128, activation="swish", kernel_regularizer=reg, name="img_repr")(img_repr)

    return keras.Model(
        inputs={"seq_in": seq_in, "img_in": img_in}, outputs=[seq_repr, img_repr], name="StrictDecoupledEncoder"
    )

class CosineClassifier(layers.Layer):
    def __init__(self, num_classes, scale=20.0, **kwargs):
        super().__init__(dtype=tf.float32, **kwargs)
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.num_classes), initializer="glorot_uniform", trainable=True, name="cosine_weights")

    def call(self, inputs):
        inputs_fp32 = tf.cast(inputs, tf.float32)
        w_fp32 = tf.cast(self.w, tf.float32)
        x_norm = tf.math.l2_normalize(inputs_fp32, axis=1)
        w_norm = tf.math.l2_normalize(w_fp32, axis=0)
        return self.scale * tf.matmul(x_norm, w_norm)

# =======================================================
# 5. 主模型类：三模态引航 & 专家路由门控 (MoE)
# =======================================================
class HeterogeneousEnsembleModel(keras.Model):
    def __init__(self, encoder, num_classes, class_weights, class_priors, feat_dim, projection_dim_modal=128, projection_dim_feat=32, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.class_priors = class_priors
        self.feat_dim = feat_dim
        self.current_stage = 1
        self.temperature = tf.Variable(0.2, trainable=False, dtype=tf.float32, name="temperature")

        self.proj_seq = keras.Sequential([layers.Dense(256, use_bias=False, input_shape=(128,)), layers.BatchNormalization(), layers.Activation("relu"), layers.Dense(projection_dim_modal, use_bias=False)])
        self.proj_img = keras.Sequential([layers.Dense(256, use_bias=False, input_shape=(128,)), layers.BatchNormalization(), layers.Activation("relu"), layers.Dense(projection_dim_modal, use_bias=False)])
        self.feat_encoder = keras.Sequential([layers.Dense(64, kernel_regularizer=regularizers.l2(1.5e-4), input_shape=(self.feat_dim,)), layers.BatchNormalization(), layers.Activation("swish")])
        self.proj_feat = keras.Sequential([layers.Dense(32, use_bias=False, input_shape=(64,)), layers.BatchNormalization(), layers.Activation("relu"), layers.Dense(projection_dim_feat, use_bias=False)])

        self.fusion_dropout_1d = layers.SpatialDropout1D(0.15)
        self.fusion_dense = keras.Sequential([layers.Dense(128, kernel_regularizer=regularizers.l2(1.5e-4), input_shape=(256,)), layers.BatchNormalization(), layers.Activation("swish"), layers.Dropout(0.5)])

        self.router = keras.Sequential([layers.Dense(64, activation="swish", input_shape=(192,)), layers.Dense(3, activation="softmax", name="expert_router", dtype=tf.float32)])
        self.classifier_cb = layers.Dense(num_classes, activation="linear", name="head_cb", dtype=tf.float32)
        self.classifier_la = layers.Dense(num_classes, activation="linear", name="head_la", dtype=tf.float32)
        self.classifier_cos = CosineClassifier(num_classes, name="head_cos")

        self.prototypes_modal = self.add_weight(shape=(num_classes, projection_dim_modal), initializer="zeros", trainable=False, name="class_prototypes_modal")
        self.prototypes_feat = self.add_weight(shape=(num_classes, projection_dim_feat), initializer="zeros", trainable=False, name="class_prototypes_feat")

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
        return [self.loss_tracker, self.val_loss_tracker, self.loss_seq_tracker, self.loss_img_tracker, self.loss_feat_tracker]

    def _fuse_features(self, seq_repr, img_repr, training=False):
        stacked = tf.stack([seq_repr, img_repr], axis=1)
        stacked = self.fusion_dropout_1d(stacked, training=training)
        flat = layers.Reshape((256,))(stacked)
        return self.fusion_dense(flat, training=training)

    def calculate_proxy_loss(self, projections, prototypes, y_onehot):
        z = tf.math.l2_normalize(tf.cast(projections, tf.float32), axis=1)
        p = tf.math.l2_normalize(tf.cast(prototypes, tf.float32), axis=1)
        logits_con = tf.matmul(z, p, transpose_b=True) / self.temperature
        labels_idx = tf.argmax(y_onehot, axis=1)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels_idx, logits_con, from_logits=True)
        )

    def call(self, inputs, training=False):
        seq_repr, img_repr = self.encoder({"seq_in": inputs["seq_in"], "img_in": inputs["img_in"]}, training=training)
        feat_repr = self.feat_encoder(inputs["feat_in"], training=training)

        if self.current_stage == 1:
            z_seq, z_img, z_feat = self.proj_seq(seq_repr, training=training), self.proj_img(img_repr, training=training), self.proj_feat(feat_repr, training=training)
            z_modal_norm = tf.math.l2_normalize((tf.cast(z_seq, tf.float32) + tf.cast(z_img, tf.float32)) / 2.0, axis=1)
            p_modal_norm = tf.math.l2_normalize(tf.cast(self.prototypes_modal, tf.float32), axis=1)
            logits_modal = tf.matmul(z_modal_norm, p_modal_norm, transpose_b=True) / self.temperature

            z_feat_norm = tf.math.l2_normalize(tf.cast(z_feat, tf.float32), axis=1)
            p_feat_norm = tf.math.l2_normalize(tf.cast(self.prototypes_feat, tf.float32), axis=1)
            logits_feat = tf.matmul(z_feat_norm, p_feat_norm, transpose_b=True) / self.temperature
            return (tf.nn.softmax(logits_modal) + tf.nn.softmax(logits_feat)) / 2.0
        else:
            fusion_features = self._fuse_features(seq_repr, img_repr, training=training)
            final_features = layers.Concatenate()([fusion_features, feat_repr])
            logits_cb = self.classifier_cb(final_features)
            logits_la = self.classifier_la(final_features) + 0.3 * tf.math.log(tf.cast(self.class_priors, tf.float32) + 1e-9)
            logits_cos = self.classifier_cos(final_features)

            prob_cb, prob_la, prob_cos = tf.nn.softmax(logits_cb), tf.nn.softmax(logits_la), tf.nn.softmax(logits_cos)
            probs = tf.stack([prob_cb, prob_la, prob_cos], axis=-1)
            routing_weights = tf.expand_dims(self.router(final_features), axis=1)
            return tf.reduce_sum(probs * routing_weights, axis=-1)

    def calculate_losses(self, x, y_onehot, training=False):
        encoder_training_mode = training if (self.current_stage == 1 or self.encoder.trainable) else False
        seq_repr, img_repr = self.encoder({"seq_in": x["seq_in"], "img_in": x["img_in"]}, training=encoder_training_mode)
        feat_repr = self.feat_encoder(x["feat_in"], training=training)

        if self.current_stage == 1:
            z_seq, z_img, z_feat = self.proj_seq(seq_repr, training=training), self.proj_img(img_repr, training=training), self.proj_feat(feat_repr, training=training)
            loss_seq = self.calculate_proxy_loss(z_seq, self.prototypes_modal, y_onehot)
            loss_img = self.calculate_proxy_loss(z_img, self.prototypes_modal, y_onehot)
            loss_feat = self.calculate_proxy_loss(z_feat, self.prototypes_feat, y_onehot)
            z_avg_modal = (z_seq + z_img) / 2.0

            if training:
                loss_seq_f32 = tf.cast(tf.stop_gradient(loss_seq), tf.float32)
                loss_img_f32 = tf.cast(tf.stop_gradient(loss_img), tf.float32)
                loss_feat_f32 = tf.cast(tf.stop_gradient(loss_feat), tf.float32)

                sum_loss = loss_seq_f32 + loss_img_f32 + loss_feat_f32 + 1e-9
                target_loss = sum_loss / 3.0 + 1e-9

                weight_seq = tf.pow(loss_seq_f32 / target_loss, 0.5)
                weight_img = tf.pow(loss_img_f32 / target_loss, 0.5)
                weight_feat = tf.pow(loss_feat_f32 / target_loss, 0.5)

                norm_factor = 3.0 / (weight_seq + weight_img + weight_feat)

                con_loss = (
                    tf.cast(weight_seq * norm_factor, loss_seq.dtype) * loss_seq +
                    tf.cast(weight_img * norm_factor, loss_img.dtype) * loss_img +
                    tf.cast(weight_feat * norm_factor, loss_feat.dtype) * loss_feat
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

            y_true_smooth = tf.cast(y_onehot, tf.float32) * (1.0 - self.label_smoothing) + (self.label_smoothing / self.num_classes)
            loss_cb = tf.reduce_mean(tf.reduce_sum(tf.cast(self.class_weights, tf.float32) * tf.cast(y_onehot, tf.float32), axis=1) * tf.keras.losses.categorical_crossentropy(y_true_smooth, logits_cb, from_logits=True))
            adjusted_logits_la = logits_la + 0.3 * tf.math.log(tf.cast(self.class_priors, tf.float32) + 1e-9)
            loss_la = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_smooth, adjusted_logits_la, from_logits=True))
            probs_cos = tf.nn.softmax(logits_cos)
            focal_weight = tf.pow(1.0 - tf.reduce_sum(probs_cos * tf.cast(y_onehot, tf.float32), axis=-1), 2.0)
            loss_cos = tf.reduce_mean(focal_weight * tf.keras.losses.categorical_crossentropy(y_true_smooth, logits_cos, from_logits=True))

            probs = tf.stack([tf.nn.softmax(logits_cb), tf.nn.softmax(adjusted_logits_la), probs_cos], axis=-1)
            routing_weights_exp = tf.expand_dims(self.router(final_features_stopped), axis=1)
            final_ensemble_prob = tf.reduce_sum(probs * routing_weights_exp, axis=-1)
            loss_router = tf.reduce_mean(tf.reduce_sum(tf.cast(self.class_weights, tf.float32) * tf.cast(y_onehot, tf.float32), axis=1) * tf.keras.losses.categorical_crossentropy(y_true_smooth, final_ensemble_prob, from_logits=False))

            z_seq, z_img, z_feat = self.proj_seq(seq_repr, training=training), self.proj_img(img_repr, training=training), self.proj_feat(feat_repr, training=training)
            anchor_loss = (self.calculate_proxy_loss(z_seq, self.prototypes_modal, y_onehot) + self.calculate_proxy_loss(z_img, self.prototypes_modal, y_onehot) + self.calculate_proxy_loss(z_feat, self.prototypes_feat, y_onehot)) / 3.0
            return loss_cb + loss_la + loss_cos + loss_router + 0.1 * anchor_loss, None, None, 0.0, 0.0, 0.0

    def train_step(self, data):
        x, y_onehot = data
        with tf.GradientTape() as tape:
            total_loss, z_avg_modal, z_feat, l_seq, l_img, l_feat = self.calculate_losses(x, y_onehot, training=True)
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.current_stage == 1 and z_avg_modal is not None:
            ctx = tf.distribute.get_replica_context()
            real_mask = tf.expand_dims(1.0 - x["is_gen"], -1)
            y_onehot_real = tf.cast(y_onehot, tf.float32) * tf.cast(real_mask, tf.float32)

            local_counts = tf.transpose(tf.reduce_sum(y_onehot_real, axis=0, keepdims=True))
            local_sum_modal = tf.matmul(y_onehot_real, tf.math.l2_normalize(tf.cast(z_avg_modal, tf.float32), axis=1), transpose_a=True)
            local_sum_feat = tf.matmul(y_onehot_real, tf.math.l2_normalize(tf.cast(z_feat, tf.float32), axis=1), transpose_a=True)

            global_counts = ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_counts)
            global_sum_modal = ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum_modal)
            global_sum_feat = ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum_feat)

            valid_mask = global_counts > 0

            centers_modal = tf.math.l2_normalize(tf.math.divide_no_nan(global_sum_modal, global_counts), axis=1)
            self.prototypes_modal.assign(
                tf.where(valid_mask, tf.math.l2_normalize(0.995 * tf.cast(self.prototypes_modal, tf.float32) + 0.005 * centers_modal, axis=1), tf.cast(self.prototypes_modal, tf.float32))
            )

            centers_feat = tf.math.l2_normalize(tf.math.divide_no_nan(global_sum_feat, global_counts), axis=1)
            self.prototypes_feat.assign(
                tf.where(valid_mask, tf.math.l2_normalize(0.995 * tf.cast(self.prototypes_feat, tf.float32) + 0.005 * centers_feat, axis=1), tf.cast(self.prototypes_feat, tf.float32))
            )

            self.loss_seq_tracker.update_state(l_seq)
            self.loss_img_tracker.update_state(l_img)
            self.loss_feat_tracker.update_state(l_feat)

        self.loss_tracker.update_state(total_loss)
        results = {"loss": self.loss_tracker.result()}
        if self.current_stage == 1:
            results.update({"con_loss_seq": self.loss_seq_tracker.result(), "con_loss_img": self.loss_img_tracker.result(), "con_loss_feat": self.loss_feat_tracker.result()})
        return results

    def test_step(self, data):
        x, y_onehot = data
        total_loss, _, _, l_seq, l_img, l_feat = self.calculate_losses(x, y_onehot, training=False)
        self.val_loss_tracker.update_state(total_loss)
        results = {"loss": self.val_loss_tracker.result()}
        if self.current_stage == 1:
            self.loss_seq_tracker.update_state(l_seq)
            self.loss_img_tracker.update_state(l_img)
            self.loss_feat_tracker.update_state(l_feat)
            results.update({"con_loss_seq": self.loss_seq_tracker.result(), "con_loss_img": self.loss_img_tracker.result(), "con_loss_feat": self.loss_feat_tracker.result()})
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
        def warmup_fn(): return self.initial_learning_rate * step / self.warmup_steps
        def decay_fn():
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.initial_learning_rate * (self.alpha + (1.0 - self.alpha) * 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress)))
        return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)

class TemperatureAnnealing(keras.callbacks.Callback):
    def __init__(self, initial_temp=0.2, final_temp=0.08, hold_epochs=20, decay_epochs=120):
        super().__init__()
        self.initial_temp, self.final_temp, self.hold_epochs, self.decay_epochs = initial_temp, final_temp, hold_epochs, decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.hold_epochs: new_temp = self.initial_temp
        elif epoch < self.hold_epochs + self.decay_epochs:
            progress = (epoch - self.hold_epochs) / self.decay_epochs
            new_temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (1 + math.cos(math.pi * progress))
        else: new_temp = self.final_temp
        self.model.temperature.assign(new_temp)
        if epoch % 5 == 0: print(f"\n🌡️ [Temp Annealing] Epoch {epoch} - T: {new_temp:.4f}")

class Stage1LossEvaluation(keras.callbacks.Callback):
    def __init__(self, model_save_path, patience=25):
        super().__init__()
        self.model_save_path, self.patience, self.best_loss, self.wait = model_save_path, patience, float("inf"), 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss", float("inf"))
        print(f" — val_loss: {current_loss:.4f} | Proxy(Seq): {logs.get('val_con_loss_seq', 0.0):.4f} | Proxy(Img): {logs.get('val_con_loss_img', 0.0):.4f} | Proxy(Feat): {logs.get('val_con_loss_feat', 0.0):.4f}")
        if current_loss < self.best_loss:
            print(f"✅ Loss improved from {self.best_loss:.4f} to {current_loss:.4f}. Saving weights.")
            self.best_loss, self.wait = current_loss, 0
            self.model.save_weights(self.model_save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\n⛔ Early stopping triggered after {self.patience} epochs.")

class F1Evaluation(keras.callbacks.Callback):
    def __init__(self, val_ds, y_val_true, model_save_path, patience=25):
        super().__init__()
        self.val_ds, self.y_val_true, self.model_save_path, self.patience, self.best_f1, self.wait = val_ds, y_val_true, model_save_path, patience, 0.0, 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0)
        current_f1 = f1_score(self.y_val_true, np.argmax(preds, axis=1), average="macro")
        if logs is not None: logs["val_macro_f1"] = current_f1
        print(f" — val_loss: {logs.get('loss', 0.0):.4f} — val_macro_f1: {current_f1:.4f}")
        if current_f1 > self.best_f1:
            print(f"✅ Metric improved from {self.best_f1:.4f} to {current_f1:.4f}. Saving weights.")
            self.best_f1, self.wait = current_f1, 0
            self.model.save_weights(self.model_save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\n⛔ Early stopping triggered after {self.patience} epochs.")

class TopKAveragingCallback(keras.callbacks.Callback):
    def __init__(self, save_dir, k=3):
        super().__init__()
        self.save_dir, self.k, self.top_k_records = save_dir, k, []

    def on_epoch_end(self, epoch, logs=None):
        current_f1 = (logs or {}).get("val_macro_f1")
        if current_f1 is None: return
        filepath = os.path.join(self.save_dir, f"topk_model_epoch_{epoch}.ckpt")

        if len(self.top_k_records) < self.k:
            self.model.save_weights(filepath)
            self.top_k_records.append((current_f1, filepath))
            self.top_k_records.sort(key=lambda x: x[0], reverse=True)
        else:
            if current_f1 > self.top_k_records[-1][0]:
                for f in glob.glob(self.top_k_records[-1][1] + "*"):
                    try: os.remove(f)
                    except Exception:
                        pass
                self.model.save_weights(filepath)
                self.top_k_records[-1] = (current_f1, filepath)
                self.top_k_records.sort(key=lambda x: x[0], reverse=True)

    def get_averaged_weights(self, target_model):
        records = self.top_k_records or [(0.0, f.replace(".index", "")) for f in glob.glob(os.path.join(self.save_dir, "topk_model_epoch_*.ckpt.index"))]
        if not records: return None
        averaged_weights = None
        for _, filepath in records:
            target_model.load_weights(filepath).expect_partial()
            current_weights = target_model.get_weights()
            averaged_weights = current_weights if averaged_weights is None else [avg_w + curr_w for avg_w, curr_w in zip(averaged_weights, current_weights)]
        return [w / len(records) for w in averaged_weights]

# =======================
# 7. 主执行函数
# =======================
def run_experiment():
    K.clear_session()
    gc.collect()

    print(f"\n🚀 Loading OGLE Data (Including is_gen & Target Feat)...")
    with h5py.File(DATA_PATH_H5, "r") as hf:
        X_seq_tr, X_seq_val, X_seq_te = [np.nan_to_num(hf[f"X_seq_interp_{k}"][:].astype(np.float32), nan=-10.0) for k in ["train", "val", "test"]]
        X_img_tr, X_img_val, X_img_te = [hf[f"X_img_interp_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]]
        X_feat_tr, X_feat_val, X_feat_te = [hf[f"X_feat_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]]
        X_mask_tr, X_mask_val, X_mask_te = [hf[f"X_mask_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]]
        y_tr, y_val, y_te = [hf[f"y_{k}"][:].astype(np.int32).squeeze() for k in ["train", "val", "test"]]
        y_tr_oh, y_val_oh, y_te_oh = [hf[f"y_{k}_onehot"][:] for k in ["train", "val", "test"]]
        is_gen_tr, is_gen_val, is_gen_te = [hf[f"is_gen_{k}"][:].astype(np.float32) for k in ["train", "val", "test"]]
        object_id_test = hf["object_id_test"][:]
        object_id_test = np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in object_id_test])

    FEAT_DIM = X_feat_tr.shape[1]
    print(f"✅ Dynamic Feature Dimension Detected: {FEAT_DIM}")

    strategy = tf.distribute.MirroredStrategy()
    print(f"\n🌐 Distributed Training Initialized. Number of devices: {strategy.num_replicas_in_sync}")
    print(f"📦 Global Batch Size set to: {GLOBAL_BATCH_SIZE}")

    # =========================================================
    # 🛑 【核心护城河】：温和的动态过采样
    # =========================================================
    original_train_indices = np.arange(len(y_tr))
    oversampled_indices = []

    print("\n⚖️  Applying Bounded Minority Oversampling for Stage 1 Balance...")
    for c in range(NUM_CLASSES):
        c_indices = original_train_indices[y_tr == c]
        c_count = len(c_indices)
        if c_count == 0: continue

        MIN_SAMPLES_TARGET = min(800, c_count * 5)

        if c_count < MIN_SAMPLES_TARGET:
            shortfall = MIN_SAMPLES_TARGET - c_count
            extra_indices = np.random.choice(c_indices, size=shortfall, replace=True)
            oversampled_indices.extend(c_indices.tolist())
            oversampled_indices.extend(extra_indices.tolist())
            print(f"   📈 Class {c} ({class_names[c]:<15}): {c_count:>5} -> {MIN_SAMPLES_TARGET}")
        else:
            oversampled_indices.extend(c_indices.tolist())

    train_indices = np.array(oversampled_indices)
    np.random.shuffle(train_indices)
    print(f"📦 Total Training Samples after Oversampling: {len(train_indices)}")

    class_weights = get_class_balanced_weights(y_tr)
    counts_arr = np.zeros(NUM_CLASSES, dtype=np.float64)
    unique_classes, counts = np.unique(y_tr, return_counts=True)
    counts_arr[unique_classes] = counts
    class_priors = tf.constant(counts_arr / np.sum(counts_arr), dtype=tf.float32)

    train_ds = create_dataset_with_indices(train_indices, X_seq_tr, X_img_tr, X_feat_tr, X_mask_tr, y_tr_oh, is_gen_tr, FEAT_DIM, is_train=True)
    val_ds = create_dataset_with_indices(np.arange(len(y_val)), X_seq_val, X_img_val, X_feat_val, X_mask_val, y_val_oh, is_gen_val, FEAT_DIM, is_train=False)
    test_ds = create_dataset_with_indices(np.arange(len(y_te)), X_seq_te, X_img_te, X_feat_te, X_mask_te, y_te_oh, is_gen_te, FEAT_DIM, is_train=False)

    seq_shape, img_shape = ((X_seq_tr.shape[1], X_seq_tr.shape[2]), (RESIZE_TARGET[0], RESIZE_TARGET[1], 3))

    with strategy.scope():
        model = HeterogeneousEnsembleModel(build_pure_di_modal_encoder(seq_shape, img_shape), NUM_CLASSES, class_weights, class_priors, feat_dim=FEAT_DIM)

        for dummy_x, dummy_y in train_ds.take(1):
            small_dummy_x = {k: v[:2] for k, v in dummy_x.items()}
            model(small_dummy_x)

        print("\n🔥 [Optimization] Performing Prototype Warm-start (Real Data ONLY!)...")
        model.current_stage = 1
        sum_modal, sum_feat, class_counts = tf.zeros((NUM_CLASSES, 128)), tf.zeros((NUM_CLASSES, 32)), tf.zeros((NUM_CLASSES, 1))

        clean_warmup_ds = create_dataset_with_indices([i for i in original_train_indices if is_gen_tr[i] == 0.0], X_seq_tr, X_img_tr, X_feat_tr, X_mask_tr, y_tr_oh, is_gen_tr, FEAT_DIM, is_train=False, custom_batch_size=PER_REPLICA_BATCH_SIZE)
        for x_batch, y_batch in clean_warmup_ds:
            seq_repr, img_repr = model.encoder({"seq_in": x_batch["seq_in"], "img_in": x_batch["img_in"]}, training=False)
            feat_repr = model.feat_encoder(x_batch["feat_in"], training=False)
            z_seq, z_img, z_feat = model.proj_seq(seq_repr, training=False), model.proj_img(img_repr, training=False), model.proj_feat(feat_repr, training=False)

            sum_modal += tf.matmul(tf.cast(y_batch, tf.float32), tf.math.l2_normalize((tf.cast(z_seq, tf.float32) + tf.cast(z_img, tf.float32)) / 2.0, axis=1), transpose_a=True)
            sum_feat += tf.matmul(tf.cast(y_batch, tf.float32), tf.math.l2_normalize(tf.cast(z_feat, tf.float32), axis=1), transpose_a=True)
            class_counts += tf.transpose(tf.reduce_sum(tf.cast(y_batch, tf.float32), axis=0, keepdims=True))

        valid_mask = class_counts > 0
        model.prototypes_modal.assign(tf.where(valid_mask, tf.math.l2_normalize(tf.math.divide_no_nan(sum_modal, class_counts), axis=1), tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 128)), axis=1)))
        model.prototypes_feat.assign(tf.where(valid_mask, tf.math.l2_normalize(tf.math.divide_no_nan(sum_feat, class_counts), axis=1), tf.math.l2_normalize(tf.random.normal((NUM_CLASSES, 32)), axis=1)))
        print("✅ Prototype Warm-start Completed.")

        # ==========================================
        # 🏁 STAGE 1 Compile
        # ==========================================
        print("\n" + "=" * 50)
        print("🏁 STAGE 1: Contrastive Pretraining (Seq + Img + Feat)")
        print("=" * 50)

        total_steps_s1 = EPOCHS_STAGE1 * (len(train_indices) // GLOBAL_BATCH_SIZE)
        base_lr_s1 = 4e-4 * (GLOBAL_BATCH_SIZE / 256.0)

        optimizer_s1 = keras.optimizers.AdamW(learning_rate=CosineDecayWithWarmup(base_lr_s1, int(total_steps_s1 * 0.1), total_steps_s1, 1e-6), weight_decay=1e-4, clipnorm=1.0)
        optimizer_s1 = mixed_precision.LossScaleOptimizer(optimizer_s1)
        model.compile(optimizer=optimizer_s1)

    stage1_ckpt_path = os.path.join(SAVE_BASE_DIR, "stage1_tri_modal_best.ckpt")

    if os.path.exists(stage1_ckpt_path + ".index") or os.path.exists(stage1_ckpt_path):
        print(f"\n⏭️ [Stage 1] Skipping... Loading existing weights.")
        model.load_weights(stage1_ckpt_path).expect_partial()
    else:
        model.fit(train_ds, epochs=EPOCHS_STAGE1, validation_data=val_ds, callbacks=[TemperatureAnnealing(), Stage1LossEvaluation(stage1_ckpt_path, patience=30)], verbose=1)

    # ==========================================
    # 🏁 STAGE 2 Compile
    # ==========================================
    print("\n" + "=" * 50)
    print("🏁 STAGE 2: Unified Fine-tuning + MoE Routing + Max-Confidence")
    print("=" * 50)

    with strategy.scope():
        if os.path.exists(stage1_ckpt_path + ".index") or os.path.exists(stage1_ckpt_path):
            model.load_weights(stage1_ckpt_path).expect_partial()

        model.current_stage = 2
        model.encoder.trainable = True
        for layer in model.encoder.layers:
            if isinstance(layer, (layers.Conv1D, layers.Conv2D, layers.SeparableConv2D, layers.MaxPooling2D)): layer.trainable = False

        total_steps_s2 = EPOCHS_STAGE2 * (len(train_indices) // GLOBAL_BATCH_SIZE)
        base_lr_s2 = 1e-4 * (GLOBAL_BATCH_SIZE / 256.0)

        optimizer_s2 = keras.optimizers.AdamW(learning_rate=CosineDecayWithWarmup(base_lr_s2, int(total_steps_s2 * 0.1), total_steps_s2, 1e-5), weight_decay=1e-4, clipnorm=1.0)
        optimizer_s2 = mixed_precision.LossScaleOptimizer(optimizer_s2)
        model.compile(optimizer=optimizer_s2)

    stage2_ckpt_path = os.path.join(SAVE_BASE_DIR, "stage2_merged_best.ckpt")
    topk_swa_cb = TopKAveragingCallback(save_dir=SAVE_BASE_DIR, k=3)

    model.fit(train_ds, epochs=EPOCHS_STAGE2, validation_data=val_ds, callbacks=[F1Evaluation(val_ds, y_val, stage2_ckpt_path, patience=30), topk_swa_cb], verbose=1)

    # ==========================================
    # 🏁 FINAL EVALUATION
    # ==========================================
    print("\n[FINAL EVALUATION: Physics-Informed MoE Ensemble]")

    if os.path.exists(stage2_ckpt_path + ".index") or os.path.exists(stage2_ckpt_path):
        model.load_weights(stage2_ckpt_path).expect_partial()

    # Freeze Standard-vs-TopK selection using validation Macro-F1 only.
    preds_best_val = model.predict(val_ds)
    y_pred_best_val = np.argmax(preds_best_val, axis=1)
    f1_best_val = f1_score(y_val, y_pred_best_val, average="macro")

    print("\n🔍 Evaluating Top-K SWA Model on Validation Dataset...")
    swa_model = HeterogeneousEnsembleModel(build_pure_di_modal_encoder(seq_shape, img_shape), NUM_CLASSES, class_weights, class_priors, feat_dim=FEAT_DIM)

    swa_model.current_stage = 1
    for dummy_x, dummy_y in train_ds.take(1):
        small_dummy_x = {k: v[:2] for k, v in dummy_x.items()}
        swa_model(small_dummy_x)
    swa_model.current_stage = 2
    for dummy_x, dummy_y in train_ds.take(1):
        small_dummy_x = {k: v[:2] for k, v in dummy_x.items()}
        swa_model(small_dummy_x)

    avg_weights = topk_swa_cb.get_averaged_weights(swa_model)
    if avg_weights is not None:
        swa_model.set_weights(avg_weights)
        preds_swa_val = swa_model.predict(val_ds)
        y_pred_swa_val = np.argmax(preds_swa_val, axis=1)
        f1_swa_val = f1_score(y_val, y_pred_swa_val, average="macro")
    else:
        f1_swa_val = float("-inf")

    if f1_swa_val > f1_best_val:
        selected_model, selected_name, selected_val_f1 = swa_model, "top3_average", f1_swa_val
    else:
        selected_model, selected_name, selected_val_f1 = model, "standard_best", f1_best_val
    selection = {
        "selection_metric": "validation_macro_f1",
        "standard_validation_macro_f1": float(f1_best_val),
        "top3_validation_macro_f1": None if not np.isfinite(f1_swa_val) else float(f1_swa_val),
        "selected_model": selected_name,
        "selected_validation_macro_f1": float(selected_val_f1),
    }
    with open(os.path.join(SAVE_BASE_DIR, "model_selection.json"), "w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    print(f"\n🏆 Selected {selected_name} on validation Macro-F1; evaluating test once.")
    final_probabilities = selected_model.predict(test_ds)
    final_preds = np.argmax(final_probabilities, axis=1)

    print("\n" + "*" * 60)
    print("🎯 CLASSIFIER RESULT: 精细小类 (Sub-Classes - 15 Classes)")
    print("*" * 60)
    print(classification_report(y_te, final_preds, target_names=class_names, digits=4))

    cm_sub = confusion_matrix(y_te, final_preds)
    calculate_per_class_metrics(cm_sub, class_names)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(confusion_matrix(y_te, final_preds, normalize="true"), display_labels=class_names).plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format=".2f")
    plt.title("MoE Sub-Class Confusion Matrix (15 Classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_BASE_DIR, "cm_final_sub_classes.png"), dpi=300)
    plt.close(fig)

    print("\n" + "*" * 60)
    print("🌍 CLASSIFIER RESULT: 宏观大类 (Major-Classes - 6 Classes)")
    print("*" * 60)
    y_te_major = np.array([sub_to_major_map[y] for y in y_te])
    final_preds_major = np.array([sub_to_major_map[y] for y in final_preds])
    print(classification_report(y_te_major, final_preds_major, target_names=major_class_names, digits=4))

    cm_major = confusion_matrix(y_te_major, final_preds_major)
    calculate_per_class_metrics(cm_major, major_class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_te_major, final_preds_major, normalize="true"), display_labels=major_class_names).plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format=".2f")
    plt.title("MoE Major-Class Confusion Matrix (6 Classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_BASE_DIR, "cm_final_major_classes.png"), dpi=300)
    plt.close(fig)

    per_class_recall = np.divide(
        np.diag(cm_sub), cm_sub.sum(axis=1), out=np.zeros(NUM_CLASSES, dtype=float), where=cm_sub.sum(axis=1) > 0
    )
    metrics = {
        **selection,
        "survey": "ogle",
        "test_macro_f1_15class": float(f1_score(y_te, final_preds, average="macro")),
        "test_balanced_accuracy_15class": float(balanced_accuracy_score(y_te, final_preds)),
        "test_gmean_recall_15class": float(np.exp(np.mean(np.log(np.clip(per_class_recall, 1e-12, 1.0))))),
        "test_macro_f1_6class": float(f1_score(y_te_major, final_preds_major, average="macro")),
        "test_balanced_accuracy_6class": float(balanced_accuracy_score(y_te_major, final_preds_major)),
        "seed": int(SEED),
    }
    with open(os.path.join(SAVE_BASE_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(SAVE_BASE_DIR, "test_predictions.npz"),
        object_id=object_id_test,
        y_true=y_te,
        y_pred=final_preds,
        probabilities=final_probabilities,
    )

    print("\nOGLE TDMN experiment completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the accepted-paper OGLE TDMN model")
    parser.add_argument("--data", default=DATA_PATH_H5)
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/ogle/tdmn_seed<seed>")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs-stage1", type=int, default=EPOCHS_STAGE1)
    parser.add_argument("--epochs-stage2", type=int, default=EPOCHS_STAGE2)
    return parser.parse_args()


def configure_from_args(args):
    global DATA_PATH_H5, SAVE_BASE_DIR, SEED, EPOCHS_STAGE1, EPOCHS_STAGE2
    DATA_PATH_H5 = str(Path(args.data).expanduser().resolve())
    output_dir = args.output_dir or REPO_ROOT / "results/ogle" / f"tdmn_seed{args.seed}"
    SAVE_BASE_DIR = str(Path(output_dir).expanduser().resolve())
    SEED = args.seed
    EPOCHS_STAGE1 = args.epochs_stage1
    EPOCHS_STAGE2 = args.epochs_stage2
    if not Path(DATA_PATH_H5).is_file():
        raise FileNotFoundError(DATA_PATH_H5)
    Path(SAVE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    (Path(SAVE_BASE_DIR) / "run_config.json").write_text(
        json.dumps({**vars(args), "data": DATA_PATH_H5, "output_dir": SAVE_BASE_DIR}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    configure_from_args(parse_args())
    run_experiment()
