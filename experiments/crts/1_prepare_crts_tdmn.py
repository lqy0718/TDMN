"""Prepare the CRTS HDF5 dataset and object-level provenance manifest.

The implementation matches the accepted-paper experiments: absolute phase is
retained, class-specific cleaning/augmentation rules are applied, and the
train/validation/test split is fixed before training-set augmentation.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from supersmoother import SuperSmoother
import glob
import random
from PIL import Image
import copy
from sklearn.preprocessing import StandardScaler
import george
from george import kernels
from scipy.optimize import minimize
import time
import sys
import gc
import h5py
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# ===================================================================
# 配置与初始化
# ===================================================================
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_root():
    """Find the baseline project root from either the launch or script location."""
    candidates = [Path.cwd().resolve(), EXPERIMENT_DIR, *EXPERIMENT_DIR.parents]
    for candidate in candidates:
        if (candidate / "data/SSS_Per_Tab.dat").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
save_dir = str(REPO_ROOT / "data/processed")
# "realdup" records that replacement copies retain the baseline's real-sample
# prototype-update semantics.  Keep this separate from the earlier HDF5/cache.
temp_cache_dir = str(REPO_ROOT / "data/cache/crts_uniform_median")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(temp_cache_dir, exist_ok=True)

SEED = 42
PAD_VALUE = -10.0
AUGMENTATION_STRATEGY = "uniform_median"
AUGMENTATION_METHOD = "combined"
SMOOTHER_MODE = "full"
SPLIT_RATIOS = (0.7, 0.2, 0.1)
MIN_OBSERVATIONS = 95
REUSE_CACHE = False
MAX_WORKERS = 8
UNIFORM_MEDIAN_ORIGINAL_COUNT = 0
UNIFORM_MEDIAN_TRAIN_COUNT = 0
DATASET_OUTPUT_PATH = str(REPO_ROOT / "data/processed/crts_tdmn.h5")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"已设置全局随机种子: {seed}")


set_global_seed(SEED)


class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


timestamp = time.strftime("%Y%m%d")


# ===================================================================
# 缓存读写辅助函数
# ===================================================================
def save_checkpoint(
    path,
    X_seq_list,
    X_seq_raw_list,
    Y_list,
    is_gen_list,
    split_tags,
    metadata_rows,
):
    print(f"正在保存临时断点数据到 {path} ...")
    with h5py.File(path, "w") as f:
        flat_seq = np.vstack(X_seq_list)
        seq_lengths = np.array([len(s) for s in X_seq_list], dtype=np.int32)
        f.create_dataset("flat_seq", data=flat_seq, compression="gzip", dtype="float32")
        f.create_dataset("seq_lengths", data=seq_lengths, dtype="int32")

        if X_seq_raw_list is not None and len(X_seq_raw_list) > 0:
            flat_seq_raw = np.vstack(X_seq_raw_list)
            f.create_dataset(
                "flat_seq_raw", data=flat_seq_raw, compression="gzip", dtype="float32"
            )

        f.create_dataset(
            "Y", data=np.array(Y_list), compression="gzip", dtype="float32"
        )
        f.create_dataset("is_generated", data=np.array(is_gen_list), dtype="bool")

        tag_map = {"train": 0, "val": 1, "test": 2}
        tags_int = np.array([tag_map.get(t, 0) for t in split_tags], dtype="int8")
        f.create_dataset("split_tags", data=tags_int, dtype="int8")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key in ("object_id", "parent_id", "origin", "source_path"):
            f.create_dataset(
                key,
                data=np.array([str(row[key]) for row in metadata_rows], dtype=object),
                dtype=string_dtype,
            )
        for key, dtype in (
            ("prototype_eligible", "bool"),
            ("nobs_raw", "int32"),
            ("nobs_clean", "int32"),
            ("time_span", "float64"),
        ):
            f.create_dataset(key, data=np.array([row[key] for row in metadata_rows]), dtype=dtype)
    print("断点保存完成。")


def load_checkpoint(path):
    print(f"发现临时缓存 {path}，正在加载...")
    with h5py.File(path, "r") as f:
        flat_seq = f["flat_seq"][:]
        seq_lengths = f["seq_lengths"][:]
        Y_arr = f["Y"][:]
        is_gen_arr = f["is_generated"][:]
        tags_int = f["split_tags"][:]
        flat_seq_raw = f["flat_seq_raw"][:] if "flat_seq_raw" in f else None
        required_metadata = (
            "object_id",
            "parent_id",
            "origin",
            "source_path",
            "prototype_eligible",
            "nobs_raw",
            "nobs_clean",
            "time_span",
        )
        if not all(key in f for key in required_metadata):
            raise RuntimeError(
                "Legacy checkpoint has no provenance. Delete it or run without --reuse-cache."
            )
        metadata_rows = []
        for i in range(len(Y_arr)):
            def decode(key):
                value = f[key][i]
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)

            metadata_rows.append(
                {
                    "object_id": decode("object_id"),
                    "parent_id": decode("parent_id"),
                    "origin": decode("origin"),
                    "source_path": decode("source_path"),
                    "prototype_eligible": bool(f["prototype_eligible"][i]),
                    "nobs_raw": int(f["nobs_raw"][i]),
                    "nobs_clean": int(f["nobs_clean"][i]),
                    "time_span": float(f["time_span"][i]),
                }
            )

    X_seq_list, X_seq_raw_list = [], []
    current_idx, current_idx_raw = 0, 0
    for length in seq_lengths:
        X_seq_list.append(flat_seq[current_idx : current_idx + length])
        current_idx += length
        if flat_seq_raw is not None:
            X_seq_raw_list.append(
                flat_seq_raw[current_idx_raw : current_idx_raw + length]
            )
            current_idx_raw += length
        else:
            X_seq_raw_list.append(flat_seq[current_idx - length : current_idx].copy())

    int_map = {0: "train", 1: "val", 2: "test"}
    split_tags = [int_map.get(t, "train") for t in tags_int]
    return (
        X_seq_list,
        X_seq_raw_list,
        list(Y_arr),
        list(is_gen_arr),
        split_tags,
        metadata_rows,
    )


# ===================================================================
# LightCurve 类
# ===================================================================
class LightCurve:
    def __init__(
        self,
        sorted=True,
        time_fmt="mjd",
        measurement="mag",
        error="err",
        file_path=None,
        meta_data=None,
    ):
        self.data = None
        self.new_df = None
        self.folded = False
        self.sorted = sorted
        self.time_span = None
        self.phase_span = None
        self.amplitude = None
        self.smoothed = False
        self.time_fmt = time_fmt
        self.measurement = measurement
        self.error = error
        self.GP_model = None
        self.generated = False
        self.origin = "real"
        self.parent_id = None
        self.prototype_eligible = True
        self.source_path = None
        self.nobs_raw = 0
        self.nobs_clean = 0
        self.smoother_mode = SMOOTHER_MODE
        self.id = None
        self.ra = None
        self.dec = None
        self.type = None
        self.period = None
        self.split_tag = None
        self.v_css = None
        self.v_amp = None
        if file_path is not None and meta_data is not None:
            self.load_data(file_path, meta_data)

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def add_column(self, column, name):
        if self.data is None:
            return
        index = self.data.index.copy()
        new_col = pd.Series(column, index=index, name=name)
        self.data = pd.concat([self.data, new_col], axis=1)

    def fold(self):
        if self.folded:
            return
        time = self.data[self.time_fmt].copy()

        # 【V17.5 核心修复】：撤销归一化，恢复绝对相位跨度 [0, period)
        phase = (time - time.iloc[0]) % self.period
        self.phase_span = self.period

        self.add_column(phase, name="phase")
        self.data = self.data.sort_values(by="phase", ignore_index=True)
        phase_vals = self.data["phase"].values
        for i in range(1, len(phase_vals)):
            if phase_vals[i] <= phase_vals[i - 1]:
                phase_vals[i] = phase_vals[i - 1] + 1e-8
        self.data["phase"] = phase_vals
        self.folded = True

    def supersmoother_fit(self):
        if self.smoothed:
            return
        x_label = "phase" if self.folded else self.time_fmt
        x, y, err = (
            np.around(self.data[x_label], 4),
            np.around(self.data[self.measurement], 4),
            np.around(self.data[self.error], 4),
        )
        temp_df = pd.DataFrame(
            {x_label: x, self.measurement: y, self.error: err}
        ).drop_duplicates(subset=x_label)

        if self.smoother_mode == "no_supersmoother":
            self.add_column(y, name=f"smoothed_{self.measurement}")
            self.amplitude = np.max(y) - np.min(y)
            self.smoothed = True
            return

        # 【同步回滚】：因为 x 恢复了真实跨度，SuperSmoother 的周期也传真实的 period
        model = SuperSmoother(period=self.period)

        try:
            model.fit(temp_df[x_label], temp_df[self.measurement], temp_df[self.error])
            smoothed_y = model.predict(x)
            self.add_column(smoothed_y, name=f"smoothed_{self.measurement}")
            self.amplitude = np.max(smoothed_y) - np.min(smoothed_y)
        except Exception:
            self.add_column(y, name=f"smoothed_{self.measurement}")
            self.amplitude = np.max(y) - np.min(y)
        self.smoothed = True

    def clean(self):
        if not self.smoothed:
            self.supersmoother_fit()

        # Ablations can disable rejection while retaining identical downstream shapes.
        if self.smoother_mode in ("no_rejection", "no_supersmoother"):
            self.nobs_clean = len(self.data)
            return

        # 【物理离散度保护机制】: RRd (3) 和 Blazkho (4) 直接跳过清洗逻辑！
        if self.type in [3, 4]:
            self.nobs_clean = len(self.data)
            return

        y_smoothed, y_original, err = (
            self.data[f"smoothed_{self.measurement}"],
            self.data[self.measurement],
            self.data[self.error],
        )
        mean_err = np.mean(err)
        is_bad = (abs(y_original - y_smoothed) >= 3 * err) | (err >= 2 * mean_err)
        self.data = self.data.drop(index=self.data[is_bad].index)
        self.nobs_clean = len(self.data)

    def load_data(self, file_path, meta_data):
        base = os.path.splitext(os.path.basename(file_path))[0]
        id_val = int(base.split("_")[0])
        meta_row = meta_data[meta_data["ID"] == id_val]
        if not meta_row.empty:
            self.id, self.ra, self.dec, self.period = (
                id_val,
                meta_row.iloc[0]["RA"],
                meta_row.iloc[0]["Dec"],
                meta_row.iloc[0]["Period"],
            )
            self.v_css, self.npts, self.v_amp = (
                meta_row.iloc[0]["V_CSS"],
                meta_row.iloc[0]["Npts"],
                meta_row.iloc[0]["V_amp"],
            )
            self.type = (
                11 if meta_row.iloc[0]["Type"] == 12 else meta_row.iloc[0]["Type"]
            )
        else:
            self.id, self.period, self.type = id_val, 1.0, 0
        df = pd.read_csv(file_path, sep=r"\s+", names=["mjd", "mag", "err"])
        self.data = df.sort_values(by="mjd").reset_index(drop=True)
        self.source_path = str(Path(file_path).resolve())
        self.parent_id = str(self.id)
        self.nobs_raw = len(self.data)
        self.nobs_clean = len(self.data)
        self.time_span = float(self.data["mjd"].max() - self.data["mjd"].min())


def load_meta_data(meta_data_path=PROJECT_ROOT / "data/SSS_Per_Tab.dat"):
    return pd.read_csv(
        meta_data_path,
        header=None,
        skiprows=3,
        sep=r"\s+",
        usecols=range(9),
        names=["SSS_ID", "ID", "RA", "Dec", "Period", "V_CSS", "Npts", "V_amp", "Type"],
    )


# ===================================================================
# GP 与 数值计算函数
# ===================================================================
def fit_GP_model_one_attempt(lc, kernel=None):
    if not lc.folded:
        raise RuntimeError("GP模型仅支持相位折叠数据。")
    if lc.GP_model is not None:
        return lc.GP_model
    x, y, err = (
        lc.data["phase"].values,
        lc.data[lc.measurement].values,
        lc.data[lc.error].values,
    )
    if kernel is None:
        kernel = np.var(y) * kernels.ExpSine2Kernel(gamma=1.0, log_period=0.0) + np.var(
            y
        ) * 0.1 * kernels.Matern32Kernel(metric=0.1, ndim=1)
    gp = george.GP(kernel)
    gp.compute(x, err)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    try:
        result = minimize(
            neg_ln_like,
            gp.get_parameter_vector(),
            jac=grad_neg_ln_like,
            method="L-BFGS-B",
        )
        gp.set_parameter_vector(result.x)
    except Exception:
        pass
    lc.GP_model = gp
    return gp


def fit_GP_model(lc, kernel=None):
    success, count, count_limit = False, 0, 20
    while not success:
        try:
            fit_GP_model_one_attempt(lc, kernel=kernel)
            success = True
        except Exception:
            if not lc.data.empty:
                lc.data = lc.data.drop(random.sample(list(lc.data.index), 1)[0])
            count += 1
            if count >= count_limit or lc.data.empty:
                raise RuntimeError("GP拟合失败。")


def generate_GP_simulation(lc, input_phase, phase_shift_ratio=0, scale_std=True):
    if lc.GP_model is None:
        fit_GP_model(lc)
    simu_lc = copy.deepcopy(lc)
    simu_lc.generated = True
    simu_lc.origin = "gp"
    simu_lc.prototype_eligible = False
    try:
        pred, pred_var = lc.GP_model.predict(
            lc.data[lc.measurement].values, input_phase, return_var=True
        )
        pred_std = np.sqrt(np.maximum(pred_var, 0.0))
    except Exception:
        return None

    simu_std = (
        pred_std * (np.mean(lc.data[lc.error]) / np.mean(pred_std))
        if scale_std and np.mean(pred_std) > 1e-9
        else pred_std
    )
    noised_pred = pred + np.array(
        [random.normalvariate(0, s) % (3 * s) for s in simu_std]
    )
    if not np.all(np.isfinite(noised_pred)):
        return None

    phase = (np.array(input_phase) + phase_shift_ratio * lc.phase_span) % lc.phase_span
    simu_lc.data = pd.DataFrame(
        np.column_stack((phase, noised_pred, simu_std, phase)),
        columns=["mjd", "mag", "err", "phase"],
    )
    simu_lc.smoothed = False
    simu_lc.supersmoother_fit()
    simu_lc.data = simu_lc.data.sort_values(by="phase", ignore_index=True)
    simu_lc.nobs_raw = len(simu_lc.data)
    simu_lc.nobs_clean = len(simu_lc.data)
    return simu_lc


def generate_Ralse_simulation(lc, phase_shift=True):
    simu_lc = copy.deepcopy(lc)
    simu_lc.generated = True
    simu_lc.origin = "rasle"
    simu_lc.prototype_eligible = False
    simu_lc.data[simu_lc.measurement] = np.random.normal(
        loc=simu_lc.data[simu_lc.measurement].values,
        scale=simu_lc.data[simu_lc.error].values,
    )
    if phase_shift:
        shift = random.random() * simu_lc.phase_span
        simu_lc.data["phase"] = (simu_lc.data["phase"] + shift) % simu_lc.phase_span
        simu_lc.data = simu_lc.data.sort_values(by="phase", ignore_index=True)
    smoothed_column = f"smoothed_{simu_lc.measurement}"
    if smoothed_column in simu_lc.data:
        simu_lc.data = simu_lc.data.drop(columns=[smoothed_column])
    simu_lc.smoothed = False
    simu_lc.supersmoother_fit()
    simu_lc.nobs_clean = len(simu_lc.data)
    return simu_lc


def array_to_image(
    sequence_data,
    orig_sequence_raw=None,
    pad_value=-10.0,
    figsize=(8, 8),
    dpi=64,
    include_smoothed=True,
):
    valid_mask = sequence_data[:, 0] > (pad_value + 0.1)
    if not np.any(valid_mask):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    data = sequence_data[valid_mask]
    data = data[np.argsort(data[:, 0])]

    x, y_smoothed, y_err, mask_vals = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    real_idx, fake_idx = mask_vals > 0.5, mask_vals <= 0.5

    raw_phase_to_mag = (
        {
            float(np.round(ph, 6)): float(mag)
            for ph, mag in orig_sequence_raw[
                orig_sequence_raw[:, 0] > (pad_value + 0.1)
            ][:, :2]
        }
        if orig_sequence_raw is not None
        else {}
    )

    r_x, r_y = [], []
    if include_smoothed and np.any(real_idx):
        for ph, m in zip(x[real_idx], mask_vals[real_idx]):
            key = float(round(ph, 6))
            if key in raw_phase_to_mag:
                r_x.append(ph)
                r_y.append(raw_phase_to_mag[key])
            else:
                r_x.append(ph)
                r_y.append(float(y_smoothed[np.where(x == ph)[0][0]]))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, facecolor="black")

    # matplotlib 会自动 scale x 轴，所以无论周期是 0.1 还是 50，图像永远是充满画布的
    if np.any(real_idx):
        ax.scatter(
            x[real_idx],
            y_smoothed[real_idx],
            c="#0000FF",
            s=30,
            marker="o",
            edgecolors="none",
            alpha=0.7,
        )
    if include_smoothed and np.any(fake_idx):
        ax.scatter(
            x[fake_idx],
            y_smoothed[fake_idx],
            c="#0000FF",
            s=30,
            marker="o",
            edgecolors="none",
            alpha=0.3,
        )

    ax.errorbar(
        x, y_smoothed, yerr=y_err, fmt="none", ecolor="#00FF00", elinewidth=1, alpha=0.4
    )

    if r_x:
        ax.scatter(
            np.array(r_x),
            np.array(r_y),
            c="#FF0000",
            s=30,
            marker="o",
            edgecolors="none",
            alpha=0.9,
        )

    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    image = Image.frombytes("RGB", (w, h), buf.tobytes()).resize(
        (256, 256), Image.Resampling.LANCZOS
    )
    plt.close(fig)
    return np.array(image)


def worker_clean_and_image(args):
    seq_smoothed, seq_raw_original, type_id, period, max_len, pad_val, smoother_mode = args

    valid_mask = seq_smoothed[:, 0] > (pad_val + 0.1)
    if not np.any(valid_mask):
        return (
            np.full((max_len, 5), pad_val, dtype=np.float32),
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros(max_len, dtype=np.float32),
        )

    valid_data = seq_smoothed[valid_mask]
    valid_data = valid_data[np.argsort(valid_data[:, 0])]

    res_seq = np.full((max_len, 5), pad_val, dtype=np.float32)
    padding_mask = np.zeros(max_len, dtype=np.float32)

    v_len = min(len(valid_data), max_len)
    res_seq[:v_len] = valid_data[:v_len]
    padding_mask[:v_len] = 1.0

    seq_for_img = np.column_stack(
        (res_seq[:, 0], res_seq[:, 2], res_seq[:, 3], res_seq[:, 4])
    )
    img_out = array_to_image(
        seq_for_img,
        orig_sequence_raw=seq_raw_original,
        pad_value=pad_val,
        include_smoothed=smoother_mode != "no_smoothed_channel",
    )

    return res_seq, img_out, padding_mask


def preprocess_errors_and_mask(X, pad_value=-10.0):
    X = X.copy()
    X_masked = np.where(X == pad_value, np.nan, X)
    mag_mean_raw = np.nanmean(X_masked[:, :, 1], axis=1, keepdims=True)
    mag_mean_smooth = np.nanmean(X_masked[:, :, 2], axis=1, keepdims=True)
    X[:, :, 1] = np.where(
        np.isnan(X_masked[:, :, 1]), pad_value, X_masked[:, :, 1] - mag_mean_raw
    )
    X[:, :, 2] = np.where(
        np.isnan(X_masked[:, :, 2]), pad_value, X_masked[:, :, 2] - mag_mean_smooth
    )
    X[:, :, 3] = np.where(np.isnan(X_masked[:, :, 3]), pad_value, X_masked[:, :, 3])
    return np.nan_to_num(X, nan=pad_value)


# 【V17.5 傅里叶数学修复】：引入 period，强制将其除掉以获得标准化相位，确保谐波频率正确
def calculate_fourier_params(phase, mag, period, order=6):
    phase, mag = np.array(phase), np.array(mag)
    # 转换为相对于周期的标准化相位，避免绝对天数带来的频率错误
    normalized_phase = phase / period

    terms = [np.ones_like(normalized_phase)]
    for i in range(1, order + 1):
        terms.append(np.cos(2 * np.pi * i * normalized_phase))
        terms.append(np.sin(2 * np.pi * i * normalized_phase))
    try:
        beta, _, _, _ = np.linalg.lstsq(np.column_stack(terms), mag, rcond=None)
    except Exception:
        return [0.0] * (2 * order - 2)
    A, phi = [], []
    for i in range(1, order + 1):
        ai, bi = beta[2 * i - 1], beta[2 * i]
        A.append(np.sqrt(ai**2 + bi**2))
        phi.append(np.arctan2(-bi, ai))
    if A[0] < 1e-7:
        return [0.0] * (2 * order - 2)

    res = []
    for i in range(1, order):
        res.append(A[i] / A[0])
        res.append((phi[i] - (i + 1) * phi[0]) % (2 * np.pi))
    return res


def get_split_sizes(original_total):
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    if original_total < 3:
        raise ValueError("Each class needs at least three eligible unique objects")
    n_train = max(1, int(np.floor(original_total * train_ratio)))
    n_val = max(1, int(np.floor(original_total * val_ratio)))
    n_test = original_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = original_total - n_val - n_test
    return n_test, n_val


def get_train_target_count(original_total):
    if AUGMENTATION_STRATEGY == "none":
        return None
    if AUGMENTATION_STRATEGY == "uniform_median":
        if original_total < UNIFORM_MEDIAN_ORIGINAL_COUNT:
            return UNIFORM_MEDIAN_TRAIN_COUNT
        return None
    if original_total < 500:
        return original_total * 4
    elif original_total < 1000:
        return original_total * 3
    elif original_total < 3000:
        return original_total * 2
    else:
        return original_total


def worker_load_and_process_file(args):
    dat_file, meta_data_broadcast, smoother_mode, minimum_observations = args
    try:
        lc = LightCurve(file_path=dat_file, meta_data=meta_data_broadcast)
        lc.smoother_mode = smoother_mode
        if len(lc) >= minimum_observations:
            lc.fold()
            lc.clean()
            if len(lc) >= minimum_observations:
                return lc, None
            return None, {
                "source_path": str(dat_file),
                "reason": "post_clean_too_short",
                "nobs_raw": lc.nobs_raw,
                "nobs_clean": len(lc),
            }
        return None, {
            "source_path": str(dat_file),
            "reason": "raw_too_short",
            "nobs_raw": len(lc),
            "nobs_clean": len(lc),
        }
    except Exception as exc:
        return None, {
            "source_path": str(dat_file),
            "reason": "load_or_clean_error",
            "error": repr(exc),
        }


def worker_augment_task(task_data):
    parent_lc, method, input_phase, worker_seed = task_data
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Preserve Rot amplitudes: apply phase shifting without photometric noise.
    if parent_lc.type == 7:
        simu_lc = copy.deepcopy(parent_lc)
        simu_lc.generated = True
        simu_lc.origin = "phase_shift"
        simu_lc.prototype_eligible = False
        # Apply phase shifting only; do not use RASLE or GP augmentation.
        shift = random.random() * simu_lc.phase_span
        simu_lc.data["phase"] = (simu_lc.data["phase"] + shift) % simu_lc.phase_span
        simu_lc.data = simu_lc.data.sort_values(by="phase", ignore_index=True)
        simu_lc.supersmoother_fit()
        simu_lc.split_tag = "train"
        simu_lc.id = f"{parent_lc.id}__phase_shift_{worker_seed}"
        return simu_lc

    # 【防抹杀机制】：3(RRd), 4(Blazkho), 5(EA), 6(Ecl), 11(Cep-II)，强制使用 Ralse 扩增
    if parent_lc.type in [3, 4, 5, 6, 11]:
        method = "Ralse"

    try:
        if method == "GP":
            simu_lc = generate_GP_simulation(parent_lc, input_phase)
        else:
            simu_lc = generate_Ralse_simulation(parent_lc, phase_shift=True)
        if simu_lc is not None:
            simu_lc.split_tag = "train"
            simu_lc.id = f"{parent_lc.id}__{simu_lc.origin}_{worker_seed}"
            simu_lc.nobs_clean = len(simu_lc.data)
        return simu_lc
    except Exception:
        return None


def worker_extract_features(lc):
    try:
        if lc.new_df is None:
            lc.new_df = lc.data.copy()
        target_col = f"smoothed_{lc.measurement}"
        if target_col not in lc.new_df.columns:
            lc.new_df[target_col] = lc.new_df[lc.measurement]

        if len(lc.new_df) >= 95:
            mags = lc.new_df[target_col].values
            phase_arr, mag_arr, err_arr, raw_mag_arr = (
                np.array(lc.new_df["phase"]),
                np.array(lc.new_df[target_col]),
                np.array(lc.new_df[lc.error]),
                np.array(lc.new_df[lc.measurement]),
            )
            mask_arr = np.ones_like(phase_arr, dtype=np.float32)

            model_smoothed_mag = (
                raw_mag_arr if lc.smoother_mode == "no_smoothed_channel" else mag_arr
            )
            sequence_data = np.column_stack(
                (phase_arr, raw_mag_arr, model_smoothed_mag, err_arr, mask_arr)
            )
            sequence_raw = np.column_stack((phase_arr, raw_mag_arr, err_arr, mask_arr))

            # 传入 lc.period 进行正确的傅里叶变换
            fourier_params = calculate_fourier_params(
                lc.new_df["phase"], lc.new_df[target_col], lc.period, order=6
            )

            features_and_label = (
                [
                    lc.period,
                    lc.amplitude,
                    lc.v_css,
                    lc.v_amp,
                    np.min(mags),
                    np.max(mags),
                    np.mean(mags),
                ]
                + fourier_params
                + [int(lc.type)]
            )

            if not np.all(np.isfinite(features_and_label)):
                return None

            return {
                "sequence": sequence_data,
                "sequence_raw": sequence_raw,
                "label": features_and_label,
                "is_gen": lc.generated,
                "tag": lc.split_tag,
                "object_id": str(lc.id),
                "parent_id": str(lc.parent_id),
                "origin": lc.origin,
                "prototype_eligible": bool(lc.prototype_eligible),
                "source_path": lc.source_path or "",
                "nobs_raw": int(lc.nobs_raw),
                "nobs_clean": int(lc.nobs_clean or len(lc.new_df)),
                "time_span": float(lc.time_span),
            }
    except Exception:
        pass
    return None


def main():
    global UNIFORM_MEDIAN_ORIGINAL_COUNT, UNIFORM_MEDIAN_TRAIN_COUNT
    max_workers = MAX_WORKERS
    print(f"设置 Worker 数量: {max_workers}")
    cache_key = (
        f"{AUGMENTATION_STRATEGY}_{AUGMENTATION_METHOD}_{SMOOTHER_MODE}_"
        f"{SEED}_{'-'.join(str(value) for value in SPLIT_RATIOS)}"
    ).replace(".", "p")
    checkpoint_file = os.path.join(temp_cache_dir, f"raw_{cache_key}.h5")

    if REUSE_CACHE and os.path.exists(checkpoint_file):
        print(
            "\n" + "=" * 50 + "\n【跳过】发现临时缓存文件，跳过数据加载...\n" + "=" * 50
        )
        (
            X_seq_list,
            X_seq_raw_list,
            Y_list,
            is_gen_list,
            split_tags,
            metadata_rows,
        ) = load_checkpoint(checkpoint_file)
    else:
        print("正在加载元数据...")
        meta_data = load_meta_data()
        all_dat_files = sorted([
            f
            for t in [str(i) for i in range(1, 11)] + ["12"]
            for f in glob.glob(
                str(PROJECT_ROOT / "data/cartlinDR2/original_data/type" / t / "*.dat")
            )
            if "_" not in os.path.basename(f)
        ])

        lc_list, excluded_rows = [], []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for lc, exclusion in tqdm(
                executor.map(
                    worker_load_and_process_file,
                    [
                        (f, meta_data, SMOOTHER_MODE, MIN_OBSERVATIONS)
                        for f in all_dat_files
                    ],
                    chunksize=5,
                ),
                total=len(all_dat_files),
                desc="Loading",
            ):
                if lc is not None:
                    lc_list.append(lc)
                elif exclusion is not None:
                    excluded_rows.append(exclusion)

        pd.DataFrame(excluded_rows).to_csv(
            str(Path(DATASET_OUTPUT_PATH).with_suffix(".excluded.csv")), index=False
        )

        target_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        categorical_lc_list = [
            [lc for lc in lc_list if lc.type == t] for t in target_types
        ]

        non_empty_class_groups = [lcs for lcs in categorical_lc_list if lcs]
        if AUGMENTATION_STRATEGY == "uniform_median":
            original_counts = np.array([len(lcs) for lcs in non_empty_class_groups])
            train_counts = np.array(
                [
                    len(lcs) - sum(get_split_sizes(len(lcs)))
                    for lcs in non_empty_class_groups
                ]
            )
            UNIFORM_MEDIAN_ORIGINAL_COUNT = int(np.median(original_counts))
            UNIFORM_MEDIAN_TRAIN_COUNT = int(np.median(train_counts))
            print(
                "\n[Uniform Median] 少数类定义: 原始样本数 < "
                f"{UNIFORM_MEDIAN_ORIGINAL_COUNT}; 训练集统一补至 "
                f"{UNIFORM_MEDIAN_TRAIN_COUNT}。"
            )

        final_lc_list, augmentation_tasks = [], []

        for lcs in categorical_lc_list:
            if not lcs:
                continue
            class_type, original_count = lcs[0].type, len(lcs)
            n_test, n_val = get_split_sizes(original_count)

            # Keep train/validation/test membership identical across strategies.
            random.Random(SEED + class_type).shuffle(lcs)
            for lc in lcs[:n_test]:
                lc.split_tag, lc.generated = "test", False
                lc.prototype_eligible = False
                final_lc_list.append(lc)
            for lc in lcs[n_test : n_test + n_val]:
                lc.split_tag, lc.generated = "val", False
                lc.prototype_eligible = False
                final_lc_list.append(lc)

            train_subset_real = lcs[n_test + n_val :]
            class_phase_pool = []
            for lc in train_subset_real:
                lc.split_tag, lc.generated = "train", False
                lc.prototype_eligible = True
                final_lc_list.append(lc)
                class_phase_pool.append(lc.data["phase"].values)

            target_count = get_train_target_count(original_count)
            needed = 0 if target_count is None else target_count - len(train_subset_real)
            print(
                f"类别 {class_type}: 原始 {original_count} | Train(真) {len(train_subset_real)} | "
                f"目标 {target_count if target_count is not None else len(train_subset_real)} | 需补充样本 {max(0, needed)}"
            )

            if needed > 0 and len(train_subset_real) > 0 and AUGMENTATION_METHOD != "none":
                duplicate_count = (
                    needed
                    if AUGMENTATION_METHOD == "duplicate"
                    else int(needed * 0.2) if AUGMENTATION_METHOD == "combined" else 0
                )
                for duplicate_index, cp in enumerate(
                    random.choices(train_subset_real, k=duplicate_count)
                ):
                    duplicate = copy.deepcopy(cp)
                    duplicate.id = f"{cp.id}__duplicate_{duplicate_index}"
                    duplicate.parent_id = str(cp.id)
                    duplicate.origin = "duplicate"
                    duplicate.generated = True
                    duplicate.prototype_eligible = False
                    final_lc_list.append(duplicate)

                for _ in range(needed - duplicate_count):
                    if AUGMENTATION_METHOD == "gp":
                        method = "GP"
                    elif AUGMENTATION_METHOD == "rasle":
                        method = "Ralse"
                    else:
                        method = "GP" if random.random() < 0.3 else "Ralse"
                    augmentation_tasks.append(
                        (
                            random.choice(train_subset_real),
                            method,
                            (
                                random.choice(class_phase_pool)
                                if class_phase_pool
                                else None
                            ),
                            SEED + len(augmentation_tasks),
                        )
                    )

        if augmentation_tasks:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for res in tqdm(
                    executor.map(worker_augment_task, augmentation_tasks, chunksize=2),
                    total=len(augmentation_tasks),
                    desc="Augmenting",
                ):
                    if res is not None:
                        final_lc_list.append(res)

        print("\n并行提取特征 (Sequence)...")
        X_seq_list, X_seq_raw_list, Y_list, is_gen_list, split_tags = [], [], [], [], []
        metadata_rows = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in tqdm(
                executor.map(worker_extract_features, final_lc_list, chunksize=50),
                total=len(final_lc_list),
                desc="Extracting",
            ):
                if res is not None:
                    X_seq_list.append(res["sequence"])
                    X_seq_raw_list.append(res["sequence_raw"])
                    Y_list.append(res["label"])
                    is_gen_list.append(res["is_gen"])
                    split_tags.append(res["tag"])
                    metadata_rows.append(
                        {
                            key: res[key]
                            for key in (
                                "object_id",
                                "parent_id",
                                "origin",
                                "source_path",
                                "prototype_eligible",
                                "nobs_raw",
                                "nobs_clean",
                                "time_span",
                            )
                        }
                    )

        del final_lc_list, categorical_lc_list
        gc.collect()
        save_checkpoint(
            checkpoint_file,
            X_seq_list,
            X_seq_raw_list,
            Y_list,
            is_gen_list,
            split_tags,
            metadata_rows,
        )

    MAX_SEQUENCE_LENGTH = max([len(s) for s in X_seq_list])
    print(
        f"\n序列清洗对齐及生成 Interp 图像与 Mask (Target Len: {MAX_SEQUENCE_LENGTH})..."
    )
    Y_all = np.array(Y_list)
    y_labels_all = Y_all[:, -1].astype(int)

    X_seq_clean_raw_list, X_img_clean_list, X_mask_list = [], [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for seq_raw_clean, img, mask in tqdm(
            executor.map(
                worker_clean_and_image,
                [
                    (
                        X_seq_list[i],
                        X_seq_raw_list[i],
                        y_labels_all[i],
                        Y_all[i, 0],
                        MAX_SEQUENCE_LENGTH,
                        PAD_VALUE,
                        SMOOTHER_MODE,
                    )
                    for i in range(len(X_seq_list))
                ],
                chunksize=50,
            ),
            total=len(X_seq_list),
            desc="Cleaning & Imaging",
        ):
            X_seq_clean_raw_list.append(seq_raw_clean)
            X_img_clean_list.append(img)
            X_mask_list.append(mask)

    print("\n最终预处理 (Mean Alignment)...")
    X_seq_clean_processed = preprocess_errors_and_mask(
        np.array(X_seq_clean_raw_list, dtype=np.float32), pad_value=PAD_VALUE
    )

    print("\n特征防泄露处理 (Data Leakage Protection)...")

    df_feat = pd.DataFrame(Y_all[:, :-1])
    df_feat["type"] = y_labels_all

    split_tags_arr = np.array(split_tags)
    train_idx = split_tags_arr == "train"

    for col in df_feat.columns[:-1]:
        median_train = df_feat[train_idx].groupby("type")[col].median()
        global_train_median = df_feat[train_idx][col].median()

        # 1. 对 Train 集合，根据类别填充
        train_fill = df_feat.loc[train_idx, "type"].map(median_train)
        df_feat.loc[train_idx, col] = (
            df_feat.loc[train_idx, col].fillna(train_fill).fillna(global_train_median)
        )

        # 2. 对 Val 和 Test 集合，强制盲填（只用全局中位数）
        val_test_idx = ~train_idx
        df_feat.loc[val_test_idx, col] = df_feat.loc[val_test_idx, col].fillna(
            global_train_median
        )

    lower_bounds = df_feat[train_idx].drop(columns=["type"]).quantile(0.01)
    upper_bounds = df_feat[train_idx].drop(columns=["type"]).quantile(0.99)
    df_feat_features = df_feat.drop(columns=["type"]).clip(
        lower=lower_bounds, upper=upper_bounds, axis=1
    )

    df_feat_features.iloc[:, 0] = np.log10(
        np.clip(df_feat_features.iloc[:, 0], 1e-5, None)
    )
    df_feat_features.iloc[:, 1] = np.log10(
        np.clip(df_feat_features.iloc[:, 1], 1e-5, None)
    )
    df_feat_features.iloc[:, 3] = np.log10(
        np.clip(df_feat_features.iloc[:, 3], 1e-5, None)
    )

    X_feature_processed = np.zeros_like(df_feat_features.values, dtype=np.float32)
    scaler = StandardScaler()

    X_feature_processed[train_idx] = scaler.fit_transform(df_feat_features[train_idx])

    for tag in ["val", "test"]:
        tag_mask = split_tags_arr == tag
        if np.any(tag_mask):
            X_feature_processed[tag_mask] = scaler.transform(df_feat_features[tag_mask])

    print("\n保存 H5 数据...")
    output_path = DATASET_OUTPUT_PATH
    masks = {
        "train": split_tags_arr == "train",
        "val": split_tags_arr == "val",
        "test": split_tags_arr == "test",
    }
    y_onehot_all = np.eye(11, dtype=np.float32)[y_labels_all - 1]
    metadata_frame = pd.DataFrame(metadata_rows)
    metadata_frame["split"] = split_tags_arr
    metadata_frame["class_index"] = y_labels_all - 1
    metadata_frame.to_csv(
        str(Path(output_path).with_suffix(".manifest.csv")), index=False
    )

    with h5py.File(output_path, "w") as f:
        f.attrs["augmentation_strategy"] = AUGMENTATION_STRATEGY
        f.attrs["augmentation_method"] = AUGMENTATION_METHOD
        f.attrs["smoother_mode"] = SMOOTHER_MODE
        f.attrs["split_ratios"] = json.dumps(SPLIT_RATIOS)
        f.attrs["minimum_observations_after_cleaning"] = MIN_OBSERVATIONS
        f.attrs["minority_original_median"] = UNIFORM_MEDIAN_ORIGINAL_COUNT
        f.attrs["minority_train_target_median"] = UNIFORM_MEDIAN_TRAIN_COUNT
        f.attrs["replacement_samples_update_prototypes"] = False
        f.attrs["seed"] = SEED
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for tag, mask in masks.items():
            f.create_dataset(
                f"X_seq_interp_{tag}",
                data=X_seq_clean_processed[mask],
                compression="gzip",
                dtype="float32",
            )
            f.create_dataset(
                f"X_mask_{tag}",
                data=np.array(X_mask_list, dtype=np.float32)[mask],
                compression="gzip",
                dtype="float32",
            )
            f.create_dataset(
                f"X_img_interp_{tag}",
                data=(np.array(X_img_clean_list)[mask] / 255.0).astype(np.float32),
                compression="gzip",
                dtype="float32",
            )
            f.create_dataset(
                f"X_feat_{tag}",
                data=X_feature_processed[mask],
                compression="gzip",
                dtype="float32",
            )
            f.create_dataset(
                f"y_{tag}",
                data=(y_labels_all - 1)[mask],
                compression="gzip",
                dtype="int32",
            )
            f.create_dataset(
                f"y_{tag}_onehot",
                data=y_onehot_all[mask],
                compression="gzip",
                dtype="float32",
            )
            f.create_dataset(
                f"is_gen_{tag}",
                data=np.array(is_gen_list, dtype=bool)[mask],
                dtype="bool",
            )
            f.create_dataset(
                f"prototype_eligible_{tag}",
                data=metadata_frame["prototype_eligible"].to_numpy(dtype=bool)[mask],
                dtype="bool",
            )
            for key in ("object_id", "parent_id", "origin", "source_path"):
                f.create_dataset(
                    f"{key}_{tag}",
                    data=metadata_frame[key].astype(str).to_numpy(dtype=object)[mask],
                    dtype=string_dtype,
                )
            for key, dtype in (
                ("nobs_raw", "int32"),
                ("nobs_clean", "int32"),
                ("time_span", "float64"),
            ):
                f.create_dataset(
                    f"{key}_{tag}",
                    data=metadata_frame[key].to_numpy()[mask],
                    dtype=dtype,
                )

    print(f"\n全部流程完成！精简安全版数据已保存至: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a provenance-preserving CRTS dataset for TDMN."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Directory containing data/SSS_Per_Tab.dat and data/cartlinDR2",
    )
    parser.add_argument(
        "--augmentation-strategy",
        choices=["dynamic", "uniform_median", "none"],
        default="uniform_median",
    )
    parser.add_argument(
        "--augmentation-method",
        choices=["combined", "duplicate", "gp", "rasle", "gp_rasle", "none"],
        default="combined",
    )
    parser.add_argument(
        "--smoother-mode",
        choices=["full", "no_smoothed_channel", "no_rejection", "no_supersmoother"],
        default="full",
    )
    parser.add_argument(
        "--split",
        default="7:2:1",
        help="Train:validation:test ratios, e.g. 7:2:1 or 6:1:3",
    )
    parser.add_argument("--min-observations", type=int, default=MIN_OBSERVATIONS)
    parser.add_argument(
        "--max-workers", type=int, default=min(multiprocessing.cpu_count(), 8)
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output-path", default=DATASET_OUTPUT_PATH)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PROJECT_ROOT = Path(args.project_root).expanduser().resolve()
    if not (PROJECT_ROOT / "data/SSS_Per_Tab.dat").is_file():
        raise FileNotFoundError(
            f"Missing metadata: {PROJECT_ROOT / 'data/SSS_Per_Tab.dat'}"
        )
    AUGMENTATION_STRATEGY = args.augmentation_strategy
    AUGMENTATION_METHOD = args.augmentation_method
    SMOOTHER_MODE = args.smoother_mode
    DATASET_OUTPUT_PATH = args.output_path
    SEED = args.seed
    MIN_OBSERVATIONS = args.min_observations
    MAX_WORKERS = args.max_workers
    REUSE_CACHE = args.reuse_cache
    split_parts = tuple(float(part) for part in args.split.split(":"))
    if len(split_parts) != 3 or any(part <= 0 for part in split_parts):
        raise ValueError("--split must contain three positive values, e.g. 7:2:1")
    split_total = sum(split_parts)
    SPLIT_RATIOS = tuple(part / split_total for part in split_parts)
    if args.cache_dir:
        temp_cache_dir = args.cache_dir
    os.makedirs(os.path.dirname(DATASET_OUTPUT_PATH) or ".", exist_ok=True)
    os.makedirs(temp_cache_dir, exist_ok=True)
    with open(
        str(Path(DATASET_OUTPUT_PATH).with_suffix(".config.json")),
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            {
                **vars(args),
                "project_root": str(PROJECT_ROOT),
                "output_path": str(Path(DATASET_OUTPUT_PATH).resolve()),
                "normalized_split_ratios": SPLIT_RATIOS,
            },
            config_file,
            ensure_ascii=False,
            indent=2,
        )
    sys.stdout = Logger(
        str(Path(DATASET_OUTPUT_PATH).with_suffix(".log"))
    )
    sys.stderr = sys.stdout
    set_global_seed(SEED)
    multiprocessing.freeze_support()
    main()
