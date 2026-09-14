"""Prepare the OGLE-IV HDF5 dataset and provenance manifest.

The script retains the accepted-paper 15-subclass/6-superclass taxonomy and
uses deterministic object-level splitting before training-set augmentation.
"""

import argparse
import json
import os
from pathlib import Path
import glob
import random
import multiprocessing
import numpy as np
import pandas as pd
import h5py
import hashlib
from tqdm import tqdm

from supersmoother import SuperSmoother
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_OGLE_DIR = str(REPO_ROOT / "data/raw/ogle4")
OUTPUT_H5 = str(REPO_ROOT / "data/processed/ogle_tdmn.h5")
PAD_VALUE = -10.0
SEED = 42
MAX_WORKERS = min(multiprocessing.cpu_count(), 16)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(SEED)

TARGET_GALAXIES = ["lmc", "smc"]
TARGET_CLASSES = ["acep", "cep", "dsct", "ecl", "rrlyr", "t2cep"]
MAIN_CLASS_MAP = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}

SUB_CLASS_MAP = {
    "ACEP_F": 1,
    "ACEP_1O": 2,
    "CEP_F": 3,
    "CEP_1O": 4,
    "CEP_1O2O": 5,
    "DSCT_SINGLEMODE": 6,
    "DSCT_MULTIMODE": 7,
    "ECL_C": 8,
    "ECL_NC": 9,
    "RRLYR_RRAB": 10,
    "RRLYR_RRC": 11,
    "RRLYR_RRD": 12,
    "T2CEP_BLHER": 13,
    "T2CEP_RVTAU": 14,
    "T2CEP_WVIR": 15,
}
INV_SUB_CLASS_MAP = {v: k for k, v in SUB_CLASS_MAP.items()}


# ===================================================================
# 核心 LightCurve 类
# ===================================================================
class LightCurve:
    def __init__(self, file_path=None, meta_row=None):
        self.data = None
        self.smoothed = False
        self.generated = False
        self.origin = "real"
        self.source_path = None
        self.nobs_raw = 0
        self.nobs_clean = 0
        if file_path is not None and meta_row is not None:
            self.load_ogle_data(file_path, meta_row)

    def load_ogle_data(self, file_path, row):
        self.id = str(row["OGLE_ID"])
        self.main_type = row["Main_Type"]
        self.sub_type = row["Sub_Type"]
        self.period = row["Period"]
        self.needs_fourier = row["needs_fourier"]

        self.feat_base = [self.period, row["I_mag"], row["I_amp"]]
        self.fourier_vals = [row["R_21"], row["phi_21"], row["R_31"], row["phi_31"]]

        df = pd.read_csv(file_path, sep=r"\s+", names=["time", "mag", "err"])
        self.data = df.sort_values(by="time").reset_index(drop=True)
        self.source_path = str(os.path.abspath(file_path))
        self.nobs_raw = len(self.data)
        self.nobs_clean = len(self.data)

    def fold(self):
        time_arr = self.data["time"].copy()
        # 保留物理绝对天数感知
        phase_abs = (time_arr - time_arr.iloc[0]) % self.period

        self.data["phase"] = phase_abs
        self.data = self.data.sort_values(by="phase", ignore_index=True)

        phase_vals = self.data["phase"].values
        for i in range(1, len(phase_vals)):
            if phase_vals[i] <= phase_vals[i - 1]:
                phase_vals[i] = phase_vals[i - 1] + 1e-8
        self.data["phase"] = phase_vals

    def supersmoother_fit(self):
        x, y, err = (
            np.around(self.data["phase"], 4),
            np.around(self.data["mag"], 4),
            np.around(self.data["err"], 4),
        )
        model = SuperSmoother(period=self.period)
        try:
            model.fit(x, y, err)
            self.data["smoothed_mag"] = model.predict(x)
        except Exception:
            self.data["smoothed_mag"] = y
        self.smoothed = True

    def clean(self):
        if not self.smoothed:
            self.supersmoother_fit()

        # 🌟 核心防线：保护多模式变星 (CEP_1O2O, DSCT_MULTIMODE, RRLYR_RRD) 的包络不被误杀
        if self.sub_type not in [5, 7, 12]:
            y_smoothed, y_original, err = (
                self.data["smoothed_mag"],
                self.data["mag"],
                self.data["err"],
            )
            mean_err = np.mean(err)
            is_bad = (abs(y_original - y_smoothed) >= 3 * err) | (err >= 2 * mean_err)
            self.data = self.data.drop(index=self.data[is_bad].index).reset_index(
                drop=True
            )
            self.supersmoother_fit()

        MAX_SAFE_LEN = 800
        if len(self.data) > MAX_SAFE_LEN:
            normalized_ratio = self.data["phase"] / self.period
            grid_id = np.clip(
                (normalized_ratio * MAX_SAFE_LEN).astype(int), 0, MAX_SAFE_LEN - 1
            )

            self.data["grid_id"] = grid_id
            self.data = self.data.drop_duplicates(subset=["grid_id"], keep="first")
            self.data = self.data.drop(columns=["grid_id"]).reset_index(drop=True)
        self.nobs_clean = len(self.data)

    def calculate_fallback_fourier(self):
        phase, mag = self.data["phase"].values, self.data["smoothed_mag"].values
        normalized_phase = phase / self.period
        terms = [np.ones_like(normalized_phase)]
        for i in range(1, 4):
            terms.append(np.cos(2 * np.pi * i * normalized_phase))
            terms.append(np.sin(2 * np.pi * i * normalized_phase))
        try:
            beta, _, _, _ = np.linalg.lstsq(np.column_stack(terms), mag, rcond=None)
            A, phi = [], []
            for i in range(1, 4):
                ai, bi = beta[2 * i - 1], beta[2 * i]
                A.append(np.sqrt(ai**2 + bi**2))
                phi.append(np.arctan2(-bi, ai))
            if A[0] >= 1e-7:
                self.fourier_vals = [
                    A[1] / A[0],
                    (phi[1] - 2 * phi[0]) % (2 * np.pi),
                    A[2] / A[0],
                    (phi[2] - 3 * phi[0]) % (2 * np.pi),
                ]
            else:
                self.fourier_vals = [0.0, 0.0, 0.0, 0.0]
        except Exception:
            self.fourier_vals = [0.0, 0.0, 0.0, 0.0]


# ===================================================================
# 元数据构建引擎
# ===================================================================
def build_ogle_meta_db(base_dir):
    def safe_float(val):
        # Returns np.nan for '-' placeholders or any non-numeric string.
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    meta_records = []

    for galaxy in TARGET_GALAXIES:
        galaxy_dir = os.path.join(base_dir, galaxy)
        if not os.path.exists(galaxy_dir):
            continue

        print(f">>> 正在重构星系元数据字典: [{galaxy.upper()}] ...")
        for cat in TARGET_CLASSES:
            cat_dir = os.path.join(galaxy_dir, cat)
            if not os.path.exists(cat_dir):
                continue

            ident_dict = {}
            ident_file = os.path.join(cat_dir, "ident.dat")
            if os.path.exists(ident_file):
                with open(ident_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            star_id = parts[0]
                            raw_sub = parts[1].upper().replace("/", "")
                            if cat == "rrlyr" and raw_sub == "ARRD":
                                raw_sub = "RRD"
                            key = f"{cat.upper()}_{raw_sub}"
                            sub_id = SUB_CLASS_MAP.get(key, -1)
                            if sub_id != -1:
                                ident_dict[star_id] = sub_id

            param_files = [
                f
                for f in glob.glob(os.path.join(cat_dir, "*.dat"))
                if "ident" not in f and "remarks" not in f
            ]
            for p_file in param_files:
                try:
                    with open(p_file, "r", encoding="utf-8") as f:
                        for line in f:
                            row = line.strip().split()
                            if len(row) < 4:
                                continue

                            star_id = row[0].strip()
                            if star_id not in ident_dict:
                                continue

                            p = np.nan
                            i_mag = np.nan
                            i_amp = np.nan
                            r21, phi21, r31, phi31 = np.nan, np.nan, np.nan, np.nan
                            needs_fourier = True

                            if cat == "ecl":
                                if len(row) < 6:
                                    continue
                                p = safe_float(row[3])
                                i_mag = safe_float(row[1])
                                i_amp = safe_float(row[5])
                            else:
                                if len(row) < 7:
                                    continue
                                p = safe_float(row[3])
                                i_mag = safe_float(row[1])
                                i_amp = safe_float(row[6])

                                if len(row) >= 11:
                                    r21_v = safe_float(row[7])
                                    phi21_v = safe_float(row[8])
                                    r31_v = safe_float(row[9])
                                    phi31_v = safe_float(row[10])
                                    # Only treat as complete if all four params are
                                    # finite; a single '-' means the fit was invalid.
                                    if not any(
                                        np.isnan(v)
                                        for v in (r21_v, phi21_v, r31_v, phi31_v)
                                    ):
                                        r21, phi21 = r21_v, phi21_v
                                        r31, phi31 = r31_v, phi31_v
                                        needs_fourier = False

                            # Period is the only non-negotiable field: without it
                            # phase-folding is impossible and the star is useless.
                            if np.isnan(p):
                                continue

                            meta_records.append(
                                {
                                    "OGLE_ID": star_id,
                                    "Galaxy": galaxy,
                                    "Cat_Folder": cat,
                                    "Main_Type": MAIN_CLASS_MAP[cat],
                                    "Sub_Type": ident_dict[star_id],
                                    "Period": p,
                                    "I_mag": i_mag,
                                    "I_amp": i_amp,
                                    "R_21": r21,
                                    "phi_21": phi21,
                                    "R_31": r31,
                                    "phi_31": phi31,
                                    "needs_fourier": needs_fourier,
                                }
                            )
                except Exception:
                    continue

    meta_df = pd.DataFrame(meta_records).drop_duplicates(subset=["OGLE_ID"])
    return meta_df


def get_split_tag(star_id):
    hash_val = int(hashlib.md5(star_id.encode("utf-8")).hexdigest(), 16) % 100
    if hash_val < 10:
        return "test"
    elif hash_val < 20:
        return "val"
    else:
        return "train"


def worker_load_and_process(args):
    dat_file, meta_row = args
    try:
        lc = LightCurve(file_path=dat_file, meta_row=meta_row)
        if lc.data is None:
            return None
        if len(lc.data) >= 10:
            lc.fold()
            lc.clean()
            if len(lc.data) >= 10:
                if lc.needs_fourier:
                    lc.calculate_fallback_fourier()
                lc.split_tag = get_split_tag(lc.id)
                return lc
    except Exception:
        pass
    return None


def worker_extract_features(lc):
    try:
        phase_arr, raw_mag_arr, mag_arr, err_arr = (
            np.array(lc.data["phase"]),
            np.array(lc.data["mag"]),
            np.array(lc.data["smoothed_mag"]),
            np.array(lc.data["err"]),
        )
        mask_arr = np.ones_like(phase_arr, dtype=np.float32)
        sequence_data = np.column_stack(
            (phase_arr, raw_mag_arr, mag_arr, err_arr, mask_arr)
        )
        label_data = lc.feat_base + lc.fourier_vals + [int(lc.sub_type)]

        # Fallback chain: recover missing catalog values from the real light curve.
        # I_mag missing → use mean of raw photometry (same physical quantity).
        if np.isnan(label_data[1]):
            label_data[1] = float(np.nanmean(raw_mag_arr))
        # I_amp missing → peak-to-peak of smoothed curve.
        if np.isnan(label_data[2]):
            label_data[2] = float(np.nanmax(mag_arr) - np.nanmin(mag_arr))

        return {
            "sequence": sequence_data,
            "label": label_data,
            "is_gen": lc.generated,
            "tag": lc.split_tag,
            "object_id": lc.id,
            "source_path": lc.source_path,
            "origin": lc.origin,
            "nobs_raw": lc.nobs_raw,
            "nobs_clean": lc.nobs_clean,
        }
    except Exception:
        pass
    return None


def array_to_image_fast(sequence_data):
    if len(sequence_data) == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    x, y_raw, y_smoothed, y_err = (
        sequence_data[:, 0],
        sequence_data[:, 1],
        sequence_data[:, 2],
        sequence_data[:, 3],
    )
    fig = Figure(figsize=(2, 2), dpi=64, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.axis("off")
    ax.plot(
        x,
        y_smoothed,
        marker="o",
        linestyle="none",
        color="#0000FF",
        markersize=1.5,
        alpha=0.7,
    )
    ax.vlines(
        x,
        y_smoothed - y_err,
        y_smoothed + y_err,
        colors="#00FF00",
        linewidth=0.5,
        alpha=0.4,
    )
    ax.plot(
        x,
        y_raw,
        marker="o",
        linestyle="none",
        color="#FF0000",
        markersize=1.5,
        alpha=0.9,
    )
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3]


def worker_clean_and_image(args):
    seq_data, max_len, pad_val = args
    valid_mask = seq_data[:, 0] > (pad_val + 0.1)
    if not np.any(valid_mask):
        return (
            np.full((max_len, 5), pad_val, dtype=np.float32),
            np.zeros((128, 128, 3), dtype=np.uint8),
            np.zeros(max_len, dtype=np.float32),
        )

    valid_data = seq_data[valid_mask]
    res_seq = np.full((max_len, 5), pad_val, dtype=np.float32)
    padding_mask = np.zeros(max_len, dtype=np.float32)
    v_len = min(len(valid_data), max_len)
    res_seq[:v_len] = valid_data[:v_len]
    padding_mask[:v_len] = 1.0
    img_out = array_to_image_fast(valid_data[:v_len])
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


def main():
    max_workers = MAX_WORKERS
    print(">>> Parsing OGLE-IV catalogues and light curves...")
    meta_df = build_ogle_meta_db(BASE_OGLE_DIR)

    task_args = []
    for _, row in meta_df.iterrows():
        star_id = row["OGLE_ID"]
        galaxy = row["Galaxy"]
        cat = row["Cat_Folder"]

        # 兼容可能存在的 I/V 波段落位
        dat_path_I = os.path.join(
            BASE_OGLE_DIR, galaxy, cat, "phot", "I", f"{star_id}.dat"
        )
        dat_path_V = os.path.join(
            BASE_OGLE_DIR, galaxy, cat, "phot", "V", f"{star_id}.dat"
        )

        target_path = None
        if os.path.exists(dat_path_I):
            target_path = dat_path_I
        elif os.path.exists(dat_path_V):
            target_path = dat_path_V

        if target_path is not None:
            task_args.append((target_path, row))

    lc_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(
            executor.map(worker_load_and_process, task_args, chunksize=100),
            total=len(task_args),
            desc="1. 物理折叠与抗OOM降采",
        ):
            if res is not None:
                lc_list.append(res)

    X_seq_list, Y_list, is_gen_list, split_tags, metadata_rows = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(
            executor.map(worker_extract_features, lc_list, chunksize=200),
            total=len(lc_list),
            desc="2. 提取严谨表格特征",
        ):
            if res is not None:
                X_seq_list.append(res["sequence"])
                Y_list.append(res["label"])
                is_gen_list.append(res["is_gen"])
                split_tags.append(res["tag"])
                metadata_rows.append(
                    {
                        "object_id": res["object_id"],
                        "source_path": res["source_path"],
                        "origin": res["origin"],
                        "nobs_raw": res["nobs_raw"],
                        "nobs_clean": res["nobs_clean"],
                    }
                )

    seq_lengths_raw = [len(s) for s in X_seq_list]
    MAX_SEQUENCE_LENGTH = max(seq_lengths_raw) if seq_lengths_raw else 0
    X_seq_clean_raw_list, X_img_clean_list, X_mask_list = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        img_tasks = [
            (X_seq_list[i], MAX_SEQUENCE_LENGTH, PAD_VALUE)
            for i in range(len(X_seq_list))
        ]
        for seq_raw_clean, img, mask in tqdm(
            executor.map(worker_clean_and_image, img_tasks, chunksize=200),
            total=len(X_seq_list),
            desc="3. 极速物理图像渲染",
        ):
            X_seq_clean_raw_list.append(seq_raw_clean)
            X_img_clean_list.append(img)
            X_mask_list.append(mask)

    X_seq_clean_processed = preprocess_errors_and_mask(
        np.array(X_seq_clean_raw_list, dtype=np.float32), pad_value=PAD_VALUE
    )
    Y_all = np.array(Y_list)
    y_sub_all = Y_all[:, -1].astype(int)
    df_feat_features = pd.DataFrame(Y_all[:, :-1])

    # 对数化依然安全，因为我们已经抛弃了 Period 或 Amp 小于等于0的坏数据
    for col in [0, 2]:
        df_feat_features.iloc[:, col] = np.log10(
            np.clip(df_feat_features.iloc[:, col], 1e-5, None)
        )

    split_tags_arr = np.array(split_tags)
    train_idx = split_tags_arr == "train"
    scaler = StandardScaler()
    X_feature_processed = np.zeros_like(df_feat_features.values, dtype=np.float32)

    if np.any(train_idx):
        X_feature_processed[train_idx] = scaler.fit_transform(
            df_feat_features[train_idx]
        )
    for tag in ["val", "test"]:
        tag_mask = split_tags_arr == tag
        if np.any(tag_mask):
            X_feature_processed[tag_mask] = scaler.transform(df_feat_features[tag_mask])

    print("\n" + "=" * 70)
    print("📊 OGLE-IV LMC+SMC 科学底线收官版构建报告")
    print("=" * 70)
    print(f"📦 样本总规模: {len(Y_all)}")
    print(f"   ┣ 训练集 (Train): {np.sum(split_tags_arr == 'train')}")
    print(f"   ┣ 验证集 (Val):   {np.sum(split_tags_arr == 'val')}")
    print(f"   ┗ 测试集 (Test):  {np.sum(split_tags_arr == 'test')}")
    print(f"\n📏 MAX 锁定上限: {MAX_SEQUENCE_LENGTH} 点 (安全无隐患)")
    print("=" * 70 + "\n")

    print(f">>> 正在将数据高速刷入 HDF5 存储: {OUTPUT_H5}")
    y_sub_onehot = np.eye(15, dtype=np.float32)[y_sub_all - 1]
    masks = {
        "train": train_idx,
        "val": split_tags_arr == "val",
        "test": split_tags_arr == "test",
    }

    metadata_frame = pd.DataFrame(metadata_rows)
    metadata_frame["split"] = split_tags_arr
    metadata_frame["class_index"] = y_sub_all - 1
    manifest_path = str(Path(OUTPUT_H5).with_suffix(".manifest.csv"))
    metadata_frame.to_csv(manifest_path, index=False)

    with h5py.File(OUTPUT_H5, "w") as f:
        f.attrs["survey"] = "OGLE-IV LMC+SMC"
        f.attrs["split_method"] = "MD5 object-id 80/10/10"
        f.attrs["seed"] = SEED
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for tag, mask in masks.items():
            if np.sum(mask) == 0:
                continue
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
                data=(y_sub_all - 1)[mask],
                compression="gzip",
                dtype="int32",
            )
            f.create_dataset(
                f"y_{tag}_onehot",
                data=y_sub_onehot[mask],
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
                data=np.full(int(mask.sum()), tag == "train", dtype=bool),
                dtype="bool",
            )
            for key in ("object_id", "source_path", "origin"):
                f.create_dataset(
                    f"{key}_{tag}",
                    data=metadata_frame[key].astype(str).to_numpy(dtype=object)[mask],
                    dtype=string_dtype,
                )
            for key in ("nobs_raw", "nobs_clean"):
                f.create_dataset(
                    f"{key}_{tag}",
                    data=metadata_frame[key].to_numpy(dtype=np.int32)[mask],
                    dtype="int32",
                )

    print(f">>> 🎯 All Done! 高质量多模态物理数据集已生成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the frozen OGLE-IV TDMN split and provenance manifest")
    parser.add_argument("--ogle-root", default=BASE_OGLE_DIR)
    parser.add_argument("--output", default=OUTPUT_H5)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    BASE_OGLE_DIR = str(Path(args.ogle_root).expanduser().resolve())
    OUTPUT_H5 = str(Path(args.output).expanduser().resolve())
    MAX_WORKERS = args.workers
    SEED = args.seed
    Path(OUTPUT_H5).parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    Path(OUTPUT_H5).with_suffix(".config.json").write_text(
        json.dumps({**vars(args), "ogle_root": BASE_OGLE_DIR, "output": OUTPUT_H5}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    multiprocessing.freeze_support()
    main()
