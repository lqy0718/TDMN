"""CPU-only provenance checks; embedded in every standalone training script."""
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(os.environ.get(
    'TDMN_CRTS_H5', REPO_ROOT / 'data/processed/crts_tdmn.h5'
))
EXPECTED_SEEDS = (42, 142, 242)
CLASS_NAMES = ['RRab', 'RRc', 'RRd', 'Blazkho', 'Ecl', 'EA', 'Rot', 'LPV',
               'delta-Scuti', 'ACep', 'Cep-II']


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prediction_cohort(path):
    # Only trusted experiment outputs. Existing TDMN files may use object strings.
    with np.load(path, allow_pickle=True) as data:
        ids = data['object_id'].astype(str).reshape(-1)
        labels = data['y_true'].astype(int).reshape(-1)
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid/duplicate test object IDs: {path}')
    return {key: int(label) for key, label in zip(ids, labels)}


def audit_hdf5(path, check_all_arrays=False):
    """No data mutations, no splitting, no test-performance-based decisions.

    Default: schema, provenance, labels, features, split membership. Optional
    full-array scan is chunked. Fingerprint is NOT a hash of all tensor bytes.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'CRTS HDF5 dataset not found: {path}')
    n_classes = len(CLASS_NAMES)
    cohorts, parent_sets, record = {}, {}, {}
    with h5py.File(path, 'r') as hf:
        for tag in ('train', 'val', 'test'):
            keys = [f'{prefix}_{tag}' for prefix in (
                'X_seq_interp', 'X_img_interp', 'X_feat', 'X_mask', 'y',
                'is_gen', 'prototype_eligible', 'object_id', 'parent_id', 'origin')]
            keys += [f'y_{tag}_onehot']
            missing = [key for key in keys if key not in hf]
            if missing:
                raise ValueError(f'CRTS HDF5 is missing required fields: {missing}')
            ids = hf[f'object_id_{tag}'].asstr()[:].reshape(-1)
            parents = hf[f'parent_id_{tag}'].asstr()[:].reshape(-1)
            origins = hf[f'origin_{tag}'].asstr()[:].reshape(-1)
            labels_raw = hf[f'y_{tag}'][:].reshape(-1)
            if not np.isfinite(labels_raw).all() or not np.equal(labels_raw, np.floor(labels_raw)).all():
                raise ValueError(f'{tag}: labels must be finite integers')
            labels = labels_raw.astype(int)
            generated = hf[f'is_gen_{tag}'][:].reshape(-1)
            eligible = hf[f'prototype_eligible_{tag}'][:].reshape(-1)
            if not np.isin(generated, [0, 1]).all() or not np.isin(eligible, [0, 1]).all():
                raise ValueError(f'{tag}: invalid provenance flags')
            generated, eligible = generated.astype(bool), eligible.astype(bool)
            n = len(ids)
            if n == 0 or any(hf[key].shape[0] != n for key in keys):
                raise ValueError(f'{tag}: empty split or inconsistent first dimensions')
            if len(set(ids)) != n or any(not item or item in ('None', 'nan') for item in ids):
                raise ValueError(f'{tag}: missing or duplicate object IDs')
            if not np.isin(labels, np.arange(n_classes)).all():
                raise ValueError(f'{tag}: labels outside 0..10')
            onehot = hf[f'y_{tag}_onehot'][:]
            if onehot.shape != (n, n_classes) or not np.allclose(onehot, np.eye(n_classes)[labels]):
                raise ValueError(f'{tag}: one-hot labels disagree with integer labels')
            seq, img, feat, mask = [hf[f'{key}_{tag}'] for key in
                                   ('X_seq_interp', 'X_img_interp', 'X_feat', 'X_mask')]
            # Current preparer stores five columns: phase, raw magnitude,
            # smoothed magnitude, error, observation flag. Training consumes
            # only columns 0..3 and uses the separate X_mask padding mask.
            # Four-column exports are also compatible; never reshape the data.
            if seq.ndim != 3 or seq.shape[-1] not in (4, 5) or mask.shape != seq.shape[:2]:
                raise ValueError(
                    f'{tag}: expected sequence (N,L,4) or (N,L,5) and mask (N,L); '
                    f'got sequence {seq.shape}, mask {mask.shape}'
                )
            if img.ndim != 4 or img.shape[-1] != 3 or feat.ndim != 2 or feat.shape[1] < 1:
                raise ValueError(f'{tag}: invalid image/feature shape')
            if not np.isfinite(feat[:]).all():
                raise ValueError(f'{tag}: non-finite scalar features')
            if tag == 'train':
                if not np.array_equal(~generated, eligible) or not np.array_equal(origins == 'real', eligible):
                    raise ValueError('train: is_gen / origin / prototype_eligible不一致；不能安全复用Full协议')
                real_ids = set(ids[eligible])
                if not np.array_equal(parents[eligible], ids[eligible]) or not set(parents) <= real_ids:
                    raise ValueError('train: augmentation parent is not a unique real training object')
                real_counts = np.bincount(labels[eligible], minlength=n_classes)
                if np.any(real_counts == 0):
                    raise ValueError('Unique real training set is missing a class')
                record['unique_real_train_counts'] = real_counts.tolist()
                record['n_unique_real_train'] = int(eligible.sum())
            elif generated.any() or eligible.any() or not np.all(origins == 'real') or not np.array_equal(parents, ids):
                raise ValueError(f'{tag}: validation/test must contain unique real objects only')
            if np.any(np.bincount(labels, minlength=n_classes) == 0):
                raise ValueError(f'{tag}: missing classes, fixed 11-class evaluation not applicable')
            cohorts[tag] = dict(zip(ids.tolist(), labels.tolist()))
            parent_sets[tag] = set(parents)
            record[tag] = {
                'n_rows': n, 'class_counts': np.bincount(labels, minlength=n_classes).tolist(),
                'n_generated': int(generated.sum()), 'n_eligible': int(eligible.sum()),
                'ordered_metadata_sha256': canonical_hash([ids.tolist(), parents.tolist(),
                    labels.tolist(), origins.tolist(), generated.tolist(), eligible.tolist()]),
                'tensor_schema': {key: {'shape': list(hf[key].shape), 'dtype': str(hf[key].dtype)}
                                  for key in keys if key.startswith(('X_', 'y_'))},
            }
            if check_all_arrays:
                for field in ('X_seq_interp', 'X_img_interp', 'X_feat', 'X_mask'):
                    dataset = hf[f'{field}_{tag}']
                    for start in range(0, n, 128):
                        block = dataset[start:start + 128]
                        if not np.isfinite(block).all():
                            raise ValueError(f'{field}_{tag}: non-finite values at rows {start}:{start + 128}')
                        if field == 'X_mask' and (not np.isin(block, [0, 1]).all() or not (block.sum(axis=1) > 0).all()):
                            raise ValueError(f'{field}_{tag}: invalid/empty masks')
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if set(cohorts[a]) & set(cohorts[b]) or parent_sets[a] & parent_sets[b]:
                raise ValueError(f'{a}/{b}: object/parent leakage detected')
        for field in ('X_seq_interp', 'X_img_interp', 'X_feat', 'X_mask'):
            shapes = [hf[f'{field}_{tag}'].shape[1:] for tag in ('train', 'val', 'test')]
            if len(set(shapes)) != 1:
                raise ValueError(f'{field}: incompatible train/val/test shapes')
        record['hdf5_attributes'] = {key: str(value) for key, value in hf.attrs.items()}
    stat = path.stat()
    record['cohort_schema_sha256'] = canonical_hash(record)
    record.update(data_path=str(path), file_size_bytes=stat.st_size,
                  file_mtime_ns=stat.st_mtime_ns, all_arrays_scanned=check_all_arrays,
                  fingerprint_scope='metadata+schema+attributes; NOT full tensor bytes')
    return record, cohorts['test']


def original_full_dir(seed, local_root=None):
    """Locate a full TDMN result directory for an optional result audit."""
    if local_root is not None:
        return Path(local_root) / 'crts' / f'seed{seed}' / f'tdmn_seed{seed}'
    return REPO_ROOT / 'results/crts' / f'tdmn_seed{seed}'


def verify_full_reference(directory, seed, test_cohort=None):
    directory = Path(directory)
    for name in ('run_config.json', 'metrics.json', 'model_selection.json', 'test_predictions.npz'):
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Full reference missing: {directory / name}')
    config = json.loads((directory / 'run_config.json').read_text())
    expected = dict(seed=seed, batch_size=256, epochs_stage1=300, epochs_stage2=120,
                    anchor_weight=0.1, logit_coefficient=0.3, frequency_source='real')
    differences = {key: [config.get(key), val] for key, val in expected.items()
                   if config.get(key) != val}
    if differences or config.get('data') != str(DEFAULT_DATA):
        raise ValueError(f'Full training configuration differs: {directory}: {differences}; data={config.get("data")}')
    selection = json.loads((directory / 'model_selection.json').read_text())
    if selection.get('selection_metric') != 'validation_macro_f1':
        raise ValueError(f'Full reference did not use validation selection: {directory}')
    cohort = prediction_cohort(directory / 'test_predictions.npz')
    if test_cohort is not None and cohort != test_cohort:
        raise ValueError(f'Full reference test IDs/labels differ: {directory}')
    return {'seed': seed, 'directory': str(directory), 'n_test': len(cohort),
            'test_cohort_sha256': canonical_hash(cohort),
            'warning': 'Original run has no HDF5/source hash; unchanged input tensors must be confirmed by user.'}

# 默认检查元数据、类别、划分和物理特征；不读取整个大型图像张量。
CHECK_ALL_ARRAYS = False
COMPUTE_FULL_FILE_SHA256 = False  # 大文件会耗时；需逐字节确认时改为True。

def main():
    audit, _ = audit_hdf5(DEFAULT_DATA, check_all_arrays=CHECK_ALL_ARRAYS)
    if COMPUTE_FULL_FILE_SHA256:
        audit['full_file_sha256'] = file_sha256(DEFAULT_DATA)
    report = audit
    destination = Path(__file__).resolve().parent / 'data_check.json'
    json_write(destination, report)
    print('检查通过：不重新切分、不生成或覆盖HDF5。')
    for tag in ('train', 'val', 'test'):
        print(f"{tag}: {audit[tag]['n_rows']} rows, counts={audit[tag]['class_counts']}")
    print(f"V5 Exp1–4均使用全部 {audit['train']['n_rows']} 条已有训练行；"
          f"其中 {audit['n_unique_real_train']} 条是唯一真实对象。")
    print(f'检查报告: {destination}')

if __name__ == '__main__':
    main()
