"""Server-side synthetic smoke test for V5-faithful scripts; no real results."""
import importlib.util
import os
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ["V5_ABLATION_ACTIVE_SEED"] = "42"


def load_module(index, folder):
    path = ROOT / folder / "train.py"
    spec = importlib.util.spec_from_file_location(f"v5_smoke_exp{index}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    folders = ["exp1_ce", "exp2_augmentation_ce",
               "exp3_proxy_contrastive", "exp4_two_stage_cb"]
    rng = np.random.default_rng(42)
    for index, folder in enumerate(folders, 1):
        module = load_module(index, folder)
        tf, keras = module.tf, module.keras
        keras.backend.clear_session()
        seq = rng.normal(size=(2, 32, 7)).astype(np.float32)
        seq[:, :, 6] = 1.0
        inputs3 = {"seq_in": tf.constant(seq),
                   "img_in": tf.constant(rng.random((2, 128, 128, 3)).astype(np.float32)),
                   "feat_in": tf.constant(rng.normal(size=(2, 17)).astype(np.float32))}
        if index == 1:
            model = module.build_exp1_model((32, 7), (128, 128, 3), 17)
            stages = (None,)
        elif index == 2:
            model = module.build_exp2_model((32, 7), (128, 128, 3), 17)
            stages = (None,)
        else:
            encoder = module.build_pure_di_modal_encoder((32, 7), (128, 128, 3))
            model = (module.ProxyContrastiveE2EModel(encoder, 17, 11) if index == 3
                     else module.DecoupledCBModel(encoder, 17, 11, tf.ones(11)))
            inputs3["is_gen"] = tf.zeros(2)
            stages = (1, 2)
            model.prototypes_modal.assign(tf.math.l2_normalize(tf.random.normal((11, 128)), axis=1))
            model.prototypes_feat.assign(tf.math.l2_normalize(tf.random.normal((11, 32)), axis=1))
        for stage in stages:
            if stage is not None:
                model.current_stage = stage
            probability = model(inputs3, training=False).numpy()
            assert probability.shape == (2, 11) and np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5)
        with tempfile.TemporaryDirectory(prefix=f"v5-exp{index}-") as scratch:
            checkpoint = str(Path(scratch) / "smoke.ckpt")
            model.save_weights(checkpoint)
            model.load_weights(checkpoint)
        print(f"PASS: V5 Exp{index}", flush=True)
    print("V5 smoke test passed. No HDF5 or scientific result was changed.")


if __name__ == "__main__":
    main()
