import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_experiment_contract(self):
        contract = json.loads(
            (ROOT / "configs/experiment_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(contract["upsilon_features"]), 16)
        self.assertEqual(len(contract["surveys"]["crts"]["class_names"]), 11)
        self.assertEqual(len(contract["surveys"]["ogle"]["class_names"]), 15)
        self.assertEqual(len(contract["surveys"]["ogle"]["major_class_names"]), 6)

    def test_required_entry_points_exist(self):
        expected = (
            "experiments/crts/1_prepare_crts_tdmn.py",
            "experiments/crts/2_train_crts_tdmn.py",
            "experiments/crts/3_extract_upsilon_features.py",
            "experiments/crts/4_train_upsilon_rf.py",
            "experiments/ogle/1_prepare_ogle_tdmn.py",
            "experiments/ogle/2_train_ogle_tdmn.py",
            "experiments/ogle/3_extract_upsilon_features.py",
            "experiments/ogle/4_train_upsilon_rf.py",
            "tools/compare_predictions.py",
            "tools/validate_tdmn_dataset.py",
        )
        for relative in expected:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_no_private_machine_paths(self):
        forbidden = (
            "/Users/" + "lqy/",
            "/home/" + "kongxiao/",
            "data45/kx/" + "lqy",
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"{value} found in {path}")

    def test_prediction_contract_is_saved(self):
        for relative in (
            "experiments/crts/2_train_crts_tdmn.py",
            "experiments/ogle/2_train_ogle_tdmn.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("test_predictions.npz", text)
            self.assertIn("object_id=object_id_test", text)
            self.assertEqual(
                text.count("final_probabilities = selected_model.predict(test_ds)"), 1
            )


if __name__ == "__main__":
    unittest.main()
