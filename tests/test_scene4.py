from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.model import MultiHeadMLP
from analyze_scene4_exposure_quality import exposure_metrics, metric_suggestion
from generate_labels import ANNOTATION_FIELDS, FRAME_FIELDS, write_csv
from generate_test_inference_report import build_hdr_cases, build_hdr_report
from label_scene4 import LabelState
from prepare_scene4 import discover_groups, image_number
from prepare_scene4_hdr_review import prepare as prepare_hdr_review
from train_scene4 import evaluate_binary, select_binary_threshold, split_name


class Scene4Test(unittest.TestCase):
    def test_hdr_image_report_uses_target_ev_operating_frame_and_full_bracket(self):
        samples = [
            {
                "scene_id": "Scene4_t085", "filename": "bright.dng",
                "time_index": 85, "exposure_index": 0, "applied_ev": 12.0,
                "predicted_hdr_probability": 0.40, "thumbnail": "bright",
            },
            {
                "scene_id": "Scene4_t085", "filename": "target.dng",
                "time_index": 85, "exposure_index": 1, "applied_ev": 10.0,
                "predicted_hdr_probability": 0.85, "thumbnail": "target",
            },
        ]
        operating = [{
            "scene_id": "Scene4_t085", "time_index": 85,
            "target_hdr_enable": 1, "predicted_hdr_enable": 1, "correct": 1,
            "applied_ev": 10.0, "target_ev": 10.0, "distance_to_target_ev": 0.0,
            "hdr_probability": 0.85, "decision_threshold": 0.75,
        }]
        cases = build_hdr_cases(samples, operating)
        self.assertEqual(cases[0]["operating_filename"], "target.dng")
        self.assertEqual(cases[0]["bracket_count"], 2)
        self.assertEqual(cases[0]["bracket_wrong_count"], 1)
        report = build_hdr_report(cases, Path("report.html"), "candidate")
        self.assertEqual(report["operating_accuracy"], 1.0)
        self.assertEqual(report["all_bracket_accuracy"], 0.5)

    def test_hdr_binary_metrics_include_confusion_matrix(self):
        result = evaluate_binary(
            np.asarray([0.9, 0.8, 0.7, 0.2], np.float32),
            np.asarray([1, 0, 1, 0], np.float32),
            np.ones(4, dtype=bool),
        )
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 0)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_hdr_threshold_is_selected_from_validation_probabilities(self):
        result = select_binary_threshold(
            np.asarray([0.60, 0.61, 0.79, 0.80], np.float32),
            np.asarray([0, 0, 1, 1], np.float32),
        )
        self.assertAlmostEqual(result["threshold"], 0.70, places=5)
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_hdr_review_preserves_sdr_and_validates_benefit_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "source", root / "hdr_review"
            frames = []
            for exposure_index, filename in enumerate(("bright.dng", "dark.dng")):
                frames.append({
                    "scene_id": "Scene4_t000", "filename": filename,
                    "frame_index": 0, "exposure_time_us": 1000 / (exposure_index + 1),
                    "sensor_gain_code": 32, "isp_gain_code": 1024,
                    "hdr_mode": "SDR", "time_index": 0,
                    "exposure_index": exposure_index, "iso": 100, "aperture": 14,
                })
            annotation = {field: "" for field in ANNOTATION_FIELDS}
            annotation.update({
                "scene_id": "Scene4_t000", "preferred_filename": "bright.dng",
                "acceptable_min_filename": "dark.dng",
                "acceptable_max_filename": "bright.dng",
                "label_confidence": "0.8", "label_source": "human_review",
            })
            write_csv(source / "frame_metadata.csv", FRAME_FIELDS, frames)
            write_csv(source / "scene_annotations.csv", ANNOTATION_FIELDS, [annotation])
            prepare_hdr_review(source, output)
            state = LabelState(output, root / "contacts")
            self.assertEqual(state.payload()["hdr_reviewed_count"], 0)
            state.save_hdr({
                "scene_id": "Scene4_t000", "hdr_benefit_if_static": "0.75",
                "hdr_enable_if_static": "1", "hdr_ratio_class": "",
                "hdr_anchor_filename": "", "hdr_label_confidence": 0.9,
                "hdr_notes": "recoverable highlights",
            })
            payload = state.payload()
            self.assertEqual(payload["hdr_reviewed_count"], 1)
            self.assertEqual(payload["scenes"][0]["annotation"]["preferred_filename"],
                             "bright.dng")
            with self.assertRaisesRegex(ValueError, "enable 應為 Off"):
                state.save_hdr({
                    "scene_id": "Scene4_t000", "hdr_benefit_if_static": "0.25",
                    "hdr_enable_if_static": "1", "hdr_ratio_class": "",
                    "hdr_anchor_filename": "", "hdr_label_confidence": 0.8,
                    "hdr_notes": "",
                })

    def test_exposure_metrics_reject_clipped_and_dark_extremes(self):
        bright = np.full((32, 32, 3), 255, np.uint8)
        gradient = np.tile(
            np.linspace(8, 247, 32, dtype=np.uint8)[None, :, None], (32, 1, 3)
        )
        dark = np.zeros((32, 32, 3), np.uint8)
        measured = []
        for index, image in enumerate((bright, gradient, dark)):
            measured.append({
                "filename": f"e{index}.dng",
                **exposure_metrics(image, 0.03, 250, 2.0, 1.0),
            })
        preferred, acceptable_dark, acceptable_bright = metric_suggestion(
            measured, score_margin=0.08, max_saturated_ratio=0.08,
            max_dark_ratio=0.45,
        )
        self.assertEqual(preferred, 1)
        self.assertEqual(acceptable_dark, 1)
        self.assertEqual(acceptable_bright, 1)

    def test_scene4_filename_grouping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for number in range(2006, 2036):
                (root / f"1P0A{number}.dng").touch()
            groups = discover_groups(root)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0].name, "1P0A2006.dng")
        self.assertEqual(groups[1][-1].name, "1P0A2035.dng")
        self.assertEqual(image_number(groups[1][0]), 2021)

    def test_unsupervised_heads_are_frozen_by_masks(self):
        model = MultiHeadMLP(input_dim=4, hidden_dim=5, ratio_count=3, seed=7)
        original_hdr = model.params["w_hdr"].copy()
        original_ratio = model.params["w_ratio"].copy()
        rng = np.random.default_rng(3)
        x = rng.random((8, 4), dtype=np.float32)
        zeros = np.zeros(8, np.float32)
        model.train(
            x, np.linspace(0, 1, 8, dtype=np.float32), zeros,
            np.zeros(8, np.int64), np.ones(8, np.float32), epochs=3, lr=0.01,
            hdr_mask=zeros, ratio_mask=zeros,
        )
        np.testing.assert_array_equal(model.params["w_hdr"], original_hdr)
        np.testing.assert_array_equal(model.params["w_ratio"], original_ratio)

    def test_unsupervised_ev_head_is_frozen_by_mask(self):
        model = MultiHeadMLP(input_dim=4, hidden_dim=5, ratio_count=3, seed=7)
        original_ev = model.params["w_ev"].copy()
        original_bias = model.params["b_ev"].copy()
        rng = np.random.default_rng(9)
        x = rng.random((8, 4), dtype=np.float32)
        zeros = np.zeros(8, np.float32)
        model.train(
            x, np.linspace(-2, 2, 8, dtype=np.float32), zeros,
            np.zeros(8, np.int64), np.ones(8, np.float32), epochs=3, lr=0.01,
            ev_mask=zeros, hdr_mask=zeros, ratio_mask=zeros,
            confidence_mask=zeros,
        )
        np.testing.assert_array_equal(model.params["w_ev"], original_ev)
        np.testing.assert_array_equal(model.params["b_ev"], original_bias)
        result = model.loss_components(
            x, np.linspace(-2, 2, 8, dtype=np.float32), zeros,
            np.zeros(8, np.int64), np.ones(8, np.float32), ev_mask=zeros,
            hdr_mask=zeros, ratio_mask=zeros, confidence_mask=zeros,
        )
        self.assertEqual(result["ev_mse"], 0.0)
        self.assertEqual(result["total"], 0.0)

    def test_loss_components_match_masked_scene4_objective(self):
        model = MultiHeadMLP(input_dim=4, hidden_dim=5, ratio_count=3, seed=7)
        rng = np.random.default_rng(4)
        x = rng.random((6, 4), dtype=np.float32)
        zeros = np.zeros(6, np.float32)
        result = model.loss_components(
            x, np.linspace(0, 1, 6, dtype=np.float32), zeros,
            np.zeros(6, np.int64), np.full(6, 0.8, np.float32),
            hdr_mask=zeros, ratio_mask=zeros,
        )
        self.assertEqual(result["hdr_bce"], 0.0)
        self.assertEqual(result["ratio_ce"], 0.0)
        self.assertAlmostEqual(
            result["total"], result["ev_mse"] + 0.1 * result["confidence_bce"]
        )

    def test_time_split_boundaries(self):
        self.assertEqual(split_name(0, 70, 85), "train")
        self.assertEqual(split_name(69, 70, 85), "train")
        self.assertEqual(split_name(70, 70, 85), "validation")
        self.assertEqual(split_name(84, 70, 85), "validation")
        self.assertEqual(split_name(85, 70, 85), "test")
        self.assertEqual(split_name(99, 70, 85), "test")


if __name__ == "__main__":
    unittest.main()
