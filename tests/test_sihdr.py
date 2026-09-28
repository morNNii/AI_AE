from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ae.sihdr import (
    RawFrameMetadata,
    calibrate_reference_luma,
    choose_exposure_indices,
    discover_sihdr_stacks,
    hdr_benefit_suggestion,
    merge_exposure_stack,
    render_exposure,
)
from build_sihdr_candidate import hdr_class, mask_unreviewed_hdr, stratified_assignments


class SihdrTest(unittest.TestCase):
    def test_unreviewed_hdr_heuristic_is_cleared_and_masked(self):
        annotation = {
            "scene_id": "SIHDR_001", "hdr_reviewed": "",
            "hdr_benefit_if_static": "0.75", "hdr_enable_if_static": "1",
            "hdr_ratio_class": "4", "hdr_anchor_filename": "frame.jpg",
            "hdr_label_confidence": "0.8", "hdr_notes": "heuristic",
        }
        sanitized = mask_unreviewed_hdr(annotation)
        self.assertEqual(hdr_class(annotation), "unknown")
        for field in (
            "hdr_benefit_if_static", "hdr_enable_if_static", "hdr_ratio_class",
            "hdr_anchor_filename", "hdr_label_confidence", "hdr_notes",
        ):
            self.assertEqual(sanitized[field], "")

    def test_scene_split_is_exact_deterministic_and_disjoint(self):
        annotations = []
        for index in range(20):
            annotations.append({
                "scene_id": f"SIHDR_{index:03d}",
                "preferred_filename": f"SIHDR_{index:03d}_e{5 + index % 6:02d}.jpg",
                "hdr_enable_if_static": "" if index % 5 == 0 else str(index % 2),
            })
        first = stratified_assignments(annotations, seed=42)
        second = stratified_assignments(annotations, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["scene_id"] for row in first}), 20)
        counts = {name: sum(row["split"] == name for row in first)
                  for name in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 14, "validation": 3, "test": 3})

    def test_discovers_five_frame_scene_stacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for scene in ("001", "003"):
                directory = root / "raw" / scene
                directory.mkdir(parents=True)
                for index in range(5):
                    (directory / f"frame{index}.CR2").touch()
            stacks = discover_sihdr_stacks(root)
        self.assertEqual(list(stacks), ["001", "003"])
        self.assertTrue(all(len(frames) == 5 for frames in stacks.values()))

    def test_radiance_merge_recovers_linear_signal_rate(self):
        radiance = np.full((12, 16), 0.2, np.float32)
        exposure_seconds = (0.25, 0.5, 1.0)
        signals = [radiance * exposure for exposure in exposure_seconds]
        metadata = [
            RawFrameMetadata(Path(f"e{index}.CR2"), exposure, 100, 14, "camera")
            for index, exposure in enumerate(exposure_seconds)
        ]
        merged, report = merge_exposure_stack(signals, metadata)
        np.testing.assert_allclose(merged, radiance, rtol=1e-5, atol=1e-6)
        self.assertEqual(report["covered_fraction"], 1.0)

    def test_reference_calibration_recovers_scale(self):
        reference = np.linspace(0.01, 1.0, 400, dtype=np.float32).reshape(20, 20)
        calibrated, scale = calibrate_reference_luma(reference, reference * 3.0)
        self.assertAlmostEqual(scale, 3.0, places=5)
        np.testing.assert_allclose(calibrated, reference * 3.0, rtol=1e-5)

    def test_exposure_selection_and_hdr_suggestion_are_deterministic(self):
        metrics = [
            {"score": 0.1, "saturated_ratio": 0.20, "dark_ratio": 0.0},
            {"score": 0.8, "saturated_ratio": 0.01, "dark_ratio": 0.02},
            {"score": 0.72, "saturated_ratio": 0.0, "dark_ratio": 0.08},
        ]
        preferred, acceptable_min, acceptable_max = choose_exposure_indices(metrics)
        self.assertEqual(preferred, 1)
        self.assertEqual((acceptable_min, acceptable_max), (2, 1))

        radiance = np.concatenate([
            np.full((10, 5), 4.0, np.float32),
            np.full((10, 5), 0.004, np.float32),
        ], axis=1)
        rendered = [render_exposure(radiance, shutter) for shutter in (8.0, 1.0, 0.125)]
        suggestion = hdr_benefit_suggestion(rendered, preferred_index=1)
        self.assertEqual(suggestion["hdr_enable_if_static"], 1)
        self.assertGreater(suggestion["recoverable_total_ratio"], 0.03)


if __name__ == "__main__":
    unittest.main()
