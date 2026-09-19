from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.controller import DelayAwareController
from ai_ae.io import calculate_log_exposure, load_camera_status, load_scene_labels, read_pgm


class CoreTest(unittest.TestCase):
    def test_pgm(self):
        image, meta = read_pgm(ROOT / "input/pattern_1/00000020_bayer.pgm")
        self.assertEqual(image.shape, (1080, 1920))
        self.assertEqual(meta["max_value"], "1023")
        self.assertEqual(meta["SAMPLE_MODE"], "BAYER_B")

    def test_examples(self):
        statuses = load_camera_status(ROOT / "examples/camera_status_example.txt")
        self.assertEqual(len(statuses), 3)
        self.assertAlmostEqual(calculate_log_exposure(statuses[0], 100, 32, 1024), 0.0)
        self.assertAlmostEqual(calculate_log_exposure(statuses[2], 100, 32, 1024),
                               np.log2(5280.0))
        self.assertIn("pattern_1", load_scene_labels(ROOT / "examples/ae_output_example.txt"))

    def test_controller_uses_future_queue(self):
        controller = DelayAwareController(1.0, 0.65, 0.35)
        out = controller.step(target_ev=4.0, hdr_benefit=0.8, hdr_ratio=4,
                              pending_queue=[1.0, 2.0])
        self.assertAlmostEqual(out.exposure_command_ev, 3.0)
        self.assertTrue(out.hdr_enabled)


if __name__ == "__main__":
    unittest.main()
