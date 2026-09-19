import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_labels import ANNOTATION_FIELDS, FRAME_FIELDS, build, initialize, write_csv


class GenerateLabelsTest(unittest.TestCase):
    def test_initialize_and_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pattern = root / "input/pattern_a"
            pattern.mkdir(parents=True)
            (pattern / "frame_000.pgm").touch()
            (pattern / "frame_001.pgm").touch()
            draft = root / "draft"
            initialize(root / "input", draft, "*.pgm")
            write_csv(draft / "frame_metadata.csv", FRAME_FIELDS, [
                {"scene_id": "pattern_a", "filename": "frame_000.pgm", "frame_index": 0,
                 "exposure_time_us": 100, "sensor_gain_code": 32,
                 "isp_gain_code": 1024, "hdr_mode": "SDR"},
                {"scene_id": "pattern_a", "filename": "frame_001.pgm", "frame_index": 1,
                 "exposure_time_us": 200, "sensor_gain_code": 32,
                 "isp_gain_code": 1024, "hdr_mode": "SDR"},
            ])
            write_csv(draft / "scene_annotations.csv", ANNOTATION_FIELDS, [{
                "scene_id": "pattern_a", "preferred_filename": "frame_001.pgm",
                "acceptable_min_filename": "frame_000.pgm",
                "acceptable_max_filename": "frame_001.pgm", "hdr_benefit_if_static": "",
                "hdr_enable_if_static": "", "hdr_ratio_class": "",
                "hdr_anchor_filename": "", "label_confidence": 0.8,
                "label_source": "unit_test",
            }])
            config = root / "config.json"
            config.write_text(json.dumps({"reference_exposure_time_us": 100,
                                          "sensor_gain_unity_code": 32,
                                          "isp_gain_unity_code": 1024}))
            output = root / "generated"
            build(draft, output, config)
            with (output / "training_samples.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["target_log_exposure"]), 1.0)
            self.assertEqual(rows[0]["hdr_label_mask"], "0")


if __name__ == "__main__":
    unittest.main()
