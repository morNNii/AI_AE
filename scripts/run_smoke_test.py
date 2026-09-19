#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.controller import DelayAwareController
from ai_ae.features import extract_features
from ai_ae.io import (CameraStatus, calculate_log_exposure, load_camera_status,
                      load_scene_labels, read_pgm)
from ai_ae.model import MultiHeadMLP


def main() -> None:
    config = json.loads((ROOT / "configs/smoke_test.json").read_text())
    statuses = load_camera_status(ROOT / "examples/camera_status_example.txt")
    labels = load_scene_labels(ROOT / "examples/ae_output_example.txt")
    ratios = config["ratio_classes"]
    exposure_args = (config["reference_exposure_time_us"],
                     config["sensor_gain_unity_code"], config["isp_gain_unity_code"])
    samples = []
    cache = {}
    for status in statuses:
        image, meta = read_pgm(ROOT / "input/pattern_1" / status.filename)
        cache[status.filename] = (image, meta)
        applied_ev = calculate_log_exposure(status, *exposure_args)
        pending = [applied_ev] * config["exposure_delay_frames"]
        feature = extract_features(image, float(meta["max_value"]), status,
                                   config["histogram_bins"], config["grid_rows"],
                                   config["grid_cols"], pending, applied_ev,
                                   config["sensor_gain_unity_code"], config["isp_gain_unity_code"])
        label = labels[status.scene_id]
        samples.append((status, feature, label))

    x = np.stack([s[1] for s in samples])
    y_ev = np.array([s[2].target_ev for s in samples], np.float32)
    y_hdr = np.array([s[2].hdr_enable for s in samples], np.float32)
    y_ratio = np.array([ratios.index(s[2].hdr_ratio) for s in samples], np.int64)
    y_conf = np.array([s[2].confidence for s in samples], np.float32)
    model = MultiHeadMLP(x.shape[1], config["hidden_dim"], len(ratios), config["seed"])
    loss = model.train(x, y_ev, y_hdr, y_ratio, y_conf,
                       config["epochs"], config["learning_rate"])
    output_dir = ROOT / "outputs/smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.npz")

    predictions = model.predict(x)
    inference_rows = []
    for i, (status, _, _) in enumerate(samples):
        ratio = ratios[int(np.argmax(predictions["ratio_prob"][i]))]
        inference_rows.append({
            "filename": status.filename,
            "target_ev": float(predictions["target_ev"][i]),
            "hdr_benefit": float(predictions["hdr_benefit"][i]),
            "hdr_ratio": ratio,
            "confidence": float(predictions["confidence"][i]),
        })

    # 靜態 sweep lookup simulator：requested EV 使用最接近的實拍 frame。
    ordered = sorted(((calculate_log_exposure(s, *exposure_args), s) for s in statuses),
                     key=lambda item: item[0])
    queue = [ordered[0][0]] * config["exposure_delay_frames"]
    controller = DelayAwareController(config["max_step_ev"], config["hdr_on_threshold"],
                                      config["hdr_off_threshold"])
    rollout = []
    for frame in range(config["rollout_frames"]):
        applied = queue.pop(0)
        nearest_ev, nearest = min(ordered, key=lambda item: abs(item[0] - applied))
        image, meta = cache[nearest.filename]
        runtime_status = CameraStatus(nearest.filename, nearest.scene_id, frame,
                                      nearest.exposure_time_us, nearest.sensor_gain_code,
                                      nearest.isp_gain_code, nearest.hdr_mode)
        feature = extract_features(image, float(meta["max_value"]), runtime_status,
                                   config["histogram_bins"], config["grid_rows"],
                                   config["grid_cols"], queue or [applied], applied,
                                   config["sensor_gain_unity_code"], config["isp_gain_unity_code"])
        pred = model.predict(feature[None, :])
        ratio = ratios[int(np.argmax(pred["ratio_prob"][0]))]
        command = controller.step(float(pred["target_ev"][0]),
                                  float(pred["hdr_benefit"][0]), ratio, queue or [applied])
        queue.append(command.exposure_command_ev)
        rollout.append({
            "frame": frame,
            "applied_ev": applied,
            "lookup_frame": nearest.filename,
            "lookup_error_ev": abs(applied - nearest_ev),
            "predicted_target_ev": float(pred["target_ev"][0]),
            "issued_command_ev": command.exposure_command_ev,
            "command_effective_frame": frame + config["exposure_delay_frames"],
            "hdr_enabled": command.hdr_enabled,
            "hdr_ratio": command.hdr_ratio,
            "pending_queue_after_issue": queue.copy(),
        })

    report = {
        "status": "PASS",
        "purpose": "僅驗證架構與 flow；不代表 AE/HDR 品質",
        "input_count": len(samples),
        "feature_dim": int(x.shape[1]),
        "initial_loss": loss[0],
        "final_loss": loss[-1],
        "inference": inference_rows,
        "rollout": rollout,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
