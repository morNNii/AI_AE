#!/usr/bin/env python3
"""為 exposure-sweep patterns 建立 label 模板，並產生模型可讀 labels。"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


FRAME_FIELDS = [
    "scene_id", "filename", "frame_index", "exposure_time_us",
    "sensor_gain_code", "isp_gain_code", "hdr_mode",
]
ANNOTATION_FIELDS = [
    "scene_id", "preferred_filename", "acceptable_min_filename",
    "acceptable_max_filename", "hdr_benefit_if_static",
    "hdr_enable_if_static", "hdr_ratio_class", "hdr_anchor_filename",
    "label_confidence", "label_source",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else 0


def initialize(input_dir: Path, draft_dir: Path, image_glob: str) -> None:
    frame_rows = []
    annotation_rows = []
    patterns = sorted(p for p in input_dir.iterdir() if p.is_dir())
    for pattern in patterns:
        images = sorted(pattern.glob(image_glob))
        if not images:
            continue
        for image in images:
            frame_rows.append({
                "scene_id": pattern.name,
                "filename": image.name,
                "frame_index": frame_index(image),
                "exposure_time_us": "",
                "sensor_gain_code": "",
                "isp_gain_code": "1024",
                "hdr_mode": "SDR",
            })
        annotation_rows.append({
            "scene_id": pattern.name,
            "preferred_filename": "",
            "acceptable_min_filename": "",
            "acceptable_max_filename": "",
            "hdr_benefit_if_static": "",
            "hdr_enable_if_static": "",
            "hdr_ratio_class": "",
            "hdr_anchor_filename": "",
            "label_confidence": "",
            "label_source": "human_review",
        })
    if not frame_rows:
        raise ValueError(f"{input_dir} 中找不到符合 {image_glob} 的 pattern 影像")
    write_csv(draft_dir / "frame_metadata.csv", FRAME_FIELDS, frame_rows)
    write_csv(draft_dir / "scene_annotations.csv", ANNOTATION_FIELDS, annotation_rows)
    print(f"已建立 {len(frame_rows)} 張 frame、{len(annotation_rows)} 個 scene 的待填模板：{draft_dir}")


def required_float(row: dict[str, str], key: str, where: str) -> float:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{where}: {key} 不可留白")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{where}: {key} 必須大於 0")
    return number


def optional_float(value: str) -> float | None:
    return None if not value.strip() else float(value)


def build(draft_dir: Path, output_dir: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference_time = float(config["reference_exposure_time_us"])
    sensor_unity = float(config["sensor_gain_unity_code"])
    isp_unity = float(config["isp_gain_unity_code"])
    frames = read_csv(draft_dir / "frame_metadata.csv")
    annotations = read_csv(draft_dir / "scene_annotations.csv")

    frame_lookup: dict[tuple[str, str], dict] = {}
    generated_frames = []
    for row in frames:
        where = f"{row.get('scene_id')}/{row.get('filename')}"
        time_us = required_float(row, "exposure_time_us", where)
        sensor_code = required_float(row, "sensor_gain_code", where)
        isp_code = required_float(row, "isp_gain_code", where)
        product = time_us * (sensor_code / sensor_unity) * (isp_code / isp_unity)
        log_exposure = float(np.log2(product / reference_time))
        out = dict(row)
        out.update(exposure_product=product, applied_log_exposure=log_exposure)
        key = (row["scene_id"], row["filename"])
        if key in frame_lookup:
            raise ValueError(f"重複 frame：{key}")
        frame_lookup[key] = out
        generated_frames.append(out)

    scenes = []
    training = []
    seen_scenes = set()
    for ann in annotations:
        scene = ann["scene_id"].strip()
        if not scene or scene in seen_scenes:
            raise ValueError(f"scene_id 空白或重複：{scene!r}")
        seen_scenes.add(scene)

        def exposure_for(field: str, required: bool = True) -> float | None:
            filename = ann.get(field, "").strip()
            if not filename and not required:
                return None
            if not filename:
                raise ValueError(f"{scene}: {field} 不可留白")
            key = (scene, filename)
            if key not in frame_lookup:
                raise ValueError(f"{scene}: {field} 找不到 frame {filename}")
            return float(frame_lookup[key]["applied_log_exposure"])

        target = exposure_for("preferred_filename")
        acceptable_a = exposure_for("acceptable_min_filename")
        acceptable_b = exposure_for("acceptable_max_filename")
        acceptable_min, acceptable_max = sorted((acceptable_a, acceptable_b))
        if not acceptable_min <= target <= acceptable_max:
            raise ValueError(f"{scene}: preferred frame 必須落在 acceptable exposure 區間")

        hdr_benefit = optional_float(ann.get("hdr_benefit_if_static", ""))
        hdr_enable = optional_float(ann.get("hdr_enable_if_static", ""))
        hdr_ratio = optional_float(ann.get("hdr_ratio_class", ""))
        hdr_anchor = exposure_for("hdr_anchor_filename", required=False)
        confidence = optional_float(ann.get("label_confidence", ""))
        if hdr_benefit is not None and not 0 <= hdr_benefit <= 1:
            raise ValueError(f"{scene}: hdr_benefit_if_static 必須在 0~1")
        if hdr_enable is not None and hdr_enable not in (0, 1):
            raise ValueError(f"{scene}: hdr_enable_if_static 必須是 0 或 1")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError(f"{scene}: label_confidence 必須在 0~1")

        scene_row = {
            "scene_id": scene,
            "target_log_exposure": target,
            "acceptable_min_log_exposure": acceptable_min,
            "acceptable_max_log_exposure": acceptable_max,
            "hdr_benefit_if_static": "" if hdr_benefit is None else hdr_benefit,
            "hdr_enable_if_static": "" if hdr_enable is None else int(hdr_enable),
            "hdr_ratio_class": "" if hdr_ratio is None else int(hdr_ratio),
            "hdr_anchor_log_exposure": "" if hdr_anchor is None else hdr_anchor,
            "label_confidence": "" if confidence is None else confidence,
            "label_source": ann.get("label_source", "").strip(),
            "hdr_label_mask": int(hdr_enable is not None),
            "hdr_ratio_mask": int(hdr_ratio is not None),
        }
        scenes.append(scene_row)
        for frame in generated_frames:
            if frame["scene_id"] == scene:
                training.append({**frame, **scene_row})

    frame_fields = FRAME_FIELDS + ["exposure_product", "applied_log_exposure"]
    scene_fields = list(scenes[0].keys())
    training_fields = list(training[0].keys())
    write_csv(output_dir / "frames.csv", frame_fields, generated_frames)
    write_csv(output_dir / "scenes.csv", scene_fields, scenes)
    write_csv(output_dir / "training_samples.csv", training_fields, training)
    print(f"PASS：產生 {len(scenes)} 個 scene、{len(training)} 筆 training samples：{output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="掃描 pattern 並建立人工待填模板")
    init.add_argument("--input", type=Path, default=Path("input"))
    init.add_argument("--draft", type=Path, default=Path("labels/draft"))
    init.add_argument("--glob", default="*.pgm")
    build_parser = sub.add_parser("build", help="驗證人工資料並產生訓練 labels")
    build_parser.add_argument("--draft", type=Path, default=Path("labels/draft"))
    build_parser.add_argument("--output", type=Path, default=Path("labels/generated"))
    build_parser.add_argument("--config", type=Path, default=Path("configs/smoke_test.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "init":
        initialize(args.input, args.draft, args.glob)
    else:
        build(args.draft, args.output, args.config)
