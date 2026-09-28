#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ae.dng import read_dng_metadata, read_dng_preview_rgb, rgb_to_luma
from ai_ae.features import extract_features
from ai_ae.io import CameraStatus, calculate_log_exposure
from generate_labels import ANNOTATION_FIELDS, FRAME_FIELDS, read_csv, write_csv


EXPOSURES_PER_TIME = 15
EXPECTED_EXPOSURES_SECONDS = (
    15.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1 / 2, 1 / 4, 1 / 8,
    1 / 15, 1 / 30, 1 / 60, 1 / 125, 1 / 250, 1 / 500,
)


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def image_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"cannot extract trailing image number from {path.name}")
    return int(match.group(1))


def discover_groups(input_dir: Path) -> list[list[Path]]:
    files = sorted(input_dir.glob("*.dng"), key=image_number)
    if not files:
        raise ValueError(f"{input_dir} contains no .dng files")
    numbers = [image_number(path) for path in files]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        raise ValueError("Scene4 DNG filenames are not a consecutive sequence")
    if len(files) % EXPOSURES_PER_TIME:
        raise ValueError(
            f"{len(files)} files cannot be grouped into {EXPOSURES_PER_TIME} exposures"
        )
    return [
        files[start:start + EXPOSURES_PER_TIME]
        for start in range(0, len(files), EXPOSURES_PER_TIME)
    ]


def blank_annotation(scene_id: str) -> dict[str, str]:
    return {
        "scene_id": scene_id,
        "preferred_filename": "",
        "acceptable_min_filename": "",
        "acceptable_max_filename": "",
        "hdr_benefit_if_static": "",
        "hdr_enable_if_static": "",
        "hdr_ratio_class": "",
        "hdr_anchor_filename": "",
        "hdr_reviewed": "",
        "hdr_label_confidence": "",
        "hdr_notes": "",
        "label_confidence": "",
        "label_source": "",
    }


def format_shutter(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"1/{round(1 / seconds)} s"


def make_contact_sheet(scene_id: str, entries: list[tuple[dict, np.ndarray]], path: Path) -> None:
    columns, rows = 5, 3
    tile_width, image_height, caption_height, header_height = 288, 192, 36, 38
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (image_height + caption_height)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 12), f"{scene_id} - choose acceptable dark / preferred / acceptable bright",
              fill=(255, 255, 255), font=font)
    for index, (row, preview) in enumerate(entries):
        x = (index % columns) * tile_width
        y = header_height + (index // columns) * (image_height + caption_height)
        image = Image.fromarray(preview).resize(
            (tile_width, image_height), Image.Resampling.LANCZOS
        )
        sheet.paste(image, (x, y))
        caption = (
            f"e{index:02d} | {format_shutter(float(row['exposure_time_us']) / 1e6)}"
            f" | {row['filename']}"
        )
        draw.rectangle((x, y + image_height, x + tile_width, y + image_height + caption_height),
                       fill=(10, 10, 10))
        draw.text((x + 5, y + image_height + 10), caption, fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88, optimize=True)


def suggest_annotation(entries: list[tuple[dict, np.ndarray]]) -> tuple[str, str, str]:
    """Return a low-confidence starting point that must be reviewed by a person."""
    scores = []
    for _, preview in entries:
        luma = rgb_to_luma(preview) / 255.0
        unsaturated = luma[luma < 0.90]
        outlier_rejected_mean = float(unsaturated.mean()) if unsaturated.size else 1.0
        height, width = luma.shape
        center = luma[
            int(height * 0.05):int(height * 0.95),
            int(width * 0.35):int(width * 0.65),
        ]
        center_mean = float(center.mean())
        clipped = float((luma >= 0.98).mean())
        shadows = float((luma <= 0.02).mean())
        score = abs(math.log2(max(outlier_rejected_mean, 1e-4) / 0.55))
        score += 0.75 * abs(math.log2(max(center_mean, 1e-4) / 0.50))
        score += 0.25 * clipped + 0.25 * shadows
        scores.append(score)
    preferred_index = int(np.argmin(scores))
    bright_index = max(0, preferred_index - 1)
    dark_index = min(len(entries) - 1, preferred_index + 1)
    return (
        entries[preferred_index][0]["filename"],
        entries[dark_index][0]["filename"],
        entries[bright_index][0]["filename"],
    )


def load_existing_annotations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["scene_id"]: row for row in read_csv(path)}


def prepare(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = discover_groups(args.input)
    if len(groups) != 100:
        print(f"WARNING: official Scene4 has 100 time steps; found {len(groups)}")

    existing = load_existing_annotations(args.draft / "scene_annotations.csv")
    annotations = []
    frame_rows = []

    for time_index, group in enumerate(groups):
        scene_id = f"Scene4_t{time_index:03d}"
        annotations.append(existing.get(scene_id, blank_annotation(scene_id)))
        for exposure_index, path in enumerate(group):
            metadata = read_dng_metadata(path)
            expected = EXPECTED_EXPOSURES_SECONDS[exposure_index]
            actual = metadata.exposure_time_us / 1_000_000.0
            if not math.isclose(actual, expected, rel_tol=0.02, abs_tol=1e-6):
                raise ValueError(
                    f"{path.name}: exposure {actual:g}s, expected {expected:g}s at e{exposure_index:02d}"
                )
            if not math.isclose(metadata.iso, 100.0) or not math.isclose(metadata.aperture, 14.0):
                raise ValueError(
                    f"{path.name}: expected ISO 100 and f/14, got ISO {metadata.iso:g}, "
                    f"f/{metadata.aperture:g}"
                )
            frame_rows.append({
                "scene_id": scene_id,
                "filename": path.name,
                "frame_index": time_index,
                "exposure_time_us": f"{metadata.exposure_time_us:.6f}",
                "sensor_gain_code": config["sensor_gain_unity_code"],
                "isp_gain_code": config["isp_gain_unity_code"],
                "hdr_mode": "SDR",
                "time_index": time_index,
                "exposure_index": exposure_index,
                "iso": f"{metadata.iso:g}",
                "aperture": f"{metadata.aperture:g}",
            })

    args.draft.mkdir(parents=True, exist_ok=True)
    write_csv(args.draft / "frame_metadata.csv", FRAME_FIELDS, frame_rows)
    write_csv(args.draft / "scene_annotations.csv", ANNOTATION_FIELDS, annotations)

    row_lookup = {(row["scene_id"], row["filename"]): row for row in frame_rows}
    annotation_lookup = {row["scene_id"]: row for row in annotations}
    feature_rows: list[np.ndarray] = []
    feature_scene_ids: list[str] = []
    feature_filenames: list[str] = []
    feature_time_indices: list[int] = []
    feature_exposure_indices: list[int] = []
    feature_applied_ev: list[float] = []

    for time_index, group in enumerate(groups):
        scene_id = f"Scene4_t{time_index:03d}"
        entries: list[tuple[dict, np.ndarray]] = []
        for exposure_index, path in enumerate(group):
            row = row_lookup[(scene_id, path.name)]
            preview = read_dng_preview_rgb(path, args.preview_series)
            entries.append((row, preview))
            if not args.skip_features:
                status = CameraStatus(
                    filename=path.name,
                    scene_id=scene_id,
                    frame_index=time_index,
                    exposure_time_us=float(row["exposure_time_us"]),
                    sensor_gain_code=float(row["sensor_gain_code"]),
                    isp_gain_code=float(row["isp_gain_code"]),
                    hdr_mode="SDR",
                )
                applied_ev = calculate_log_exposure(
                    status,
                    float(config["reference_exposure_time_us"]),
                    float(config["sensor_gain_unity_code"]),
                    float(config["isp_gain_unity_code"]),
                )
                feature_rows.append(extract_features(
                    rgb_to_luma(preview), 255.0, status,
                    int(config["histogram_bins"]), int(config["grid_rows"]),
                    int(config["grid_cols"]),
                    [applied_ev] * int(config["exposure_delay_frames"]), applied_ev,
                    float(config["sensor_gain_unity_code"]),
                    float(config["isp_gain_unity_code"]),
                ))
                feature_scene_ids.append(scene_id)
                feature_filenames.append(path.name)
                feature_time_indices.append(time_index)
                feature_exposure_indices.append(exposure_index)
                feature_applied_ev.append(applied_ev)

        if not args.skip_contact_sheets:
            make_contact_sheet(
                scene_id, entries, args.artifacts / "contact_sheets" / f"{scene_id}.jpg"
            )
        annotation = annotation_lookup[scene_id]
        if (not annotation.get("preferred_filename", "").strip() or
                annotation.get("label_source") == "auto_preview_heuristic"):
            preferred, darker, brighter = suggest_annotation(entries)
            annotation.update({
                "preferred_filename": preferred,
                "acceptable_min_filename": darker,
                "acceptable_max_filename": brighter,
                "label_confidence": "0.25",
                "label_source": "auto_preview_heuristic",
            })
        if (time_index + 1) % 10 == 0 or time_index + 1 == len(groups):
            print(f"prepared {time_index + 1}/{len(groups)} time steps", flush=True)

    write_csv(args.draft / "scene_annotations.csv", ANNOTATION_FIELDS, annotations)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if not args.skip_features:
        np.savez_compressed(
            args.artifacts / "features.npz",
            features=np.stack(feature_rows).astype(np.float32),
            scene_ids=np.asarray(feature_scene_ids),
            filenames=np.asarray(feature_filenames),
            time_indices=np.asarray(feature_time_indices, np.int16),
            exposure_indices=np.asarray(feature_exposure_indices, np.int8),
            applied_ev=np.asarray(feature_applied_ev, np.float32),
            input_domain=np.asarray(f"DNG embedded preview series {args.preview_series}"),
        )

    feature_path = args.artifacts / "features.npz"
    contact_path = args.artifacts / "contact_sheets"
    reviewed_count = sum(row.get("label_source") == "human_review" for row in annotations)
    report = {
        "status": ("READY_FOR_LABEL_BUILD" if reviewed_count == len(groups)
                   else "WAITING_FOR_HUMAN_LABELS"),
        "input": portable_path(args.input),
        "dng_count": len(frame_rows),
        "time_step_count": len(groups),
        "exposures_per_time_step": EXPOSURES_PER_TIME,
        "human_reviewed_time_steps": reviewed_count,
        "preview_series": args.preview_series,
        "feature_cache": portable_path(feature_path) if feature_path.exists() else None,
        "contact_sheets": portable_path(contact_path) if contact_path.exists() else None,
        "annotation_csv": portable_path(args.draft / "scene_annotations.csv"),
        "note": ("All time steps are human-reviewed."
                 if reviewed_count == len(groups)
                 else "Auto suggestions are low-confidence starting points and must be reviewed."),
    }
    (args.artifacts / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Scene4 DNG files, contact sheets, draft labels, and feature cache."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "input/Scene4")
    parser.add_argument("--draft", type=Path, default=ROOT / "labels/scene4_draft")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/scene4_labeling")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene4.json")
    parser.add_argument("--preview-series", type=int, default=2)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
