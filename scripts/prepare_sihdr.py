#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.features import extract_features
from ai_ae.io import CameraStatus, calculate_log_exposure
from ai_ae.sihdr import (
    SCENE4_SHUTTERS_SECONDS,
    calibrate_reference_luma,
    choose_exposure_indices,
    discover_sihdr_stacks,
    exposure_quality,
    hdr_benefit_suggestion,
    merge_exposure_stack,
    preview_rgb,
    read_cr2_linear_luma,
    read_cr2_metadata,
    read_exr_rgb,
    render_exposure,
    rgb_to_linear_luma,
)


FRAME_FIELDS = [
    "scene_id", "filename", "frame_index", "exposure_time_us",
    "sensor_gain_code", "isp_gain_code", "hdr_mode", "time_index",
    "exposure_index", "iso", "aperture",
]
ANNOTATION_FIELDS = [
    "scene_id", "preferred_filename", "acceptable_min_filename",
    "acceptable_max_filename", "hdr_benefit_if_static",
    "hdr_enable_if_static", "hdr_ratio_class", "hdr_anchor_filename",
    "hdr_reviewed", "hdr_label_confidence", "hdr_notes",
    "label_confidence", "label_source",
]


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_shutter(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"1/{round(1 / seconds):g} s"


def create_contact_sheet(scene_id: str, frames: list[dict], images: list[np.ndarray],
                         output: Path) -> None:
    tile_width, tile_height, caption_height = 300, 200, 40
    canvas = Image.new("RGB", (tile_width * 5, (tile_height + caption_height) * 3),
                       "#07111f")
    draw = ImageDraw.Draw(canvas)
    for index, (frame, image) in enumerate(zip(frames, images)):
        tile = Image.fromarray(preview_rgb(image), "RGB")
        tile.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % 5) * tile_width
        y = (index // 5) * (tile_height + caption_height)
        canvas.paste(tile, (x + (tile_width - tile.width) // 2,
                            y + (tile_height - tile.height) // 2))
        color = "#8fd7ff" if frame["is_preferred"] else "#d9e6ee"
        draw.text((x + 8, y + tile_height + 4),
                  f"e{index:02d} | {format_shutter(frame['shutter_seconds'])}",
                  fill=color)
        draw.text((x + 8, y + tile_height + 20),
                  f"score {frame['score']:.3f}", fill="#8ca5b2")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=88, optimize=True)


def audit_dataset(dataset_root: Path, artifacts: Path) -> tuple[list[dict], list[dict]]:
    stacks = discover_sihdr_stacks(dataset_root)
    source_rows = []
    scene_rows = []
    reference_root = dataset_root / "reference"
    for time_index, (key, paths) in enumerate(stacks.items()):
        metadata = [read_cr2_metadata(path) for path in paths]
        ordered = sorted(metadata, key=lambda item: item.exposure_seconds, reverse=True)
        reference = reference_root / f"{key}.exr"
        scene_rows.append({
            "scene_key": key,
            "scene_id": f"SIHDR_{key}",
            "time_index": time_index,
            "raw_count": len(ordered),
            "reference_path": portable_path(reference),
            "reference_present": int(reference.exists()),
            "camera_models": ";".join(sorted({item.camera_model for item in ordered})),
            "iso_values": ";".join(f"{value:g}" for value in sorted({item.iso for item in ordered})),
            "aperture_values": ";".join(
                f"{value:g}" for value in sorted({item.aperture for item in ordered})
            ),
            "min_exposure_seconds": min(item.exposure_seconds for item in ordered),
            "max_exposure_seconds": max(item.exposure_seconds for item in ordered),
        })
        for raw_index, item in enumerate(ordered):
            source_rows.append({
                "scene_key": key,
                "scene_id": f"SIHDR_{key}",
                "time_index": time_index,
                "raw_exposure_index": raw_index,
                "raw_path": portable_path(item.path),
                "filename": item.path.name,
                "exposure_seconds": item.exposure_seconds,
                "iso": item.iso,
                "aperture": item.aperture,
                "camera_model": item.camera_model,
            })
    write_csv(artifacts / "source_raw_manifest.csv", list(source_rows[0]), source_rows)
    write_csv(artifacts / "scene_audit.csv", list(scene_rows[0]), scene_rows)
    return source_rows, scene_rows


def prepare(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.artifacts.mkdir(parents=True, exist_ok=True)
    source_rows, audited_scenes = audit_dataset(args.input, args.artifacts)
    report = {
        "status": "AUDITED" if args.audit_only else "PREPARING",
        "dataset": "SI-HDR",
        "dataset_root": portable_path(args.input),
        "scene_count": len(audited_scenes),
        "raw_frame_count": len(source_rows),
        "raw_frames_per_scene": dict(Counter(row["raw_count"] for row in audited_scenes)),
        "reference_count": sum(row["reference_present"] for row in audited_scenes),
        "missing_reference_count": sum(not row["reference_present"] for row in audited_scenes),
        "target_shutters_seconds": list(SCENE4_SHUTTERS_SECONDS),
        "simulation": {
            "output_kind": "simulated linear camera-luma for AE statistics",
            "target_iso": 100,
            "target_aperture": args.target_aperture,
            "stored_full_resolution_bayer": False,
            "warning": (
                "Rendered frames are synthetic and must not be mixed with real Scene4 "
                "DNG inputs without domain-aware validation."
            ),
        },
    }
    if args.audit_only:
        (args.artifacts / "prepare_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    selected = audited_scenes
    if args.scene_limit is not None:
        selected = selected[:args.scene_limit]
    if args.radiance_source == "reference":
        missing = [row["scene_key"] for row in selected if not row["reference_present"]]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} EXR references below {args.input / 'reference'}; "
                "download and extract official reference.zip first"
            )

    source_by_scene: dict[str, list[dict]] = {}
    for row in source_rows:
        source_by_scene.setdefault(row["scene_key"], []).append(row)

    features: list[np.ndarray] = []
    feature_scene_ids: list[str] = []
    feature_filenames: list[str] = []
    feature_time_indices: list[int] = []
    feature_exposure_indices: list[int] = []
    feature_applied_ev: list[float] = []
    frame_rows: list[dict] = []
    annotation_rows: list[dict] = []
    metric_rows: list[dict] = []
    simulation_rows: list[dict] = []
    scene_reports: list[dict] = []
    previews = args.artifacts / "previews"
    contacts = args.artifacts / "contact_sheets"
    previews.mkdir(parents=True, exist_ok=True)

    for output_time_index, scene in enumerate(selected):
        key = scene["scene_key"]
        scene_id = scene["scene_id"]
        raw_rows = sorted(
            source_by_scene[key], key=lambda row: row["exposure_seconds"], reverse=True
        )
        metadata = [
            read_cr2_metadata(resolve_project_path(row["raw_path"]))
            for row in raw_rows
        ]
        signals = []
        sensor_info = None
        for item in metadata:
            signal, current_info = read_cr2_linear_luma(
                item.path, width=args.width, height=args.height
            )
            signals.append(signal)
            sensor_info = sensor_info or current_info
        raw_radiance, merge_info = merge_exposure_stack(
            signals, metadata, target_aperture=args.target_aperture
        )
        reference_scale = None
        reference_path = args.input / "reference" / f"{key}.exr"
        if args.radiance_source == "reference":
            reference_rgb = read_exr_rgb(reference_path)
            reference_luma = rgb_to_linear_luma(reference_rgb)
            radiance, reference_scale = calibrate_reference_luma(
                reference_luma, raw_radiance
            )
        else:
            radiance = raw_radiance

        rendered = [
            render_exposure(radiance, shutter)
            for shutter in SCENE4_SHUTTERS_SECONDS
        ]
        metrics = [exposure_quality(image) for image in rendered]
        preferred, acceptable_min, acceptable_max = choose_exposure_indices(metrics)
        hdr = hdr_benefit_suggestion(rendered, preferred)
        current_frames = []
        for exposure_index, (shutter, image, quality) in enumerate(zip(
                SCENE4_SHUTTERS_SECONDS, rendered, metrics)):
            filename = f"{scene_id}_e{exposure_index:02d}.jpg"
            preview_path = previews / filename
            Image.fromarray(preview_rgb(image), "RGB").save(
                preview_path, "JPEG", quality=args.jpeg_quality, optimize=True
            )
            frame_index = output_time_index * len(SCENE4_SHUTTERS_SECONDS) + exposure_index
            status = CameraStatus(
                filename=filename,
                scene_id=scene_id,
                frame_index=frame_index,
                exposure_time_us=shutter * 1_000_000.0,
                sensor_gain_code=float(config["sensor_gain_unity_code"]),
                isp_gain_code=float(config["isp_gain_unity_code"]),
                hdr_mode="SDR",
            )
            applied_ev = calculate_log_exposure(
                status,
                float(config["reference_exposure_time_us"]),
                float(config["sensor_gain_unity_code"]),
                float(config["isp_gain_unity_code"]),
            )
            features.append(extract_features(
                image, 1.0, status,
                int(config["histogram_bins"]),
                int(config["grid_rows"]),
                int(config["grid_cols"]),
                pending_ev=[applied_ev] * int(config["exposure_delay_frames"]),
                applied_ev=applied_ev,
                sensor_gain_unity_code=float(config["sensor_gain_unity_code"]),
                isp_gain_unity_code=float(config["isp_gain_unity_code"]),
            ))
            feature_scene_ids.append(scene_id)
            feature_filenames.append(filename)
            feature_time_indices.append(output_time_index)
            feature_exposure_indices.append(exposure_index)
            feature_applied_ev.append(applied_ev)
            frame = {
                "scene_id": scene_id,
                "filename": filename,
                "frame_index": frame_index,
                "exposure_time_us": f"{shutter * 1_000_000.0:.6f}",
                "sensor_gain_code": config["sensor_gain_unity_code"],
                "isp_gain_code": config["isp_gain_unity_code"],
                "hdr_mode": "SDR",
                "time_index": output_time_index,
                "exposure_index": exposure_index,
                "iso": 100,
                "aperture": args.target_aperture,
            }
            frame_rows.append(frame)
            current_frames.append({
                **frame,
                **quality,
                "shutter_seconds": shutter,
                "is_preferred": exposure_index == preferred,
            })
            metric_rows.append({
                **frame,
                **quality,
                "simulation_domain": args.radiance_source,
            })
            simulation_rows.append({
                "scene_id": scene_id,
                "filename": filename,
                "preview_path": portable_path(preview_path),
                "source_reference": (
                    portable_path(reference_path) if reference_path.exists() else ""
                ),
                "source_raw_stack": ";".join(
                    portable_path(item.path) for item in metadata
                ),
                "is_simulated": 1,
                "simulation_kind": "linear HDR radiance scaled by virtual shutter",
                "radiance_source": args.radiance_source,
            })

        annotation_rows.append({
            "scene_id": scene_id,
            "preferred_filename": current_frames[preferred]["filename"],
            "acceptable_min_filename": current_frames[acceptable_min]["filename"],
            "acceptable_max_filename": current_frames[acceptable_max]["filename"],
            "hdr_benefit_if_static": hdr["hdr_benefit_if_static"],
            "hdr_enable_if_static": (
                "" if hdr["hdr_enable_if_static"] is None
                else hdr["hdr_enable_if_static"]
            ),
            "hdr_ratio_class": "",
            "hdr_anchor_filename": "",
            "hdr_reviewed": 0,
            "hdr_label_confidence": 0.6,
            "hdr_notes": (
                "AUTO from static SI-HDR radiance; requires review before formal training. "
                f"recoverable={hdr['recoverable_total_ratio']:.6f}"
            ),
            "label_confidence": 0.55,
            "label_source": "sihdr_reference_heuristic",
        })
        if not args.skip_contact_sheets:
            create_contact_sheet(
                scene_id, current_frames, rendered, contacts / f"{scene_id}.jpg"
            )
        scene_reports.append({
            "scene_id": scene_id,
            "source_key": key,
            "preferred_exposure_index": preferred,
            "acceptable_min_exposure_index": acceptable_min,
            "acceptable_max_exposure_index": acceptable_max,
            "hdr_suggestion": hdr,
            "raw_merge": merge_info,
            "reference_scale": reference_scale,
            "sensor": sensor_info,
        })
        print(
            f"[{output_time_index + 1}/{len(selected)}] {scene_id}: "
            f"preferred=e{preferred:02d}, HDR={hdr['hdr_enable_if_static']}"
        )

    write_csv(args.draft / "frame_metadata.csv", FRAME_FIELDS, frame_rows)
    write_csv(args.draft / "scene_annotations.csv", ANNOTATION_FIELDS, annotation_rows)
    write_csv(args.artifacts / "exposure_metrics.csv", list(metric_rows[0]), metric_rows)
    write_csv(args.artifacts / "simulation_manifest.csv",
              list(simulation_rows[0]), simulation_rows)
    np.savez_compressed(
        args.artifacts / "features.npz",
        features=np.stack(features).astype(np.float32),
        scene_ids=np.asarray(feature_scene_ids),
        filenames=np.asarray(feature_filenames),
        time_indices=np.asarray(feature_time_indices, np.int16),
        exposure_indices=np.asarray(feature_exposure_indices, np.int8),
        applied_ev=np.asarray(feature_applied_ev, np.float32),
        input_domain=np.asarray(
            "SI-HDR EXR radiance calibrated to Canon CR2; simulated ISO100 f/14 luma"
            if args.radiance_source == "reference"
            else "SI-HDR Canon CR2 stack merge; simulated ISO100 f/14 luma"
        ),
    )
    report.update({
        "status": "READY_FOR_HUMAN_REVIEW",
        "processed_scene_count": len(selected),
        "simulated_frame_count": len(frame_rows),
        "feature_shape": [len(features), int(features[0].size)],
        "radiance_source": args.radiance_source,
        "draft": portable_path(args.draft),
        "feature_cache": portable_path(args.artifacts / "features.npz"),
        "previews": portable_path(previews),
        "contact_sheets": (
            None if args.skip_contact_sheets else portable_path(contacts)
        ),
        "label_status": (
            "heuristic only; review SDR and HDR suggestions before formal mixed-dataset training"
        ),
        "scene_reports": scene_reports,
    })
    (args.artifacts / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit SI-HDR CR2 stacks and simulate the Scene4 15-shutter input pattern "
            "from calibrated HDR radiance."
        )
    )
    parser.add_argument("--input", type=Path, default=ROOT / "input/raw/sihdr")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/sihdr_prepared")
    parser.add_argument("--draft", type=Path, default=ROOT / "labels/sihdr_draft")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sihdr.json")
    parser.add_argument("--radiance-source", choices=("reference", "raw"),
                        default="reference")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=683)
    parser.add_argument("--target-aperture", type=float, default=14.0)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = prepare(parse_args())
    print(json.dumps({key: value for key, value in result.items()
                      if key != "scene_reports"}, indent=2, ensure_ascii=False))
