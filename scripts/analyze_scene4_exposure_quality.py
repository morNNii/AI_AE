#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.dng import read_dng_preview_rgb, rgb_to_luma
from generate_labels import ANNOTATION_FIELDS, read_csv, write_csv


METRIC_FIELDS = [
    "scene_id", "filename", "exposure_index", "entropy", "saturated_ratio",
    "dark_ratio", "mean_luma", "quality_score",
]
SUGGESTION_FIELDS = [
    "scene_id",
    "original_preferred_filename", "original_preferred_index",
    "original_acceptable_min_filename", "original_acceptable_min_index",
    "original_acceptable_max_filename", "original_acceptable_max_index",
    "metric_preferred_filename", "metric_preferred_index",
    "metric_acceptable_min_filename", "metric_acceptable_min_index",
    "metric_acceptable_max_filename", "metric_acceptable_max_index",
    "preferred_index_delta", "preferred_entropy", "preferred_saturated_ratio",
    "preferred_dark_ratio", "preferred_mean_luma", "preferred_quality_score",
]


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def exposure_metrics(rgb: np.ndarray, dark_threshold: float,
                     saturated_threshold: int, saturation_weight: float,
                     dark_weight: float) -> dict[str, float]:
    luma = rgb_to_luma(rgb) / 255.0
    histogram, _ = np.histogram(luma, bins=256, range=(0.0, 1.0))
    probability = histogram.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / 8.0)
    saturated_ratio = float(np.any(rgb >= saturated_threshold, axis=2).mean())
    dark_ratio = float((luma <= dark_threshold).mean())
    mean_luma = float(luma.mean())
    quality_score = entropy - saturation_weight * saturated_ratio - dark_weight * dark_ratio
    return {
        "entropy": entropy,
        "saturated_ratio": saturated_ratio,
        "dark_ratio": dark_ratio,
        "mean_luma": mean_luma,
        "quality_score": quality_score,
    }


def metric_suggestion(rows: list[dict], score_margin: float,
                      max_saturated_ratio: float,
                      max_dark_ratio: float) -> tuple[int, int, int]:
    scores = np.asarray([row["quality_score"] for row in rows], np.float64)
    preferred = int(np.argmax(scores))
    floor = float(scores[preferred] - score_margin)

    def acceptable(index: int) -> bool:
        row = rows[index]
        return (
            row["quality_score"] >= floor
            and row["saturated_ratio"] <= max_saturated_ratio
            and row["dark_ratio"] <= max_dark_ratio
        )

    bright = preferred
    while bright > 0 and acceptable(bright - 1):
        bright -= 1
    dark = preferred
    while dark + 1 < len(rows) and acceptable(dark + 1):
        dark += 1
    return preferred, dark, bright


def initialize_review_annotations(source_rows: list[dict], output_path: Path) -> int:
    existing = {
        row["scene_id"]: row for row in read_csv(output_path)
    } if output_path.exists() else {}
    initialized = []
    reviewed = 0
    for source in source_rows:
        current = existing.get(source["scene_id"])
        if current and current.get("label_source") == "human_review":
            initialized.append(current)
            reviewed += 1
            continue
        row = dict(source)
        row["label_source"] = "metric_review_pending"
        initialized.append(row)
    write_csv(output_path, ANNOTATION_FIELDS, initialized)
    return reviewed


def analyze(args: argparse.Namespace) -> None:
    frames = read_csv(args.source_draft / "frame_metadata.csv")
    annotations = read_csv(args.source_draft / "scene_annotations.csv")
    if not frames or not annotations:
        raise ValueError("source draft is empty; prepare and review Scene4 labels first")
    if {row.get("label_source") for row in annotations} != {"human_review"}:
        raise ValueError("all source annotations must be human_review before metric comparison")

    frames_by_scene: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        frames_by_scene[frame["scene_id"]].append(frame)
    annotation_by_scene = {row["scene_id"]: row for row in annotations}

    args.output_draft.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        args.source_draft / "frame_metadata.csv",
        args.output_draft / "frame_metadata.csv",
    )
    shutil.copy2(
        args.source_draft / "scene_annotations.csv",
        args.output_draft / "original_scene_annotations.csv",
    )
    reviewed = initialize_review_annotations(
        annotations, args.output_draft / "scene_annotations.csv"
    )

    metric_rows = []
    suggestion_rows = []
    deltas = []
    for scene_number, scene_id in enumerate(sorted(frames_by_scene), start=1):
        scene_frames = sorted(
            frames_by_scene[scene_id], key=lambda row: int(row["exposure_index"])
        )
        measured = []
        for frame in scene_frames:
            image_path = args.input / frame["filename"]
            rgb = read_dng_preview_rgb(image_path, args.preview_series)
            metrics = exposure_metrics(
                rgb, args.dark_threshold, args.saturated_threshold,
                args.saturation_weight, args.dark_weight,
            )
            row = {
                "scene_id": scene_id,
                "filename": frame["filename"],
                "exposure_index": int(frame["exposure_index"]),
                **metrics,
            }
            measured.append(row)
            metric_rows.append(row)

        preferred, dark, bright = metric_suggestion(
            measured, args.score_margin, args.max_saturated_ratio,
            args.max_dark_ratio,
        )
        original = annotation_by_scene[scene_id]
        index_by_filename = {
            row["filename"]: int(row["exposure_index"]) for row in scene_frames
        }
        original_preferred = index_by_filename[original["preferred_filename"]]
        selected = measured[preferred]
        delta = preferred - original_preferred
        deltas.append(delta)
        suggestion_rows.append({
            "scene_id": scene_id,
            "original_preferred_filename": original["preferred_filename"],
            "original_preferred_index": original_preferred,
            "original_acceptable_min_filename": original["acceptable_min_filename"],
            "original_acceptable_min_index": index_by_filename[
                original["acceptable_min_filename"]
            ],
            "original_acceptable_max_filename": original["acceptable_max_filename"],
            "original_acceptable_max_index": index_by_filename[
                original["acceptable_max_filename"]
            ],
            "metric_preferred_filename": measured[preferred]["filename"],
            "metric_preferred_index": preferred,
            "metric_acceptable_min_filename": measured[dark]["filename"],
            "metric_acceptable_min_index": dark,
            "metric_acceptable_max_filename": measured[bright]["filename"],
            "metric_acceptable_max_index": bright,
            "preferred_index_delta": delta,
            "preferred_entropy": selected["entropy"],
            "preferred_saturated_ratio": selected["saturated_ratio"],
            "preferred_dark_ratio": selected["dark_ratio"],
            "preferred_mean_luma": selected["mean_luma"],
            "preferred_quality_score": selected["quality_score"],
        })
        if scene_number % 10 == 0 or scene_number == len(frames_by_scene):
            print(f"analyzed {scene_number}/{len(frames_by_scene)} time steps", flush=True)

    def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(args.output_draft / "frame_metrics.csv", METRIC_FIELDS, metric_rows)
    write_rows(
        args.output_draft / "metric_suggestions.csv", SUGGESTION_FIELDS,
        suggestion_rows,
    )

    delta_counts = Counter(deltas)
    report = {
        "status": "READY_FOR_SECOND_HUMAN_REVIEW",
        "source_annotations": portable_path(args.source_draft / "scene_annotations.csv"),
        "review_annotations": portable_path(args.output_draft / "scene_annotations.csv"),
        "frame_metrics": portable_path(args.output_draft / "frame_metrics.csv"),
        "metric_suggestions": portable_path(args.output_draft / "metric_suggestions.csv"),
        "time_step_count": len(suggestion_rows),
        "frame_count": len(metric_rows),
        "already_reviewed": reviewed,
        "exact_preferred_match_count": int(delta_counts[0]),
        "metric_brighter_than_original_count": sum(delta < 0 for delta in deltas),
        "metric_darker_than_original_count": sum(delta > 0 for delta in deltas),
        "mean_absolute_index_delta": float(np.mean(np.abs(deltas))),
        "preferred_index_delta_distribution": {
            str(key): delta_counts[key] for key in sorted(delta_counts)
        },
        "metric_definition": {
            "entropy": "normalized Shannon entropy of preview luminance (0..1)",
            "saturated_ratio": (
                f"fraction of pixels with any RGB channel >= {args.saturated_threshold}"
            ),
            "dark_ratio": (
                f"fraction of pixels with preview luminance <= {args.dark_threshold:g}"
            ),
            "quality_score": (
                f"entropy - {args.saturation_weight:g} * saturated_ratio - "
                f"{args.dark_weight:g} * dark_ratio"
            ),
        },
        "note": (
            "Metric suggestions are review aids, not ground truth. Existing Version 0 "
            "human labels are preserved in original_scene_annotations.csv."
        ),
    }
    (args.output_draft / "metric_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create entropy/saturation/darkness suggestions for a second Scene4 review."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "input/Scene4")
    parser.add_argument("--source-draft", type=Path,
                        default=ROOT / "labels/scene4_draft")
    parser.add_argument("--output-draft", type=Path,
                        default=ROOT / "labels/scene4_metric_review")
    parser.add_argument("--preview-series", type=int, default=2)
    parser.add_argument("--dark-threshold", type=float, default=0.03)
    parser.add_argument("--saturated-threshold", type=int, default=250)
    parser.add_argument("--saturation-weight", type=float, default=2.0)
    parser.add_argument("--dark-weight", type=float, default=1.0)
    parser.add_argument("--score-margin", type=float, default=0.08)
    parser.add_argument("--max-saturated-ratio", type=float, default=0.08)
    parser.add_argument("--max-dark-ratio", type=float, default=0.45)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
