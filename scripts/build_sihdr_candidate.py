#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_labels import ANNOTATION_FIELDS, FRAME_FIELDS, build, read_csv, write_csv


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def preferred_index(annotation: dict[str, str]) -> int:
    match = re.search(r"_e(\d\d)\.", annotation["preferred_filename"])
    if not match:
        raise ValueError(
            f"{annotation['scene_id']}: cannot parse preferred exposure index from "
            f"{annotation['preferred_filename']}"
        )
    return int(match.group(1))


def exposure_bin(index: int) -> str:
    if index == 0:
        return "boundary_long"
    if index == 14:
        return "boundary_short"
    if index <= 4:
        return "long"
    if index <= 9:
        return "medium"
    return "short"


def hdr_class(annotation: dict[str, str]) -> str:
    if annotation.get("hdr_reviewed", "").strip() != "1":
        return "unknown"
    value = annotation.get("hdr_enable_if_static", "").strip()
    return "unknown" if not value else ("on" if int(float(value)) else "off")


def mask_unreviewed_hdr(annotation: dict[str, str]) -> dict[str, str]:
    sanitized = dict(annotation)
    if sanitized.get("hdr_reviewed", "").strip() != "1":
        for field in (
            "hdr_benefit_if_static", "hdr_enable_if_static", "hdr_ratio_class",
            "hdr_anchor_filename", "hdr_label_confidence", "hdr_notes",
        ):
            sanitized[field] = ""
    return sanitized


def stratified_assignments(annotations: list[dict[str, str]], seed: int) -> list[dict]:
    count = len(annotations)
    train_count = round(count * 0.70)
    validation_count = round(count * 0.15)
    targets = {
        "train": train_count,
        "validation": validation_count,
        "test": count - train_count - validation_count,
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for annotation in annotations:
        index = preferred_index(annotation)
        groups[(hdr_class(annotation), exposure_bin(index))].append(annotation)
    rng = random.Random(seed)
    remaining = dict(targets)
    assignments = []
    for stratum, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        rng.shuffle(rows)
        desired = {
            name: len(rows) * targets[name] / count for name in targets
        }
        assigned = Counter()
        for annotation in rows:
            available = [name for name, capacity in remaining.items() if capacity > 0]
            split = max(
                available,
                key=lambda name: (
                    desired[name] - assigned[name],
                    remaining[name] / max(targets[name], 1),
                    name,
                ),
            )
            assigned[split] += 1
            remaining[split] -= 1
            assignments.append({
                "scene_id": annotation["scene_id"],
                "split": split,
                "stratum": f"{stratum[0]}_{stratum[1]}",
                "hdr_class": stratum[0],
                "preferred_exposure_index": preferred_index(annotation),
                "target_at_boundary": int(preferred_index(annotation) in (0, 14)),
            })
    if any(remaining.values()):
        raise RuntimeError(f"failed to fill split targets: {remaining}")
    return sorted(assignments, key=lambda row: row["scene_id"])


def validate_features(source: Path, included_scenes: set[str]) -> int:
    with np.load(source) as cache:
        scene_ids = cache["scene_ids"].astype(str)
        cached_scenes = set(scene_ids)
    if cached_scenes != included_scenes:
        missing = sorted(included_scenes - cached_scenes)
        extra = sorted(cached_scenes - included_scenes)
        raise ValueError(
            f"feature cache scene mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    return int(len(scene_ids))


def build_candidate(args: argparse.Namespace) -> dict:
    frames = read_csv(args.draft / "frame_metadata.csv")
    annotations = read_csv(args.draft / "scene_annotations.csv")
    training_annotations = [mask_unreviewed_hdr(row) for row in annotations]
    boundary = []
    for annotation in annotations:
        index = preferred_index(annotation)
        if index in (0, 14):
            boundary.append({
                "scene_id": annotation["scene_id"],
                "preferred_exposure_index": index,
                "target_ev_mask": 0,
                "hdr_retained": 1,
                "reason": "SDR target is censored by the 15-shutter boundary",
            })
    included_ids = {row["scene_id"] for row in annotations}
    boundary_ids = {row["scene_id"] for row in boundary}
    assignments = stratified_assignments(training_annotations, args.seed)

    write_csv(args.output / "split_assignments.csv", list(assignments[0]), assignments)
    write_csv(args.output / "boundary_scenes.csv", list(boundary[0]), boundary)
    with tempfile.TemporaryDirectory(prefix="ai_ae_sihdr_candidate_") as temp_dir:
        candidate_draft = Path(temp_dir)
        write_csv(candidate_draft / "frame_metadata.csv", FRAME_FIELDS, frames)
        write_csv(
            candidate_draft / "scene_annotations.csv", ANNOTATION_FIELDS,
            training_annotations,
        )
        build(candidate_draft, args.output, args.config)
    for filename in ("scenes.csv", "training_samples.csv"):
        path = args.output / filename
        rows = read_csv(path)
        for row in rows:
            row["target_ev_mask"] = int(row["scene_id"] not in boundary_ids)
        write_csv(path, list(rows[0]), rows)
    feature_count = validate_features(args.features, included_ids)

    by_split = Counter(row["split"] for row in assignments)
    by_split_hdr: dict[str, Counter] = defaultdict(Counter)
    for row in assignments:
        by_split_hdr[row["split"]][row["hdr_class"]] += 1
    sdr_human_count = sum(
        row.get("label_source", "").strip() == "human_review"
        for row in annotations
    )
    hdr_human_count = sum(
        row.get("hdr_reviewed", "").strip() == "1" for row in annotations
    )
    if (sdr_human_count == len(annotations)
            and hdr_human_count == len(annotations)):
        status = "HUMAN_SDR_HDR_READY"
        warning = (
            "All SDR and HDR-benefit labels are human-reviewed. Ambiguous HDR "
            "decisions and unlabeled ratios remain masked from loss."
        )
    elif sdr_human_count == len(annotations):
        status = "HUMAN_SDR_READY"
        warning = (
            "All SDR targets are human-reviewed. Unreviewed HDR fields are cleared "
            "and masked from loss."
        )
    else:
        status = "CANDIDATE_READY"
        warning = (
            "Some SDR targets are radiance-derived heuristics. Boundary scenes "
            "remain available to reviewed HDR supervision but are masked from EV loss."
        )
    report = {
        "status": status,
        "source_scene_count": len(annotations),
        "included_scene_count": len(annotations),
        "sdr_human_reviewed_scene_count": sdr_human_count,
        "hdr_human_reviewed_scene_count": hdr_human_count,
        "ev_supervised_scene_count": len(annotations) - len(boundary),
        "ev_masked_boundary_scene_count": len(boundary),
        "hdr_retained_boundary_scene_count": len(boundary),
        "training_sample_count": feature_count,
        "split_scene_counts": dict(by_split),
        "split_hdr_counts": {
            split: dict(counts) for split, counts in by_split_hdr.items()
        },
        "features": portable_path(args.features),
        "labels": portable_path(args.output / "training_samples.csv"),
        "split_assignments": portable_path(args.output / "split_assignments.csv"),
        "warning": warning,
    }
    (args.output / "candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SI-HDR candidate with masked shutter-boundary EV targets and "
            "deterministic scene-level stratified splits."
        )
    )
    parser.add_argument("--draft", type=Path, default=ROOT / "labels/sihdr_draft")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "labels/sihdr_generated_candidate")
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs/sihdr_prepared/features.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sihdr.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build_candidate(parse_args()), indent=2, ensure_ascii=False))
