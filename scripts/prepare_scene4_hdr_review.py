#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_labels import ANNOTATION_FIELDS, read_csv, write_csv


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def prepare(source: Path, output: Path) -> None:
    frames_path = source / "frame_metadata.csv"
    annotations_path = source / "scene_annotations.csv"
    if not frames_path.exists() or not annotations_path.exists():
        raise FileNotFoundError("source Scene4 draft is incomplete")
    source_rows = read_csv(annotations_path)
    if {row.get("label_source") for row in source_rows} != {"human_review"}:
        raise ValueError("all source SDR labels must be human_review")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frames_path, output / "frame_metadata.csv")
    shutil.copy2(annotations_path, output / "original_scene_annotations.csv")
    existing = {
        row["scene_id"]: row for row in read_csv(output / "scene_annotations.csv")
    } if (output / "scene_annotations.csv").exists() else {}

    rows = []
    reviewed = 0
    for source_row in source_rows:
        current = existing.get(source_row["scene_id"])
        if current and current.get("hdr_reviewed") == "1":
            rows.append(current)
            reviewed += 1
            continue
        row = dict(source_row)
        row.update({
            "hdr_benefit_if_static": "",
            "hdr_enable_if_static": "",
            "hdr_ratio_class": "",
            "hdr_anchor_filename": "",
            "hdr_reviewed": "",
            "hdr_label_confidence": "",
            "hdr_notes": "",
        })
        rows.append(row)
    write_csv(output / "scene_annotations.csv", ANNOTATION_FIELDS, rows)
    report = {
        "status": "READY_FOR_HDR_HUMAN_REVIEW",
        "scene_count": len(rows),
        "hdr_reviewed_count": reviewed,
        "source_sdr_labels": portable_path(annotations_path),
        "hdr_review_labels": portable_path(output / "scene_annotations.csv"),
        "policy": {
            "benefit_0_or_0.25": "HDR off",
            "benefit_0.5": "ambiguous; enable missing and loss masked",
            "benefit_0.75_or_1": "HDR on",
            "ratio_and_anchor": "leave missing until real or simulated HDR outputs exist",
        },
    }
    (output / "hdr_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated Scene4 draft for static HDR benefit review."
    )
    parser.add_argument("--source", type=Path, default=ROOT / "labels/scene4_draft")
    parser.add_argument("--output", type=Path, default=ROOT / "labels/scene4_hdr_review")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.output)
