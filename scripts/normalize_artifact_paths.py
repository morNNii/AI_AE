#!/usr/bin/env python3
"""Replace workspace-absolute paths in versioned text artifacts with relative paths."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}


def normalize_tree(base: Path) -> tuple[int, int]:
    checked = 0
    changed = 0
    root_native = str(ROOT.resolve())
    replacements = (
        (root_native.replace("\\", "\\\\") + "\\\\", ""),
        (root_native + "\\", ""),
        (root_native.replace("\\", "/") + "/", ""),
    )
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        checked += 1
        original = path.read_text(encoding="utf-8-sig")
        normalized = original
        for prefix, replacement in replacements:
            normalized = normalized.replace(prefix, replacement)
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")
            changed += 1
    return checked, changed


if __name__ == "__main__":
    total_checked = 0
    total_changed = 0
    for directory in (ROOT / "outputs", ROOT / "labels"):
        checked, changed = normalize_tree(directory)
        total_checked += checked
        total_changed += changed
    print(f"checked={total_checked} changed={total_changed}")
