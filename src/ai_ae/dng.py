from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


EXIF_IFD = 34665
EXPOSURE_TIME = 33434
F_NUMBER = 33437
ISO_SPEED = 34855
DATE_TIME_ORIGINAL = 36867


@dataclass(frozen=True)
class DngMetadata:
    exposure_time_us: float
    iso: float
    aperture: float
    captured_at: str


def read_dng_metadata(path: Path) -> DngMetadata:
    """Read the camera settings needed by the Scene4 pipeline."""
    with Image.open(path) as image:
        exif = image.getexif().get_ifd(EXIF_IFD)
    missing = [
        name
        for tag, name in (
            (EXPOSURE_TIME, "ExposureTime"),
            (F_NUMBER, "FNumber"),
            (ISO_SPEED, "ISOSpeedRatings"),
        )
        if tag not in exif
    ]
    if missing:
        raise ValueError(f"{path}: missing EXIF fields: {', '.join(missing)}")
    return DngMetadata(
        exposure_time_us=float(exif[EXPOSURE_TIME]) * 1_000_000.0,
        iso=float(exif[ISO_SPEED]),
        aperture=float(exif[F_NUMBER]),
        captured_at=str(exif.get(DATE_TIME_ORIGINAL, "")),
    )


def read_dng_preview_rgb(path: Path, series_index: int = 2) -> np.ndarray:
    """Decode a DNG embedded preview as uint8 RGB.

    Scene4 contains a 1024x683 processed preview at series index 2. Using the
    preview makes one-time feature extraction practical while retaining the
    exposure differences needed by the baseline model.
    """
    with tifffile.TiffFile(path) as dng:
        if not 0 <= series_index < len(dng.series):
            shapes = [series.shape for series in dng.series]
            raise ValueError(
                f"{path}: preview series {series_index} is unavailable; series={shapes}"
            )
        preview = dng.series[series_index].asarray()

    if preview.ndim == 2:
        preview = np.repeat(preview[..., None], 3, axis=2)
    if preview.ndim != 3 or preview.shape[2] < 3:
        raise ValueError(f"{path}: unsupported preview shape {preview.shape}")
    preview = preview[..., :3]
    if preview.dtype != np.uint8:
        info = np.iinfo(preview.dtype)
        preview = np.clip(preview.astype(np.float32) / info.max * 255.0, 0, 255).astype(
            np.uint8
        )
    return preview


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected RGB image, got {rgb.shape}")
    return (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )


def read_dng_preview_luma(path: Path, series_index: int = 2) -> np.ndarray:
    return rgb_to_luma(read_dng_preview_rgb(path, series_index))
