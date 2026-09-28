from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image


SCENE4_SHUTTERS_SECONDS = (
    15.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1 / 2, 1 / 4, 1 / 8,
    1 / 15, 1 / 30, 1 / 60, 1 / 125, 1 / 250, 1 / 500,
)


@dataclass(frozen=True)
class RawFrameMetadata:
    path: Path
    exposure_seconds: float
    iso: float
    aperture: float
    camera_model: str


def _tag_float(tag: object) -> float:
    value = getattr(tag, "values", tag)
    if isinstance(value, (list, tuple)):
        value = value[0]
    numerator = getattr(value, "num", None)
    denominator = getattr(value, "den", None)
    if numerator is not None and denominator is not None:
        return float(numerator) / float(denominator)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return float(Fraction(text))


def _first_tag(tags: dict, names: tuple[str, ...], path: Path) -> object:
    for name in names:
        if name in tags:
            return tags[name]
    raise ValueError(f"{path}: missing EXIF tag; expected one of {names}")


def read_cr2_metadata(path: Path) -> RawFrameMetadata:
    import exifread

    with path.open("rb") as file:
        tags = exifread.process_file(file, details=False)
    exposure = _tag_float(_first_tag(
        tags, ("EXIF ExposureTime", "Image ExposureTime"), path
    ))
    iso = _tag_float(_first_tag(
        tags, ("EXIF ISOSpeedRatings", "Image ISOSpeedRatings"), path
    ))
    aperture = _tag_float(_first_tag(tags, ("EXIF FNumber",), path))
    model_tag = tags.get("Image Model", "unknown")
    if exposure <= 0 or iso <= 0 or aperture <= 0:
        raise ValueError(f"{path}: exposure, ISO and aperture must be positive")
    return RawFrameMetadata(path, exposure, iso, aperture, str(model_tag).strip())


def discover_sihdr_stacks(dataset_root: Path) -> dict[str, list[Path]]:
    raw_root = dataset_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"SI-HDR raw directory not found: {raw_root}")
    stacks: dict[str, list[Path]] = {}
    for directory in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        frames = sorted(directory.glob("*.CR2"))
        if not frames:
            frames = sorted(directory.glob("*.cr2"))
        if frames:
            if len(frames) != 5:
                raise ValueError(
                    f"{directory}: official SI-HDR scenes must have 5 CR2 files; "
                    f"found {len(frames)}"
                )
            stacks[directory.name] = frames
    if not stacks:
        raise ValueError(f"no CR2 exposure stacks found below {raw_root}")
    return stacks


def _resize_float(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    return np.asarray(
        pil.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    )


def read_cr2_linear_luma(path: Path, width: int = 1024,
                         height: int = 683) -> tuple[np.ndarray, dict]:
    """Decode a CR2 mosaic into normalized linear camera-luma.

    The output is a compact luma plane for AE statistics. It is not a
    demosaiced/color-corrected photograph and is not written as a fake DNG.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32, copy=False)
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            raise ValueError(f"{path}: unsupported CFA pattern {pattern}")
        color_desc = raw.color_desc.decode("ascii", errors="replace")
        black = np.asarray(raw.black_level_per_channel, np.float32)
        per_channel_white = raw.camera_white_level_per_channel
        if per_channel_white is None or any(value is None for value in per_channel_white):
            white = np.full(len(black), float(raw.white_level), np.float32)
        else:
            white = np.asarray(per_channel_white, np.float32)
        h = (mosaic.shape[0] // 2) * 2
        w = (mosaic.shape[1] // 2) * 2
        luma = np.zeros((h // 2, w // 2), np.float32)
        counts = {"R": 0, "G": 0, "B": 0}
        for y in range(2):
            for x in range(2):
                color_index = int(pattern[y, x])
                color = color_desc[color_index].upper()
                if color not in counts:
                    raise ValueError(f"{path}: unsupported CFA color {color!r}")
                counts[color] += 1
        channel_weights = {"R": 0.2126, "G": 0.7152, "B": 0.0722}
        for y in range(2):
            for x in range(2):
                color_index = int(pattern[y, x])
                color = color_desc[color_index].upper()
                denominator = max(float(white[color_index] - black[color_index]), 1.0)
                plane = (mosaic[y:h:2, x:w:2] - black[color_index]) / denominator
                weight = channel_weights[color] / counts[color]
                luma += np.clip(plane, 0.0, 1.0) * weight
        info = {
            "raw_shape": [int(mosaic.shape[0]), int(mosaic.shape[1])],
            "cfa_pattern": pattern.astype(int).tolist(),
            "color_desc": color_desc,
            "black_level_per_channel": black.astype(int).tolist(),
            "white_level_per_channel": white.astype(int).tolist(),
        }
    return _resize_float(luma, width, height), info


def merge_exposure_stack(signals: list[np.ndarray], metadata: list[RawFrameMetadata],
                         target_aperture: float = 14.0,
                         noise_floor: float = 0.01,
                         saturation: float = 0.985) -> tuple[np.ndarray, dict]:
    if len(signals) != len(metadata) or not signals:
        raise ValueError("signals and metadata must have the same non-zero length")
    shape = signals[0].shape
    if any(signal.shape != shape for signal in signals):
        raise ValueError("all exposure planes must have the same shape")
    signal_stack = np.stack(signals).astype(np.float32)
    rates = []
    for signal, meta in zip(signal_stack, metadata):
        effective_time = meta.exposure_seconds * (meta.iso / 100.0)
        aperture_scale = (meta.aperture / target_aperture) ** 2
        rates.append(signal / max(effective_time, 1e-12) * aperture_scale)
    rate_stack = np.stack(rates).astype(np.float32)
    middle_weight = np.maximum(1.0 - np.abs(signal_stack * 2.0 - 1.0), 0.0) ** 2
    valid = (signal_stack > noise_floor) & (signal_stack < saturation)
    weights = middle_weight * valid
    weight_sum = weights.sum(axis=0)
    merged = (rate_stack * weights).sum(axis=0) / np.maximum(weight_sum, 1e-12)
    uncovered = weight_sum <= 0
    if np.any(uncovered):
        closest = np.argmin(np.abs(signal_stack - 0.5), axis=0)
        fallback = np.take_along_axis(rate_stack, closest[None, ...], axis=0)[0]
        merged[uncovered] = fallback[uncovered]
    finite = np.isfinite(merged)
    merged = np.where(finite, np.maximum(merged, 0.0), 0.0).astype(np.float32)
    return merged, {
        "covered_fraction": float(np.mean(~uncovered)),
        "finite_fraction": float(np.mean(finite)),
        "target_aperture": float(target_aperture),
        "noise_floor": float(noise_floor),
        "saturation_cutoff": float(saturation),
    }


def read_exr_rgb(path: Path) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as file:
        if len(file.parts) != 1:
            raise ValueError(f"{path}: expected one EXR part; found {len(file.parts)}")
        channels = file.channels()
        if "RGB" in channels:
            rgb = channels["RGB"].pixels
        elif all(name in channels for name in ("R", "G", "B")):
            rgb = np.stack([channels[name].pixels for name in ("R", "G", "B")], axis=-1)
        else:
            raise ValueError(f"{path}: missing RGB channels; found {sorted(channels)}")
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"{path}: unexpected RGB shape {rgb.shape}")
    return np.nan_to_num(rgb[..., :3], nan=0.0, posinf=0.0, neginf=0.0)


def rgb_to_linear_luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def calibrate_reference_luma(reference_luma: np.ndarray,
                             raw_radiance: np.ndarray) -> tuple[np.ndarray, float]:
    reference = _resize_float(reference_luma, raw_radiance.shape[1], raw_radiance.shape[0])
    valid = (
        np.isfinite(reference) & np.isfinite(raw_radiance)
        & (reference > np.quantile(reference[reference > 0], 0.05))
        & (raw_radiance > np.quantile(raw_radiance[raw_radiance > 0], 0.05))
    )
    if int(valid.sum()) < 100:
        raise ValueError("not enough valid pixels to calibrate EXR radiance scale")
    log_scale = np.median(
        np.log(np.maximum(raw_radiance[valid], 1e-12))
        - np.log(np.maximum(reference[valid], 1e-12))
    )
    scale = float(np.exp(log_scale))
    return np.maximum(reference * scale, 0.0).astype(np.float32), scale


def render_exposure(radiance_per_second: np.ndarray,
                    shutter_seconds: float) -> np.ndarray:
    if shutter_seconds <= 0:
        raise ValueError("shutter_seconds must be positive")
    return np.clip(radiance_per_second * shutter_seconds, 0.0, 1.0).astype(np.float32)


def exposure_quality(signal: np.ndarray) -> dict[str, float]:
    signal = np.clip(np.asarray(signal, np.float32), 0.0, 1.0)
    display = np.sqrt(signal)
    histogram, _ = np.histogram(display, bins=64, range=(0.0, 1.0))
    probability = histogram.astype(np.float64) / max(histogram.sum(), 1)
    nonzero = probability > 0
    entropy = float(-(probability[nonzero] * np.log2(probability[nonzero])).sum() / 6.0)
    saturated_ratio = float(np.mean(signal >= 0.985))
    dark_ratio = float(np.mean(signal <= 0.01))
    score = entropy - 2.0 * saturated_ratio - dark_ratio
    return {
        "entropy": entropy,
        "saturated_ratio": saturated_ratio,
        "dark_ratio": dark_ratio,
        "mean_linear_luma": float(signal.mean()),
        "score": float(score),
    }


def choose_exposure_indices(metrics: list[dict[str, float]]) -> tuple[int, int, int]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    preferred = max(range(len(metrics)), key=lambda index: metrics[index]["score"])
    best_score = metrics[preferred]["score"]
    acceptable = [
        index for index, item in enumerate(metrics)
        if item["score"] >= best_score - 0.12
        and item["saturated_ratio"] <= max(metrics[preferred]["saturated_ratio"] + 0.03, 0.05)
        and item["dark_ratio"] <= max(metrics[preferred]["dark_ratio"] + 0.08, 0.20)
    ]
    if preferred not in acceptable:
        acceptable.append(preferred)
    # The shutter list is bright-to-dark. Return filenames later, while the
    # label generator sorts their log-exposure values into min/max.
    return preferred, max(acceptable), min(acceptable)


def hdr_benefit_suggestion(rendered: list[np.ndarray], preferred_index: int) -> dict:
    preferred = rendered[preferred_index]
    brightest = rendered[0]
    darkest = rendered[-1]
    recoverable_highlights = float(np.mean((preferred >= 0.985) & (darkest < 0.95)))
    recoverable_shadows = float(np.mean((preferred <= 0.01) & (brightest > 0.03)))
    recoverable = recoverable_highlights + recoverable_shadows
    if recoverable < 0.005:
        benefit, enable = 0.25, 0
    elif recoverable < 0.03:
        benefit, enable = 0.5, None
    else:
        benefit, enable = 0.75, 1
    return {
        "hdr_benefit_if_static": benefit,
        "hdr_enable_if_static": enable,
        "recoverable_highlight_ratio": recoverable_highlights,
        "recoverable_shadow_ratio": recoverable_shadows,
        "recoverable_total_ratio": recoverable,
    }


def preview_rgb(signal: np.ndarray) -> np.ndarray:
    display = np.sqrt(np.clip(signal, 0.0, 1.0))
    gray = np.round(display * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)
