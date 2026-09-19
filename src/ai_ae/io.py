from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraStatus:
    filename: str
    scene_id: str
    frame_index: int
    exposure_time_us: float
    sensor_gain_code: float
    isp_gain_code: float
    hdr_mode: str


@dataclass(frozen=True)
class SceneLabel:
    scene_id: str
    target_ev: float
    acceptable_min_ev: float
    acceptable_max_ev: float
    hdr_benefit: float
    hdr_enable: int
    hdr_ratio: int
    hdr_anchor_ev: float
    confidence: float
    source: str


def _data_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line.split()


def load_camera_status(path: Path) -> list[CameraStatus]:
    rows = []
    for x in _data_lines(path):
        if len(x) != 7:
            raise ValueError(f"{path}: camera status 應有 7 欄，實際為 {len(x)}")
        rows.append(CameraStatus(x[0], x[1], int(x[2]), float(x[3]), float(x[4]),
                                 float(x[5]), x[6]))
    return rows


def calculate_log_exposure(status: CameraStatus, reference_time_us: float,
                           sensor_gain_unity_code: float,
                           isp_gain_unity_code: float) -> float:
    """將硬體 code 換成倍率後，計算相對固定 reference 的 log2 exposure。"""
    sensor_gain = status.sensor_gain_code / sensor_gain_unity_code
    isp_gain = status.isp_gain_code / isp_gain_unity_code
    relative_product = (status.exposure_time_us / reference_time_us) * sensor_gain * isp_gain
    if relative_product <= 0:
        raise ValueError("exposure time 與 gain 必須大於 0")
    return float(np.log2(relative_product))


def load_scene_labels(path: Path) -> dict[str, SceneLabel]:
    labels = {}
    for x in _data_lines(path):
        if len(x) != 10:
            raise ValueError(f"{path}: AE output 應有 10 欄，實際為 {len(x)}")
        label = SceneLabel(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]),
                           int(x[5]), int(x[6]), float(x[7]), float(x[8]), x[9])
        labels[label.scene_id] = label
    return labels


def read_pgm(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """讀取 P5 PGM，支援 8-bit 與 big-endian 16-bit 容器。"""
    metadata: dict[str, str] = {}
    tokens: list[bytes] = []
    with path.open("rb") as f:
        if f.readline().strip() != b"P5":
            raise ValueError(f"{path} 不是 P5 PGM")
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} header 不完整")
            if line.startswith(b"#"):
                text = line[1:].decode("ascii", errors="replace").strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata[key.strip()] = value.strip()
            else:
                tokens.extend(line.split())
        width, height, max_value = map(int, tokens[:3])
        dtype = ">u2" if max_value > 255 else "u1"
        pixels = np.frombuffer(f.read(), dtype=dtype, count=width * height)
    if pixels.size != width * height:
        raise ValueError(f"{path} pixel 數量不符：{pixels.size} != {width * height}")
    metadata.update(width=str(width), height=str(height), max_value=str(max_value))
    return pixels.reshape(height, width).astype(np.float32), metadata
