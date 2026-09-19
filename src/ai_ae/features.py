from __future__ import annotations

import numpy as np

from .io import CameraStatus


def extract_features(image: np.ndarray, max_value: float, status: CameraStatus,
                     histogram_bins: int, grid_rows: int, grid_cols: int,
                     pending_ev: list[float], applied_ev: float,
                     sensor_gain_unity_code: float, isp_gain_unity_code: float) -> np.ndarray:
    normalized = np.clip(image / max_value, 0.0, 1.0)
    histogram, _ = np.histogram(normalized, bins=histogram_bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / normalized.size

    h = (normalized.shape[0] // grid_rows) * grid_rows
    w = (normalized.shape[1] // grid_cols) * grid_cols
    grid = normalized[:h, :w].reshape(grid_rows, h // grid_rows,
                                      grid_cols, w // grid_cols).mean(axis=(1, 3))
    state = np.array([
        applied_ev / 16.0,
        np.log2(max(status.exposure_time_us, 1.0)) / 20.0,
        np.log2(max((status.sensor_gain_code / sensor_gain_unity_code) *
                    (status.isp_gain_code / isp_gain_unity_code), 1e-6)) / 8.0,
        pending_ev[0] / 16.0,
        pending_ev[-1] / 16.0,
    ], dtype=np.float32)
    return np.concatenate([histogram, grid.astype(np.float32).ravel(), state])
