from __future__ import annotations

from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class MultiHeadMLP:
    """NumPy smoke-test 模型：共享 encoder + EV/HDR/ratio/confidence heads。"""

    def __init__(self, input_dim: int, hidden_dim: int, ratio_count: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim)).astype(np.float32),
            "b1": np.zeros(hidden_dim, np.float32),
            "w_ev": rng.normal(0, 0.1, (hidden_dim, 1)).astype(np.float32),
            "b_ev": np.zeros(1, np.float32),
            "w_hdr": rng.normal(0, 0.1, (hidden_dim, 1)).astype(np.float32),
            "b_hdr": np.zeros(1, np.float32),
            "w_ratio": rng.normal(0, 0.1, (hidden_dim, ratio_count)).astype(np.float32),
            "b_ratio": np.zeros(ratio_count, np.float32),
            "w_conf": rng.normal(0, 0.1, (hidden_dim, 1)).astype(np.float32),
            "b_conf": np.zeros(1, np.float32),
        }

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        h = np.maximum(x @ self.params["w1"] + self.params["b1"], 0.0)
        ratio_logits = h @ self.params["w_ratio"] + self.params["b_ratio"]
        ratio_prob = np.exp(ratio_logits - ratio_logits.max(axis=1, keepdims=True))
        ratio_prob /= ratio_prob.sum(axis=1, keepdims=True)
        return {
            "target_ev": (h @ self.params["w_ev"] + self.params["b_ev"]).ravel(),
            "hdr_benefit": sigmoid(h @ self.params["w_hdr"] + self.params["b_hdr"]).ravel(),
            "ratio_prob": ratio_prob,
            "confidence": sigmoid(h @ self.params["w_conf"] + self.params["b_conf"]).ravel(),
        }

    def train(self, x: np.ndarray, target_ev: np.ndarray, hdr: np.ndarray,
              ratio_index: np.ndarray, confidence: np.ndarray, epochs: int, lr: float) -> list[float]:
        n = x.shape[0]
        history = []
        for _ in range(epochs):
            z = x @ self.params["w1"] + self.params["b1"]
            h = np.maximum(z, 0.0)
            ev = (h @ self.params["w_ev"] + self.params["b_ev"]).ravel()
            hdr_p = sigmoid(h @ self.params["w_hdr"] + self.params["b_hdr"]).ravel()
            conf_p = sigmoid(h @ self.params["w_conf"] + self.params["b_conf"]).ravel()
            logits = h @ self.params["w_ratio"] + self.params["b_ratio"]
            ratio_p = np.exp(logits - logits.max(axis=1, keepdims=True))
            ratio_p /= ratio_p.sum(axis=1, keepdims=True)
            one_hot = np.zeros_like(ratio_p); one_hot[np.arange(n), ratio_index] = 1.0
            eps = 1e-7
            loss = np.mean((ev - target_ev) ** 2)
            loss += 0.2 * -np.mean(hdr * np.log(hdr_p + eps) + (1 - hdr) * np.log(1 - hdr_p + eps))
            loss += 0.2 * -np.mean(np.log(ratio_p[np.arange(n), ratio_index] + eps))
            loss += 0.1 * -np.mean(confidence * np.log(conf_p + eps) + (1 - confidence) * np.log(1 - conf_p + eps))
            history.append(float(loss))

            d_ev = (2.0 / n) * (ev - target_ev)[:, None]
            d_hdr = (0.2 / n) * (hdr_p - hdr)[:, None]
            d_ratio = (0.2 / n) * (ratio_p - one_hot)
            d_conf = (0.1 / n) * (conf_p - confidence)[:, None]
            dh = (d_ev @ self.params["w_ev"].T + d_hdr @ self.params["w_hdr"].T +
                  d_ratio @ self.params["w_ratio"].T + d_conf @ self.params["w_conf"].T)
            grads = {
                "w_ev": h.T @ d_ev, "b_ev": d_ev.sum(axis=0),
                "w_hdr": h.T @ d_hdr, "b_hdr": d_hdr.sum(axis=0),
                "w_ratio": h.T @ d_ratio, "b_ratio": d_ratio.sum(axis=0),
                "w_conf": h.T @ d_conf, "b_conf": d_conf.sum(axis=0),
                "w1": x.T @ (dh * (z > 0)), "b1": (dh * (z > 0)).sum(axis=0),
            }
            for key, grad in grads.items():
                self.params[key] -= lr * np.clip(grad, -5.0, 5.0)
        return history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.params)

