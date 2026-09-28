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

    def loss_components(self, x: np.ndarray, target_ev: np.ndarray,
                        hdr: np.ndarray, ratio_index: np.ndarray,
                        confidence: np.ndarray,
                        ev_mask: np.ndarray | None = None,
                        hdr_mask: np.ndarray | None = None,
                        ratio_mask: np.ndarray | None = None,
                        confidence_mask: np.ndarray | None = None) -> dict[str, float]:
        """Return the same full-batch objective components used by ``train``."""
        n = x.shape[0]
        ev_mask = np.ones(n, np.float32) if ev_mask is None else ev_mask.astype(np.float32)
        hdr_mask = np.ones(n, np.float32) if hdr_mask is None else hdr_mask.astype(np.float32)
        ratio_mask = (np.ones(n, np.float32) if ratio_mask is None
                      else ratio_mask.astype(np.float32))
        confidence_mask = (np.ones(n, np.float32) if confidence_mask is None
                           else confidence_mask.astype(np.float32))
        prediction = self.predict(x)
        eps = 1e-7
        ev_error = (prediction["target_ev"] - target_ev) ** 2
        ev_mse = float((ev_error * ev_mask).sum() / max(float(ev_mask.sum()), 1.0))
        hdr_loss = -(hdr * np.log(prediction["hdr_benefit"] + eps) +
                     (1 - hdr) * np.log(1 - prediction["hdr_benefit"] + eps))
        ratio_loss = -np.log(
            prediction["ratio_prob"][np.arange(n), ratio_index] + eps
        )
        confidence_loss = -(confidence * np.log(prediction["confidence"] + eps) +
                            (1 - confidence) *
                            np.log(1 - prediction["confidence"] + eps))
        hdr_bce = float((hdr_loss * hdr_mask).sum() / max(float(hdr_mask.sum()), 1.0))
        ratio_ce = float(
            (ratio_loss * ratio_mask).sum() / max(float(ratio_mask.sum()), 1.0)
        )
        confidence_bce = float(
            (confidence_loss * confidence_mask).sum() /
            max(float(confidence_mask.sum()), 1.0)
        )
        return {
            "total": ev_mse + 0.2 * hdr_bce + 0.2 * ratio_ce + 0.1 * confidence_bce,
            "ev_mse": ev_mse,
            "hdr_bce": hdr_bce,
            "ratio_ce": ratio_ce,
            "confidence_bce": confidence_bce,
        }

    def train(self, x: np.ndarray, target_ev: np.ndarray, hdr: np.ndarray,
              ratio_index: np.ndarray, confidence: np.ndarray, epochs: int, lr: float,
              ev_mask: np.ndarray | None = None,
              hdr_mask: np.ndarray | None = None,
              ratio_mask: np.ndarray | None = None,
              confidence_mask: np.ndarray | None = None) -> list[float]:
        n = x.shape[0]
        ev_mask = np.ones(n, np.float32) if ev_mask is None else ev_mask.astype(np.float32)
        hdr_mask = np.ones(n, np.float32) if hdr_mask is None else hdr_mask.astype(np.float32)
        ratio_mask = (np.ones(n, np.float32) if ratio_mask is None
                      else ratio_mask.astype(np.float32))
        confidence_mask = (np.ones(n, np.float32) if confidence_mask is None
                           else confidence_mask.astype(np.float32))
        ev_denominator = max(float(ev_mask.sum()), 1.0)
        hdr_denominator = max(float(hdr_mask.sum()), 1.0)
        ratio_denominator = max(float(ratio_mask.sum()), 1.0)
        confidence_denominator = max(float(confidence_mask.sum()), 1.0)
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
            loss = float((((ev - target_ev) ** 2) * ev_mask).sum() / ev_denominator)
            hdr_loss = -(hdr * np.log(hdr_p + eps) +
                         (1 - hdr) * np.log(1 - hdr_p + eps))
            ratio_loss = -np.log(ratio_p[np.arange(n), ratio_index] + eps)
            confidence_loss = -(confidence * np.log(conf_p + eps) +
                                (1 - confidence) * np.log(1 - conf_p + eps))
            loss += 0.2 * float((hdr_loss * hdr_mask).sum() / hdr_denominator)
            loss += 0.2 * float((ratio_loss * ratio_mask).sum() / ratio_denominator)
            loss += 0.1 * float((confidence_loss * confidence_mask).sum() /
                                confidence_denominator)
            history.append(float(loss))

            d_ev = (2.0 / ev_denominator) * (
                (ev - target_ev) * ev_mask
            )[:, None]
            d_hdr = (0.2 / hdr_denominator) * (
                (hdr_p - hdr) * hdr_mask
            )[:, None]
            d_ratio = (0.2 / ratio_denominator) * (
                (ratio_p - one_hot) * ratio_mask[:, None]
            )
            d_conf = (0.1 / confidence_denominator) * (
                (conf_p - confidence) * confidence_mask
            )[:, None]
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

