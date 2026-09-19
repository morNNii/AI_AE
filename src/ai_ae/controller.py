from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControllerOutput:
    exposure_command_ev: float
    hdr_enabled: bool
    hdr_ratio: int


class DelayAwareController:
    def __init__(self, max_step_ev: float, hdr_on: float, hdr_off: float):
        self.max_step_ev = max_step_ev
        self.hdr_on = hdr_on
        self.hdr_off = hdr_off
        self.hdr_enabled = False

    def step(self, target_ev: float, hdr_benefit: float, hdr_ratio: int,
             pending_queue: list[float]) -> ControllerOutput:
        future_ev = pending_queue[-1]
        command = future_ev + float(np.clip(target_ev - future_ev,
                                             -self.max_step_ev, self.max_step_ev))
        if self.hdr_enabled and hdr_benefit < self.hdr_off:
            self.hdr_enabled = False
        elif not self.hdr_enabled and hdr_benefit > self.hdr_on:
            self.hdr_enabled = True
        return ControllerOutput(command, self.hdr_enabled, hdr_ratio)

