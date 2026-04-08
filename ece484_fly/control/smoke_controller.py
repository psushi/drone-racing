"""Minimal controller for simulation smoke tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ece484_fly.control.controller import Controller


class SmokeController(Controller):
    """Returns near no-op commands with the right shape for the configured control mode."""

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._control_mode = config.env.control_mode

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        if self._control_mode == "state":
            return np.zeros(13, dtype=float)
        return np.zeros(4, dtype=float)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        # Keep simulation running until the environment terminates/truncates naturally.
        return False
