"""Simple controller that moves the drone straight up and back down."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ece484_fly.control.controller import Controller


class AltitudeBounceController(Controller):
    """Commands a vertical up/down maneuver while holding horizontal position."""

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._control_mode = config.env.control_mode
        self._dt = 1.0 / config.env.freq
        self._time = 0.0

        self._start_pos = np.array(obs["pos"], dtype=float)
        self._start_yaw = self._quat_to_yaw(obs["quat"])

        high_limit = float(config.env.track.safety_limits.pos_limit_high[2])
        self._low_z = float(max(self._start_pos[2], 0.15))
        self._high_z = float(min(self._start_pos[2] + 0.75, high_limit - 0.2))
        if self._high_z <= self._low_z + 0.05:
            self._high_z = self._low_z + 0.2

        self._settle_time = 0.5
        self._climb_time = 2.0
        self._hover_time = 0.75
        self._descend_time = 2.0
        self._finish_time = 0.5

        self._hover_thrust = 0.33
        self._kp_z = 0.9
        self._kd_z = 0.35

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        z_des, vz_des = self._vertical_profile(self._time)

        if self._control_mode == "state":
            action = np.zeros(13, dtype=float)
            action[0] = self._start_pos[0]
            action[1] = self._start_pos[1]
            action[2] = z_des
            action[5] = vz_des
            action[9] = self._start_yaw
            return action

        z = float(obs["pos"][2])
        vz = float(obs["vel"][2])
        thrust = self._hover_thrust + self._kp_z * (z_des - z) + self._kd_z * (vz_des - vz)
        thrust = float(np.clip(thrust, 0.0, 0.6))
        return np.array([0.0, 0.0, self._start_yaw, thrust], dtype=float)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._time += self._dt

        landed = (
            self._time >= self._total_time
            and abs(float(obs["pos"][2]) - self._low_z) < 0.05
            and abs(float(obs["vel"][2])) < 0.1
        )
        return landed

    @property
    def _total_time(self) -> float:
        return (
            self._settle_time
            + self._climb_time
            + self._hover_time
            + self._descend_time
            + self._finish_time
        )

    def _vertical_profile(self, t: float) -> tuple[float, float]:
        if t < self._settle_time:
            return self._low_z, 0.0

        t -= self._settle_time
        dz = self._high_z - self._low_z

        if t < self._climb_time:
            alpha = t / self._climb_time
            return self._low_z + alpha * dz, dz / self._climb_time

        t -= self._climb_time
        if t < self._hover_time:
            return self._high_z, 0.0

        t -= self._hover_time
        if t < self._descend_time:
            alpha = t / self._descend_time
            return self._high_z - alpha * dz, -dz / self._descend_time

        return self._low_z, 0.0

    @staticmethod
    def _quat_to_yaw(quat: NDArray[np.floating]) -> float:
        x, y, z, w = np.asarray(quat, dtype=float)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
