"""Minimal controller for simulation smoke tests."""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ece484_fly.control.controller import Controller


class WaypointController(Controller):
    """Simple waypoint controller."""
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._control_mode = config.env.control_mode
        self.phase = 0

    def compute_control(self, obs, info=None) -> NDArray[np.floating]:

        if self._control_mode == "attitude":
            raise NotImplementedError
        if self.phase == 0:
            target_gate = obs["target_gate"]
            gate_pos = obs["gates_pos"][target_gate]
            gate_quat = Rotation.from_quat(obs["gates_quat"][target_gate])
            yaw_only = Rotation.from_euler("z", gate_quat.as_euler("xyz")[2])
            r_180 = Rotation.from_euler("z", 180, degrees=True)
            target_quat = yaw_only * r_180
            target_yaw = target_quat.as_euler("xyz")[2]
            offset = yaw_only.apply(np.array([-0.2, 0, 0]))
            target_pos = gate_pos + offset

            diff = target_pos - obs["pos"]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 0.5:
                target_pos = obs["pos"] + diff / diff_norm * 0.5
            else:
                self.phase = 1
        elif self.phase == 1:
            target_gate = obs["target_gate"]
            gate_quat = Rotation.from_quat(obs["gates_quat"][target_gate])
            yaw_only = Rotation.from_euler('z', gate_quat.as_euler('xyz')[2])
            target_pos = obs["pos"] + yaw_only.apply(np.array([0.1, 0, 0]))
            target_yaw = yaw_only.as_euler("xyz")[2]



        # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]`
        return np.array([target_pos[0], target_pos[1], target_pos[2],0,0,0,0,0,0, 0, 0,0,0])

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        # Keep simulation running until the environment terminates/truncates naturally.
        return False
