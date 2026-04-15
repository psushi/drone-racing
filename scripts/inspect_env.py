"""Inspect environment action ranges and basic physical quantities."""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
import numpy as np

import ece484_fly.envs  # noqa: F401
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def inspect_env(
    config: str = "level1.toml",
    seed: int = 0,
    device: str = "auto",
) -> None:
    """Print action-space and drone-parameter information for the sim env."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    device = select_device(device)
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        device=device,
    )

    try:
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        print("Control mode:", cfg.env.control_mode)
        print("Action low:", action_low)
        print("Action high:", action_high)

        drone_mass = np.asarray(env.unwrapped.drone_mass, dtype=np.float32)
        hover_thrust = drone_mass * np.float32(9.81)
        print("Drone mass (kg):", drone_mass)
        print("Approx hover thrust (N):", hover_thrust)

        if cfg.env.control_mode == "attitude":
            thrust_low = action_low[0]
            thrust_high = action_high[0]
            thrust_mid = 0.5 * (thrust_low + thrust_high)
            hover_normalized = 2.0 * (hover_thrust - thrust_low) / (thrust_high - thrust_low) - 1.0
            print("Thrust action range:", np.array([thrust_low, thrust_high], dtype=np.float32))
            print("Thrust midpoint:", thrust_mid)
            print("Hover normalized in [-1, 1]:", hover_normalized)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(inspect_env)
