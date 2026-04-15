"""Manually fly a single drone in sim and inspect reward/progress terms."""

from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import ece484_fly.envs  # noqa: F401
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def _read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def _distance_to_target_gate(obs: dict) -> float:
    target_gate = int(obs["target_gate"])
    if target_gate == -1:
        return 0.0
    return float(np.linalg.norm(obs["gates_pos"][target_gate] - obs["pos"]))


def debug_reward(
    config: str = "level1.toml",
    seed: int = 0,
    step_size_xy: float = 0.10,
    step_size_z: float = 0.05,
    yaw_step: float = 0.15,
    render: bool = True,
    device: str = "auto",
) -> None:
    """Run a manual-control debug loop for inspecting reward behavior."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.env.control_mode = "state"
    cfg.sim.render = render
    device = select_device(device)

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode="state",
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        device=device,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset(seed=seed)
    target_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
    target_yaw = 0.0

    print("Manual reward debug")
    print("Controls: w/s x, a/d y, r/f z, q/e yaw, space hover, c reset, x quit")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        prev_dist = _distance_to_target_gate(obs)
        step_idx = 0
        while True:
            key = _read_key()
            if key == "x":
                break
            if key == "w":
                target_pos[0] += step_size_xy
            elif key == "s":
                target_pos[0] -= step_size_xy
            elif key == "a":
                target_pos[1] += step_size_xy
            elif key == "d":
                target_pos[1] -= step_size_xy
            elif key == "r":
                target_pos[2] += step_size_z
            elif key == "f":
                target_pos[2] -= step_size_z
            elif key == "q":
                target_yaw += yaw_step
            elif key == "e":
                target_yaw -= yaw_step
            elif key == " ":
                target_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
            elif key == "c":
                obs, info = env.reset()
                target_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
                target_yaw = 0.0
                prev_dist = _distance_to_target_gate(obs)
                step_idx = 0
                print("Environment reset.")
                continue

            action = np.array(
                [
                    target_pos[0],
                    target_pos[1],
                    target_pos[2],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    target_yaw,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            curr_dist = _distance_to_target_gate(obs)
            dist_delta = prev_dist - curr_dist

            if step_idx % 5 == 0 or key is not None or terminated or truncated:
                print(
                    f"step={step_idx} key={key!r} reward={float(reward):+.4f} "
                    f"target_gate={int(obs['target_gate'])} dist={curr_dist:.3f} "
                    f"delta={dist_delta:+.4f} pos={np.round(obs['pos'], 3).tolist()}"
                )

            if render:
                env.render()
                time.sleep(1 / 60)

            if terminated or truncated:
                print("Episode ended, resetting.")
                obs, info = env.reset()
                target_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
                target_yaw = 0.0
                prev_dist = _distance_to_target_gate(obs)
                step_idx = 0
                continue

            prev_dist = curr_dist
            step_idx += 1
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(debug_reward)
