"""Validate attitude action ordering and sign by commanding one axis at a time."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import ece484_fly.envs  # noqa: F401
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def _mean_delta(history: list[np.ndarray]) -> np.ndarray:
    if len(history) < 2:
        return np.zeros(3, dtype=np.float32)
    deltas = np.diff(np.stack(history, axis=0), axis=0)
    return deltas.mean(axis=0)


def validate(
    config: str = "level1.toml",
    seed: int = 0,
    device: str = "auto",
    render: bool = True,
    hold_steps: int = 60,
    settle_steps: int = 30,
    angle_test_mag: float = 0.25,
    yaw_test_mag: float = 0.75,
    thrust_bias: float = 0.05,
) -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.env.control_mode = "attitude"
    cfg.sim.render = render
    device = select_device(device)

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode="attitude",
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
        device=device,
    )
    env = JaxToNumpy(env)

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_mid = 0.5 * (action_low + action_high)

    print("Attitude action validation")
    print(f"Action low : {action_low}")
    print(f"Action high: {action_high}")
    print(f"Action mid : {action_mid}")
    print("Expected order in env code: [roll, pitch, yaw, thrust]")
    print()

    hover_action = action_mid.copy()
    hover_action[3] = np.clip(action_mid[3] + thrust_bias, action_low[3], action_high[3])

    tests = [
        ("baseline_hoverish", hover_action.copy()),
        ("thrust_high", hover_action.copy()),
        ("thrust_low", hover_action.copy()),
        ("roll_pos", hover_action.copy()),
        ("roll_neg", hover_action.copy()),
        ("pitch_pos", hover_action.copy()),
        ("pitch_neg", hover_action.copy()),
        ("yaw_pos", hover_action.copy()),
        ("yaw_neg", hover_action.copy()),
    ]

    tests[1][1][3] = action_high[3]
    tests[2][1][3] = action_low[3]
    tests[3][1][0] = np.clip(angle_test_mag, action_low[0], action_high[0])
    tests[4][1][0] = np.clip(-angle_test_mag, action_low[0], action_high[0])
    tests[5][1][1] = np.clip(angle_test_mag, action_low[1], action_high[1])
    tests[6][1][1] = np.clip(-angle_test_mag, action_low[1], action_high[1])
    tests[7][1][2] = np.clip(yaw_test_mag, action_low[2], action_high[2])
    tests[8][1][2] = np.clip(-yaw_test_mag, action_low[2], action_high[2])

    for name, action in tests:
        obs, _ = env.reset(seed=seed)
        pos_hist: list[np.ndarray] = []

        for _ in range(settle_steps):
            obs, _, terminated, truncated, _ = env.step(hover_action)
            if render:
                env.render()
                time.sleep(1 / 120)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed)

        start_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
        start_quat = np.asarray(obs["quat"], dtype=np.float32).copy()

        for _ in range(hold_steps):
            obs, reward, terminated, truncated, _ = env.step(action)
            pos_hist.append(np.asarray(obs["pos"], dtype=np.float32).copy())
            if render:
                env.render()
                time.sleep(1 / 120)
            if terminated or truncated:
                break

        end_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
        end_quat = np.asarray(obs["quat"], dtype=np.float32).copy()
        mean_dp = _mean_delta(pos_hist)

        print(name)
        print(f"  action    : {np.round(action, 4).tolist()}")
        print(f"  start_pos : {np.round(start_pos, 4).tolist()}")
        print(f"  end_pos   : {np.round(end_pos, 4).tolist()}")
        print(f"  delta_pos : {np.round(end_pos - start_pos, 4).tolist()}")
        print(f"  mean_step : {np.round(mean_dp, 5).tolist()}")
        print(f"  start_quat: {np.round(start_quat, 4).tolist()}")
        print(f"  end_quat  : {np.round(end_quat, 4).tolist()}")
        print(f"  reward    : {float(reward):+.4f}")
        print()

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(validate)
