"""Render a saved rsl_rl policy in a single simulation environment."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import gymnasium
import numpy as np
import torch
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from tensordict import TensorDict

import ece484_fly.envs  # noqa: F401
from ece484_fly.train.obs import flatten_obs
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config
from rsl_rl.models import MLPModel


logger = logging.getLogger(__name__)


def build_actor(policy_obs: torch.Tensor) -> MLPModel:
    return MLPModel(
        obs=TensorDict({"policy": policy_obs}, batch_size=[policy_obs.shape[0]]),
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        obs_set="actor",
        output_dim=4,
        hidden_dims=[256, 256, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules.GaussianDistribution",
            "init_std": 0.2,
            "std_type": "log",
        },
    )


def watch_policy(
    checkpoint_path: str = "artifacts/rsl_rl/policy_final.pt",
    config: str = "level1.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
) -> None:
    """Load a saved rsl_rl policy and render rollouts in a single env."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = True
    cfg.sim.pause = pause
    env_device = select_device(device)
    torch_device = "cuda" if env_device == "gpu" and torch.cuda.is_available() else "cpu"
    print("Using env device:", env_device)
    print("Using torch device:", torch_device)

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
        device=env_device,
    )
    env = JaxToNumpy(env)

    obs, info = env.reset(seed=seed)
    flat_obs = flatten_obs(obs, vectorized=False)
    actor = build_actor(torch.as_tensor(flat_obs[None, :], dtype=torch.float32, device=torch_device))
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    episode = 0
    step_idx = 0
    try:
        while True:
            policy_obs = flatten_obs(obs, vectorized=False)
            td = TensorDict(
                {"policy": torch.as_tensor(policy_obs[None, :], dtype=torch.float32, device=torch_device)},
                batch_size=[1],
            )
            with torch.no_grad():
                action = actor(td).squeeze(0).cpu().numpy().astype(np.float32)
            applied_action = action_low + 0.5 * (np.tanh(action) + 1.0) * (action_high - action_low)

            obs, reward, terminated, truncated, info = env.step(applied_action)
            env.render()
            if step_idx % 20 == 0:
                print(
                    f"episode={episode} step={step_idx} reward={float(reward):+.4f} "
                    f"target_gate={int(obs['target_gate'])}"
                )
            if terminated or truncated:
                episode += 1
                step_idx = 0
                print(f"Resetting after episode {episode}.")
                obs, info = env.reset()
                continue
            step_idx += 1
            time.sleep(1 / 60)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(watch_policy)
