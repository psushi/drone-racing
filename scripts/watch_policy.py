"""Render a saved policy in a single simulation environment."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import ece484_fly.envs  # noqa: F401
from ece484_fly.train import flatten_obs
from ece484_fly.train.actor_critic_models import ActorCritic
from ece484_fly.train.utils import normalize_actions, select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def watch_policy(
    checkpoint_path: str = "artifacts/policy_jax.msgpack",
    config: str = "level1.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
) -> None:
    """Load a saved policy and render rollouts in a single env."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = True
    cfg.sim.pause = pause
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)
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
    env = JaxToNumpy(env)

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 22)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    obs, info = env.reset(seed=seed)
    episode = 0
    step_idx = 0
    try:
        while True:
            policy_obs = flatten_obs(obs, vectorized=False)
            pi, value = model.apply(params, jnp.asarray(policy_obs)[None, :])
            action = np.asarray(pi.mean()[0], dtype=np.float32)
            applied_action = normalize_actions(action, action_low, action_high)
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
