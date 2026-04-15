"""Small CPU smoke tests for the JAX env wrappers."""

from __future__ import annotations

import logging

import fire
import jax
import jax.numpy as jnp
import numpy as np

from ece484_fly.envs.jax_env import FunctionalJaxVecDroneRaceEnv, JaxVecDroneRaceEnv


logger = logging.getLogger(__name__)


def smoke_jax_env(
    config: str = "level1.toml",
    num_envs: int = 4,
    seed: int = 0,
    steps: int = 3,
) -> None:
    """Reset and step the thin JAX env wrapper on CPU."""
    env = JaxVecDroneRaceEnv(config=config, num_envs=num_envs, seed=seed, device="cpu")
    try:
        obs, info = env.reset(seed=seed)
        print("JAX devices:", jax.devices())
        print("Obs type:", type(obs["pos"]))
        print("Obs device:", obs["pos"].device)
        print("Obs shape:", obs["pos"].shape)
        action = jnp.zeros((env.num_envs, env.single_action_space.shape[0]), dtype=jnp.float32)
        for step_idx in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step_idx} reward_shape={reward.shape} reward_device={reward.device} "
                f"done_mean={float(jnp.mean(jnp.logical_or(terminated, truncated))):.4f}"
            )
    finally:
        env.close()


def smoke_scan_env(
    config: str = "level1.toml",
    num_envs: int = 4,
    seed: int = 0,
    steps: int = 8,
) -> None:
    """Reset and roll out the functional JAX env inside ``lax.scan`` on CPU."""
    env = FunctionalJaxVecDroneRaceEnv(config=config, num_envs=num_envs, seed=seed, device="cpu")
    state, obs = env.reset(seed=seed)
    action_dim = env.single_action_space.shape[0]
    print("JAX devices:", jax.devices())
    print("Obs device:", obs["pos"].device)
    print("Obs shape:", obs["pos"].shape)

    def scan_step(carry, _):
        state = carry
        action = jnp.zeros((env.num_envs, action_dim), dtype=jnp.float32)
        state, obs, reward, terminated, truncated, _ = env.step_fn(state, action)
        done = jnp.logical_or(terminated, truncated)
        return state, {
            "reward": reward,
            "done": done,
            "target_gate": obs["target_gate"],
        }

    state, traj = jax.lax.scan(scan_step, state, xs=None, length=steps)
    print("Trajectory reward shape:", traj["reward"].shape)
    print("Trajectory reward device:", traj["reward"].device)
    print("Mean reward:", float(jnp.mean(traj["reward"])))
    print("Mean done rate:", float(jnp.mean(traj["done"])))
    print("Final target gates:", np.asarray(traj["target_gate"][-1]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "thin": smoke_jax_env,
            "scan": smoke_scan_env,
        }
    )
