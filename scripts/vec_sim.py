"""Inspect the vectorized drone racing environment."""

from __future__ import annotations
import jax
from ece484_fly.train.actor_critic_models import ActorCritic, Transition

import logging
from pathlib import Path

import fire
import gymnasium
import jax.numpy as jnp

import ece484_fly.envs  # noqa: F401
from ece484_fly.utils import load_config
from ece484_fly.train import flatten_obs
import numpy as np


logger = logging.getLogger(__name__)


def inspect_vec_env(
    config: str = "level1.toml",
    num_envs: int = 4,
    seed: int = 0,
) -> None:
    """Create the vectorized env and print observation/action metadata."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    env = gymnasium.make_vec(
        cfg.env.id,
        num_envs=num_envs,
        vectorization_mode="vector_entry_point",
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    )

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    prng = jax.random.PRNGKey(seed)
    params = model.init(prng, jnp.zeros((env.num_envs, 22)))
    try:
        obs, info = env.reset(seed=seed)
        policy_obs = flatten_obs(obs, vectorized=True)
        print("Flatten obs:",policy_obs.shape)
        pi, value  = model.apply(params, policy_obs)
        action = pi.sample(seed=seed)
        log_prob = pi.log_prob(action)
        print("Action shape:",action.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        policy_obs:np.ndarray = flatten_obs(obs, vectorized=True)
        transition = Transition(policy_obs, action, np.asarray(value), reward, terminated, log_prob)
        # print all the shapes

    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(inspect_vec_env)
