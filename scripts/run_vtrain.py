"""Inspect the vectorized drone racing environment."""

from __future__ import annotations
import optax
from ece484_fly.train.train import flatten_rollout_batch, make_minibatches, make_update_fn
from ece484_fly.train.ppo import compute_gae
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


def run_train(
    config: str = "level1.toml",
    num_envs: int = 4,
    seed: int = 0,
) -> None:
    """Create the vectorized env and print observation/action metadata."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    env = gymnasium.make_vec(
        cfg.env.id,
        num_envs= num_envs if cfg.train.num_envs is None else cfg.train.num_envs,
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
    prng, key = jax.random.split(prng)
    params = model.init(prng, jnp.zeros((env.num_envs, 22)))
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm),
        optax.adamw(learning_rate=cfg.train.lr, eps=1e-8),
    )
    opt_state = tx.init(params)
    update_step = make_update_fn(model, tx, cfg.train.clip_eps, cfg.train.vf_coef, cfg.train.ent_coef)
    obs, info = env.reset(seed=seed)
    try:
        for i in range(cfg.train.num_iterations):
            print("Iteration:",i)
            observations = []
            log_probs = []
            values = []
            dones = []
            rewards = []
            actions = []
            for i in range(cfg.train.num_steps):
                policy_obs = flatten_obs(obs, vectorized=True)
                pi, value  = model.apply(params, jnp.asarray(policy_obs))
                prng, key = jax.random.split(prng)
                action = pi.sample(seed=key)
                log_prob = pi.log_prob(action)
                observations.append(policy_obs)
                log_probs.append(log_prob)
                values.append(value)
                actions.append(action)

                (obs, reward, terminated, truncated, info) = env.step(np.asarray(action))
                done = np.logical_or(terminated, truncated)
                dones.append(done)
                rewards.append(reward)

            last_obs = flatten_obs(obs, vectorized=True)
            _, last_value = model.apply(params, jnp.asarray(last_obs))
            values = np.stack(values)
            log_probs = np.stack(log_probs)
            dones = np.stack(dones)
            rewards = np.stack(rewards)
            actions = jnp.stack(actions)
            observations = jnp.stack(observations)


            # Compute the advantages and returns
            advantages, returns = compute_gae(
                values=values,
                rewards=rewards,
                dones=dones,
                gamma=cfg.train.gamma,
                lambda_=cfg.train.lambda_,
                last_value=np.asarray(last_value),
            )

            flat_obs = flatten_rollout_batch(
                observations, 
                actions, 
                jnp.asarray(log_probs),
                jnp.asarray(values),
                jnp.asarray(advantages),
                jnp.asarray(returns)
            )


            for i in range(cfg.train.num_epochs):
                print("Epoch:", i)
                prng, key = jax.random.split(prng)
                minibatches = make_minibatches(flat_obs, key, cfg.train.num_minibatches)
                for minibatch in minibatches:
                    params, opt_state, metrics = update_step(params, opt_state, minibatch)
                    if i % 10 == 0:
                        print("Actor loss:", metrics["actor_loss"].item())
                        print("Value loss:", metrics["value_loss"].item())
                        print("Entropy:", metrics["entropy"].item())

    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(run_train)
