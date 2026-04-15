"""Train PPO on a single drone racing environment."""

from __future__ import annotations
from ece484_fly.train.utils import normalize_actions, select_device

import logging
from pathlib import Path

import fire
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax

import ece484_fly.envs  # noqa: F401
from ece484_fly.train import flatten_obs
from ece484_fly.train.actor_critic_models import ActorCritic
from ece484_fly.train.ppo import compute_gae
from ece484_fly.train.train import flatten_rollout_batch, make_minibatches, make_update_fn
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def run_train(
    config: str = "level1.toml",
    seed: int = 0,
    device: str = "auto",
) -> None:
    """Create the single-env trainer and run PPO."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
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

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    prng = jax.random.PRNGKey(seed)
    prng, key = jax.random.split(prng)
    params = model.init(prng, jnp.zeros((1, 22)))
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm),
        optax.adamw(learning_rate=cfg.train.lr, eps=1e-8),
    )
    opt_state = tx.init(params)
    update_step = make_update_fn(model, tx, cfg.train.clip_eps, cfg.train.vf_coef, cfg.train.ent_coef)
    obs, info = env.reset(seed=seed)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    try:
        for i in range(cfg.train.num_iterations):
            print("Iteration:", i)
            observations = []
            log_probs = []
            values = []
            dones = []
            rewards = []
            actions = []
            for i in range(cfg.train.num_steps):
                prev_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
                policy_obs = flatten_obs(obs, vectorized=False)
                policy_obs_batch = jnp.asarray(policy_obs)[None, :]
                pi, value = model.apply(params, policy_obs_batch)
                prng, key = jax.random.split(prng)
                action = pi.sample(seed=key)
                log_prob = pi.log_prob(action)
                observations.append(np.asarray(policy_obs)[None, :])
                log_probs.append(np.asarray(log_prob))
                values.append(np.asarray(value))
                actions.append(np.asarray(action))
                raw_action = np.asarray(action[0], dtype=np.float32)
                action = normalize_actions(raw_action, action_low, action_high)
                obs, reward, terminated, truncated, info = env.step(action)
                if i < 10:
                    print("Raw action:", raw_action)
                    print("Applied action:", action)
                    print("Reward:", reward)
                    print("Prev pos:", prev_pos)
                    print("Curr pos:", np.asarray(obs["pos"], dtype=np.float32))
                    print("Delta pos:", np.asarray(obs["pos"], dtype=np.float32) - prev_pos)
                done = terminated or truncated
                dones.append(np.asarray([done], dtype=bool))
                rewards.append(np.asarray([reward], dtype=np.float32))
                if done:
                    print("Episode ended. Resetting env.")
                    obs, info = env.reset()

            last_obs = flatten_obs(obs, vectorized=False)
            _, last_value = model.apply(params, jnp.asarray(last_obs)[None, :])
            values = np.stack(values)
            log_probs = np.stack(log_probs)
            dones = np.stack(dones)
            rewards = np.stack(rewards)
            actions = jnp.stack(actions)
            observations = jnp.asarray(np.stack(observations))

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
                jnp.asarray(returns),
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
