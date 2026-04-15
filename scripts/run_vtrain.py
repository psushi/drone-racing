"""Inspect the vectorized drone racing environment."""

from __future__ import annotations
from flax import serialization
from ece484_fly.train.utils import normalize_actions
from ece484_fly.envs.drone_race import VecDroneRaceEnv
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
from rich.live import Live
from rich.table import Table

import ece484_fly.envs  # noqa: F401
from ece484_fly.utils import load_config
from ece484_fly.train import flatten_obs
import numpy as np
from gymnasium.vector import VectorEnv


logger = logging.getLogger(__name__)


def make_metrics_table(metrics: dict[str, float | int]) -> Table:
    table = Table(title="Training")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    return table


def run_train(
    config: str = "level1.toml",
    num_envs: int = 100,
    seed: int = 0,
    checkpoint_path: str = "artifacts/policy.msgpack",
) -> None:
    """Create the vectorized env and print observation/action metadata."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    env: VectorEnv = gymnasium.make_vec(
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
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    total_reward_sum = 0.0
    total_reward_count = 0
    try:
        live_metrics = {
            "iteration": 0,
            "mean_reward": 0.0,
            "running_mean_reward": 0.0,
            "done_rate": 0.0,
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        with Live(make_metrics_table(live_metrics), refresh_per_second=4) as live:
            for iter_idx in range(cfg.train.num_iterations):
                observations = []
                log_probs = []
                values = []
                dones = []
                rewards = []
                actions = []
                for _ in range(cfg.train.num_steps):
                    policy_obs = flatten_obs(obs, vectorized=True)
                    pi, value = model.apply(params, jnp.asarray(policy_obs))
                    prng, key = jax.random.split(prng)
                    action = pi.sample(seed=key)
                    log_prob = pi.log_prob(action)
                    observations.append(policy_obs)
                    log_probs.append(log_prob)
                    values.append(value)
                    actions.append(action)

                    action = normalize_actions(action, action_low, action_high)
                    obs, reward, terminated, truncated, info = env.step(np.asarray(action))
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

                metrics = None
                for _ in range(cfg.train.num_epochs):
                    prng, key = jax.random.split(prng)
                    minibatches = make_minibatches(flat_obs, key, cfg.train.num_minibatches)
                    for minibatch in minibatches:
                        params, opt_state, metrics = update_step(params, opt_state, minibatch)

                if metrics is not None:
                    total_reward_sum += float(np.sum(rewards))
                    total_reward_count += int(rewards.size)
                    live_metrics = {
                        "iteration": iter_idx,
                        "mean_reward": float(np.mean(rewards)),
                        "running_mean_reward": total_reward_sum / total_reward_count,
                        "done_rate": float(np.mean(dones)),
                        "actor_loss": float(metrics["actor_loss"]),
                        "value_loss": float(metrics["value_loss"]),
                        "entropy": float(metrics["entropy"]),
                    }
                    live.update(make_metrics_table(live_metrics))
            print(f"Final running mean reward: {total_reward_sum / total_reward_count:.4f}")
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.write_bytes(serialization.to_bytes(params))
            print(f"Saved checkpoint to {checkpoint_file}")

    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(run_train)
