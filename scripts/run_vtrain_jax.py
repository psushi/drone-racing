"""Vectorized PPO trainer using the functional JAX env wrapper for rollout collection."""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from rich.live import Live
from rich.table import Table

from ece484_fly.envs.jax_env import FunctionalJaxVecDroneRaceEnv
from ece484_fly.train.actor_critic_models import ActorCritic
from ece484_fly.train.obs import flatten_obs_jax
from ece484_fly.train.ppo import compute_gae_jax
from ece484_fly.train.train import flatten_rollout_batch, make_minibatches, make_update_fn
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


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


def scale_actions_jax(actions: jax.Array, action_low: jax.Array, action_high: jax.Array) -> jax.Array:
    norm_actions = jnp.tanh(actions)
    return action_low + 0.5 * (norm_actions + 1.0) * (action_high - action_low)


def run_train(
    config: str = "level1.toml",
    num_envs: int = 100,
    seed: int = 0,
    checkpoint_path: str = "artifacts/policy_jax.msgpack",
    device: str = "auto",
) -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)

    env = FunctionalJaxVecDroneRaceEnv(
        config=config,
        num_envs=num_envs if cfg.train.num_envs is None else cfg.train.num_envs,
        seed=seed,
        device=device,
    )
    action_low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    prng = jax.random.PRNGKey(seed)
    params = model.init(prng, jnp.zeros((env.num_envs, 22), dtype=jnp.float32))
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm),
        optax.adamw(learning_rate=cfg.train.lr, eps=1e-8),
    )
    opt_state = tx.init(params)
    update_step = make_update_fn(model, tx, cfg.train.clip_eps, cfg.train.vf_coef, cfg.train.ent_coef)
    env_state, obs = env.reset(seed=seed)

    def rollout_step(carry, _):
        params, env_state, obs, rng = carry
        policy_obs = flatten_obs_jax(obs, vectorized=True)
        pi, value = model.apply(params, policy_obs)
        rng, key = jax.random.split(rng)
        raw_action = pi.sample(seed=key)
        log_prob = pi.log_prob(raw_action)
        action = scale_actions_jax(raw_action, action_low, action_high)
        env_state, next_obs, reward, terminated, truncated, _ = env.step_fn(env_state, action)
        done = jnp.logical_or(terminated, truncated)
        transition = {
            "obs": policy_obs,
            "actions": raw_action,
            "old_log_probs": log_prob,
            "old_values": value,
            "rewards": reward,
            "dones": done,
        }
        return (params, env_state, next_obs, rng), transition

    collect_rollout = jax.jit(
        lambda params, env_state, obs, rng: jax.lax.scan(
            rollout_step,
            (params, env_state, obs, rng),
            xs=None,
            length=cfg.train.num_steps,
        )
    )

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
                (params_ref, env_state, obs, prng), traj = collect_rollout(params, env_state, obs, prng)
                del params_ref
                last_obs = flatten_obs_jax(obs, vectorized=True)
                _, last_value = model.apply(params, last_obs)
                advantages, returns = compute_gae_jax(
                    values=traj["old_values"],
                    rewards=traj["rewards"],
                    dones=traj["dones"],
                    gamma=cfg.train.gamma,
                    lambda_=cfg.train.lambda_,
                    last_value=last_value,
                )

                flat_obs = flatten_rollout_batch(
                    traj["obs"],
                    traj["actions"],
                    traj["old_log_probs"],
                    traj["old_values"],
                    advantages,
                    returns,
                )

                metrics = None
                for _ in range(cfg.train.num_epochs):
                    prng, key = jax.random.split(prng)
                    minibatches = make_minibatches(flat_obs, key, cfg.train.num_minibatches)
                    for minibatch in minibatches:
                        params, opt_state, metrics = update_step(params, opt_state, minibatch)

                if metrics is not None:
                    rewards_np = jax.device_get(traj["rewards"])
                    dones_np = jax.device_get(traj["dones"])
                    total_reward_sum += float(rewards_np.sum())
                    total_reward_count += int(rewards_np.size)
                    live_metrics = {
                        "iteration": iter_idx,
                        "mean_reward": float(rewards_np.mean()),
                        "running_mean_reward": total_reward_sum / total_reward_count,
                        "done_rate": float(dones_np.mean()),
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
