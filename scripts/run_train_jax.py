"""Fully JAX-native PPO trainer for the functional drone racing env."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from rich.live import Live
from rich.table import Table

from ece484_fly.envs.jax_env import FunctionalJaxVecDroneRaceEnv
from ece484_fly.train.actor_critic_models import ActorCritic
from ece484_fly.train.experiment_io import (
    choose_runtime_config_path,
    default_metadata,
    normalize_checkpoint_path,
    write_experiment_sidecar,
)
from ece484_fly.train.obs import POLICY_OBS_DIM, flatten_obs_jax
from ece484_fly.train.ppo import compute_gae_jax, ppo_loss
from ece484_fly.train.utils import select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class RolloutBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    next_value: jax.Array
    reward: jax.Array
    terminated: jax.Array
    done: jax.Array
    passed: jax.Array
    task_complete: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    last_obs: jax.Array
    rng: jax.Array


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


def default_wandb_run_name(checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix == ".msgpack":
        return checkpoint.parent.name or checkpoint.stem
    return checkpoint.name


def completed_episode_stats(
    passed: np.ndarray,
    done: np.ndarray,
    task_complete: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return completed-episode gate stats and task completion reliability."""
    num_steps, num_envs = passed.shape
    episode_gate_counts: list[int] = []
    episode_task_success: list[bool] = []
    running_gate_counts = np.zeros(num_envs, dtype=np.int32)

    for step_idx in range(num_steps):
        running_gate_counts += passed[step_idx].astype(np.int32)
        finished_envs = np.nonzero(done[step_idx])[0]
        if finished_envs.size > 0:
            episode_gate_counts.extend(running_gate_counts[finished_envs].tolist())
            episode_task_success.extend(task_complete[step_idx, finished_envs].astype(bool).tolist())
            running_gate_counts[finished_envs] = 0

    if not episode_gate_counts:
        return 0.0, 0.0, 0.0, 0

    counts = np.asarray(episode_gate_counts, dtype=np.float32)
    success = np.asarray(episode_task_success, dtype=bool)
    return float(counts.mean()), float((counts >= 1).mean()), float(success.mean()), int(counts.size)


def scale_actions_jax(actions: jax.Array, action_low: jax.Array, action_high: jax.Array) -> jax.Array:
    return jnp.clip(actions, action_low, action_high)


def _select_obs_by_mask(
    reset_obs: dict[str, jax.Array],
    final_obs: dict[str, jax.Array],
    mask: jax.Array,
) -> dict[str, jax.Array]:
    return jax.tree_util.tree_map(
        lambda reset_x, final_x: jnp.where(
            mask.reshape(mask.shape + (1,) * (reset_x.ndim - 1)),
            final_x,
            reset_x,
        ),
        reset_obs,
        final_obs,
    )


def create_train_state(
    rng: jax.Array,
    model: ActorCritic,
    obs_dim: int,
    lr: float,
    max_grad_norm: float,
) -> TrainState:
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr, eps=1e-8),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def flatten_batch(rollout: RolloutBatch, advantages: jax.Array, returns: jax.Array) -> dict[str, jax.Array]:
    return {
        "obs": rollout.obs.reshape((-1, rollout.obs.shape[-1])),
        "action": rollout.action.reshape((-1, rollout.action.shape[-1])),
        "old_log_prob": rollout.log_prob.reshape((-1,)),
        "old_value": rollout.value.reshape((-1,)),
        "advantages": advantages.reshape((-1,)),
        "returns": returns.reshape((-1,)),
    }


def make_minibatches(batch: dict[str, jax.Array], rng: jax.Array, num_minibatches: int) -> dict[str, jax.Array]:
    batch_size = batch["obs"].shape[0]
    if batch_size % num_minibatches != 0:
        raise ValueError(f"Batch size {batch_size} not divisible by {num_minibatches=}")
    minibatch_size = batch_size // num_minibatches
    perm = jax.random.permutation(rng, batch_size)
    shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]),
        shuffled,
    )


def make_update_step(model: ActorCritic, clip_eps: float, vf_coef: float):
    @jax.jit
    def update_step(train_state: TrainState, minibatch: dict[str, jax.Array], ent_coef: jax.Array):
        def loss_fn(params):
            return ppo_loss(
                params,
                model,
                minibatch["obs"],
                minibatch["action"],
                minibatch["old_log_prob"],
                minibatch["old_value"],
                minibatch["advantages"],
                minibatch["returns"],
                clip_eps,
                vf_coef,
                ent_coef,
            )

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {
            "total_loss": total_loss,
            "actor_loss": aux[0],
            "value_loss": aux[1],
            "entropy": aux[2],
        }
        return train_state, metrics

    return update_step


def make_collect_rollout_fn(
    model: ActorCritic,
    env: FunctionalJaxVecDroneRaceEnv,
    action_low: jax.Array,
    action_high: jax.Array,
    num_steps: int,
):
    def collect_rollout(runner_state: RunnerState):
        def env_step(carry, _):
            train_state, env_state, last_obs, rng = carry
            rng, action_key = jax.random.split(rng)

            pi, value = model.apply(train_state.params, last_obs)
            raw_action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(raw_action)
            action = scale_actions_jax(raw_action, action_low, action_high)

            env_state, next_obs_dict, reward, terminated, truncated, info = env.step_fn(env_state, action)
            done = jnp.logical_or(terminated, truncated)
            next_obs = flatten_obs_jax(next_obs_dict, vectorized=True)
            bootstrap_obs_dict = _select_obs_by_mask(
                next_obs_dict,
                info["final_observation"],
                truncated,
            )
            bootstrap_obs = flatten_obs_jax(bootstrap_obs_dict, vectorized=True)
            _, next_value = model.apply(train_state.params, bootstrap_obs)

            transition = RolloutBatch(
                obs=last_obs,
                action=raw_action,
                log_prob=log_prob,
                value=value,
                next_value=next_value,
                reward=reward,
                terminated=terminated,
                done=done,
                passed=info["passed"],
                task_complete=info["final_observation"]["target_gate"] == -1,
            )
            next_carry = RunnerState(train_state, env_state, next_obs, rng)
            return next_carry, transition

        return jax.lax.scan(env_step, runner_state, None, length=num_steps)

    return jax.jit(collect_rollout)


def make_train_iteration_fn(
    model: ActorCritic,
    env: FunctionalJaxVecDroneRaceEnv,
    action_low: jax.Array,
    action_high: jax.Array,
    num_steps: int,
    num_minibatches: int,
    update_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coef: float,
):
    collect_rollout = make_collect_rollout_fn(model, env, action_low, action_high, num_steps)
    update_step = make_update_step(model, clip_eps, vf_coef)

    @jax.jit
    def train_iteration(runner_state: RunnerState, ent_coef: jax.Array):
        runner_state, rollout = collect_rollout(runner_state)
        advantages, returns = compute_gae_jax(
            rollout.value,
            rollout.reward,
            rollout.next_value,
            rollout.done,
            rollout.terminated,
            gamma,
            gae_lambda,
        )
        batch = flatten_batch(rollout, advantages, returns)

        def epoch_step(carry, _):
            train_state, rng = carry
            rng, mb_key = jax.random.split(rng)
            minibatches = make_minibatches(batch, mb_key, num_minibatches)

            def minibatch_step(train_state: TrainState, minibatch: dict[str, jax.Array]):
                return update_step(train_state, minibatch, ent_coef)

            train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
            mean_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
            return (train_state, rng), mean_metrics

        (train_state, rng), metrics = jax.lax.scan(
            epoch_step,
            (runner_state.train_state, runner_state.rng),
            None,
            length=update_epochs,
        )
        runner_state = RunnerState(train_state, runner_state.env_state, runner_state.last_obs, rng)
        return runner_state, rollout, metrics

    return train_iteration


def run_train(
    config: str = "level1_flat.toml",
    num_envs: int | None = None,
    seed: int | None = None,
    checkpoint_path: str = "artifacts/policy_jax/model.msgpack",
    init_checkpoint_path: str = "",
    device: str = "auto",
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_mode: str = "",
) -> None:
    checkpoint_path = str(normalize_checkpoint_path(checkpoint_path))
    init_checkpoint_path = str(normalize_checkpoint_path(init_checkpoint_path)) if init_checkpoint_path else ""
    repo_root = Path(__file__).parents[1]
    resolved_config_path = choose_runtime_config_path(
        repo_root,
        checkpoint_path,
        config,
        prefer_checkpoint_sidecar=False,
    )
    cfg = load_config(resolved_config_path)
    cfg_seed = cfg.env.get("seed", 0)
    if seed is None:
        if cfg_seed == "random":
            raise ValueError(
                "Training reproducibility requires a fixed seed. Set env.seed to an integer in the config "
                "or pass --seed explicitly."
            )
        seed = int(cfg_seed)
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_project or str(wandb_cfg.get("project", ""))
    wandb_mode = wandb_mode or str(wandb_cfg.get("mode", "online"))
    wandb_run_name = wandb_run_name or str(wandb_cfg.get("run_name", "")) or default_wandb_run_name(
        checkpoint_path
    )
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)
    print(f"Using config: {resolved_config_path}")
    ent_anneal_iters = int(cfg.train.get("ent_anneal_iters", cfg.train.num_iterations))
    min_ent_coef = float(cfg.train.get("min_ent_coef", 5e-4))
    if ent_anneal_iters <= 0:
        raise ValueError(f"train.ent_anneal_iters must be positive, got {ent_anneal_iters}")

    env = FunctionalJaxVecDroneRaceEnv(
        config=str(resolved_config_path),
        num_envs=cfg.train.num_envs if num_envs is None else num_envs,
        seed=seed,
        device=device,
    )
    transitions_per_iter = env.num_envs * cfg.train.num_steps
    print(
        "Training setup:",
        f"num_envs={env.num_envs}",
        f"num_steps={cfg.train.num_steps}",
        f"transitions_per_iter={transitions_per_iter}",
        f"ent_anneal_iters={ent_anneal_iters}",
    )
    sidecar_config_path, sidecar_metadata_path = write_experiment_sidecar(
        checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path,
        cfg=cfg,
        metadata=default_metadata(
            repo_root=repo_root,
            seed=seed,
            device=device,
            num_envs_effective=env.num_envs,
            transitions_per_iter=transitions_per_iter,
        ),
    )
    print(f"Saved run config to {sidecar_config_path}")
    print(f"Saved run metadata to {sidecar_metadata_path}")
    action_low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    train_state = create_train_state(
        init_key,
        model,
        obs_dim=POLICY_OBS_DIM,
        lr=cfg.train.lr,
        max_grad_norm=cfg.train.max_grad_norm,
    )
    if init_checkpoint_path:
        train_state = train_state.replace(
            params=serialization.from_bytes(
                train_state.params,
                Path(init_checkpoint_path).read_bytes(),
            )
        )
        print(f"Initialized policy from {init_checkpoint_path}")
    env_state, obs_dict = env.reset(seed=seed)
    last_obs = flatten_obs_jax(obs_dict, vectorized=True)
    runner_state = RunnerState(train_state, env_state, last_obs, rng)

    wandb_run = None
    if wandb_project:
        if wandb is None:
            print("wandb logging requested, but wandb is not installed in the current Pixi env.")
        else:
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=wandb_mode,
                config={
                    "config_file": str(resolved_config_path),
                    "seed": seed,
                    "device": device,
                    "checkpoint_path": checkpoint_path,
                    "init_checkpoint_path": init_checkpoint_path,
                    "wandb_project": wandb_project,
                    "wandb_run_name": wandb_run_name,
                    "wandb_mode": wandb_mode,
                    "train": dict(cfg.train),
                    "env_reward": dict(cfg.env.get("reward") or {}),
                    "num_envs_effective": env.num_envs,
                    "transitions_per_iter": transitions_per_iter,
                    "policy_obs_dim": POLICY_OBS_DIM,
                    "ent_anneal_iters_effective": ent_anneal_iters,
                    "min_ent_coef_effective": min_ent_coef,
                },
            )

    train_iteration = make_train_iteration_fn(
        model=model,
        env=env,
        action_low=action_low,
        action_high=action_high,
        num_steps=cfg.train.num_steps,
        num_minibatches=cfg.train.num_minibatches,
        update_epochs=cfg.train.num_epochs,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.lambda_,
        clip_eps=cfg.train.clip_eps,
        vf_coef=cfg.train.vf_coef,
    )

    total_reward_sum = 0.0
    total_reward_count = 0
    final_metrics: dict[str, float | int] = {}
    best_checkpoint_iteration = -1
    best_checkpoint_reliability = float("-inf")
    best_metrics = {
        "best_avg_gates_passed": float("-inf"),
        "best_episode_gates_mean": float("-inf"),
        "best_episode_pass_rate": float("-inf"),
        "best_completion_reliability": float("-inf"),
        "best_running_mean_reward": float("-inf"),
    }
    try:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        final_checkpoint_file = checkpoint_file.with_name(f"{checkpoint_file.stem}.final{checkpoint_file.suffix}")
        live_metrics = {
            "iteration": 0,
            "mean_reward": 0.0,
            "running_mean_reward": 0.0,
            "done_rate": 0.0,
            "avg_gates_passed": 0.0,
            "episode_gates_mean": 0.0,
            "episode_pass_rate": 0.0,
            "completion_reliability": 0.0,
            "episodes_completed": 0,
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ent_coef": float(cfg.train.ent_coef),
        }
        with Live(make_metrics_table(live_metrics), refresh_per_second=4) as live:
            for iter_idx in range(cfg.train.num_iterations):
                anneal_progress = min(iter_idx / max(ent_anneal_iters - 1, 1), 1.0)
                # Front-load exploration, then decay aggressively so the policy can
                # consolidate once it starts reliably passing gates. Annealing is
                # decoupled from total run length so short and long runs are comparable.
                ent_decay = (1.0 - anneal_progress) ** 2
                current_ent_coef = min_ent_coef + (cfg.train.ent_coef - min_ent_coef) * ent_decay
                runner_state, rollout, metrics = train_iteration(
                    runner_state, jnp.asarray(current_ent_coef, dtype=jnp.float32)
                )
                rewards = jax.device_get(rollout.reward)
                dones = jax.device_get(rollout.done)
                passed = jax.device_get(rollout.passed)
                task_complete = jax.device_get(rollout.task_complete)
                epoch_metrics = jax.device_get(metrics)
                (
                    episode_gates_mean,
                    episode_pass_rate,
                    completion_reliability,
                    episodes_completed,
                ) = completed_episode_stats(
                    passed,
                    dones,
                    task_complete,
                )

                total_reward_sum += float(rewards.sum())
                total_reward_count += int(rewards.size)
                live_metrics = {
                    "iteration": iter_idx,
                    "mean_reward": float(rewards.mean()),
                    "running_mean_reward": total_reward_sum / total_reward_count,
                    "done_rate": float(dones.mean()),
                    "avg_gates_passed": float(passed.sum(axis=0).mean()),
                    "episode_gates_mean": episode_gates_mean,
                    "episode_pass_rate": episode_pass_rate,
                    "completion_reliability": completion_reliability,
                    "episodes_completed": episodes_completed,
                    "actor_loss": float(epoch_metrics["actor_loss"][-1]),
                    "value_loss": float(epoch_metrics["value_loss"][-1]),
                    "entropy": float(epoch_metrics["entropy"][-1]),
                    "ent_coef": float(current_ent_coef),
                }
                final_metrics = dict(live_metrics)
                best_metrics["best_avg_gates_passed"] = max(
                    best_metrics["best_avg_gates_passed"],
                    live_metrics["avg_gates_passed"],
                )
                best_metrics["best_episode_gates_mean"] = max(
                    best_metrics["best_episode_gates_mean"],
                    live_metrics["episode_gates_mean"],
                )
                best_metrics["best_episode_pass_rate"] = max(
                    best_metrics["best_episode_pass_rate"],
                    live_metrics["episode_pass_rate"],
                )
                best_metrics["best_completion_reliability"] = max(
                    best_metrics["best_completion_reliability"],
                    live_metrics["completion_reliability"],
                )
                best_metrics["best_running_mean_reward"] = max(
                    best_metrics["best_running_mean_reward"],
                    live_metrics["running_mean_reward"],
                )
                if live_metrics["completion_reliability"] > best_checkpoint_reliability:
                    best_checkpoint_reliability = live_metrics["completion_reliability"]
                    best_checkpoint_iteration = iter_idx
                    checkpoint_file.write_bytes(serialization.to_bytes(runner_state.train_state.params))
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            **live_metrics,
                            "transitions_per_iter": transitions_per_iter,
                            "total_transitions": (iter_idx + 1) * transitions_per_iter,
                        },
                        step=iter_idx,
                    )
                live.update(make_metrics_table(live_metrics))

        print(f"Final running mean reward: {total_reward_sum / total_reward_count:.4f}")
        final_checkpoint_file.write_bytes(serialization.to_bytes(runner_state.train_state.params))
        write_experiment_sidecar(
            checkpoint_path=checkpoint_path,
            resolved_config_path=resolved_config_path,
            cfg=cfg,
            metadata={
                **default_metadata(
                    repo_root=repo_root,
                    seed=seed,
                    device=device,
                    num_envs_effective=env.num_envs,
                    transitions_per_iter=transitions_per_iter,
                ),
                "completed_iterations": int(cfg.train.num_iterations),
                "init_checkpoint_path": init_checkpoint_path,
                "final_running_mean_reward": total_reward_sum / total_reward_count,
                "final_metrics": final_metrics,
                "best_metrics": best_metrics,
                "best_checkpoint_iteration": best_checkpoint_iteration,
                "best_checkpoint_reliability": best_checkpoint_reliability,
                "best_checkpoint_path": str(checkpoint_file),
                "final_checkpoint_path": str(final_checkpoint_file),
            },
        )
        print(f"Saved best checkpoint to {checkpoint_file}")
        print(f"Saved final checkpoint to {final_checkpoint_file}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(run_train)
