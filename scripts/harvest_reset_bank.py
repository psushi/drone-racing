"""Harvest policy rollout states into a reset bank."""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from drone_racing_rl.envs.jax_env import FunctionalJaxVecDroneRaceEnv
from drone_racing_rl.train.actor_critic_models import ActorCritic
from drone_racing_rl.train.experiment_io import (
    checkpoint_directory,
    choose_runtime_config_path,
    normalize_checkpoint_path,
)
from drone_racing_rl.train.obs import POLICY_OBS_DIM, flatten_obs_jax
from drone_racing_rl.train.utils import normalize_actions, select_device


logger = logging.getLogger(__name__)


def harvest_reset_bank(
    checkpoint_path: str = "artifacts/policy_jax/model.msgpack",
    config: str = "level1_flat.toml",
    output_path: str = "",
    seed: int = 0,
    device: str = "auto",
    episodes: int = 32,
    max_states: int = 4096,
    sample_every: int = 2,
    min_target_gate: int = 1,
    min_speed: float = 0.5,
    max_speed: float = 8.0,
    max_ang_speed: float = 12.0,
) -> None:
    checkpoint_path = str(normalize_checkpoint_path(checkpoint_path))
    repo_root = Path(__file__).parents[1]
    resolved_config_path = choose_runtime_config_path(repo_root, checkpoint_path, config)
    output_file = (
        Path(output_path)
        if output_path
        else checkpoint_directory(checkpoint_path) / "reset_bank.npz"
    )

    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)
    print(f"Using config: {resolved_config_path}")

    env = FunctionalJaxVecDroneRaceEnv(
        config=str(resolved_config_path),
        num_envs=1,
        seed=seed,
        device=device,
    )
    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, POLICY_OBS_DIM)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())

    env_state, obs = env.reset(seed=seed)
    collected_pos: list[np.ndarray] = []
    collected_quat: list[np.ndarray] = []
    collected_vel: list[np.ndarray] = []
    collected_ang_vel: list[np.ndarray] = []
    collected_target_gate: list[np.ndarray] = []

    episode_count = 0
    step_idx = 0
    try:
        while episode_count < episodes and len(collected_pos) < max_states:
            if step_idx % sample_every == 0:
                obs_np = jax.device_get(obs)
                pos = np.asarray(obs_np["pos"][0], dtype=np.float32)
                quat = np.asarray(obs_np["quat"][0], dtype=np.float32)
                vel = np.asarray(obs_np["vel"][0], dtype=np.float32)
                ang_vel = np.asarray(obs_np["ang_vel"][0], dtype=np.float32)
                target_gate = int(np.asarray(obs_np["target_gate"])[0])
                speed = float(np.linalg.norm(vel))
                ang_speed = float(np.linalg.norm(ang_vel))
                if (
                    target_gate >= min_target_gate
                    and min_speed <= speed <= max_speed
                    and ang_speed <= max_ang_speed
                ):
                    collected_pos.append(pos.copy())
                    collected_quat.append(quat.copy())
                    collected_vel.append(vel.copy())
                    collected_ang_vel.append(ang_vel.copy())
                    collected_target_gate.append(np.asarray(target_gate, dtype=np.int32))

            policy_obs = flatten_obs_jax(obs, vectorized=True)
            pi, _ = model.apply(params, policy_obs)
            raw_action = np.asarray(pi.mean()[0], dtype=np.float32)
            applied_action = normalize_actions(raw_action, action_low, action_high)
            env_state, obs, _, terminated, truncated, _ = env.step_fn(
                env_state,
                jnp.asarray(applied_action, dtype=jnp.float32),
            )
            done = bool(jax.device_get(jnp.logical_or(terminated, truncated))[0])
            step_idx += 1
            if done:
                episode_count += 1
                step_idx = 0

        if not collected_pos:
            raise RuntimeError("No reset states were harvested. Relax the filters or collect more episodes.")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_file,
            pos=np.stack(collected_pos, axis=0).astype(np.float32),
            quat=np.stack(collected_quat, axis=0).astype(np.float32),
            vel=np.stack(collected_vel, axis=0).astype(np.float32),
            ang_vel=np.stack(collected_ang_vel, axis=0).astype(np.float32),
            target_gate=np.asarray(collected_target_gate, dtype=np.int32),
        )
        print(f"Saved reset bank with {len(collected_pos)} states to {output_file}")
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(harvest_reset_bank)
