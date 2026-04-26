"""Render a saved policy using the functional JAX env path."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from drone_racing_rl.envs.jax_env import FunctionalJaxVecDroneRaceEnv
from drone_racing_rl.train.actor_critic_models import ActorCritic
from drone_racing_rl.train.experiment_io import choose_runtime_config_path, normalize_checkpoint_path
from drone_racing_rl.train.obs import POLICY_OBS_DIM, flatten_obs_jax
from drone_racing_rl.train.utils import normalize_actions, select_device
from drone_racing_rl.utils import load_config


logger = logging.getLogger(__name__)


def watch_policy(
    checkpoint_path: str = "artifacts/policy_jax/model.msgpack",
    config: str = "level1_flat.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
    sample_actions: bool = False,
    disable_ang_speed_limit: bool = True,
    gate_shift_x: float = 0.0,
    gate_shift_y: float = 0.0,
    gate_shift_z: float = 0.0,
    reset_shift_forward: float = 0.0,
    reset_shift_lateral: float = 0.0,
    reset_shift_vertical: float = 0.0,
) -> None:
    """Load a saved policy and render rollouts using the training env path."""
    checkpoint_path = str(normalize_checkpoint_path(checkpoint_path))
    repo_root = Path(__file__).parents[1]
    resolved_config_path = choose_runtime_config_path(repo_root, checkpoint_path, config)
    cfg = load_config(resolved_config_path)
    if disable_ang_speed_limit:
        reward_cfg = cfg.env.get("reward")
        if reward_cfg is None:
            cfg.env.reward = {"max_angular_speed": 1e6}
        else:
            reward_cfg["max_angular_speed"] = 1e6
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)
    print(f"Using config: {resolved_config_path}")
    if gate_shift_x or gate_shift_y or gate_shift_z or reset_shift_forward or reset_shift_lateral or reset_shift_vertical:
        print("watch_policy now uses the functional training env path; gate/reset shift overrides are ignored.")
    if disable_ang_speed_limit:
        print("watch_policy override: angular-speed limit disabled")

    env = FunctionalJaxVecDroneRaceEnv(
        config=str(resolved_config_path),
        num_envs=1,
        seed=seed,
        device=device,
    )
    if disable_ang_speed_limit:
        env.env.reward_config["max_angular_speed"] = 1e6
        env.env.data = env.env.data.replace(
            max_angular_speed=jnp.asarray([1e6], dtype=jnp.float32),
        )

    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)
    env_state, obs = env.reset(seed=seed)
    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, POLICY_OBS_DIM)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())
    policy_rng = jax.random.PRNGKey(seed + 1)

    episode = 0
    step_idx = 0
    try:
        while True:
            policy_obs = flatten_obs_jax(obs, vectorized=True)
            pi, _ = model.apply(params, policy_obs)
            if sample_actions:
                policy_rng, action_key = jax.random.split(policy_rng)
                action = np.asarray(pi.sample(seed=action_key)[0], dtype=np.float32)
            else:
                action = np.asarray(pi.mean()[0], dtype=np.float32)
            applied_action = normalize_actions(action, action_low, action_high)
            env_state, obs, reward, terminated, truncated, info = env.step_fn(
                env_state,
                jnp.asarray(applied_action, dtype=jnp.float32),
            )
            obs_np = jax.device_get(obs)
            info_np = jax.device_get(info)
            reward_np = float(jax.device_get(reward)[0])
            terminated_np = bool(jax.device_get(terminated)[0])
            truncated_np = bool(jax.device_get(truncated)[0])
            speed = float(np.linalg.norm(np.asarray(obs_np["vel"][0], dtype=np.float32)))
            ang_speed = float(np.linalg.norm(np.asarray(obs_np["ang_vel"][0], dtype=np.float32)))
            env.env.sim.data = env_state.sim_data
            env.env.sim.mjx_data = env_state.mjx_data
            env.env.render()
            if step_idx % 20 == 0:
                print(
                    f"episode={episode} step={step_idx} reward={reward_np:+.4f} "
                    f"target_gate={int(obs_np['target_gate'][0])} "
                    f"speed={speed:.3f} ang_speed={ang_speed:.3f}"
                )
            if terminated_np or truncated_np:
                final_obs = info_np.get("final_observation", obs_np)
                final_vel = np.asarray(
                    info_np.get("raw_final_vel", final_obs["vel"])[0],
                    dtype=np.float32,
                )
                final_ang_vel = np.asarray(
                    info_np.get("raw_final_ang_vel", final_obs["ang_vel"])[0],
                    dtype=np.float32,
                )
                final_pos = np.asarray(
                    info_np.get("raw_final_pos", final_obs["pos"])[0],
                    dtype=np.float32,
                )
                final_target_gate = int(np.asarray(final_obs["target_gate"])[0])
                final_speed = float(np.linalg.norm(final_vel))
                final_ang_speed = float(np.linalg.norm(final_ang_vel))
                already_disabled = bool(np.asarray(info_np.get("already_disabled", [False]))[0])
                no_target_left = bool(np.asarray(info_np.get("no_target_left", [False]))[0])
                speed_limit = bool(np.asarray(info_np.get("speed_limit", [False]))[0])
                angular_speed_limit = bool(np.asarray(info_np.get("angular_speed_limit", [False]))[0])
                ground_crash = bool(np.asarray(info_np.get("ground_crash", [False]))[0])
                out_of_bounds = bool(np.asarray(info_np.get("out_of_bounds", [False]))[0])
                contact = bool(np.asarray(info_np.get("contact", [False]))[0])
                invalid_state = bool(np.asarray(info_np.get("invalid_state", [False]))[0])
                passed = bool(np.asarray(info_np.get("passed", [False]))[0])
                episode += 1
                step_idx = 0
                print(
                    f"Resetting after episode {episode}. "
                    f"terminated={terminated_np} truncated={truncated_np} "
                    f"target_gate={final_target_gate} pos={np.round(final_pos, 3).tolist()} "
                    f"speed={final_speed:.3f} ang_speed={final_ang_speed:.3f} "
                    f"already_disabled={already_disabled} no_target_left={no_target_left} "
                    f"speed_limit={speed_limit} ang_speed_limit={angular_speed_limit} "
                    f"ground={ground_crash} oob={out_of_bounds} contact={contact} "
                    f"invalid={invalid_state} passed={passed}"
                )
                continue
            step_idx += 1
            time.sleep(1 / 60)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(watch_policy)
