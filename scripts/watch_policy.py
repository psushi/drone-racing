"""Render a saved policy in a single simulation environment."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import MethodType

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
from ece484_fly.train.obs import POLICY_OBS_DIM
from ece484_fly.train.utils import normalize_actions, select_device
from ece484_fly.utils import load_config


logger = logging.getLogger(__name__)


def _quat_apply_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return vec + 2.0 * (q_w * uv + uuv)


def _install_reset_shift(env, forward: float, lateral: float, vertical: float) -> None:
    if not (forward or lateral or vertical):
        return

    base_env = env.unwrapped
    original_sample_reset_state = base_env._sample_reset_state
    gate_quat = np.asarray(base_env.gates["quat"][0], dtype=np.float32)
    gate_forward = _quat_apply_np(gate_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    gate_lateral = _quat_apply_np(gate_quat, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def shifted_reset_state(self, key):
        reset_state = original_sample_reset_state(key)
        if len(reset_state) == 3:
            pos, quat, target_gate = reset_state
        else:
            pos, quat = reset_state
            target_gate = None
        offset = (
            forward * gate_forward.reshape((1, 1, 3))
            + lateral * gate_lateral.reshape((1, 1, 3))
            + vertical * world_up.reshape((1, 1, 3))
        )
        if target_gate is None:
            return pos + offset, quat
        return pos + offset, quat, target_gate

    base_env._sample_reset_state = MethodType(shifted_reset_state, base_env)


def watch_policy(
    checkpoint_path: str = "artifacts/policy_jax.msgpack",
    config: str = "level1.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
    gate_shift_x: float = 0.0,
    gate_shift_y: float = 0.0,
    gate_shift_z: float = 0.0,
    reset_shift_forward: float = 0.0,
    reset_shift_lateral: float = 0.0,
    reset_shift_vertical: float = 0.0,
) -> None:
    """Load a saved policy and render rollouts in a single env."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = True
    cfg.sim.pause = pause
    if len(cfg.env.track.gates) > 0:
        gate_cfg = cfg.env.track.gates[0]
        gate0 = list(gate_cfg["pos"] if isinstance(gate_cfg, dict) else gate_cfg.pos)
        gate0[0] += gate_shift_x
        gate0[1] += gate_shift_y
        gate0[2] += gate_shift_z
        if isinstance(gate_cfg, dict):
            gate_cfg["pos"] = gate0
        else:
            gate_cfg.pos = gate0
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)
    if gate_shift_x or gate_shift_y or gate_shift_z:
        print(
            "Shifted first gate by "
            f"dx={gate_shift_x:+.3f}, dy={gate_shift_y:+.3f}, dz={gate_shift_z:+.3f}"
        )
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        reward_config=cfg.env.get("reward"),
        reset_config=cfg.env.get("reset"),
        seed=seed,
        device=device,
    )
    _install_reset_shift(env, reset_shift_forward, reset_shift_lateral, reset_shift_vertical)
    env = JaxToNumpy(env)
    if reset_shift_forward or reset_shift_lateral or reset_shift_vertical:
        print(
            "Shifted reset by "
            f"forward={reset_shift_forward:+.3f}, lateral={reset_shift_lateral:+.3f}, "
            f"vertical={reset_shift_vertical:+.3f}"
        )

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    obs, info = env.reset(seed=seed)
    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, POLICY_OBS_DIM)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())

    episode = 0
    step_idx = 0
    try:
        while True:
            policy_obs = flatten_obs(obs, vectorized=False)
            action = np.asarray(model.apply(params, jnp.asarray(policy_obs)[None, :])[0].mean()[0], dtype=np.float32)
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
