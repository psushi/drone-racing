"""Step a learned attitude policy and print full reward-term breakdown."""

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
from ece484_fly.envs.utils import gate_passed
from ece484_fly.train import flatten_obs
from ece484_fly.train.actor_critic_models import ActorCritic
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


def _reward_terms(
    last_pos: np.ndarray,
    pos: np.ndarray,
    gate_pos: np.ndarray,
    passed: bool,
    disabled: bool,
    prev_disabled: bool,
) -> dict[str, float]:
    progress_reward_scale = 2.0
    gate_pass_bonus = 10.0
    crash_penalty = 5.0
    step_penalty = 0.02
    gate_height_penalty_scale = 0.5

    prev_target_dist = float(np.linalg.norm(last_pos - gate_pos))
    curr_target_dist = float(np.linalg.norm(pos - gate_pos))
    progress_reward = progress_reward_scale * (prev_target_dist - curr_target_dist)
    newly_disabled = float(bool(disabled and not prev_disabled))
    gate_height_penalty = gate_height_penalty_scale * abs(float(pos[2] - gate_pos[2]))

    if disabled:
        progress_reward = 0.0

    total = (
        progress_reward
        + gate_pass_bonus * float(passed)
        - crash_penalty * newly_disabled
        - step_penalty
        - gate_height_penalty
    )
    return {
        "progress": progress_reward,
        "gate_bonus": gate_pass_bonus * float(passed),
        "crash": -crash_penalty * newly_disabled,
        "step": -step_penalty,
        "gate_height": -gate_height_penalty,
        "total": total,
        "curr_target_dist": curr_target_dist,
        "target_pos_z": float(gate_pos[2]),
    }


def debug_reward_attitude(
    checkpoint_path: str = "artifacts/policy_jax.msgpack",
    config: str = "level1.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
    print_every: int = 5,
    render: bool = False,
    gate_shift_x: float = 0.0,
    gate_shift_y: float = 0.0,
    gate_shift_z: float = 0.0,
    reset_shift_forward: float = 0.0,
    reset_shift_lateral: float = 0.0,
    reset_shift_vertical: float = 0.0,
) -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = render
    cfg.sim.pause = pause
    cfg.env.control_mode = "attitude"
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
        control_mode="attitude",
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
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

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 22)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    prev_disabled = False
    last_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
    step_idx = 0

    try:
        while True:
            policy_obs = flatten_obs(obs, vectorized=False)
            pi, _ = model.apply(params, jnp.asarray(policy_obs)[None, :])
            raw_action = np.asarray(pi.mean()[0], dtype=np.float32)
            applied_action = normalize_actions(raw_action, action_low, action_high)

            target_gate = int(obs["target_gate"])
            gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=np.float32)
            gate_quat = np.asarray(obs["gates_quat"][target_gate], dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(applied_action)
            if render:
                env.render()

            pos = np.asarray(obs["pos"], dtype=np.float32)
            vel = np.asarray(obs["vel"], dtype=np.float32)
            disabled = bool(terminated or truncated)
            next_target_gate = int(obs["target_gate"])
            passed = bool(
                target_gate != -1
                and gate_passed(
                    pos[None, None, :],
                    last_pos[None, None, :],
                    gate_pos[None, None, :],
                    gate_quat[None, None, :],
                    (0.45, 0.45),
                )[0, 0]
            )

            terms = _reward_terms(
                last_pos,
                pos,
                gate_pos,
                passed,
                disabled,
                prev_disabled,
            )

            if step_idx % print_every == 0 or passed or disabled:
                print(
                    f"step={step_idx} reward={float(reward):+.4f} "
                    f"target_gate={target_gate}->{next_target_gate} pos={np.round(pos, 3).tolist()}"
                )
                print(
                    "  "
                    f"progress={terms['progress']:+.4f} gate={terms['gate_bonus']:+.1f}"
                )
                print(
                    "  "
                    f"crash={terms['crash']:+.1f} step={terms['step']:+.4f} "
                    f"gate_h={terms['gate_height']:+.4f} total={terms['total']:+.4f}"
                )
                print(
                    "  "
                    f"dist={terms['curr_target_dist']:.4f} gate_z={gate_pos[2]:.4f} "
                    f"target_z={terms['target_pos_z']:.4f} drone_z={pos[2]:.4f}"
                )
                print(
                    "  "
                    f"action={np.round(applied_action, 4).tolist()} "
                    f"vel={np.round(vel, 4).tolist()}"
                )

            if disabled:
                print("Episode ended, resetting.")
                obs, _ = env.reset()
                prev_disabled = False
                last_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
                step_idx = 0
                continue

            prev_disabled = disabled
            last_pos = pos
            step_idx += 1
            if render:
                time.sleep(1 / 60)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(debug_reward_attitude)
