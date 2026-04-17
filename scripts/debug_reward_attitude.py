"""Step a learned attitude policy and print full reward-term breakdown."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import ece484_fly.envs  # noqa: F401
from ece484_fly.envs.race_core import RaceCoreEnv
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


def _normalize_action_np(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return 2.0 * (action - action_low) / (action_high - action_low) - 1.0


def _reward_terms(
    last_pos: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    vel: np.ndarray,
    ang_vel: np.ndarray,
    gate_pos: np.ndarray,
    gate_quat: np.ndarray,
    passed: bool,
    course_complete: bool,
    disabled: bool,
    prev_disabled: bool,
    normalized_action: np.ndarray,
    prev_action: np.ndarray,
) -> dict[str, float]:
    progress_reward_scale = 2.0
    gate_pass_bonus = 50.0
    finish_bonus = 200.0
    speed_crossing_scale = 5.0
    heading_alignment_scale = 0.15
    crash_penalty = 10.0
    step_penalty = 0.02
    tilt_penalty_scale = 0.05
    ang_vel_penalty_scale = 0.02
    smoothness_penalty_scale = 0.005
    target_offset = 0.0
    absolute_dist_scale = 0.2
    gate_height_penalty_scale = 1.0

    gate_forward = _quat_apply_np(gate_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    target_pos = gate_pos + target_offset * gate_forward
    prev_target_dist = float(np.linalg.norm(last_pos - target_pos))
    curr_target_dist = float(np.linalg.norm(pos - target_pos))
    progress_reward = progress_reward_scale * (prev_target_dist - curr_target_dist)

    target_dir = target_pos - pos
    target_dir = target_dir / max(float(np.linalg.norm(target_dir)), 1e-6)
    drone_forward = _quat_apply_np(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    heading_alignment = heading_alignment_scale * float(np.dot(drone_forward, target_dir))

    drone_up = _quat_apply_np(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    tilt_penalty = tilt_penalty_scale * float(np.linalg.norm(drone_up[:2]))
    ang_vel_penalty = ang_vel_penalty_scale * float(np.linalg.norm(ang_vel))
    action_smoothness_penalty = smoothness_penalty_scale * float(np.mean((normalized_action - prev_action) ** 2))
    speed_at_crossing = speed_crossing_scale * float(np.linalg.norm(vel)) * float(passed)
    newly_disabled = float(bool(disabled and not prev_disabled))
    absolute_dist = absolute_dist_scale * curr_target_dist
    gate_height_penalty = gate_height_penalty_scale * abs(float(pos[2] - gate_pos[2]))

    if disabled:
        progress_reward = 0.0
        heading_alignment = 0.0

    total = (
        progress_reward
        + heading_alignment
        + gate_pass_bonus * float(passed)
        + speed_at_crossing
        + finish_bonus * float(course_complete)
        - crash_penalty * newly_disabled
        - step_penalty
        - tilt_penalty
        - ang_vel_penalty
        - action_smoothness_penalty
        - absolute_dist
        - gate_height_penalty
    )
    return {
        "progress": progress_reward,
        "heading": heading_alignment,
        "gate_bonus": gate_pass_bonus * float(passed),
        "speed_crossing": speed_at_crossing,
        "finish_bonus": finish_bonus * float(course_complete),
        "crash": -crash_penalty * newly_disabled,
        "step": -step_penalty,
        "tilt": -tilt_penalty,
        "ang_vel": -ang_vel_penalty,
        "smoothness": -action_smoothness_penalty,
        "absolute_dist": -absolute_dist,
        "gate_height": -gate_height_penalty,
        "total": total,
        "curr_target_dist": curr_target_dist,
        "target_pos_z": float(target_pos[2]),
    }


def debug_reward_attitude(
    checkpoint_path: str = "artifacts/policy_jax_full.msgpack",
    config: str = "level1.toml",
    seed: int = 0,
    pause: bool = False,
    device: str = "auto",
    print_every: int = 5,
    render: bool = False,
) -> None:
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = render
    cfg.sim.pause = pause
    cfg.env.control_mode = "attitude"
    device = select_device(device)
    print("JAX devices:", jax.devices())
    print("Using device:", device)

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
    env = JaxToNumpy(env)

    model = ActorCritic(action_dim=4, hidden_dim=(128, 128), activation="tanh")
    dummy_params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 22)))
    params = serialization.from_bytes(dummy_params, Path(checkpoint_path).read_bytes())

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    prev_action = np.zeros((4,), dtype=np.float32)
    prev_disabled = False
    last_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
    step_idx = 0

    try:
        while True:
            policy_obs = flatten_obs(obs, vectorized=False)
            pi, _ = model.apply(params, jnp.asarray(policy_obs)[None, :])
            raw_action = np.asarray(pi.mean()[0], dtype=np.float32)
            applied_action = normalize_actions(raw_action, action_low, action_high)
            normalized_action = _normalize_action_np(applied_action, action_low, action_high)

            target_gate = int(obs["target_gate"])
            gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=np.float32)
            gate_quat = np.asarray(obs["gates_quat"][target_gate], dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(applied_action)
            if render:
                env.render()

            pos = np.asarray(obs["pos"], dtype=np.float32)
            quat = np.asarray(obs["quat"], dtype=np.float32)
            vel = np.asarray(obs["vel"], dtype=np.float32)
            ang_vel = np.asarray(obs["ang_vel"], dtype=np.float32)
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
            course_complete = next_target_gate == -1 and passed

            terms = _reward_terms(
                last_pos,
                pos,
                quat,
                vel,
                ang_vel,
                gate_pos,
                gate_quat,
                passed,
                course_complete,
                disabled,
                prev_disabled,
                normalized_action,
                prev_action,
            )

            if step_idx % print_every == 0 or passed or disabled:
                print(
                    f"step={step_idx} reward={float(reward):+.4f} "
                    f"target_gate={target_gate}->{next_target_gate} pos={np.round(pos, 3).tolist()}"
                )
                print(
                    "  "
                    f"progress={terms['progress']:+.4f} heading={terms['heading']:+.4f} "
                    f"gate={terms['gate_bonus']:+.1f} speed={terms['speed_crossing']:+.4f} "
                    f"finish={terms['finish_bonus']:+.1f}"
                )
                print(
                    "  "
                    f"crash={terms['crash']:+.1f} step={terms['step']:+.4f} "
                    f"tilt={terms['tilt']:+.4f} ang_vel={terms['ang_vel']:+.4f} "
                    f"smooth={terms['smoothness']:+.4f} abs={terms['absolute_dist']:+.4f} "
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
                prev_action = np.zeros((4,), dtype=np.float32)
                prev_disabled = False
                last_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
                step_idx = 0
                continue

            prev_action = normalized_action
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
