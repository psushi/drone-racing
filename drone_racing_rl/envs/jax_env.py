"""JAX-facing wrappers around the drone racing environments.

This module provides two layers:

* thin wrappers that bypass Gymnasium and return JAX arrays
* a functional vectorized wrapper that carries explicit JAX state and exposes pure ``reset_fn`` and
  ``step_fn`` suitable for use inside ``jax.lax.scan``
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.sim import seed_sim, sync_sim2mjx
from flax.struct import dataclass
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from drone_racing_rl.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv
from drone_racing_rl.envs.race_core import EnvData, RaceCoreEnv
from drone_racing_rl.envs.utils import gate_passed
from drone_racing_rl.utils import load_config


@dataclass
class FunctionalJaxVecEnvState:
    """Explicit JAX state for vectorized drone racing rollouts."""

    sim_data: object
    mjx_data: object
    env_data: EnvData
    gate_nominal_pos: jax.Array


class JaxVecDroneRaceEnv:
    """Thin JAX-oriented wrapper around ``VecDroneRaceEnv``."""

    def __init__(
        self,
        config: str = "level1.toml",
        num_envs: int | None = None,
        seed: int = 0,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        cfg = load_config(Path(__file__).parents[2] / "config" / config)
        self.cfg = cfg
        self.env = VecDroneRaceEnv(
            num_envs=cfg.train.num_envs if num_envs is None else num_envs,
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
            max_episode_steps=int(cfg.env.get("max_episode_steps", 1500)),
            device=device,
        )
        self.num_envs = self.env.num_envs
        self.action_space = self.env.action_space
        self.single_action_space = self.env.single_action_space

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        info = {k: jnp.asarray(v) for k, v in info.items()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(jnp.asarray(action))
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        reward = jnp.asarray(reward)
        terminated = jnp.asarray(terminated)
        truncated = jnp.asarray(truncated)
        info = {k: jnp.asarray(v) for k, v in info.items()}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class FunctionalJaxVecDroneRaceEnv:
    """Pure JAX-facing vector env suitable for ``jax.lax.scan`` rollouts.

    This wrapper reuses the existing Crazyflow / RaceCore logic, but it does not step through the
    mutable Gym interface. Instead it carries ``SimData``, ``mjx_data``, and ``EnvData`` explicitly
    and exposes pure reset and step functions.
    """

    def __init__(
        self,
        config: str = "level1.toml",
        num_envs: int | None = None,
        seed: int = 0,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        cfg = load_config(Path(__file__).parents[2] / "config" / config)
        self.cfg = cfg
        self.env = VecDroneRaceEnv(
            num_envs=cfg.train.num_envs if num_envs is None else num_envs,
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
            max_episode_steps=int(cfg.env.get("max_episode_steps", 1500)),
            device=device,
        )
        if self.env.track.randomize:
            raise NotImplementedError("Functional JAX env does not yet support track.randomize=True")

        self.num_envs = self.env.num_envs
        self.n_drones = self.env.sim.n_drones
        self.action_space = self.env.action_space
        self.single_action_space = self.env.single_action_space
        self._sim_steps_per_env_step = self.env.sim.freq // self.env.freq
        self._default_sim_data = self.env.sim.default_data
        self._mjx_model = self.env.sim.mjx_model
        self._action_low = jnp.asarray(self.single_action_space.low, dtype=jnp.float32).reshape((1, 1, -1))
        self._action_high = jnp.asarray(self.single_action_space.high, dtype=jnp.float32).reshape((1, 1, -1))
        self._all_worlds_mask = jnp.ones((self.num_envs,), dtype=bool)
        self._gate_nominal_pos = jnp.asarray(self.env.gates["nominal_pos"], dtype=jnp.float32)
        self._gate_nominal_quat = jnp.asarray(self.env.gates["nominal_quat"], dtype=jnp.float32)
        self._obstacle_nominal_pos = jnp.asarray(self.env.obstacles["nominal_pos"], dtype=jnp.float32)
        gate_z_randomization = cfg.env.track.get("gate_z_randomization", ConfigDict())
        self._gate_z_min = float(gate_z_randomization.get("min", 0.0))
        self._gate_z_max = float(gate_z_randomization.get("max", 0.0))
        self._reset_gate_indices = jnp.asarray(self.env.reset_gate_indices, dtype=jnp.int32)
        self._reset_gate_probs = jnp.asarray(self.env.reset_gate_probs, dtype=jnp.float32)
        self._reset_post_prev_prob = float(self.env.reset_post_prev_prob)
        self._reset_post_prev_speed_min = float(self.env.reset_post_prev_speed_min)
        self._reset_post_prev_speed_max = float(self.env.reset_post_prev_speed_max)
        self._reset_pre_gate_speed = float(self.env.reset_pre_gate_speed)
        self._reset_post_prev_distance_min = float(self.env.reset_post_prev_distance_min)
        self._reset_post_prev_distance_max = float(self.env.reset_post_prev_distance_max)
        self._reset_yaw_jitter = float(self.env.reset_yaw_jitter)
        self._reset_bank_prob = float(self.env.reset_bank_prob)
        self._reset_bank_pos = self.env.reset_bank_pos
        self._reset_bank_quat = self.env.reset_bank_quat
        self._reset_bank_vel = self.env.reset_bank_vel
        self._reset_bank_ang_vel = self.env.reset_bank_ang_vel
        self._reset_bank_target_gate = self.env.reset_bank_target_gate

        self._world_up = jnp.asarray(np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape((1, 1, 3)))
        self._reset_gate_quat = jnp.asarray(np.asarray(self.env.gates["quat"], dtype=np.float32), dtype=jnp.float32)

        self.reset_fn = jax.jit(self._build_reset_fn())
        self.step_fn = jax.jit(self._build_step_fn())

    def _initial_state(self) -> FunctionalJaxVecEnvState:
        return FunctionalJaxVecEnvState(
            sim_data=self.env.sim.data,
            mjx_data=self.env.sim.mjx_data,
            env_data=self.env.data,
            gate_nominal_pos=jnp.broadcast_to(
                self._gate_nominal_pos[None, ...],
                (self.num_envs, *self._gate_nominal_pos.shape),
            ),
        )

    def _sample_reset_state(
        self, key: jax.Array, gate_pos: jax.Array, gate_quat: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        keys = jax.random.split(key, 10)
        gate_forward = jax.vmap(RaceCoreEnv._quat_apply, in_axes=(0, 0))(
            gate_quat,
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), gate_quat.shape[:-1] + (3,)),
        )
        gate_lateral = jax.vmap(RaceCoreEnv._quat_apply, in_axes=(0, 0))(
            gate_quat,
            jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32), gate_quat.shape[:-1] + (3,)),
        )
        hover_center = gate_pos - 1.25 * gate_forward
        logits = jnp.log(self._reset_gate_probs)
        gate_choice = jax.random.categorical(keys[3], logits, shape=(self.num_envs,))
        reset_target_gate = self._reset_gate_indices[gate_choice]
        prev_gate = jnp.maximum(reset_target_gate - 1, 0)
        use_post_prev = jax.random.bernoulli(
            keys[4],
            p=self._reset_post_prev_prob,
            shape=(self.num_envs,),
        ) & (reset_target_gate > 0)

        env_idx = jnp.arange(self.num_envs)
        pre_center = hover_center[env_idx, reset_target_gate]
        pre_forward = gate_forward[env_idx, reset_target_gate]
        pre_lateral = gate_lateral[env_idx, reset_target_gate]
        pre_quat = gate_quat[env_idx, reset_target_gate]

        post_distance = jax.random.uniform(
            keys[5],
            (self.num_envs, 1),
            minval=self._reset_post_prev_distance_min,
            maxval=self._reset_post_prev_distance_max,
        )
        post_center = gate_pos[env_idx, prev_gate] + post_distance * gate_forward[env_idx, prev_gate]
        post_forward = gate_forward[env_idx, prev_gate]
        post_lateral = gate_lateral[env_idx, prev_gate]
        post_quat = gate_quat[env_idx, prev_gate]

        reset_center = jnp.where(use_post_prev[:, None], post_center, pre_center)[:, None, :]
        reset_forward = jnp.where(use_post_prev[:, None], post_forward, pre_forward)[:, None, :]
        reset_lateral = jnp.where(use_post_prev[:, None], post_lateral, pre_lateral)[:, None, :]
        reset_quat = jnp.where(use_post_prev[:, None], post_quat, pre_quat)[:, None, :]
        along_track = jax.random.uniform(keys[0], (self.num_envs, 1, 1), minval=-0.10, maxval=0.10)
        lateral = jax.random.uniform(keys[1], (self.num_envs, 1, 1), minval=-0.05, maxval=0.05)
        vertical = jax.random.uniform(keys[2], (self.num_envs, 1, 1), minval=-0.02, maxval=0.02)
        reset_pos = (
            reset_center
            - along_track * reset_forward
            + lateral * reset_lateral
            + vertical * self._world_up
        )
        post_speed = jax.random.uniform(
            keys[6],
            (self.num_envs,),
            minval=self._reset_post_prev_speed_min,
            maxval=self._reset_post_prev_speed_max,
        )
        pre_speed = jnp.asarray(self._reset_pre_gate_speed, dtype=jnp.float32)
        reset_speed = jnp.where(use_post_prev, post_speed, pre_speed)
        reset_vel = reset_speed[:, None, None] * reset_forward
        reset_ang_vel = jnp.zeros_like(reset_vel)
        yaw_jitter = jax.random.uniform(
            keys[7],
            (self.num_envs, 1, 1),
            minval=-self._reset_yaw_jitter,
            maxval=self._reset_yaw_jitter,
        )
        half_yaw = 0.5 * yaw_jitter
        yaw_quat = jnp.concatenate(
            [
                jnp.zeros_like(half_yaw),
                jnp.zeros_like(half_yaw),
                jnp.sin(half_yaw),
                jnp.cos(half_yaw),
            ],
            axis=-1,
        )
        reset_quat = jnp.where(
            use_post_prev[:, None, None],
            RaceCoreEnv._quat_multiply(yaw_quat, reset_quat),
            reset_quat,
        )
        if self._reset_bank_pos is not None and self._reset_bank_prob > 0.0:
            bank_idx = jax.random.randint(
                keys[8],
                (self.num_envs,),
                minval=0,
                maxval=self._reset_bank_pos.shape[0],
            )
            use_bank = jax.random.bernoulli(
                keys[9],
                p=self._reset_bank_prob,
                shape=(self.num_envs,),
            )
            bank_pos = self._reset_bank_pos[bank_idx][:, None, :]
            bank_quat = self._reset_bank_quat[bank_idx][:, None, :]
            bank_vel = self._reset_bank_vel[bank_idx][:, None, :]
            bank_ang_vel = self._reset_bank_ang_vel[bank_idx][:, None, :]
            bank_target_gate = self._reset_bank_target_gate[bank_idx]
            reset_pos = jnp.where(use_bank[:, None, None], bank_pos, reset_pos)
            reset_quat = jnp.where(use_bank[:, None, None], bank_quat, reset_quat)
            reset_vel = jnp.where(use_bank[:, None, None], bank_vel, reset_vel)
            reset_ang_vel = jnp.where(use_bank[:, None, None], bank_ang_vel, reset_ang_vel)
            reset_target_gate = jnp.where(use_bank, bank_target_gate, reset_target_gate)
        return reset_pos, reset_quat, reset_target_gate, reset_vel, reset_ang_vel

    def _sample_gate_nominal_pos(self, key: jax.Array) -> jax.Array:
        if self._gate_z_min == 0.0 and self._gate_z_max == 0.0:
            return jnp.broadcast_to(
                self._gate_nominal_pos[None, ...],
                (self.num_envs, *self._gate_nominal_pos.shape),
            )
        z_offset = jax.random.uniform(
            key,
            (self.num_envs, self._gate_nominal_pos.shape[0]),
            minval=self._gate_z_min,
            maxval=self._gate_z_max,
        )
        gate_nominal_pos = jnp.broadcast_to(
            self._gate_nominal_pos[None, ...],
            (self.num_envs, *self._gate_nominal_pos.shape),
        )
        return gate_nominal_pos.at[:, :, 2].add(z_offset)

    def _observe(self, state: FunctionalJaxVecEnvState) -> dict[str, jax.Array]:
        gates_pos, gates_quat, obstacles_pos = RaceCoreEnv._obs(
            state.mjx_data.mocap_pos,
            state.mjx_data.mocap_quat,
            state.env_data.gates_visited,
            state.env_data.gate_mj_ids,
            state.gate_nominal_pos,
            self._gate_nominal_quat,
            state.env_data.obstacles_visited,
            state.env_data.obstacle_mj_ids,
            self._obstacle_nominal_pos,
        )
        pos, quat, vel, ang_vel = RaceCoreEnv._sanitize_drone_obs(
            state.sim_data.states.pos,
            state.sim_data.states.quat,
            state.sim_data.states.vel,
            state.sim_data.states.ang_vel,
            state.env_data.disabled_drones,
        )
        return {
            "pos": pos[:, 0],
            "quat": quat[:, 0],
            "vel": vel[:, 0],
            "ang_vel": ang_vel[:, 0],
            "target_gate": state.env_data.target_gate[:, 0],
            "gates_pos": gates_pos[:, 0],
            "gates_quat": gates_quat[:, 0],
            "gates_visited": state.env_data.gates_visited[:, 0],
            "obstacles_pos": obstacles_pos[:, 0],
            "obstacles_visited": state.env_data.obstacles_visited[:, 0],
        }

    def _apply_action(self, sim_data: object, action: jax.Array) -> tuple[object, jax.Array]:
        action = jnp.asarray(action, dtype=jnp.float32).reshape((self.num_envs, self.n_drones, -1))
        if "action" in self.env.disturbances:
            key, subkey = jax.random.split(sim_data.core.rng_key)
            action = action + self.env.disturbances["action"](subkey, action.shape)
            sim_data = sim_data.replace(core=sim_data.core.replace(rng_key=key))

        match self.env.sim.control:
            case "attitude":
                controls = sim_data.controls.replace(
                    attitude=sim_data.controls.attitude.replace(staged_cmd=action)
                )
            case "state":
                controls = sim_data.controls.replace(
                    state=sim_data.controls.state.replace(staged_cmd=action)
                )
            case _:
                raise ValueError(f"Unsupported control mode: {self.env.sim.control}")
        sim_data = sim_data.replace(controls=controls)
        normalized_action = 2.0 * (action - self._action_low) / (self._action_high - self._action_low) - 1.0
        return sim_data, normalized_action

    def _reset_subset(
        self,
        sim_data: object,
        mjx_data: object,
        env_data: EnvData,
        gate_nominal_pos: jax.Array,
        mask: jax.Array,
    ) -> FunctionalJaxVecEnvState:
        sim_data = self.env.sim._reset(sim_data, self._default_sim_data, mask)
        key, subkey, subkey2, subkey3 = jax.random.split(sim_data.core.rng_key, 4)
        sim_data = sim_data.replace(core=sim_data.core.replace(rng_key=key))
        sampled_gate_nominal_pos = self._sample_gate_nominal_pos(subkey3)
        next_gate_nominal_pos = jnp.where(
            mask[:, None, None],
            sampled_gate_nominal_pos,
            gate_nominal_pos,
        )
        gate_nominal_quat = jnp.broadcast_to(
            self._gate_nominal_quat[None, ...],
            (self.num_envs, *self._gate_nominal_quat.shape),
        )

        reset_pos, reset_quat, reset_target_gate, reset_vel, reset_ang_vel = self._sample_reset_state(
            subkey,
            next_gate_nominal_pos,
            gate_nominal_quat,
        )
        pos = jnp.where(mask[:, None, None], reset_pos, sim_data.states.pos)
        quat = jnp.where(mask[:, None, None], reset_quat, sim_data.states.quat)
        vel = jnp.where(mask[:, None, None], reset_vel, sim_data.states.vel)
        ang_vel = jnp.where(mask[:, None, None], reset_ang_vel, sim_data.states.ang_vel)
        sim_data = sim_data.replace(states=sim_data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel))
        mjx_data = self.env.randomize_track(
            mjx_data,
            mask,
            next_gate_nominal_pos,
            self._gate_nominal_quat,
            self._obstacle_nominal_pos,
            subkey2,
        )
        env_data = RaceCoreEnv._reset_env_data(
            env_data,
            sim_data.states.pos,
            mjx_data.mocap_pos,
            reset_target_gate[:, None],
            mask,
        )
        return FunctionalJaxVecEnvState(
            sim_data=sim_data,
            mjx_data=mjx_data,
            env_data=env_data,
            gate_nominal_pos=next_gate_nominal_pos,
        )

    def _build_reset_fn(self):
        def reset_fn(
            state: FunctionalJaxVecEnvState,
            mask: jax.Array | None = None,
        ) -> tuple[FunctionalJaxVecEnvState, dict[str, jax.Array]]:
            mask = self._all_worlds_mask if mask is None else mask
            next_state = self._reset_subset(
                state.sim_data,
                state.mjx_data,
                state.env_data,
                state.gate_nominal_pos,
                mask,
            )
            return next_state, self._observe(next_state)

        return reset_fn

    def _build_step_fn(self):
        def step_fn(
            state: FunctionalJaxVecEnvState,
            action: jax.Array,
        ) -> tuple[FunctionalJaxVecEnvState, dict[str, jax.Array], jax.Array, jax.Array, jax.Array, dict]:
            sim_data, normalized_action = self._apply_action(state.sim_data, action)
            sim_data = self.env.sim._step(sim_data, n_steps=self._sim_steps_per_env_step)
            sim_data = RaceCoreEnv._warp_disabled_drones(sim_data, state.env_data.disabled_drones)
            sim_data, mjx_data = sync_sim2mjx(sim_data, state.mjx_data, self._mjx_model)
            contacts = mjx_data._impl.contact.dist < 0

            n_gates = len(state.env_data.gate_mj_ids)
            gates_pos = mjx_data.mocap_pos[:, state.env_data.gate_mj_ids]
            gates_quat = mjx_data.mocap_quat[:, state.env_data.gate_mj_ids][..., [1, 2, 3, 0]]
            gate_ids = state.env_data.gate_mj_ids[state.env_data.target_gate % n_gates]
            gate_pos = gates_pos[jnp.arange(gates_pos.shape[0])[:, None], gate_ids]
            gate_quat = gates_quat[jnp.arange(gates_quat.shape[0])[:, None], gate_ids]
            passed = gate_passed(
                sim_data.states.pos,
                state.env_data.last_drone_pos,
                gate_pos,
                gate_quat,
                (0.45, 0.45),
            )

            env_data, disable_flags = RaceCoreEnv._step_env(
                state.env_data,
                sim_data.states.pos,
                sim_data.states.quat,
                sim_data.states.vel,
                sim_data.states.ang_vel,
                normalized_action,
                mjx_data.mocap_pos,
                mjx_data.mocap_quat,
                contacts,
                self.env.sim.freq,
            )

            reward = env_data.rewards[:, 0]
            terminated = env_data.disabled_drones[:, 0]
            truncated = RaceCoreEnv._truncated(
                env_data.steps, env_data.max_episode_steps, self.n_drones
            )[:, 0]
            next_state = FunctionalJaxVecEnvState(
                sim_data=sim_data,
                mjx_data=mjx_data,
                env_data=env_data,
                gate_nominal_pos=state.gate_nominal_pos,
            )
            final_obs = self._observe(next_state)
            if self.env.autoreset:
                next_state = self._reset_subset(
                    next_state.sim_data,
                    next_state.mjx_data,
                    next_state.env_data,
                    next_state.gate_nominal_pos,
                    next_state.env_data.marked_for_reset,
                )
            obs = self._observe(next_state)
            info = {
                "passed": passed[:, 0],
                "final_observation": final_obs,
                "raw_final_pos": sim_data.states.pos[:, 0],
                "raw_final_vel": sim_data.states.vel[:, 0],
                "raw_final_ang_vel": sim_data.states.ang_vel[:, 0],
                "already_disabled": disable_flags.already_disabled[:, 0],
                "no_target_left": disable_flags.no_target_left[:, 0],
                "speed_limit": disable_flags.speed_limit[:, 0],
                "angular_speed_limit": disable_flags.angular_speed_limit[:, 0],
                "ground_crash": disable_flags.ground_crash[:, 0],
                "out_of_bounds": disable_flags.out_of_bounds[:, 0],
                "contact": disable_flags.contact[:, 0],
                "invalid_state": disable_flags.invalid_state[:, 0],
            }
            return next_state, obs, reward, terminated, truncated, info

        return step_fn

    def reset(
        self,
        seed: int | None = None,
        state: FunctionalJaxVecEnvState | None = None,
    ) -> tuple[FunctionalJaxVecEnvState, dict[str, jax.Array]]:
        if state is None:
            state = self._initial_state()
        if seed is not None:
            sim_data = seed_sim(state.sim_data, seed, self.env.device)
            state = FunctionalJaxVecEnvState(
                sim_data=sim_data,
                mjx_data=state.mjx_data,
                env_data=state.env_data,
                gate_nominal_pos=state.gate_nominal_pos,
            )
        return self.reset_fn(state, self._all_worlds_mask)

    def close(self) -> None:
        self.env.close()


class JaxDroneRaceEnv:
    """Thin JAX-oriented wrapper around ``DroneRaceEnv``."""

    def __init__(
        self,
        config: str = "level1.toml",
        seed: int = 0,
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        cfg = load_config(Path(__file__).parents[2] / "config" / config)
        self.cfg = cfg
        self.env = DroneRaceEnv(
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            control_mode=cfg.env.control_mode,
            track=cfg.env.track,
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            reward_config=cfg.env.get("reward"),
            seed=seed,
            device=device,
        )
        self.action_space = self.env.action_space

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        info = {k: jnp.asarray(v) for k, v in info.items()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(jnp.asarray(action))
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        reward = jnp.asarray(reward)
        terminated = jnp.asarray(terminated)
        truncated = jnp.asarray(truncated)
        info = {k: jnp.asarray(v) for k, v in info.items()}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
