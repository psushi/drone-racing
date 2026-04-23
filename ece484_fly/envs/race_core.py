"""Core environment for drone racing simulations.

This module provides the shared logic for simulating drone racing environments. It defines a core
environment class that wraps our drone simulation, drone control, gate tracking, and collision
detection. The module serves as the base for both single-drone and multi-drone racing environments.

The environment is designed to be configurable, supporting:

* Different control modes (state or attitude)
* Customizable tracks with gates and obstacles
* Various randomization options for robust policy training
* Disturbance modeling for realistic flight conditions
* Vectorized execution for parallel training

This module is primarily used as a base for the higher-level environments in
:mod:`~ece484_fly.envs.drone_race` which provide Gymnasium-compatible interfaces for control techniques.
"""

from __future__ import annotations

import copy as copy
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.sim import use_box_collision
from drone_controllers.mellinger.params import ForceTorqueParams
from flax.struct import dataclass
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ece484_fly.envs.randomize import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)
from ece484_fly.envs.utils import gate_passed, generate_random_track, load_track

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array, Device
    from ml_collections import ConfigDict
    from mujoco import MjSpec
    from mujoco.mjx import Data
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region EnvData


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment.

    This dataclass stores the dynamic and static state of the environment that is not directly
    part of the physics simulation. It includes information about gate progress, drone status,
    and environment boundaries. Static variables are initialized once and do not change during the
    episode.

    Args:
        target_gate: Current target gate index for each drone in each environment
        gates_visited: Boolean flags indicating which gates have been visited by each drone
        obstacles_visited: Boolean flags indicating which obstacles have been detected
        last_drone_pos: Previous positions of drones, used for gate passing detection
        marked_for_reset: Flags indicating which environments need to be reset
        disabled_drones: Flags indicating which drones have crashed or are otherwise disabled
        contact_masks: Masks for contact detection between drones and objects
        pos_limit_low: Lower position limits for the environment
        pos_limit_high: Upper position limits for the environment
        gate_mj_ids: MuJoCo IDs for the gates
        obstacle_mj_ids: MuJoCo IDs for the obstacles
        max_episode_steps: Maximum number of steps per episode
        sensor_range: Range at which drones can detect gates and obstacles
        rewards: Reward for each drone in each environment for the current transition
    """

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    segment_start_pos: Array
    marked_for_reset: Array
    disabled_drones: Array
    steps: Array
    last_action: Array
    # Static variables
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    gate_mj_ids: Array
    obstacle_mj_ids: Array
    max_episode_steps: Array
    sensor_range: Array
    rewards: Array
    progress_scale: Array
    perception_weight: Array
    perception_distance_scale: Array
    safety_weight: Array
    ang_vel_penalty_scale: Array
    gate_width: Array
    safety_activation_distance: Array
    crash_penalty: Array
    gate_pass_bonus: Array
    completion_bonus: Array
    vel_penalty_scale: Array
    max_linear_speed: Array
    max_angular_speed: Array

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_drones: int,
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        gate_mj_ids: Array,
        obstacle_mj_ids: Array,
        max_episode_steps: int,
        sensor_range: float,
        pos_limit_low: Array,
        pos_limit_high: Array,
        progress_scale: float,
        perception_weight: float,
        perception_distance_scale: float,
        safety_weight: float,
        ang_vel_penalty_scale: float,
        gate_width: float,
        safety_activation_distance: float,
        crash_penalty: float,
        gate_pass_bonus: float,
        completion_bonus: float,
        vel_penalty_scale: float,
        max_linear_speed: float,
        max_angular_speed: float,
        device: Device,
    ) -> EnvData:
        """Create a new environment data struct with default values."""
        return cls(
            target_gate=jnp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jnp.zeros((n_envs, n_drones, n_gates), dtype=bool, device=device),
            obstacles_visited=jnp.zeros((n_envs, n_drones, n_obstacles), dtype=bool, device=device),
            last_drone_pos=jnp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            segment_start_pos=jnp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            marked_for_reset=jnp.zeros(n_envs, dtype=bool, device=device),
            disabled_drones=jnp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jnp.array(contact_masks, dtype=bool, device=device),
            steps=jnp.zeros(n_envs, dtype=int, device=device),
            last_action=jnp.zeros((n_envs, n_drones, 4), dtype=jnp.float32, device=device),
            pos_limit_low=jnp.array(pos_limit_low, dtype=np.float32, device=device),
            pos_limit_high=jnp.array(pos_limit_high, dtype=np.float32, device=device),
            gate_mj_ids=jnp.array(gate_mj_ids, dtype=int, device=device),
            obstacle_mj_ids=jnp.array(obstacle_mj_ids, dtype=int, device=device),
            max_episode_steps=jnp.array([max_episode_steps], dtype=int, device=device),
            sensor_range=jnp.array([sensor_range], dtype=jnp.float32, device=device),
            rewards=jnp.zeros((n_envs, n_drones), dtype=jnp.float32, device=device),
            progress_scale=jnp.array([progress_scale], dtype=jnp.float32, device=device),
            perception_weight=jnp.array([perception_weight], dtype=jnp.float32, device=device),
            perception_distance_scale=jnp.array(
                [perception_distance_scale], dtype=jnp.float32, device=device
            ),
            safety_weight=jnp.array([safety_weight], dtype=jnp.float32, device=device),
            ang_vel_penalty_scale=jnp.array(
                [ang_vel_penalty_scale], dtype=jnp.float32, device=device
            ),
            gate_width=jnp.array([gate_width], dtype=jnp.float32, device=device),
            safety_activation_distance=jnp.array(
                [safety_activation_distance], dtype=jnp.float32, device=device
            ),
            crash_penalty=jnp.array([crash_penalty], dtype=jnp.float32, device=device),
            gate_pass_bonus=jnp.array([gate_pass_bonus], dtype=jnp.float32, device=device),
            completion_bonus=jnp.array([completion_bonus], dtype=jnp.float32, device=device),
            vel_penalty_scale=jnp.array([vel_penalty_scale], dtype=jnp.float32, device=device),
            max_linear_speed=jnp.array([max_linear_speed], dtype=jnp.float32, device=device),
            max_angular_speed=jnp.array([max_angular_speed], dtype=jnp.float32, device=device),
        )


@dataclass
class DisableReasonFlags:
    disabled: Array
    already_disabled: Array
    no_target_left: Array
    speed_limit: Array
    angular_speed_limit: Array
    ground_crash: Array
    out_of_bounds: Array
    contact: Array
    invalid_state: Array


@dataclass
class ResetSamplerConfig:
    reset_gate_indices: Array
    reset_gate_probs: Array
    post_prev_prob: float
    post_prev_speed_min: float
    post_prev_speed_max: float
    pre_gate_speed: float
    post_prev_distance_min: float
    post_prev_distance_max: float
    yaw_jitter: float
    along_track_min: float = -0.10
    along_track_max: float = 0.10
    lateral_min: float = -0.05
    lateral_max: float = 0.05
    vertical_min: float = -0.02
    vertical_max: float = 0.02
    pre_gate_distance: float = 1.25
    world_up: Array | None = None
    bank_prob: float = 0.0
    bank_pos: Array | None = None
    bank_quat: Array | None = None
    bank_vel: Array | None = None
    bank_ang_vel: Array | None = None
    bank_target_gate: Array | None = None


@dataclass
class ResetSample:
    pos: Array
    quat: Array
    target_gate: Array
    vel: Array
    ang_vel: Array


def build_action_space(control_mode: Literal["state", "attitude"], drone_model: str) -> spaces.Box:
    """Create the action space for the environment.

    Args:
        control_mode: The control mode to use. Either "state" for full-state control
            or "attitude" for attitude control.
        drone_model: Drone model of the environment.

    Returns:
        A Box space representing the action space for the specified control mode.
    """
    if control_mode == "state":
        return spaces.Box(low=-1, high=1, shape=(13,))
    elif control_mode == "attitude":
        params = ForceTorqueParams.load(drone_model)
        thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
        return spaces.Box(
            np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
            np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
        )
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")


def build_observation_space(n_gates: int, n_obstacles: int) -> spaces.Dict:
    """Create the observation space for the environment.

    The observation space is a dictionary containing the drone state, gate information,
    and obstacle information.

    Args:
        n_gates: Number of gates in the environment.
        n_obstacles: Number of obstacles in the environment.
    """
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "quat": spaces.Box(low=-1, high=1, shape=(4,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "target_gate": spaces.Discrete(n_gates, start=-1),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_quat": spaces.Box(low=-1, high=1, shape=(n_gates, 4)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
    }
    return spaces.Dict(obs_spec)


# region Core Env


class RaceCoreEnv:
    """The core environment for drone racing simulations.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.

    The environment provides:

    * A customizable track with gates and obstacles
    * Configurable simulation and control frequencies
    * Support for different physics models (e.g., identified dynamics, analytical dynamics)
    * Randomization of drone properties and initial conditions
    * Disturbance modeling for realistic flight conditions
    * Symbolic expressions for advanced control techniques (optional)

    The environment tracks the drone's progress through the gates and provides termination
    conditions based on gate passages and collisions.

    The observation space is a dictionary with the following keys:

    * pos: Drone position
    * quat: Drone orientation as a quaternion (x, y, z, w)
    * vel: Drone linear velocity
    * ang_vel: Drone angular velocity
    * gates_pos: Positions of the gates
    * gates_quat: Orientations of the gates
    * gates_visited: Flags indicating if the drone already was/ is in the sensor range of the
      gates and the true position is known
    * obstacles_pos: Positions of the obstacles
    * obstacles_visited: Flags indicating if the drone already was/ is in the sensor range of the
      obstacles and the true position is known
    * target_gate: The current target gate index

    The action space consists of a desired full-state command
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] that is tracked by the drone's
    low-level controller, or a desired collective thrust and attitude command [collective thrust,
    roll, pitch, yaw].
    """

    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"

    def __init__(
        self,
        n_envs: int,
        n_drones: int,
        freq: int,
        sim_config: ConfigDict,
        sensor_range: float,
        track: ConfigDict,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        reward_config: ConfigDict | None = None,
        reset_config: ConfigDict | None = None,
        seed: str | int = "random",
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the DroneRacingEnv.

        Args:
            n_envs: Number of worlds in the vectorized environment.
            n_drones: Number of drones.
            freq: Environment step frequency.
            sim_config: Configuration dictionary for the simulation.
            sensor_range: Sensor range for gate and obstacle detection.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            track: Track configuration.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            reward_config: Reward shaping configuration.
            reset_config: Reset-state sampling configuration.
            seed: "random" for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode. Needs to be tracked manually for
                vectorized environments.
            device: Device used for the environment and the simulation.
        """
        super().__init__()
        if type(seed) is str:
            seed: int = np.random.SeedSequence().entropy if seed == "random" else hash(seed)
            seed &= 0xFFFFFFFF  # Limit seed to 32 bit for jax.random
        self.sim = Sim(
            n_worlds=n_envs,
            n_drones=n_drones,
            physics=sim_config.physics,
            drone_model=sim_config.drone_model,
            control=control_mode,
            freq=sim_config.freq,
            state_freq=freq,
            attitude_freq=sim_config.attitude_freq,
            rng_key=seed,
            device=device,
        )
        use_box_collision(self.sim, True)
        self.cam_config = {
            "distance": sim_config.camera_view[0],
            "azimuth": sim_config.camera_view[1],
            "elevation": sim_config.camera_view[2],
            "lookat": sim_config.camera_view[3:],
        }

        # Sanitize args
        if sim_config.freq % freq != 0:
            raise ValueError(f"({sim_config.freq=}) is no multiple of ({freq=})")

        # Env settings
        self.freq = freq
        self.seed = seed
        self.autoreset = True  # Can be overridden by subclasses
        self.device = jax.devices(device)[0]
        self.sensor_range = sensor_range
        self.track = track
        self.gates, self.obstacles, self.drone = load_track(track)
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        reward_cfg = {} if reward_config is None else reward_config
        self.reward_config = {
            "progress_scale": float(reward_cfg.get("progress_scale", 10.0)),
            "perception_weight": float(
                reward_cfg.get("perception_weight", reward_cfg.get("alignment_weight", 1.0))
            ),
            "perception_distance_scale": float(reward_cfg.get("perception_distance_scale", 1.0)),
            "safety_weight": float(reward_cfg.get("safety_weight", 1.0)),
            "ang_vel_penalty_scale": float(reward_cfg.get("ang_vel_penalty_scale", 0.01)),
            "gate_width": float(reward_cfg.get("gate_width", 0.4)),
            "safety_activation_distance": float(
                reward_cfg.get("safety_activation_distance", 2.5)
            ),
            "crash_penalty": float(reward_cfg.get("crash_penalty", 5.0)),
            "gate_pass_bonus": float(reward_cfg.get("gate_pass_bonus", 5.0)),
            "completion_bonus": float(reward_cfg.get("completion_bonus", 0.0)),
            "vel_penalty_scale": float(reward_cfg.get("vel_penalty_scale", 0.01)),
            "max_linear_speed": float(reward_cfg.get("max_linear_speed", 8.0)),
            "max_angular_speed": float(reward_cfg.get("max_angular_speed", 20.0)),
        }
        reset_cfg = {} if reset_config is None else reset_config
        n_gates_total = len(track.gates)
        gate_indices = reset_cfg.get("gate_indices", list(range(n_gates_total)))
        if len(gate_indices) == 0:
            raise ValueError("reset_config.gate_indices must not be empty")
        gate_indices = np.asarray(gate_indices, dtype=np.int32)
        if np.any(gate_indices < 0) or np.any(gate_indices >= n_gates_total):
            raise ValueError("reset_config.gate_indices contains out-of-range gate ids")
        gate_probs = reset_cfg.get("gate_probs")
        if gate_probs is None:
            gate_probs = np.ones(len(gate_indices), dtype=np.float32) / len(gate_indices)
        else:
            gate_probs = np.asarray(gate_probs, dtype=np.float32)
            if gate_probs.shape != gate_indices.shape:
                raise ValueError("reset_config.gate_probs must match gate_indices length")
            prob_sum = float(gate_probs.sum())
            if prob_sum <= 0.0:
                raise ValueError("reset_config.gate_probs must sum to a positive value")
            gate_probs = gate_probs / prob_sum
        self.reset_gate_indices = gate_indices
        self.reset_gate_probs = gate_probs
        post_prev_prob = float(reset_cfg.get("post_prev_prob", 0.5))
        if not 0.0 <= post_prev_prob <= 1.0:
            raise ValueError("reset_config.post_prev_prob must be in [0, 1]")
        self.reset_post_prev_prob = post_prev_prob
        self.reset_post_prev_speed_min = float(
            reset_cfg.get("post_prev_speed_min", reset_cfg.get("post_prev_speed", 0.0))
        )
        self.reset_post_prev_speed_max = float(
            reset_cfg.get("post_prev_speed_max", reset_cfg.get("post_prev_speed", 0.0))
        )
        self.reset_pre_gate_speed = float(reset_cfg.get("pre_gate_speed", 0.0))
        self.reset_post_prev_distance_min = float(reset_cfg.get("post_prev_distance_min", 0.35))
        self.reset_post_prev_distance_max = float(reset_cfg.get("post_prev_distance_max", 0.95))
        self.reset_yaw_jitter = float(reset_cfg.get("yaw_jitter", 0.15))
        self.reset_bank_prob = float(reset_cfg.get("bank_prob", 0.0))
        if not 0.0 <= self.reset_bank_prob <= 1.0:
            raise ValueError("reset_config.bank_prob must be in [0, 1]")
        self.reset_bank_pos = None
        self.reset_bank_quat = None
        self.reset_bank_vel = None
        self.reset_bank_ang_vel = None
        self.reset_bank_target_gate = None
        bank_path = reset_cfg.get("bank_path")
        if bank_path:
            bank = np.load(bank_path)
            required = {"pos", "quat", "vel", "target_gate"}
            missing = required - set(bank.files)
            if missing:
                raise ValueError(f"reset bank missing arrays: {sorted(missing)}")
            self.reset_bank_pos = jnp.asarray(bank["pos"], dtype=jnp.float32)
            self.reset_bank_quat = jnp.asarray(bank["quat"], dtype=jnp.float32)
            self.reset_bank_vel = jnp.asarray(bank["vel"], dtype=jnp.float32)
            self.reset_bank_ang_vel = jnp.asarray(
                bank["ang_vel"] if "ang_vel" in bank.files else np.zeros_like(bank["vel"]),
                dtype=jnp.float32,
            )
            self.reset_bank_target_gate = jnp.asarray(bank["target_gate"], dtype=jnp.int32)
            if self.reset_bank_pos.shape[0] == 0:
                raise ValueError("reset bank is empty")
            if self.reset_bank_pos.shape[-1] != 3 or self.reset_bank_quat.shape[-1] != 4:
                raise ValueError("reset bank has invalid shapes")

        # Load the track into the simulation and compile the reset and step functions with hooks
        self._setup_sim(randomizations)

        # Create the environment data struct.
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        contact_masks = self._load_contact_masks(self.sim)
        m = self.sim.mj_model
        gate_ids = [int(m.body(f"gate:{i}").mocapid.squeeze()) for i in range(n_gates)]
        obstacle_ids = [int(m.body(f"obstacle:{i}").mocapid.squeeze()) for i in range(n_obstacles)]
        self.data = EnvData.create(
            n_envs=n_envs,
            n_drones=n_drones,
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            contact_masks=contact_masks,
            gate_mj_ids=gate_ids,
            obstacle_mj_ids=obstacle_ids,
            max_episode_steps=max_episode_steps,
            sensor_range=sensor_range,
            pos_limit_low=[-3, -3, -1e-3],
            pos_limit_high=[3, 3, 2.5],
            progress_scale=self.reward_config["progress_scale"],
            perception_weight=self.reward_config["perception_weight"],
            perception_distance_scale=self.reward_config["perception_distance_scale"],
            safety_weight=self.reward_config["safety_weight"],
            ang_vel_penalty_scale=self.reward_config["ang_vel_penalty_scale"],
            gate_width=self.reward_config["gate_width"],
            safety_activation_distance=self.reward_config["safety_activation_distance"],
            crash_penalty=self.reward_config["crash_penalty"],
            gate_pass_bonus=self.reward_config["gate_pass_bonus"],
            completion_bonus=self.reward_config["completion_bonus"],
            vel_penalty_scale=self.reward_config["vel_penalty_scale"],
            max_linear_speed=self.reward_config["max_linear_speed"],
            max_angular_speed=self.reward_config["max_angular_speed"],
            device=self.device,
        )
        self.randomize_track = build_track_randomization_fn(randomizations, gate_ids, obstacle_ids)

    def _reset(
        self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional reset options. Not used.
            mask: Mask of worlds to reset.

        Returns:
            Observation and info.
        """
        if seed is not None:
            self.sim.seed(seed)
            self._np_random = np.random.default_rng(seed)  # Also update gymnasium's rng
        # Randomization of the drone is compiled into the sim reset pipeline, so we don't need to
        # explicitly do it here
        self.sim.reset(mask=mask)
        key, subkey, subkey2 = jax.random.split(self.sim.data.core.rng_key, 3)
        # Generate random track
        track = generate_random_track(self.track, subkey2) if self.track.randomize else self.track
        self.gates, self.obstacles, self.drone = load_track(track)
        # Randomize the track
        self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        reset_pos, reset_quat, reset_target_gate, reset_vel, reset_ang_vel = self._sample_reset_state(subkey)

        @jax.jit
        def update_sim_data(
            data: SimData, mjx_data: Data, key: jax.random.PRNGKey
        ) -> tuple[SimData, Data]:
            pos = data.states.pos.at[...].set(reset_pos)
            quat = data.states.quat.at[...].set(reset_quat)
            vel = data.states.vel.at[...].set(reset_vel)
            ang_vel = data.states.ang_vel.at[...].set(reset_ang_vel)
            data = data.replace(states=data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel))

            mjx_data = self.randomize_track(
                mjx_data,
                mask,
                self.gates["nominal_pos"],
                self.gates["nominal_quat"],
                self.obstacles["nominal_pos"],
                key,
            )
            return data, mjx_data

        self.sim.data, self.sim.mjx_data = update_sim_data(self.sim.data, self.sim.mjx_data, subkey)

        # Reset the environment data
        self.data = self._reset_env_data(
            self.data,
            self.sim.data.states.pos,
            self.sim.mjx_data.mocap_pos,
            reset_target_gate[:, None],
            mask,
        )

        return self.obs(), self.info()

    @staticmethod
    def _sample_reset_state_from_config(
        key: jax.random.PRNGKey,
        gate_pos: Array,
        gate_quat: Array,
        n_worlds: int,
        config: ResetSamplerConfig,
    ) -> ResetSample:
        """Sample reset state from gate poses and a structured reset config."""
        keys = jax.random.split(key, 10)
        gate_forward = jax.vmap(RaceCoreEnv._quat_apply, in_axes=(0, 0))(
            gate_quat,
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), gate_quat.shape),
        )
        gate_lateral = jax.vmap(RaceCoreEnv._quat_apply, in_axes=(0, 0))(
            gate_quat,
            jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32), gate_quat.shape),
        )
        hover_center = gate_pos - config.pre_gate_distance * gate_forward

        gate_choice = jax.random.categorical(
            keys[3],
            jnp.log(config.reset_gate_probs),
            shape=(n_worlds,),
        )
        reset_target_gate = config.reset_gate_indices[gate_choice]
        prev_gate = jnp.maximum(reset_target_gate - 1, 0)
        use_post_prev = jax.random.bernoulli(
            keys[4],
            p=config.post_prev_prob,
            shape=(n_worlds,),
        ) & (reset_target_gate > 0)

        pre_center = hover_center[reset_target_gate]
        pre_forward = gate_forward[reset_target_gate]
        pre_lateral = gate_lateral[reset_target_gate]
        pre_quat = gate_quat[reset_target_gate]

        post_distance = jax.random.uniform(
            keys[5],
            (n_worlds, 1),
            minval=config.post_prev_distance_min,
            maxval=config.post_prev_distance_max,
        )
        post_center = gate_pos[prev_gate] + post_distance * gate_forward[prev_gate]
        post_forward = gate_forward[prev_gate]
        post_lateral = gate_lateral[prev_gate]
        post_quat = gate_quat[prev_gate]

        reset_center = jnp.where(use_post_prev[:, None], post_center, pre_center)[:, None, :]
        reset_forward = jnp.where(use_post_prev[:, None], post_forward, pre_forward)[:, None, :]
        reset_lateral = jnp.where(use_post_prev[:, None], post_lateral, pre_lateral)[:, None, :]
        reset_quat = jnp.where(use_post_prev[:, None], post_quat, pre_quat)[:, None, :]

        world_up = config.world_up
        if world_up is None:
            world_up = jnp.asarray(np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape((1, 1, 3)))

        along_track = jax.random.uniform(
            keys[0], (n_worlds, 1, 1), minval=config.along_track_min, maxval=config.along_track_max
        )
        lateral = jax.random.uniform(
            keys[1], (n_worlds, 1, 1), minval=config.lateral_min, maxval=config.lateral_max
        )
        vertical = jax.random.uniform(
            keys[2], (n_worlds, 1, 1), minval=config.vertical_min, maxval=config.vertical_max
        )
        reset_pos = reset_center - along_track * reset_forward + lateral * reset_lateral + vertical * world_up

        post_speed = jax.random.uniform(
            keys[6],
            (n_worlds,),
            minval=config.post_prev_speed_min,
            maxval=config.post_prev_speed_max,
        )
        pre_speed = jnp.asarray(config.pre_gate_speed, dtype=jnp.float32)
        reset_speed = jnp.where(use_post_prev, post_speed, pre_speed)
        reset_vel = reset_speed[:, None, None] * reset_forward
        reset_ang_vel = jnp.zeros_like(reset_vel)

        yaw_jitter = jax.random.uniform(
            keys[7],
            (n_worlds, 1, 1),
            minval=-config.yaw_jitter,
            maxval=config.yaw_jitter,
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

        if config.bank_pos is not None and config.bank_prob > 0.0:
            bank_idx = jax.random.randint(
                keys[8],
                (n_worlds,),
                minval=0,
                maxval=config.bank_pos.shape[0],
            )
            use_bank = jax.random.bernoulli(
                keys[9],
                p=config.bank_prob,
                shape=(n_worlds,),
            )
            bank_pos = config.bank_pos[bank_idx][:, None, :]
            bank_quat = config.bank_quat[bank_idx][:, None, :]
            bank_vel = config.bank_vel[bank_idx][:, None, :]
            bank_ang_vel = config.bank_ang_vel[bank_idx][:, None, :]
            bank_target_gate = config.bank_target_gate[bank_idx]
            reset_pos = jnp.where(use_bank[:, None, None], bank_pos, reset_pos)
            reset_quat = jnp.where(use_bank[:, None, None], bank_quat, reset_quat)
            reset_vel = jnp.where(use_bank[:, None, None], bank_vel, reset_vel)
            reset_ang_vel = jnp.where(use_bank[:, None, None], bank_ang_vel, reset_ang_vel)
            reset_target_gate = jnp.where(use_bank, bank_target_gate, reset_target_gate)

        return ResetSample(
            pos=reset_pos,
            quat=reset_quat,
            target_gate=reset_target_gate,
            vel=reset_vel,
            ang_vel=reset_ang_vel,
        )

    def _sample_reset_state(self, key: jax.random.PRNGKey) -> tuple[Array, Array, Array, Array, Array]:
        """Sample reset states before the target gate or just after the previous gate."""
        sample = self._sample_reset_state_from_config(
            key=key,
            gate_pos=jnp.asarray(self.gates["pos"], dtype=jnp.float32),
            gate_quat=jnp.asarray(self.gates["quat"], dtype=jnp.float32),
            n_worlds=self.sim.n_worlds,
            config=ResetSamplerConfig(
                reset_gate_indices=jnp.asarray(self.reset_gate_indices, dtype=jnp.int32),
                reset_gate_probs=jnp.asarray(self.reset_gate_probs, dtype=jnp.float32),
                post_prev_prob=self.reset_post_prev_prob,
                post_prev_speed_min=self.reset_post_prev_speed_min,
                post_prev_speed_max=self.reset_post_prev_speed_max,
                pre_gate_speed=self.reset_pre_gate_speed,
                post_prev_distance_min=self.reset_post_prev_distance_min,
                post_prev_distance_max=self.reset_post_prev_distance_max,
                yaw_jitter=self.reset_yaw_jitter,
                world_up=jnp.asarray(np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape((1, 1, 3))),
                bank_prob=self.reset_bank_prob,
                bank_pos=self.reset_bank_pos,
                bank_quat=self.reset_bank_quat,
                bank_vel=self.reset_bank_vel,
                bank_ang_vel=self.reset_bank_ang_vel,
                bank_target_gate=self.reset_bank_target_gate,
            ),
        )
        return sample.pos, sample.quat, sample.target_gate, sample.vel, sample.ang_vel

    def _step(self, action: Array) -> tuple[dict[str, Array], float, bool, bool, dict]:
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        normalized_action = self._normalize_action(action)
        self.apply_action(action)
        self.sim.step(self.sim.freq // self.freq)
        # Warp drones that have crashed outside the track to prevent them from interfering with
        # other drones still in the race
        self.sim.data = self._warp_disabled_drones(self.sim.data, self.data.disabled_drones)
        # Apply the environment logic. Check which drones are now disabled, check which gates have
        # been passed, and update the target gate.
        drone_pos = self.sim.data.states.pos
        drone_quat = self.sim.data.states.quat
        drone_vel = self.sim.data.states.vel
        drone_ang_vel = self.sim.data.states.ang_vel
        mocap_pos, mocap_quat = self.sim.mjx_data.mocap_pos, self.sim.mjx_data.mocap_quat
        contacts = self.sim.contacts()
        n_gates = len(self.data.gate_mj_ids)
        gates_pos = mocap_pos[:, self.data.gate_mj_ids]
        gates_quat = mocap_quat[:, self.data.gate_mj_ids][..., [1, 2, 3, 0]]
        gate_ids = self.data.gate_mj_ids[self.data.target_gate % n_gates]
        gate_pos = gates_pos[jnp.arange(gates_pos.shape[0])[:, None], gate_ids]
        gate_quat = gates_quat[jnp.arange(gates_quat.shape[0])[:, None], gate_ids]
        passed = gate_passed(
            drone_pos,
            self.data.last_drone_pos,
            gate_pos,
            gate_quat,
            (0.45, 0.45),
        )
        # Apply the environment logic with updated simulation data.
        self.data, disable_flags = self._step_env(
            self.data,
            drone_pos,
            drone_quat,
            drone_vel,
            drone_ang_vel,
            normalized_action,
            mocap_pos,
            mocap_quat,
            contacts,
            self.sim.freq,
        )
        reward = self.reward()
        terminated = self.terminated()
        truncated = self.truncated()
        final_obs = self.obs()
        info = {
            "passed": passed,
            "final_observation": final_obs,
            "raw_final_pos": drone_pos,
            "raw_final_vel": drone_vel,
            "raw_final_ang_vel": drone_ang_vel,
            "already_disabled": disable_flags.already_disabled,
            "no_target_left": disable_flags.no_target_left,
            "speed_limit": disable_flags.speed_limit,
            "angular_speed_limit": disable_flags.angular_speed_limit,
            "ground_crash": disable_flags.ground_crash,
            "out_of_bounds": disable_flags.out_of_bounds,
            "contact": disable_flags.contact,
            "invalid_state": disable_flags.invalid_state,
        }
        marked_for_reset = self.data.marked_for_reset
        # Reset finished worlds immediately so the next policy call sees a fresh observation while
        # reward/done still describe the transition that just ended.
        if self.autoreset and marked_for_reset.any():
            obs, _ = self._reset(mask=marked_for_reset)
        else:
            obs = self.obs()
        return obs, reward, terminated, truncated, info

    def _normalize_action(self, action: Array) -> Array:
        """Map the applied action into [-1, 1] per dimension for control regularization."""
        base_action_space = (
            self.single_action_space if hasattr(self, "single_action_space") else self.action_space
        )
        action_low = jnp.asarray(base_action_space.low, dtype=jnp.float32).reshape((1, 1, -1))
        action_high = jnp.asarray(base_action_space.high, dtype=jnp.float32).reshape((1, 1, -1))
        action = jnp.asarray(action, dtype=jnp.float32).reshape((self.sim.n_worlds, self.sim.n_drones, -1))
        return 2.0 * (action - action_low) / (action_high - action_low) - 1.0

    def apply_action(self, action: Array):
        """Apply the commanded state action to the simulation."""
        # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
        # https://github.com/jax-ml/jax/discussions/6055
        # Tracking issue:
        # https://github.com/jax-ml/jax/issues/29810
        # Forcing a copy here is less efficient, but avoids the warning.
        action = np.reshape(action, (self.sim.n_worlds, self.sim.n_drones, -1), copy=True)
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        match self.sim.control:
            case "attitude":
                self.sim.attitude_control(action)
            case "state":
                self.sim.state_control(action)
            case _:
                raise ValueError(f"Unsupported control mode: {self.sim.control}")

    def render(self):
        """Render the environment."""
        self.sim.render(cam_config=self.cam_config)

    def close(self):
        """Close the environment by stopping the drone and landing back at the starting position."""
        self.sim.close()

    def obs(self) -> dict[str, Array]:
        """Return the observation of the environment."""
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        gates_pos, gates_quat, obstacles_pos = self._obs(
            self.sim.mjx_data.mocap_pos,
            self.sim.mjx_data.mocap_quat,
            self.data.gates_visited,
            self.data.gate_mj_ids,
            self.gates["nominal_pos"],
            self.gates["nominal_quat"],
            self.data.obstacles_visited,
            self.data.obstacle_mj_ids,
            self.obstacles["nominal_pos"],
        )
        pos, quat, vel, ang_vel = self._sanitize_drone_obs(
            self.sim.data.states.pos,
            self.sim.data.states.quat,
            self.sim.data.states.vel,
            self.sim.data.states.ang_vel,
            self.data.disabled_drones,
        )
        obs = {
            "pos": pos,
            "quat": quat,
            "vel": vel,
            "ang_vel": ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> Array:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return self.data.rewards

    def terminated(self) -> Array:
        """Check if the episode is terminated.

        Returns:
            True if all drones have been disabled, else False.
        """
        return self.data.disabled_drones

    def truncated(self) -> Array:
        """Array of booleans indicating if the episode is truncated."""
        return self._truncated(self.data.steps, self.data.max_episode_steps, self.sim.n_drones)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    @property
    def drone_mass(self) -> NDArray[np.floating]:
        """The mass of the drones in the environment."""
        return np.asarray(self.sim.default_data.params.mass[..., 0])

    @staticmethod
    @jax.jit
    def _reset_env_data(
        data: EnvData,
        drone_pos: Array,
        mocap_pos: Array,
        target_gate: Array,
        mask: Array | None = None,
    ) -> EnvData:
        """Reset auxiliary variables of the environment data."""
        mask = jnp.ones(data.steps.shape, dtype=bool) if mask is None else mask
        target_gate = jnp.where(mask[..., None], target_gate, data.target_gate)
        last_drone_pos = jnp.where(mask[..., None, None], drone_pos, data.last_drone_pos)
        segment_start_pos = jnp.where(mask[..., None, None], drone_pos, data.segment_start_pos)
        disabled_drones = jnp.where(mask[..., None], False, data.disabled_drones)
        steps = jnp.where(mask, 0, data.steps)
        last_action = jnp.where(mask[..., None, None], 0.0, data.last_action)
        # Check which gates are in range of the drone
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = jnp.linalg.norm(dpos, axis=-1) < data.sensor_range
        gates_visited = jnp.where(mask[..., None, None], gates_visited, data.gates_visited)
        # And which obstacles are in range
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = jnp.linalg.norm(dpos, axis=-1) < data.sensor_range
        obstacles_visited = jnp.where(
            mask[..., None, None], obstacles_visited, data.obstacles_visited
        )
        return data.replace(
            target_gate=target_gate,
            last_drone_pos=last_drone_pos,
            segment_start_pos=segment_start_pos,
            disabled_drones=disabled_drones,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            rewards=jnp.where(mask[..., None], 0.0, data.rewards),
            steps=steps,
            last_action=last_action,
            marked_for_reset=jnp.where(mask, 0, data.marked_for_reset),  # Unmark after env reset
        )

    @staticmethod
    def _quat_apply(quat: Array, vec: Array) -> Array:
        """Rotate a world-space vector by a quaternion in [x, y, z, w] order."""
        q_xyz = quat[..., :3]
        q_w = quat[..., 3:4]
        uv = jnp.cross(q_xyz, vec)
        uuv = jnp.cross(q_xyz, uv)
        return vec + 2.0 * (q_w * uv + uuv)

    @staticmethod
    def _quat_multiply(lhs: Array, rhs: Array) -> Array:
        """Quaternion product for [x, y, z, w] quaternions."""
        lx, ly, lz, lw = lhs[..., 0:1], lhs[..., 1:2], lhs[..., 2:3], lhs[..., 3:4]
        rx, ry, rz, rw = rhs[..., 0:1], rhs[..., 1:2], rhs[..., 2:3], rhs[..., 3:4]
        xyz = jnp.concatenate(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            axis=-1,
        )
        w = lw * rw - lx * rx - ly * ry - lz * rz
        return jnp.concatenate([xyz, w], axis=-1)

    @staticmethod
    @jax.jit
    def _compute_reward(
        segment_start_pos: Array,
        last_drone_pos: Array,
        drone_pos: Array,
        drone_quat: Array,
        drone_vel: Array,
        drone_ang_vel: Array,
        gate_pos: Array,
        gate_quat: Array,
        passed: Array,
        course_complete: Array,
        disabled_drones: Array,
        prev_disabled_drones: Array,
        normalized_action: Array,
        prev_action: Array,
        progress_scale: Array,
        perception_weight: Array,
        perception_distance_scale: Array,
        safety_weight: Array,
        ang_vel_penalty_scale: Array,
        gate_width: Array,
        safety_activation_distance: Array,
        crash_penalty: Array,
        gate_pass_bonus: Array,
        completion_bonus: Array,
        vel_penalty_scale: Array,
    ) -> Array:
        """Compute the transition reward for the current step."""
        prev_gate_offset = gate_pos - last_drone_pos
        curr_gate_offset = gate_pos - drone_pos
        prev_gate_dist = jnp.linalg.norm(prev_gate_offset, axis=-1)
        curr_gate_dist = jnp.linalg.norm(curr_gate_offset, axis=-1)
        distance_progress = progress_scale * (prev_gate_dist - curr_gate_dist)

        prev_gate_dir = prev_gate_offset / jnp.maximum(prev_gate_dist[..., None], 1e-6)
        safe_gate_dir = curr_gate_offset / jnp.maximum(curr_gate_dist[..., None], 1e-6)
        drone_forward = RaceCoreEnv._quat_apply(
            drone_quat,
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), drone_pos.shape),
        )
        prev_gate_view_alignment = jnp.sum(drone_forward * prev_gate_dir, axis=-1)
        gate_view_alignment = jnp.sum(drone_forward * safe_gate_dir, axis=-1)
        prev_perception_distance_gain = jnp.tanh(
            prev_gate_dist / jnp.maximum(perception_distance_scale, 1e-6)
        )
        perception_distance_gain = jnp.tanh(
            curr_gate_dist / jnp.maximum(perception_distance_scale, 1e-6)
        )
        absolute_centering = perception_distance_gain * gate_view_alignment
        differential_centering = (
            perception_distance_gain * gate_view_alignment
            - prev_perception_distance_gain * prev_gate_view_alignment
        )
        # Keep one centering knob. Most of the term is persistent centering,
        # with a smaller differential component to avoid static camping.
        perception_reward = perception_weight * (
            0.75 * absolute_centering + 0.25 * differential_centering
        )

        gate_normal = RaceCoreEnv._quat_apply(
            gate_quat,
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32), gate_pos.shape),
        )
        rel_gate = drone_pos - gate_pos
        plane_distance = jnp.abs(jnp.sum(rel_gate * gate_normal, axis=-1))
        lateral_offset = jnp.linalg.norm(
            rel_gate - jnp.sum(rel_gate * gate_normal, axis=-1, keepdims=True) * gate_normal,
            axis=-1,
        )
        f = jnp.maximum(1.0 - plane_distance / safety_activation_distance, 0.0)
        v = jnp.maximum((1.0 - f) * (gate_width / 6.0), 0.05)
        safety_reward = -f * (1.0 - jnp.exp(-0.5 * (lateral_offset**2) / v))
        ang_vel_penalty = ang_vel_penalty_scale * jnp.linalg.norm(drone_ang_vel, axis=-1)
        vel_penalty = vel_penalty_scale * jnp.linalg.norm(drone_vel, axis=-1)

        distance_progress = jnp.where(disabled_drones, 0.0, distance_progress)
        perception_reward = jnp.where(disabled_drones, 0.0, perception_reward)
        safety_reward = jnp.where(disabled_drones, 0.0, safety_reward)
        newly_disabled = disabled_drones & ~prev_disabled_drones
        return (
            distance_progress
            + perception_reward
            + safety_weight * safety_reward
            # + gate_pass_bonus * passed.astype(jnp.float32)
            + completion_bonus * course_complete.astype(jnp.float32)
            - ang_vel_penalty
            - vel_penalty
            - crash_penalty * newly_disabled.astype(jnp.float32)
        )

    @staticmethod
    @jax.jit
    def _step_env(
        data: EnvData,
        drone_pos: Array,
        drone_quat: Array,
        drone_vel: Array,
        drone_ang_vel: Array,
        normalized_action: Array,
        mocap_pos: Array,
        mocap_quat: Array,
        contacts: Array,
        freq: int,
    ) -> tuple[EnvData, DisableReasonFlags]:
        """Step the environment data."""
        n_gates = len(data.gate_mj_ids)
        disable_flags = RaceCoreEnv._disable_reason_flags(
            drone_pos,
            drone_quat,
            drone_vel,
            drone_ang_vel,
            contacts,
            data,
        )
        disabled_drones = disable_flags.disabled
        gates_pos = mocap_pos[:, data.gate_mj_ids]
        obstacles_pos = mocap_pos[:, data.obstacle_mj_ids]
        # We need to convert the mocap quat from MuJoCo order to scipy order
        gates_quat = mocap_quat[:, data.gate_mj_ids][..., [1, 2, 3, 0]]
        # Extract the gate poses of the current target gates and check if the drones have passed
        # them between the last and current position
        gate_ids = data.gate_mj_ids[data.target_gate % n_gates]
        gate_pos = gates_pos[jnp.arange(gates_pos.shape[0])[:, None], gate_ids]
        gate_quat = gates_quat[jnp.arange(gates_quat.shape[0])[:, None], gate_ids]
        passed = gate_passed(drone_pos, data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
        course_complete = passed & (data.target_gate == (n_gates - 1))
        rewards = RaceCoreEnv._compute_reward(
            data.segment_start_pos,
            data.last_drone_pos,
            drone_pos,
            drone_quat,
            drone_vel,
            drone_ang_vel,
            gate_pos,
            gate_quat,
            passed,
            course_complete,
            disabled_drones,
            data.disabled_drones,
            normalized_action,
            data.last_action,
            data.progress_scale,
            data.perception_weight,
            data.perception_distance_scale,
            data.safety_weight,
            data.ang_vel_penalty_scale,
            data.gate_width,
            data.safety_activation_distance,
            data.crash_penalty,
            data.gate_pass_bonus,
            data.completion_bonus,
            data.vel_penalty_scale,
        )
        # Update the target gate index. Increment by one if drones have passed a gate
        target_gate = data.target_gate + passed * ~disabled_drones
        target_gate = jnp.where(target_gate >= n_gates, -1, target_gate)
        segment_start_pos = jnp.where(passed[..., None], gate_pos, data.segment_start_pos)
        steps = data.steps + 1
        truncated = steps >= data.max_episode_steps
        marked_for_reset = jnp.all(disabled_drones | truncated[..., None], axis=-1)
        # Update which gates and obstacles are or have been in range of the drone
        sensor_range = data.sensor_range
        dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
        gates_visited = data.gates_visited | (jnp.linalg.norm(dpos, axis=-1) < sensor_range)
        dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
        obstacles_visited = data.obstacles_visited | (jnp.linalg.norm(dpos, axis=-1) < sensor_range)
        data = data.replace(
            last_drone_pos=drone_pos,
            segment_start_pos=segment_start_pos,
            target_gate=target_gate,
            disabled_drones=disabled_drones,
            marked_for_reset=marked_for_reset,
            gates_visited=gates_visited,
            obstacles_visited=obstacles_visited,
            rewards=rewards,
            steps=steps,
            last_action=normalized_action,
        )
        return data, disable_flags

    @staticmethod
    @jax.jit
    def _obs(
        mocap_pos: Array,
        mocap_quat: Array,
        gates_visited: Array,
        gate_mocap_ids: Array,
        nominal_gate_pos: NDArray,
        nominal_gate_quat: NDArray,
        obstacles_visited: Array,
        obstacle_mocap_ids: Array,
        nominal_obstacle_pos: NDArray,
    ) -> tuple[Array, Array]:
        """Get the nominal or real gate positions and orientations depending on the sensor range."""
        mask, real_pos = gates_visited[..., None], mocap_pos[:, gate_mocap_ids]
        real_quat = mocap_quat[:, gate_mocap_ids][..., [1, 2, 3, 0]]
        if nominal_gate_pos.ndim == 2:
            nominal_gate_pos = nominal_gate_pos[None]
        if nominal_gate_quat.ndim == 2:
            nominal_gate_quat = nominal_gate_quat[None]
        gates_pos = jnp.where(mask, real_pos[:, None], nominal_gate_pos[:, None])
        gates_quat = jnp.where(mask, real_quat[:, None], nominal_gate_quat[:, None])
        mask, real_pos = obstacles_visited[..., None], mocap_pos[:, obstacle_mocap_ids]
        obstacles_pos = jnp.where(mask, real_pos[:, None], nominal_obstacle_pos[None, None])
        return gates_pos, gates_quat, obstacles_pos

    @staticmethod
    @partial(jax.jit, static_argnames="n_drones")
    def _truncated(steps: Array, max_episode_steps: Array, n_drones: int) -> Array:
        return jnp.tile((steps >= max_episode_steps)[..., None], (1, n_drones))

    @staticmethod
    def _sanitize_drone_obs(
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        disabled: Array,
    ) -> tuple[Array, Array, Array, Array]:
        invalid = disabled | RaceCoreEnv._invalid_drone_state(pos, quat, vel, ang_vel)
        safe_quat = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        pos = jnp.where(invalid[..., None], 0.0, pos)
        quat = jnp.where(invalid[..., None], safe_quat, quat)
        vel = jnp.where(invalid[..., None], 0.0, vel)
        ang_vel = jnp.where(invalid[..., None], 0.0, ang_vel)
        return pos, quat, vel, ang_vel

    @staticmethod
    def _invalid_drone_state(pos: Array, quat: Array, vel: Array, ang_vel: Array) -> Array:
        quat_norm = jnp.linalg.norm(quat, axis=-1)
        pos_finite = jnp.all(jnp.isfinite(pos), axis=-1)
        quat_finite = jnp.all(jnp.isfinite(quat), axis=-1)
        vel_finite = jnp.all(jnp.isfinite(vel), axis=-1)
        ang_vel_finite = jnp.all(jnp.isfinite(ang_vel), axis=-1)
        quat_valid = jnp.isfinite(quat_norm) & (quat_norm > 1e-8)
        return ~(pos_finite & quat_finite & vel_finite & ang_vel_finite & quat_valid)

    @staticmethod
    def _disable_reason_flags(
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        contacts: Array,
        data: EnvData,
    ) -> DisableReasonFlags:
        already_disabled = data.disabled_drones
        below_bounds = jnp.any(pos < data.pos_limit_low, axis=-1)
        above_bounds = jnp.any(pos > data.pos_limit_high, axis=-1)
        out_of_bounds = below_bounds | above_bounds
        ground_crash = pos[..., 2] <= 0.02
        linear_speed = jnp.linalg.norm(vel, axis=-1)
        angular_speed = jnp.linalg.norm(ang_vel, axis=-1)
        speed_limit = linear_speed > data.max_linear_speed
        angular_speed_limit = angular_speed > data.max_angular_speed
        no_target_left = data.target_gate == -1
        invalid_state = RaceCoreEnv._invalid_drone_state(pos, quat, vel, ang_vel)
        contact = jnp.any(contacts[:, None, :] & data.contact_masks, axis=-1)
        disabled = (
            already_disabled
            | out_of_bounds
            | ground_crash
            | speed_limit
            | angular_speed_limit
            | no_target_left
            | invalid_state
            | contact
        )
        return DisableReasonFlags(
            disabled,
            already_disabled,
            no_target_left,
            speed_limit,
            angular_speed_limit,
            ground_crash,
            out_of_bounds,
            contact,
            invalid_state,
        )

    @staticmethod
    def _disabled_drones(
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        contacts: Array,
        data: EnvData,
    ) -> Array:
        return RaceCoreEnv._disable_reason_flags(pos, quat, vel, ang_vel, contacts, data).disabled

    @staticmethod
    @jax.jit
    def _warp_disabled_drones(data: SimData, mask: Array) -> SimData:
        """Warp the disabled drones below the ground."""
        pos = jax.numpy.where(mask[..., None], -1, data.states.pos)
        return data.replace(states=data.states.replace(pos=pos))

    def _setup_sim(self, randomizations: dict):
        """Setup the simulation data and build the reset and step functions with custom hooks."""
        gate_spec = mujoco.MjSpec.from_file(str(self.gate_spec_path))
        obstacle_spec = mujoco.MjSpec.from_file(str(self.obstacle_spec_path))
        self._load_track_into_sim(gate_spec, obstacle_spec)
        # Set the initial drone states
        pos = self.sim.data.states.pos.at[...].set(self.drone["pos"])
        quat = self.sim.data.states.quat.at[...].set(self.drone["quat"])
        vel = self.sim.data.states.vel.at[...].set(self.drone["vel"])
        ang_vel = self.sim.data.states.ang_vel.at[...].set(self.drone["ang_vel"])
        states = self.sim.data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        self.sim.data = self.sim.data.replace(states=states)
        self.sim.build_default_data()
        # Build the reset randomizations and disturbances into the sim itself
        self.sim.reset_pipeline = self.sim.reset_pipeline + (build_reset_fn(randomizations),)
        self.sim.build_reset_fn()
        if "dynamics" in self.disturbances:
            disturbance_fn = build_dynamics_disturbance_fn(self.disturbances["dynamics"])
            self.sim.step_pipeline = (
                self.sim.step_pipeline[:2] + (disturbance_fn,) + self.sim.step_pipeline[2:]
            )
            self.sim.build_step_fn()

    def _load_track_into_sim(self, gate_spec: MjSpec, obstacle_spec: MjSpec):
        """Load the track into the simulation."""
        frame = self.sim.spec.worldbody.add_frame()
        n_gates, n_obstacles = len(self.gates["pos"]), len(self.obstacles["pos"])
        for i in range(n_gates):
            gate_body = gate_spec.body("gate")
            if gate_body is None:
                raise ValueError("Gate body not found in gate spec")
            gate = frame.attach_body(gate_body, "", f":{i}")
            gate.pos = self.gates["pos"][i]
            # Convert from scipy order to MuJoCo order
            gate.quat = self.gates["quat"][i][[3, 0, 1, 2]]
            gate.mocap = True  # Make mocap to modify the position of static bodies during sim
        for i in range(n_obstacles):
            obstacle_body = obstacle_spec.body("obstacle")
            if obstacle_body is None:
                raise ValueError("Obstacle body not found in obstacle spec")
            obstacle = frame.attach_body(obstacle_body, "", f":{i}")
            obstacle.pos = self.obstacles["pos"][i]
            obstacle.mocap = True
        self.sim.build_mjx()

    @staticmethod
    def _load_contact_masks(sim: Sim) -> Array:  # , data: EnvData
        """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
        sim.contacts()  # Trigger initial contact information computation
        contact = sim.mjx_data._impl.contact
        n_contacts = len(contact.geom1[0])
        masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)
        # We only need one world to create the mask
        geom1, geom2 = (contact.geom1[0], contact.geom2[0])
        for i in range(sim.n_drones):
            geom_start = sim.mj_model.body_geomadr[sim.mj_model.body(f"drone:{i}").id]
            geom_count = sim.mj_model.body_geomnum[sim.mj_model.body(f"drone:{i}").id]
            geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
            geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
            masks[i, :] = geom1_valid | geom2_valid
        geom_start = sim.mj_model.body_geomadr[sim.mj_model.body("world").id]
        geom_count = sim.mj_model.body_geomnum[sim.mj_model.body("world").id]
        geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
        geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)

        masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
        return masks


# region Factories


def rng_spec2fn(fn_spec: dict) -> Callable:
    """Convert a function spec to a wrapped and scaled function from jax.random."""
    offset, scale = np.array(fn_spec.get("offset", 0)), np.array(fn_spec.get("scale", 1))
    kwargs = fn_spec.get("kwargs", {})
    if "shape" in kwargs:
        raise KeyError("Shape must not be specified for randomization functions.")
    kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    jax_fn = partial(getattr(jax.random, fn_spec["fn"]), **kwargs)

    def random_fn(*args: Any, **kwargs: Any) -> Array:
        return jax_fn(*args, **kwargs) * scale + offset

    return random_fn


def build_reset_fn(randomizations: dict) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns += (randomize_drone_pos_fn(rng),)
            case "drone_rpy":
                randomization_fns += (randomize_drone_quat_fn(rng),)
            case "drone_mass":
                randomization_fns += (randomize_drone_mass_fn(rng),)
            case "drone_inertia":
                randomization_fns += (randomize_drone_inertia_fn(rng),)
            case "gate_pos" | "gate_rpy" | "obstacle_pos":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_fn(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_fn


def build_track_randomization_fn(
    randomizations: dict, gate_mocap_ids: list[int], obstacle_mocap_ids: list[int]
) -> Callable[[Data, Array, jax.random.PRNGKey], Data]:
    """Build the track randomization function for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "gate_pos":
                randomization_fns += (randomize_gate_pos_fn(rng, gate_mocap_ids),)
            case "gate_rpy":
                randomization_fns += (randomize_gate_rpy_fn(rng, gate_mocap_ids),)
            case "obstacle_pos":
                randomization_fns += (randomize_obstacle_pos_fn(rng, obstacle_mocap_ids),)
            case "drone_pos" | "drone_rpy" | "drone_mass" | "drone_inertia":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    @jax.jit
    def track_randomization(
        data: Data,
        mask: Array,
        nominal_gate_pos: Array,
        nominal_gate_quat: Array,
        nominal_obstacle_pos: Array,
        key: jax.random.PRNGKey,
    ) -> Data:
        gate_quat = jnp.roll(nominal_gate_quat, 1, axis=-1)  # Convert from scipy to MuJoCo order

        # Reset to default track positions first
        data = data.replace(mocap_pos=data.mocap_pos.at[:, gate_mocap_ids].set(nominal_gate_pos))
        data = data.replace(mocap_quat=data.mocap_quat.at[:, gate_mocap_ids].set(gate_quat))
        data = data.replace(
            mocap_pos=data.mocap_pos.at[:, obstacle_mocap_ids].set(nominal_obstacle_pos)
        )
        keys = jax.random.split(key, len(randomization_fns))
        for key, randomize_fn in zip(keys, randomization_fns, strict=True):
            data = randomize_fn(data, mask, key)
        return data

    return track_randomization


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance
