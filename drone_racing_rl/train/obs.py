import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

POLICY_OBS_DIM = 27
"""Policy observation layout:

[body_vel(3), body_rates(3), rotation_matrix(9), target_gate_encoding(6), next_gate_encoding(6)]
"""

BODY_VEL_SCALE = 5.0
BODY_RATE_SCALE = 6.0
GATE_RADIUS_SCALE = 3.0
OBS_CLIP = 5.0


def encode_gate_pos(rel_pos: np.ndarray, gate_normal: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """in spherical coordinates, r, sin(theta), cos(theta), sin(phi), cos(phi), cos(alpha)"""
    r = np.linalg.norm(rel_pos, axis=-1)
    x, y, z = rel_pos[..., 0], rel_pos[..., 1], rel_pos[..., 2]

    safe_r = np.maximum(r, epsilon)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, rho)
    cos_alpha = np.sum(gate_normal * rel_pos, axis=-1)
    cos_alpha = cos_alpha / safe_r
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    features = np.stack(
        [r, np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi), cos_alpha],
        axis=-1,
    ).astype(np.float32)
    zero_mask = r < epsilon

    if np.any(zero_mask):
        features[zero_mask] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)

    return features


def _normalize_gate_encoding_np(features: np.ndarray) -> np.ndarray:
    features = features.copy()
    features[..., 0] = features[..., 0] / GATE_RADIUS_SCALE
    return np.clip(features, -OBS_CLIP, OBS_CLIP)


def _normalize_kinematics_np(body_vel: np.ndarray, body_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_vel = np.clip(body_vel / BODY_VEL_SCALE, -OBS_CLIP, OBS_CLIP)
    body_rates = np.clip(body_rates / BODY_RATE_SCALE, -OBS_CLIP, OBS_CLIP)
    return body_vel, body_rates





# | `pos` | `(3,)` | Drone position `[x, y, z]` in meters |
# | `quat` | `(4,)` | Drone orientation as quaternion `[qx, qy, qz, qw]` |
# | `vel` | `(3,)` | Drone velocity `[vx, vy, vz]` in m/s |
# | `ang_vel` | `(3,)` | Drone angular velocity `[wx, wy, wz]` in rad/s |
# | `target_gate` | `int` | Index of the next gate to pass (`-1` if all gates passed) |
# | `gates_pos` | `(n_gates, 3)` | Positions of all gates |
# | `gates_quat` | `(n_gates, 4)` | Orientations of all gates as quaternions |
# | `gates_visited` | `(n_gates,)` | Boolean flags for which gates have been passed |
# | `obstacles_pos` | `(n_obstacles, 3)` | Positions of all obstacles |
# | `obstacles_visited` | `(n_obstacles,)` | Boolean flags for obstacles detected |



def flatten_obs(observations: dict, vectorized: bool) -> np.ndarray:
    """Flatten the observation dictionary into a single array.
    
    Args:
        observations: The observation dictionary.
        vectorized: Whether the observations are vectorized or not.

    Returns:
        A flattened array of the observations.
        format: (all in drone body frame)
        - body_vel (3)
        - body_rates (3) (angular velocity)
        - orientation (9) (rotation matrix)
        - relative position to target gate (6) (in polar coordinates, r, sin(theta), cos(theta), sin(phi), cos(phi), cos(alpha)) Alpha is the angle between drone to gate center and the gate normal
        - position of next target gate relative to target gate (6)
        Total dimension: 27
    """
    obs = {}
    N = observations["pos"].shape[0] if vectorized else 1
    for key in observations.keys():
        if key == "target_gate":
            x = np.asarray(observations[key], dtype=np.int32).reshape((N,))
            obs[key] = x
            continue
        x = np.asarray(observations[key], dtype=np.float32)
        obs[key] = x if vectorized else x[None, ...] # Add batch dim

    batch_idx = np.arange(N)
    R_wb = R.from_quat(obs["quat"]) # (N,4)
    world_vel = obs["vel"] # (N,3)
    world_ang_vel = obs["ang_vel"] # (N,3)
    pos = obs["pos"] # (N,3)
    gates_pos = obs["gates_pos"] # (N,num_gates,3)
    num_gates = gates_pos.shape[1] # num_gates
    target_gates = obs["target_gate"] # (N,)

    assert num_gates != 0


    # Wrap around gates.
    target_gates = np.where(target_gates < 0, 0, target_gates)
    next_target_gates = (target_gates + 1) % num_gates
    batch_idx = np.arange(N)

    # Drone rotation
    R_bw = R_wb.inv()

    # Drone vel and orientation in body frame.
    body_vel = R_bw.apply(world_vel) # (OBS), (N,3)
    body_rates = R_bw.apply(world_ang_vel) # (OBS), (N,3)
    body_vel, body_rates = _normalize_kinematics_np(body_vel, body_rates)
    rotation_matrix = R_wb.as_matrix().reshape(N,9) # (OBS) (N, 9)


    # Target gate
    gate_rot = R.from_quat(obs["gates_quat"][batch_idx,target_gates])
    gate_normal = gate_rot.apply(np.array([1,0,0], dtype=np.float32))
    target_pos = gates_pos[batch_idx,target_gates] # (N,3)
    rel_body_gate_pos = R_bw.apply(target_pos - pos) # (N,3) (OBS)
    gate_normal_body = R_bw.apply(gate_normal) # (N, 3)
    target_encoding = encode_gate_pos(rel_body_gate_pos, gate_normal_body, epsilon=1e-6) # (N,6)
    target_encoding = _normalize_gate_encoding_np(target_encoding)


    # Next target gate
    rotation = gate_rot.inv()
    next_target_pos = obs["gates_pos"][batch_idx,next_target_gates] # (N,3)
    next_gate_rot = R.from_quat(obs["gates_quat"][batch_idx,next_target_gates])
    next_gate_normal = next_gate_rot.apply(np.array([1,0,0], dtype=np.float32))
    next_target_pos = rotation.apply(next_target_pos - target_pos) # (N,3)
    next_gate_normal_target = rotation.apply(next_gate_normal) # (N,3)
    next_target_encoding = encode_gate_pos(next_target_pos, next_gate_normal_target, epsilon=1e-6) # (N,6)
    next_target_encoding = _normalize_gate_encoding_np(next_target_encoding)



    out =  np.concatenate([body_vel, body_rates, rotation_matrix, target_encoding, next_target_encoding],axis=-1)
    return out if vectorized else out[0]


def _quat_conjugate_jax(quat: jax.Array) -> jax.Array:
    xyz = -quat[..., :3]
    w = quat[..., 3:]
    return jnp.concatenate([xyz, w], axis=-1)


def _quat_apply_jax(quat: jax.Array, vec: jax.Array) -> jax.Array:
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    uv = jnp.cross(q_xyz, vec)
    uuv = jnp.cross(q_xyz, uv)
    return vec + 2.0 * (q_w * uv + uuv)


def _quat_to_rotmat_jax(quat: jax.Array) -> jax.Array:
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return jnp.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    )


def encode_gate_pos_jax(
    rel_pos: jax.Array,
    gate_normal: jax.Array,
    epsilon: float = 1e-6,
) -> jax.Array:
    r = jnp.linalg.norm(rel_pos, axis=-1)
    x, y, z = rel_pos[..., 0], rel_pos[..., 1], rel_pos[..., 2]

    safe_r = jnp.maximum(r, epsilon)
    rho = jnp.hypot(x, y)
    theta = jnp.arctan2(y, x)
    phi = jnp.arctan2(z, rho)
    cos_alpha = jnp.sum(gate_normal * rel_pos, axis=-1) / safe_r
    cos_alpha = jnp.clip(cos_alpha, -1.0, 1.0)

    features = jnp.stack(
        [r, jnp.sin(theta), jnp.cos(theta), jnp.sin(phi), jnp.cos(phi), cos_alpha],
        axis=-1,
    )
    default_features = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    zero_mask = r < epsilon
    return jnp.where(zero_mask[..., None], default_features, features)


def _normalize_gate_encoding_jax(features: jax.Array) -> jax.Array:
    radius = features[..., :1] / GATE_RADIUS_SCALE
    angles = features[..., 1:]
    return jnp.clip(jnp.concatenate([radius, angles], axis=-1), -OBS_CLIP, OBS_CLIP)


def _normalize_kinematics_jax(body_vel: jax.Array, body_rates: jax.Array) -> tuple[jax.Array, jax.Array]:
    body_vel = jnp.clip(body_vel / BODY_VEL_SCALE, -OBS_CLIP, OBS_CLIP)
    body_rates = jnp.clip(body_rates / BODY_RATE_SCALE, -OBS_CLIP, OBS_CLIP)
    return body_vel, body_rates


def flatten_obs_jax(observations: dict[str, jax.Array], vectorized: bool) -> jax.Array:
    """JAX-native version of ``flatten_obs`` suitable for JIT/scans.

    Output layout matches ``POLICY_OBS_DIM`` exactly.
    """
    if not vectorized:
        observations = {
            key: (value[None, ...] if key != "target_gate" else value[None])
            for key, value in observations.items()
        }

    batch_idx = jnp.arange(observations["pos"].shape[0])
    pos = jnp.asarray(observations["pos"], dtype=jnp.float32)
    world_quat = jnp.asarray(observations["quat"], dtype=jnp.float32)
    world_vel = jnp.asarray(observations["vel"], dtype=jnp.float32)
    world_ang_vel = jnp.asarray(observations["ang_vel"], dtype=jnp.float32)
    gates_pos = jnp.asarray(observations["gates_pos"], dtype=jnp.float32)
    gates_quat = jnp.asarray(observations["gates_quat"], dtype=jnp.float32)
    num_gates = gates_pos.shape[1]
    if num_gates == 0:
        raise ValueError("flatten_obs_jax requires at least one gate")

    target_gates = jnp.asarray(observations["target_gate"], dtype=jnp.int32)
    target_gates = jnp.where(target_gates < 0, 0, target_gates)
    next_target_gates = (target_gates + 1) % num_gates

    r_bw = _quat_conjugate_jax(world_quat)
    body_vel = _quat_apply_jax(r_bw, world_vel)
    body_rates = _quat_apply_jax(r_bw, world_ang_vel)
    body_vel, body_rates = _normalize_kinematics_jax(body_vel, body_rates)
    rotation_matrix = _quat_to_rotmat_jax(world_quat)

    target_pos = gates_pos[batch_idx, target_gates]
    target_gate_quat = gates_quat[batch_idx, target_gates]
    target_gate_normal_world = _quat_apply_jax(
        target_gate_quat,
        jnp.tile(jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32), (pos.shape[0], 1)),
    )
    rel_body_gate_pos = _quat_apply_jax(r_bw, target_pos - pos)
    target_gate_normal_body = _quat_apply_jax(r_bw, target_gate_normal_world)
    target_encoding = _normalize_gate_encoding_jax(
        encode_gate_pos_jax(rel_body_gate_pos, target_gate_normal_body)
    )

    target_gate_inv = _quat_conjugate_jax(target_gate_quat)
    next_target_pos = gates_pos[batch_idx, next_target_gates]
    next_target_quat = gates_quat[batch_idx, next_target_gates]
    next_gate_normal_world = _quat_apply_jax(
        next_target_quat,
        jnp.tile(jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32), (pos.shape[0], 1)),
    )
    next_target_rel = _quat_apply_jax(target_gate_inv, next_target_pos - target_pos)
    next_gate_normal_target = _quat_apply_jax(target_gate_inv, next_gate_normal_world)
    next_target_encoding = _normalize_gate_encoding_jax(
        encode_gate_pos_jax(next_target_rel, next_gate_normal_target)
    )

    out = jnp.concatenate(
        [body_vel, body_rates, rotation_matrix, target_encoding, next_target_encoding],
        axis=-1,
    ).astype(jnp.float32)
    return out if vectorized else out[0]
