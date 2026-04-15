import copy
import jax.numpy as jnp
import jax
import numpy as np
from scipy.spatial.transform import Rotation as R


def k_nearest_obstacles(obstacle_pos: np.ndarray, pos: np.ndarray, k: int) -> np.ndarray:
    """Find the k nearest obstacles to the drone."""

    # if K is larger than actual obstacles, pad with zeros
    k_eff = min(k, obstacle_pos.shape[1])
    N = obstacle_pos.shape[0]
    dist = np.linalg.norm(obstacle_pos - pos[:, None, :], axis=-1)
    sorted_dist = np.argsort(dist, axis=-1) # (N, Obstacles)
    batch_idx = np.arange(N)[:, None] # (N,1) 
    obs_idx = sorted_dist[:, :k_eff] # (N, k)
    nearest = obstacle_pos[batch_idx, obs_idx, :] # (N, k, 3)
    if k_eff < k:
        pad = np.zeros((N, k - k_eff, 3), dtype=nearest.dtype)
        nearest = np.concatenate([nearest, pad], axis=1)

    assert nearest.shape == (N, k, 3)
    return nearest


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
        - relative position to gate: (3)
        - gate normal axis: (3)
        - relative position to obstacles(k nearest): (3)
        - nearest obstacles: (3 * K) 
        - velocity: (3)
        - angular velocity: (3)
        - gravity dir: (3)
        - progress: (1) 0 to 1 
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

    K = 2 
    target_gates = obs["target_gate"] # (N,)
    batch_idx = np.arange(N)
    pos = obs["pos"] # (N,3)
    body_quat = obs["quat"] # (N,4)
    R_bw = R.from_quat(body_quat).inv() # world to body 

    gate_pos = obs["gates_pos"][batch_idx,target_gates] # (N,3)
    rel_gate_pos_w = gate_pos - pos # (N,3)
    rel_gate_pos_b = R_bw.apply(rel_gate_pos_w) # (N,3) (OBS)


    nearest_obs_w = k_nearest_obstacles(obs["obstacles_pos"],pos,k=K) # (N,K,3) 
    nearest_obs_b = np.stack(
        [R_bw[i].apply(nearest_obs_w[i]) for i in range(N)],
        axis=0,
    ).reshape(N, -1) # (N, K, 3) -> (N, K*3)

    vel_b = R_bw.apply(obs["vel"]) # (N,3) (OBS)
    ang_vel_b = R_bw.apply(obs["ang_vel"]) # (N,3) (OBS)


    R_wg = R.from_quat(obs["gates_quat"][batch_idx,target_gates])
    gate_normal_b = R_bw.apply(R_wg.apply(np.array([1,0,0]))) # gate normal in body frame (OBS) (N, 3)

    gravity =  R_bw.apply(np.array([0,0,-1], dtype=np.float32)) #(N, 3)
    gates_visited = obs["gates_visited"] # (N,Num gates) (OBS)
    progress = gates_visited.sum(axis=-1,keepdims=True).astype(np.float32) / gates_visited.shape[-1] # (N, 1) # (OBS)

    out =  np.concatenate([rel_gate_pos_b, gate_normal_b, nearest_obs_b, vel_b,ang_vel_b,gravity, progress],axis=-1)
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


def k_nearest_obstacles_jax(obstacle_pos: jax.Array, pos: jax.Array, k: int) -> jax.Array:
    """Find the k nearest obstacles to the drone in JAX."""
    n_obstacles = obstacle_pos.shape[1]
    dist = jnp.linalg.norm(obstacle_pos - pos[:, None, :], axis=-1)
    sorted_idx = jnp.argsort(dist, axis=-1)
    k_eff = min(k, n_obstacles)
    nearest = jnp.take_along_axis(
        obstacle_pos,
        sorted_idx[:, :k_eff, None],
        axis=1,
    )
    if k_eff < k:
        pad = jnp.zeros((obstacle_pos.shape[0], k - k_eff, 3), dtype=nearest.dtype)
        nearest = jnp.concatenate([nearest, pad], axis=1)
    return nearest


def flatten_obs_jax(observations: dict[str, jax.Array], vectorized: bool) -> jax.Array:
    """JAX-native version of ``flatten_obs`` suitable for JIT/scans."""
    if not vectorized:
        observations = {
            key: (value[None, ...] if key != "target_gate" else value[None])
            for key, value in observations.items()
        }

    K = 2
    batch_idx = jnp.arange(observations["pos"].shape[0])
    pos = jnp.asarray(observations["pos"], dtype=jnp.float32)
    body_quat = jnp.asarray(observations["quat"], dtype=jnp.float32)
    r_bw = _quat_conjugate_jax(body_quat)
    target_gates = jnp.asarray(observations["target_gate"], dtype=jnp.int32)

    gate_pos = observations["gates_pos"][batch_idx, target_gates]
    rel_gate_pos_w = gate_pos - pos
    rel_gate_pos_b = _quat_apply_jax(r_bw, rel_gate_pos_w)

    nearest_obs_w = k_nearest_obstacles_jax(jnp.asarray(observations["obstacles_pos"], dtype=jnp.float32), pos, k=K)
    nearest_obs_rel_w = nearest_obs_w - pos[:, None, :]
    nearest_obs_b = _quat_apply_jax(
        jnp.repeat(r_bw[:, None, :], K, axis=1),
        nearest_obs_rel_w,
    ).reshape(pos.shape[0], -1)

    vel_b = _quat_apply_jax(r_bw, jnp.asarray(observations["vel"], dtype=jnp.float32))
    ang_vel_b = _quat_apply_jax(r_bw, jnp.asarray(observations["ang_vel"], dtype=jnp.float32))

    gate_quat = jnp.asarray(observations["gates_quat"], dtype=jnp.float32)[batch_idx, target_gates]
    gate_normal_w = _quat_apply_jax(gate_quat, jnp.tile(jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32), (pos.shape[0], 1)))
    gate_normal_b = _quat_apply_jax(r_bw, gate_normal_w)

    gravity_world = jnp.tile(jnp.array([[0.0, 0.0, -1.0]], dtype=jnp.float32), (pos.shape[0], 1))
    gravity_b = _quat_apply_jax(r_bw, gravity_world)

    gates_visited = jnp.asarray(observations["gates_visited"], dtype=jnp.float32)
    progress = jnp.sum(gates_visited, axis=-1, keepdims=True) / gates_visited.shape[-1]

    out = jnp.concatenate(
        [rel_gate_pos_b, gate_normal_b, nearest_obs_b, vel_b, ang_vel_b, gravity_b, progress],
        axis=-1,
    )
    return out if vectorized else out[0]
