from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
class Transition(NamedTuple):
    """Transition data for the actor critic model."""
    obs: np.ndarray
    action: np.ndarray
    value: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    log_prob: np.ndarray


def compute_gae(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
    last_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = values.shape[0] # trajectory length
    N = values.shape[1] # number of environments
    advantages = np.zeros((T,N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)): 
        next_value = last_value if t == T-1 else values[t+1]
        not_done = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * not_done - values[t] 
        gae = delta + gamma * lambda_ * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

def ppo_loss(
    params,
    model,
    obs,
    actions,
    old_log_probs,
    old_values,
    advantages,
    returns,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    pi, values = model.apply(params, obs)
    entropy = pi.entropy().mean()
    log_prob = pi.log_prob(actions)
    ratio = jnp.exp(log_prob - old_log_probs)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    value_pred_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    value_loss1 = (values - returns)**2
    value_loss2 = (value_pred_clipped - returns)**2
    value_loss = 0.5 * jnp.maximum(value_loss1, value_loss2).mean()

    # minimise actor loss, value_loss, incentive to have some amount of entropy scaled by the coefficient to encourage exploration.
    total_loss = actor_loss + vf_coef *  value_loss - ent_coef * entropy

    return total_loss, (actor_loss, value_loss, entropy)

