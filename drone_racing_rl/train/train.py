import flax
from drone_racing_rl.train.actor_critic_models import ppo_loss
from sympy.simplify.fu import L
import jax
import optax
import jax.numpy as jnp




def create_train_state(
    rng: jax.Array,
    model: flax.linen.Module,
    obs_dim: int,
    action_dim: int,
    lr: float,
    max_grad_norm: float,
):
    params = model.init(rng, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=lr, eps=1e-8),
    )
    opt_state = tx.init(params)
    return params, opt_state, tx



def make_update_fn(model, tx:optax.GradientTransformation, clip_eps:float, vf_coef:float, ent_coef:float):
    @jax.jit
    def update_step(params,opt_state, batch: dict[str, jax.Array]):
        def loss_fn(params):
            total_loss, (actor_loss, value_loss, entropy) = ppo_loss(
                params,
                model,
                batch["obs"],
                batch["actions"],
                batch["old_log_probs"],
                batch["old_values"],
                batch["advantages"],
                batch["returns"],
                clip_eps,
                vf_coef,
                ent_coef,
            )
            return total_loss, (actor_loss, value_loss, entropy)

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {
            "total_loss": total_loss,
            "actor_loss": aux[0],
            "value_loss": aux[1],
            "entropy": aux[2],
        }
        return params, opt_state, metrics
    return update_step
        


def flatten_rollout_batch(
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    old_values: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
) -> dict[str, jax.Array]:
    return {
        "obs": obs.reshape((-1, obs.shape[-1])),
        "actions": actions.reshape((-1, actions.shape[-1])),
        "old_log_probs": old_log_probs.reshape(-1),
        "old_values": old_values.reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
    }



def make_minibatches(batch: dict[str, jax.Array], rng: jax.Array, num_minibatches: int):
    """Shuffle and split a flattened PPO batch into minibatches.

    Expected shapes:
        obs:           (B, obs_dim)
        action:        (B, action_dim)
        old_log_prob:  (B,)
        old_value:     (B,)
        advantages:    (B,)
        returns:       (B,)
    """
    batch_size = batch["obs"].shape[0]
    assert batch_size % num_minibatches == 0
    minibatch_size = batch_size // num_minibatches

    perm = jax.random.permutation(rng, batch_size)
    shuffled = {k: v[perm] for k, v in batch.items()}

    minibatches = []
    for i in range(num_minibatches):
        start = i * minibatch_size
        end = start + minibatch_size
        minibatches.append({k: v[start:end] for k, v in shuffled.items()})
    return minibatches
