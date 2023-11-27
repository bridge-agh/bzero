import jax
import jax.numpy as jnp
from pgx import State
import chex
from chex import PRNGKey
import mctx

from functools import partial

from type_aliases import Reward, Action
import env_wrapper as env


def winning_action_mask(state: State) -> chex.Array:
    actions = jnp.arange(env.num_actions)
    _, _, rewards, _ = jax.vmap(env.step, (None, 0))(state, actions)
    chex.assert_shape(rewards, [env.num_actions])
    return rewards == 1


def lookahead_policy(state: State) -> chex.Array:
    return sum((
        state.legal_action_mask.astype(jnp.float32) * 100,
        winning_action_mask(state).astype(jnp.float32) * 200,
    ))


def rollout_value(rng: PRNGKey, state: State) -> Reward:
    def step(state: State, rng: PRNGKey) -> tuple[State, Reward]:
        action = jax.random.categorical(rng, lookahead_policy(state))
        state, _, _, _ = env.step(state, action)
        return state, state.rewards
    _, rewards = jax.lax.scan(step, state, jax.random.split(rng, env.max_steps))
    chex.assert_shape(rewards, [env.max_steps, 2])
    return rewards.sum(axis=0)[state.current_player]


def batched_root_fn(rng: PRNGKey, state: State):
    rngs = jax.random.split(rng, state.current_player.shape[0])
    return mctx.RootFnOutput(
        prior_logits=jax.vmap(lookahead_policy)(state),
        value=jax.vmap(rollout_value)(rngs, state),
        embedding=state,
    )


def batched_recurrent_fn(_, rng: PRNGKey, action: Action, state: State):
    rngs = jax.random.split(rng, state.current_player.shape[0])
    new_state, observation, reward, done = jax.vmap(env.step)(state, action)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0).astype(jnp.float32),
        prior_logits=jax.vmap(lookahead_policy)(new_state),
        value=jnp.where(done, 0.0, jax.vmap(rollout_value)(rngs, new_state)).astype(jnp.float32),
    )
    return recurrent_fn_output, new_state


def batched_compute_policy(rng: PRNGKey, state: State, num_simulations: int):
    policy_rng, root_rng = jax.random.split(rng)
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=policy_rng,
        root=batched_root_fn(root_rng, state),
        recurrent_fn=batched_recurrent_fn,
        num_simulations=num_simulations,
        max_depth=env.max_steps,
        qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        invalid_actions=1.0 - state.legal_action_mask.astype(jnp.float32),
        dirichlet_fraction=0.0,
    )
    return policy_output
