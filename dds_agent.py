import jax
import jax.numpy as jnp

import chex

from pgx import State
from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

import bridge_env as env


def argmax_reverse(x):
    chex.assert_rank(x, 1)
    return x.shape[0] - jnp.argmax(x[::-1]) - 1


def get_reward_for_bid(state: State, bid: chex.Array) -> chex.Array:
    player = state.current_player
    chex.assert_shape(state._init_rng, [])
    state, _ = env.reset(state._init_rng)
    we_play_first = state.current_player == player
    action = bid + BID_OFFSET_NUM
    state, obs, rew, done = env.step(state, jnp.where(we_play_first, action, PASS_ACTION_NUM))
    state, obs, rew, done = env.step(state, jnp.where(we_play_first, PASS_ACTION_NUM, action))
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    reward_we_play_first = state.rewards[player]
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    reward_we_play_second = state.rewards[player]
    return jnp.where(we_play_first, reward_we_play_first, reward_we_play_second)


def get_best_bid(state):
    rewards = jax.vmap(get_reward_for_bid, in_axes=[None, 0])(state, jnp.arange(35))
    return argmax_reverse(rewards)


def dds_policy(rng: chex.PRNGKey, state: State) -> chex.Array:
    already_bid = state._env_state._step_count > 2
    chex.assert_shape(already_bid, [None])
    best_bid = jax.vmap(get_best_bid)(state)
    best_action = jnp.where(
        already_bid,
        PASS_ACTION_NUM,
        best_bid + BID_OFFSET_NUM,
    )
    return jax.nn.one_hot(best_action, env.num_actions) * 200 + jax.nn.one_hot(PASS_ACTION_NUM, env.num_actions) * 100
