import jax
import jax.numpy as jnp

import chex

from copy import deepcopy

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State, id_to_pair
import bridge_env as env


def argmax_reverse(x):
    chex.assert_rank(x, 1)
    return jnp.where(
        jnp.any(x > 0),
        x.shape[0] - jnp.argmax(x[::-1]) - 1,
        PASS_ACTION_NUM - BID_OFFSET_NUM,
    )


def argmax(x):
    chex.assert_rank(x, 1)
    return jnp.where(
        jnp.any(x > 0),
        jnp.argmax(x),
        PASS_ACTION_NUM - BID_OFFSET_NUM,
    )


def get_reward_for_bid(state: State, bid: chex.Array) -> chex.Array:
    """
    Simulates with the given bid and returns the reward
    of the player that made the bid.
    It assumes that other players pass after the bid (if not,
    then to win we have to make the bid higher).
    """

    state = deepcopy(state)
    player = state.current_player

    action = bid + BID_OFFSET_NUM
    state, obs, rew, done = env.step(state, action)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)

    return state.rewards[player]


def get_internal_reward_for_bid(state: State, bid: chex.Array) -> chex.Array:
    """
    Simulates with the given bid and returns the reward of internal env
    of the player that made the bid.
    It assumes that other players pass after the bid (if not,
    then to win we have to make the bid higher).
    """

    state = deepcopy(state)
    player_id = state._env_state.current_player

    action = bid + BID_OFFSET_NUM
    state, obs, rew, done = env.step(state, action)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)

    return state._env_state.rewards[player_id]
