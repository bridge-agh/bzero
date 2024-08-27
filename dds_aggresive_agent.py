import jax
import jax.numpy as jnp

import chex

from copy import deepcopy

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State
import bridge_env as env


def argmax_reverse(x):
    chex.assert_rank(x, 1)
    return jnp.where(
        jnp.any(x > 0),
        x.shape[0] - jnp.argmax(x[::-1]) - 1,
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


# def safely_get_reward_for_bid(state: State, bid: chex.Array) -> chex.Array:
#     return jax.lax.cond(bid < 0, lambda: -1., lambda: get_reward_for_bid(state, bid))

def get_best_bid(state: State):
    """
    Finds the best bid for the current player in the given state.
    It maps every bid to the reward of the player that made the bid
    and returns the bid that maximizes the reward.
    """

    legal_bids = state.legal_action_mask[BID_OFFSET_NUM:]
    legal_bids_indexes = jnp.where(legal_bids, jnp.arange(35), -1)
    # rewards = jax.vmap(safely_get_reward_for_bid, in_axes=[None, 0])(state, legal_bids_indexes)
    rewards = jax.vmap(lambda bid: jax.lax.cond(bid < 0, lambda: -1., lambda: get_reward_for_bid(state, bid)))(legal_bids_indexes)

    return argmax_reverse(rewards)


def dds_policy(rng: chex.PRNGKey, state: State) -> chex.Array:
    # best bid is the highest bid that win the game
    best_bid = jax.vmap(get_best_bid)(state)
    best_action = best_bid + BID_OFFSET_NUM

    # logits: best action = 2 or -2, pass = 1, other = 0
    # summarizing, if every best action doesn't win the game, then pass
    logits = jax.nn.one_hot(best_action, env.num_actions) * 2 + jax.nn.one_hot(PASS_ACTION_NUM, env.num_actions)
    # action_mask = state.legal_action_mask
    # logits_masked = jnp.where(action_mask, logits, -1)
    return logits.argmax(axis=-1)
