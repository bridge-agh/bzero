import jax
import jax.numpy as jnp

import chex

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State
import bridge_env as env
from dds_common import argmax_reverse, get_reward_for_bid


def get_best_bid(state: State):
    """
    Finds the best bid for the current player in the given state.
    It maps every bid to the reward of the player that made the bid
    and returns the bid that maximizes the reward.
    """

    legal_bids = state.legal_action_mask[BID_OFFSET_NUM:]
    legal_bids_indexes = jnp.where(legal_bids, jnp.arange(35), -1)
    
    rewards = jax.vmap(lambda bid: jax.lax.cond(bid < 0, lambda: -1., lambda: get_reward_for_bid(state, bid)))(legal_bids_indexes)

    return argmax_reverse(rewards)


def make_dds_policy(rng: chex.PRNGKey, state: State) -> chex.Array:
    # best bid is the highest bid that win the game
    best_bid = jax.vmap(get_best_bid)(state)
    best_action = best_bid + BID_OFFSET_NUM

    # logits: best action = 2 or -2, pass = 1, other = 0
    # summarizing, if every best action doesn't win the game, then pass
    logits = jax.nn.one_hot(best_action, env.num_actions) * 2 + jax.nn.one_hot(PASS_ACTION_NUM, env.num_actions)
    return logits.argmax(axis=-1)
