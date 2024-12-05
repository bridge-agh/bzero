import jax
import jax.numpy as jnp

import chex

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State
import bridge_env as env
from dds_common import argmax, get_internal_reward_for_bid, get_reward_for_bid


DDS_MISS_CHANCE = 0.35


def get_best_bid(state: State, rng: chex.PRNGKey) -> chex.Array:
    """
    Finds the best (lowest) bid for the current player in the given state.
    It maps every bid to the reward of the player that made the bid
    and returns the bid that maximizes the reward.
    """

    legal_bids = state.legal_action_mask[BID_OFFSET_NUM:]
    legal_bids_indexes = jnp.where(legal_bids, jnp.arange(35), -1)

    internal_rewards = jax.vmap(
        lambda bid: jax.lax.cond(bid < 0, lambda: jnp.nan, lambda: get_internal_reward_for_bid(state, bid))
    )(legal_bids_indexes)

    indices = jnp.argsort(internal_rewards)

    # create view of internal_rewards using indices
    int_rew_view_bids = jax.vmap(lambda x, y: x[y], in_axes=(None, 0))(internal_rewards, indices)

    # find first index of nan
    int_rew_view = jnp.isnan(int_rew_view_bids)
    argmax_ix = jnp.argmax(int_rew_view)

    real_argmax_ix = jax.lax.cond(
        jnp.equal(argmax_ix, 0),
        lambda: jax.lax.cond(
            int_rew_view[0],
            lambda: 0,  # pass if all nan (forbidden actions)
            lambda: 35,  # we will choose from whole action space (pass included)
        ),
        lambda: argmax_ix,
    )

    # find first value greater or equal to 0
    real_argmin_ix = jnp.argmax(jnp.where(int_rew_view_bids >= 0, jnp.bool_(True), jnp.bool_(False)))

    rng1, rng2 = jax.random.split(rng)
    action = jax.lax.cond(
        jax.random.bernoulli(rng1, DDS_MISS_CHANCE),
        lambda: indices[
            jax.random.randint(rng2, (), 0, real_argmin_ix)
        ],  # miss play / miss interpretation of enemy/partner card - pick random bid with negative outcome
        lambda: jax.lax.cond(  # we check if there is any bid that win the game, if not, we pass
            jnp.equal(real_argmax_ix, 0)
            | jnp.less(int_rew_view_bids[real_argmin_ix], 0)
            | jnp.equal(real_argmax_ix, real_argmin_ix),
            lambda: PASS_ACTION_NUM - BID_OFFSET_NUM,
            lambda: indices[
                jax.random.randint(rng2, (), real_argmin_ix, real_argmax_ix)
            ],  # correct play - pick random bid with positive outcome
        ),
    )

    # jax.debug.print(
    #     "{}\n{}\n{}\n{}\n{}\n{}\n",
    #     int_rew_view_bids,
    #     argmax_ix,
    #     real_argmax_ix,
    #     real_argmin_ix,
    #     action,
    #     internal_rewards[action],
    # )

    # jax.debug.print("{}", action)

    return action


def after_bid_check(state: State, rng: chex.PRNGKey) -> chex.Array:
    return jax.lax.cond(
        jnp.greater_equal(state._env_state._turn, 4),
        lambda: PASS_ACTION_NUM - BID_OFFSET_NUM,
        lambda: get_best_bid(state, rng),
    )


def make_dds_policy(rng: chex.PRNGKey, state: State) -> chex.Array:
    # best bid is the lowest bid that win the game
    best_bid = jax.vmap(after_bid_check, in_axes=(0, None))(state, rng)

    best_action = best_bid + BID_OFFSET_NUM

    # logits: best action = 2 or -2, pass = 1, other = 0
    # summarizing, if every best action doesn't win the game, then pass
    logits = jax.nn.one_hot(best_action, env.num_actions) * 2 + jax.nn.one_hot(PASS_ACTION_NUM, env.num_actions)
    return logits.argmax(axis=-1)
