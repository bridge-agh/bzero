import chex
import jax
import jax.numpy as jnp

from functools import partial
from tqdm import tqdm

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State
from dds_common import argmax_reverse
from run_tournament import make_bzero_policy, make_random_policy, make_mcts_policy
import bridge_env as env
from type_aliases import Done, Reward
from dds_peaceful_agent import make_dds_policy as make_dds_peaceful_policy
from dds_aggressive_agent import make_dds_policy as make_dds_aggressive_policy


# def get_reward_for_bid_first_players(state, bid, player):
#     chex.assert_shape(state._init_rng, [])

#     state, _ = env.reset(state._init_rng)

#     we_play_first = state.current_player == player
#     action = bid + BID_OFFSET_NUM

#     state, obs, rew, done = env.step(state, jnp.where(we_play_first, action, PASS_ACTION_NUM))
#     state, obs, rew, done = env.step(state, jnp.where(we_play_first, PASS_ACTION_NUM, action))
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     reward_we_play_first = state.rewards[player]
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     reward_we_play_second = state.rewards[player]

#     return jnp.where(we_play_first, reward_we_play_first, reward_we_play_second)


# def get_reward_for_bid_second_players(state, bid, player):
#     chex.assert_shape(state._init_rng, [])

#     state, _ = env.reset(state._init_rng)

#     we_play_first = state.current_player == player
#     action = bid + BID_OFFSET_NUM

#     state, _ = env.reset(state._init_rng)
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     state, obs, rew, done = env.step(state, jnp.where(we_play_first, action, PASS_ACTION_NUM))
#     state, obs, rew, done = env.step(state, jnp.where(we_play_first, PASS_ACTION_NUM, action))
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     reward_we_play_first = state.rewards[player]
#     state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
#     reward_we_play_second = state.rewards[player]

#     return jnp.where(we_play_first, reward_we_play_first, reward_we_play_second)


# def loop(args):
#     _, subkey = jax.random.split(args[1])
#     state, _ = env.reset(subkey)
#     first_player_0 = state.current_player == 0

#     rewards0 = jax.vmap(get_reward_for_bid_first_players, in_axes=[None, 0, None])(state, jnp.arange(35), 0)
#     rewards1 = jax.vmap(get_reward_for_bid_first_players, in_axes=[None, 0, None])(state, jnp.arange(35), 1)
#     rewards2 = jax.vmap(get_reward_for_bid_second_players, in_axes=[None, 0, None])(state, jnp.arange(35), 0)
#     rewards3 = jax.vmap(get_reward_for_bid_second_players, in_axes=[None, 0, None])(state, jnp.arange(35), 1)

#     max_bid_0 = argmax_reverse(rewards0)
#     max_bid_1 = argmax_reverse(rewards1)
#     max_bid_2 = argmax_reverse(rewards2)
#     max_bid_3 = argmax_reverse(rewards3)

#     chex.assert_shape(max_bid_0, [])

#     # second player must have higher bid but also bid ealier
#     winnable = jax.lax.cond(
#         first_player_0,
#         lambda: (max_bid_3 >= max_bid_2) & (max_bid_3 > max_bid_0)
#         | (max_bid_1 >= max_bid_2) & (max_bid_1 >= max_bid_0),
#         lambda: (max_bid_3 > max_bid_2) & (max_bid_3 > max_bid_0) | (max_bid_1 > max_bid_2) & (max_bid_1 > max_bid_0),
#     )

#     return (winnable, subkey)


# def get_winnable_game(rng):
#     "Returns state of a game where the second player can win"

#     a = jax.lax.while_loop(lambda cond: jnp.logical_not(cond[0]), loop, (False, rng))

#     return env.reset(a[1])


def get_repeated_bid(bid_history, player):

    def inside_loop(args):
        _, ix, last_bid, _ = args

        new_bid = bid_history[ix]

        is_new_bid_not_action = jnp.greater_equal(new_bid, BID_OFFSET_NUM)
        is_last_bid_not_action = jnp.greater_equal(last_bid, BID_OFFSET_NUM)
        is_over_last_bid = jnp.greater(new_bid, last_bid)

        is_new_bid_null = jnp.equal(new_bid, -1)  # finishes loop, not found repeat

        return jax.lax.cond(
            is_new_bid_null,
            lambda: (True, ix, last_bid, False),
            lambda: jax.lax.cond(
                jnp.all(jnp.array([is_new_bid_not_action, is_last_bid_not_action, is_over_last_bid])),
                lambda: (True, jnp.subtract(ix, 4), new_bid, True),  # found repeat
                lambda: (False, jnp.add(ix, 4), new_bid, False),  # continue loop
            ),
        )

    result = jax.lax.while_loop(
        lambda cond: jnp.logical_not(cond[0]), inside_loop, (False, player, 0, False)
    )  # is_while_completed, current_index_to_check (at the end turn in which place bid), last_bid (at then end bid which should be placed), is_bid_repeated
    _, ix, bid, is_bid_repeated = result

    return jax.lax.cond(
        is_bid_repeated,
        lambda: jnp.array((ix, bid)),
        lambda: jnp.array((-1, -1)),
    )


# def aloop(single_move, state, action_subkey):

#     chex.assert_shape(state, [1])
#     chex.assert_shape(action_subkey, ())

#     return


def find_repeated_state(single_move, repeated, reset_subkey, action_subkey):
    state, _ = jax.vmap(env.reset)(jnp.array([reset_subkey]))
    action_subkeys = jax.random.split(action_subkey, env.max_steps)

    # play until repeated bid
    state = jax.lax.fori_loop(0, repeated[0], lambda i, carry: single_move(carry, action_subkeys[i])[0], state)

    chex.assert_shape(state.observation, (1, 480))

    # play higher repeated bid
    state, _, _, _ = jax.vmap(env.step, in_axes=(0, None))(state, repeated[1])

    # play until end of the game
    state = jax.lax.fori_loop(
        repeated[0] + 1, env.max_steps, lambda i, carry: single_move(carry, action_subkeys[i])[0], state
    )

    # TODO return reward from loop as it is overwritten by zeros
    # create function to sum rewards in loop, at the end carry will hold sum of rewards (the moment reward was returned by env)

    is_done = state.terminated | state.truncated
    chex.assert_shape(is_done, (1,))

    is_done = is_done[0]
    chex.assert_shape(is_done, ())

    return jax.lax.cond(
        is_done,
        lambda: (True, state),
        lambda: (False, state),
    )


def skip_loop(single_move, args):
    rng = args[3]

    rng, reset_subkey = jax.random.split(rng)
    state, _ = jax.vmap(env.reset)(jnp.array([reset_subkey]))

    rng, action_subkey = jax.random.split(rng)
    state, out = jax.lax.scan(single_move, state, jax.random.split(action_subkey, env.max_steps))
    rewards, done = out

    chex.assert_shape(rewards, [env.max_steps, 1, 2])
    chex.assert_shape(done, [env.max_steps, 1])

    reward = rewards[:, :, 0].sum()
    is_done = done.any()

    bid_history = state._env_state._bidding_history[-1]

    target_player = jnp.where(state._env_state._shuffled_players[-1] == 0, size=1)[0][0]  # extract from tuple and array
    chex.assert_shape(target_player, ())

    partner_player = jnp.mod(jnp.add(target_player, 2), 4)

    repeated = get_repeated_bid(
        bid_history, partner_player
    )  # if (-1, -1) then no repeated bid -> we should find another game

    is_found, new_state = jax.lax.cond(
        jnp.all(jnp.equal(repeated, jnp.array((-1, -1)))),
        lambda: (False, state),
        lambda: find_repeated_state(single_move, repeated, reset_subkey, action_subkey),
    )

    not_found_return = (False, state, 0.0, rng, state)

    chex.assert_shape(state.observation, (1, 480))
    chex.assert_shape(new_state.observation, (1, 480))

    return jax.lax.cond(
        is_done,
        lambda: jax.lax.cond(
            is_found,
            lambda: (True, new_state, reward, rng, state),
            lambda: not_found_return,
        ),
        lambda: not_found_return,
    )


def get_game_with_skipped_bid(single_move, rng):
    """
    Returns state of a game where the partner played a bid two times, but reduced two one higher bid
    (first bid is skipped or in other words is reducing knowledge of the actual player)

    and

    result of game for original state where two bids are played.
    """

    loop = partial(skip_loop, single_move)

    state, _ = jax.vmap(env.reset)(jnp.array([rng]))
    result = jax.lax.while_loop(lambda cond: jnp.logical_not(cond[0]), loop, (False, state, 0.0, rng, state))

    new_state, original_reward = result[1], result[2]
    state = result[4]

    jax.debug.print(
        "new_state:\n{}\norig_reward: {}\nreward: {}\nisdone: {}\nint_rewards: {}\nstate: {}\n\n",
        new_state._env_state._bidding_history,
        original_reward,
        new_state.rewards,
        new_state.terminated | new_state.truncated,
        new_state._env_state.rewards,
        state._env_state.rewards,
    )

    chex.assert_shape(new_state.observation, (1, 480))
    chex.assert_shape(original_reward, ())

    # new_state = jnp.squeeze(new_state, 0)

    # chex.assert_shape(new_state.observation, (480,))

    return new_state, original_reward


def evaluate_pvp_winnable(rng: chex.PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: chex.PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0, rng1 = jax.random.split(rng)

        action0 = policy1(rng0, state)
        action1 = policy2(rng1, state)

        action = jnp.where(state.current_player == 0, action0, action1)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)

        return new_state, (new_state.rewards, new_done)

    get_games = partial(get_game_with_skipped_bid, single_move)

    rng, subkey = jax.random.split(rng)
    state, orig_rewards = jax.vmap(get_games)(jax.random.split(subkey, batch_size))
    # first = state
    # state, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    # state = jnp.squeeze(state, 1)

    rewards = state.rewards
    chex.assert_shape(rewards, [batch_size, 1, 2])
    chex.assert_shape(orig_rewards, [batch_size])
    # chex.assert_shape(done, [env.max_steps, batch_size])
    net_rewards = rewards[:, :, 0].sum(axis=1)
    # episode_done = done.any(axis=0)
    return orig_rewards, net_rewards


if __name__ == "__main__":
    for policy_name, policy in [
        ("bzero", make_bzero_policy()),
        # ('random', random_policy),
        # ('mcts128', make_mcts_policy(128)),
        # ('mcts512', make_mcts_policy(512)),
    ]:
        eval_func = jax.jit(
            partial(
                evaluate_pvp_winnable,
                policy1=make_bzero_policy(),
                policy2=policy,
                batch_size=32,
            )
        )

        rng = jax.random.key(0)
        game_history_orig = []
        game_history = []

        for _ in tqdm(range(1)):
            rng, subkey = jax.random.split(rng)
            orig_results, results = eval_func(subkey)
            for orig_result, result in zip(orig_results, results):
                game_history_orig.append(orig_result.astype(jnp.int32).item())
                game_history.append(result.astype(jnp.int32).item())

        print(game_history_orig)
        print(game_history)

        winrate_orig = jnp.mean(jnp.array(game_history_orig) < 0)
        winrate = jnp.mean(jnp.array(game_history) < 0)
        print(f"{policy_name} orig_winrate: {winrate_orig} winrate: {winrate}")
