import chex
import jax
import jax.numpy as jnp

from functools import partial
from tqdm import tqdm

from pgx import State
from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from run_tournament import dds_policy, make_bzero_policy, random_policy, make_mcts_policy
import bridge_env as env
from type_aliases import Done, Reward
from dds_agent import argmax_reverse


def get_player_reward_for_bid(state, bid, player):
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


def loop(args):
    _, subkey = jax.random.split(args[1])
    state, _ = env.reset(subkey)

    max_bid_0 = argmax_reverse(jax.vmap(get_player_reward_for_bid, in_axes=[None, 0, None])(state, jnp.arange(35), 0))
    max_bid_1 = argmax_reverse(jax.vmap(get_player_reward_for_bid, in_axes=[None, 0, None])(state, jnp.arange(35), 1))

    chex.assert_shape(max_bid_0, [])
    winnable = jnp.less(max_bid_0, max_bid_1)

    return (winnable, subkey)


def is_not_winnable_game(args):
    return jnp.logical_not(args[0])


def get_winnable_game(rng):
    a = jax.lax.while_loop(is_not_winnable_game, loop, (False, rng))

    return env.reset(a[1])


def evaluate_pvp_winnable(rng: chex.PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: chex.PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0, rng1 = jax.random.split(rng)

        action0 = policy1(rng0, state)
        action1 = policy2(rng1, state)
        action = jnp.where(state.current_player == 0, action0, action1)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(
            state, action
        )
        return new_state, (new_state.rewards, new_done)

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(get_winnable_game)(jax.random.split(subkey, batch_size))
    first = state
    _, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    rewards, done = out
    chex.assert_shape(rewards, [env.max_steps, batch_size, 2])
    chex.assert_shape(done, [env.max_steps, batch_size])
    net_rewards = rewards[:, :, 0].sum(axis=0)
    episode_done = done.any(axis=0)
    return net_rewards, episode_done


if __name__ == '__main__':
    eval_func = jax.jit(
        partial(
            evaluate_pvp_winnable,
            policy1=dds_policy,
            policy2=make_bzero_policy(),
            # policy2=random_policy,
            # policy2=make_mcts_policy(32),
            # policy2=make_mcts_policy(2048),
            batch_size=64,
        )
    )

    rng = jax.random.key(0)
    game_history = []

    for _ in tqdm(range(5)):
        rng, subkey = jax.random.split(rng)
        results, dones = eval_func(subkey)
        for result, done in zip(results, dones):
            if done:
                game_history.append(result.astype(jnp.int32).item())

        bzero_winrate = jnp.mean(jnp.array(game_history) < 0)
        print('winrate:', bzero_winrate)
