import chex
import jax
import jax.numpy as jnp

from functools import partial
from tqdm import tqdm

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from bridge_bidding_2p import State
from dds_common import argmax_reverse
from run_tournament import make_bzero_policy, make_bzero_xl_policy, make_random_policy, make_mcts_policy, make_brl_policy
import bridge_env as env
from type_aliases import Done, Reward
from dds_peaceful_agent import make_dds_policy as make_dds_peaceful_policy
from dds_aggressive_agent import make_dds_policy as make_dds_aggressive_policy


def get_reward_for_bid_first_players(state, bid, player):
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


def get_reward_for_bid_second_players(state, bid, player):
    chex.assert_shape(state._init_rng, [])

    state, _ = env.reset(state._init_rng)

    we_play_first = state.current_player == player
    action = bid + BID_OFFSET_NUM

    state, _ = env.reset(state._init_rng)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
    state, obs, rew, done = env.step(state, PASS_ACTION_NUM)
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
    first_player_0 = state.current_player == 0

    rewards0 = jax.vmap(get_reward_for_bid_first_players, in_axes=[None, 0, None])(state, jnp.arange(35), 0)
    rewards1 = jax.vmap(get_reward_for_bid_first_players, in_axes=[None, 0, None])(state, jnp.arange(35), 1)
    rewards2 = jax.vmap(get_reward_for_bid_second_players, in_axes=[None, 0, None])(state, jnp.arange(35), 0)
    rewards3 = jax.vmap(get_reward_for_bid_second_players, in_axes=[None, 0, None])(state, jnp.arange(35), 1)

    max_bid_0 = argmax_reverse(rewards0)
    max_bid_1 = argmax_reverse(rewards1)
    max_bid_2 = argmax_reverse(rewards2)
    max_bid_3 = argmax_reverse(rewards3)

    chex.assert_shape(max_bid_0, [])

    # second player must have higher bid but also bid ealier
    winnable = jax.lax.cond(
        first_player_0,
        lambda: (max_bid_3 >= max_bid_2) & (max_bid_3 > max_bid_0)
        | (max_bid_1 >= max_bid_2) & (max_bid_1 >= max_bid_0),
        lambda: (max_bid_3 > max_bid_2) & (max_bid_3 > max_bid_0) | (max_bid_1 > max_bid_2) & (max_bid_1 > max_bid_0),
    )

    return (winnable, subkey)


def get_winnable_game(rng):
    "Return state of a game where the second player can win"

    a = jax.lax.while_loop(lambda cond: jnp.logical_not(cond[0]), loop, (False, rng))

    return env.reset(a[1])


def evaluate_pvp_winnable(rng: chex.PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: chex.PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0, rng1 = jax.random.split(rng)

        action0 = policy1(rng0, state)
        action1 = policy2(rng1, state)

        action = jnp.where(state.current_player == 0, action0, action1)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)

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


import glob


if __name__ == "__main__":
    for policy_name, policy in [
        # *[
        #     (f"bzero-{model}", make_bzero_xl_policy(p=model))
        #     for model in glob.glob("model-*.pkl")
        # ]

        # *[
        #     (f"bzero-{model}", make_bzero_policy(p=model))
        #     for model in glob.glob("bzero-checkpoints/model-*.pkl")
        # ]

        # 0.5571823
        ("bzero", make_bzero_policy()),

        # 0.6309662
        ("bzero-xl", make_bzero_xl_policy()),

        # 0.55368805
        ("brl-from-scratch-rl", make_brl_policy("brl/bridge_models/model-from-scratch-rl.pkl")),

        # 0.6663132
        ("brl-pretrained-rl-with-fsp", make_brl_policy("brl/bridge_models/model-pretrained-rl-with-fsp.pkl")),

        # 0.6919403
        ("brl-pretrained-rl-with-pfsp", make_brl_policy("brl/bridge_models/model-pretrained-rl-with-pfsp.pkl")),

        # 0.693161
        ("brl-pretrained-rl", make_brl_policy("brl/bridge_models/model-pretrained-rl.pkl")),

        # 0.6678467
        ("brl-sl", make_brl_policy("brl/bridge_models/model-sl.pkl")),




        # 0.51067924 - argmax
        # 0.5039654 - random
        # ("bzero", make_bzero_policy()),

        # 0.21313858
        # 0.46661186
        # ("new-bzero", make_new_bzero_policy()),

        # 0.69322777
        # 0.68261147
        # ("brl-pretrained-rl-with-pfsp", make_brl_policy("brl/bridge_models/model-pretrained-rl-with-pfsp.pkl")),

        # 0.6670723
        # 0.65826416
        # ("brl-pretrained-rl-with-fsp", make_brl_policy("brl/bridge_models/model-pretrained-rl-with-fsp.pkl")),

        # 0.6944046
        # 0.6820278
        # ("brl-pretrained-rl", make_brl_policy("brl/bridge_models/model-pretrained-rl.pkl")),

        # 0.5551243
        # 0.54642105
        # ("brl-from-scratch-rl", make_brl_policy("brl/bridge_models/model-from-scratch-rl.pkl")),

        # ('random', random_policy),
        # ('mcts128', make_mcts_policy(128)),
        # ('mcts512', make_mcts_policy(2**12)),
    ]:
        eval_func = jax.jit(
            partial(
                evaluate_pvp_winnable,
                policy1=make_dds_aggressive_policy,
                policy2=policy,
                batch_size=4096,
            )
        )

        rng = jax.random.key(0)
        game_history = []

        for _ in tqdm(range(32)):
            rng, subkey = jax.random.split(rng)
            results, dones = eval_func(subkey)
            for result, done in zip(results, dones):
                if done:
                    game_history.append(result.astype(jnp.int32).item())

        winrate = jnp.mean(jnp.array(game_history) < 0)
        print(f"{policy_name} winrate:", winrate)
