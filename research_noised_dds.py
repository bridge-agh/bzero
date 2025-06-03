import chex
import jax
import jax.numpy as jnp

from functools import partial
from tqdm import tqdm

from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

import bridge_env as env
from bridge_bidding_2p import State
from type_aliases import Done, Reward

from run_tournament import make_bzero_policy, make_random_policy, make_mcts_policy
from dds_peaceful_agent import make_dds_policy as make_dds_peaceful_policy
from dds_noised_agent import make_dds_policy as make_dds_noised_policy

from copy import deepcopy


def evaluate_pvp_winnable(rng: chex.PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: chex.PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0, rng1 = jax.random.split(rng)

        action0 = policy1(rng0, state)
        action1 = policy2(rng1, state)

        action = jnp.where(state.current_player == 0, action0, action1)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)

        return new_state, (new_state.rewards, new_done)

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state
    _, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    rewards, done = out
    chex.assert_shape(rewards, [env.max_steps, batch_size, 2])
    chex.assert_shape(done, [env.max_steps, batch_size])
    net_rewards = rewards[:, :, 0].sum(axis=0)
    episode_done = done.any(axis=0)
    return net_rewards, episode_done


if __name__ == "__main__":
    for policy_name, policy in [
        ("dds", make_dds_noised_policy),
        # ('random', random_policy),
        # ('mcts128', make_mcts_policy(128)),
        # ('mcts512', make_mcts_policy(512)),
    ]:
        eval_func = jax.jit(
            partial(
                evaluate_pvp_winnable,
                policy1=make_dds_peaceful_policy,
                policy2=policy,
                batch_size=64,
            )
        )

        rng = jax.random.key(0)
        game_history = []

        for _ in tqdm(range(64)):
            rng, subkey = jax.random.split(rng)
            results, dones = eval_func(subkey)
            # print(results)
            for result, done in zip(results, dones):
                if done:
                    game_history.append(result.astype(jnp.int32).item())

        winrate = jnp.mean(jnp.array(game_history) < 0)
        print(f"{policy_name} winrate:", winrate)
