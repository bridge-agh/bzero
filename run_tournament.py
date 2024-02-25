import jax
import jax.numpy as jnp

import numpy as np
import wandb

from functools import partial
from pprint import pprint
import pickle

import chex
from chex import PRNGKey

from pgx import State
from pgx.bridge_bidding import BID_OFFSET_NUM, PASS_ACTION_NUM

from type_aliases import Observation, Reward, Done, Action
import bridge_env as env
import mcts_agent
import az_agent
from dds_agent import dds_policy


def evaluate_pvp(rng: PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0a, rng0b, rng1a, rng1b = jax.random.split(rng, 4)

        action_mask = state.legal_action_mask

        logits0 = policy1(rng0a, state)
        logits0_masked = jnp.where(action_mask, logits0, -1e9)
        action0 = jax.random.categorical(rng0b, logits0_masked)

        logits1 = policy2(rng1a, state)
        logits1_masked = jnp.where(action_mask, logits1, -1e9)
        action1 = jax.random.categorical(rng1b, logits1_masked)

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


def random_policy(rng: PRNGKey, state: State) -> chex.Array:
    return jnp.zeros(env.num_actions)


def make_mcts_policy(num_simulations: int):
    def mcts_policy(rng: PRNGKey, state: State) -> chex.Array:
        out = mcts_agent.batched_compute_policy(rng, state, num_simulations)
        return out.action_weights
    return mcts_policy


def make_bzero_policy():
    with open('models/bridge_v1.pkl', 'rb') as f:
        variables = pickle.load(f)
    def bzero_policy(rng: PRNGKey, state: State) -> chex.Array:
        outputs, _ = az_agent.forward.apply(variables.params, variables.state, rng, state.observation, is_training=False)
        return outputs.pi
    return bzero_policy


def expected(A, B):
    """
    Calculate expected score of A in a match against B

    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    return 1 / (1 + 10 ** ((B - A) / 400))


def elo(old, exp, score, k=32):
    """
    Calculate the new Elo rating for a player

    :param old: The previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match
    :param k: The k-factor for Elo (default: 32)
    """
    return old + k * (score - exp)


def elo_update(elo_a, elo_b, result):
    """
    Update the Elo ratings based on the results of a series of games

    :param elo_a: The previous Elo rating for player A
    :param elo_b: The previous Elo rating for player B
    :param result: (-1, 0, 1) for (loss, draw, win) for the game
    """
    actual = result / 2 + 0.5
    exp = expected(elo_a, elo_b)
    new_a = elo(elo_a, exp, actual)
    new_b = elo(elo_b, 1 - exp, 1 - actual)
    return new_a, new_b


def compute_elo_ratings(player_names, history):
    num_players = len(player_names)
    avg_elo_ratings = np.zeros(num_players)
    for _ in range(100):
        elo_ratings = np.ones(num_players) * 1000
        for p0, p1, result in np.random.permutation(history):
            elo_ratings[p0], elo_ratings[p1] = elo_update(elo_ratings[p0], elo_ratings[p1], result)
        avg_elo_ratings += elo_ratings
    avg_elo_ratings /= 100
    return {name: rating for name, rating in zip(player_names, avg_elo_ratings)}


def main():
    player_names = [
        "random",
        "mcts-32",
        "mcts-128",
        "dds",
        "bzero",
    ]

    num_players = len(player_names)

    policies = [
        random_policy,
        make_mcts_policy(32),
        make_mcts_policy(128),
        dds_policy,
        make_bzero_policy(),
    ]

    eval_funcs = [
        [
            jax.jit(partial(evaluate_pvp, policy1=policies[p0], policy2=policies[p1], batch_size=64))
            for p1 in range(num_players)
        ]
        for p0 in range(num_players)
    ]

    game_history = []

    rng = jax.random.key(0)

    wandb.init(project="bridge-elo")

    try:
        while True:
            for p0 in range(num_players - 1):
                for p1 in range(p0 + 1, num_players):
                    rng, subkey = jax.random.split(rng)
                    results, dones = eval_funcs[p0][p1](subkey)

                    for result, done in zip(results, dones):
                        if done:
                            game_history.append([p0, p1, result.astype(jnp.int32).item()])

                    elo_ratings = compute_elo_ratings(player_names, game_history)

                    logs = elo_ratings
                    logs["num_games"] = len(game_history)

                    pprint(logs)
                    wandb.log(logs)

            with open(f"game_history-{len(game_history)}.pkl", "wb") as f:
                pickle.dump(game_history, f)
    except KeyboardInterrupt:
        pass
    finally:
        wandb.finish()
        with open(f"game_history-{len(game_history)}.pkl", "wb") as f:
            pickle.dump(game_history, f)


if __name__ == "__main__":
    main()
