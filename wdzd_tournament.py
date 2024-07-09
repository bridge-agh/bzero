import jax
import jax.numpy as jnp

import numpy as np
import wandb

from functools import partial
from pprint import pprint
import pickle

import chex
from chex import PRNGKey

import pgx
from pgx import State

from type_aliases import Observation, Reward, Done, Action
import bridge_env as env
import az_agent


@chex.dataclass(frozen=True)
class StateRecord:
    state: State
    policy1: chex.Array
    policy2: chex.Array
    action: Action


def evaluate_pvp(rng: PRNGKey, policy1, policy2, batch_size: int):
    def single_move(state: State, rng: PRNGKey) -> tuple[State, tuple[Reward, Done]]:
        rng0, rng1 = jax.random.split(rng)

        action0, logits0 = policy1(rng0, state)
        action1, logits1 = policy2(rng1, state)
        action = jnp.where(state.current_player == 0, action0, action1)

        record = StateRecord(
            state=state,
            policy1=jax.nn.softmax(logits0),
            policy2=jax.nn.softmax(logits1),
            action=action,
        )

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)
        return new_state, (new_state.rewards, new_done, record)

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state
    _, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    rewards, done, records = out
    chex.assert_shape(rewards, [env.max_steps, batch_size, 2])
    chex.assert_shape(done, [env.max_steps, batch_size])
    chex.assert_shape(records.state.observation, [env.max_steps, batch_size, 8, 8, 2])
    net_rewards = rewards[:, :, 0].sum(axis=0)
    episode_done = done.any(axis=0)
    return net_rewards, episode_done, records


def make_pgx_policy(deterministic):
    model = pgx.make_baseline_model('othello_v0')
    def pgx_policy(rng: PRNGKey, state: State) -> chex.Array:
        logits, _ = model(state.observation)
        action_mask = state.legal_action_mask
        logits_masked = jnp.where(action_mask, logits, -1e9)
        if deterministic:
            return logits_masked.argmax(axis=-1), logits_masked
        else:
            return jax.random.categorical(rng, logits_masked), logits_masked
    return pgx_policy


def make_bzero_policy(deterministic, p='models/othello_v1.pkl'):
    with open(p, 'rb') as f:
        variables = pickle.load(f)
    def bzero_policy(rng: PRNGKey, state: State) -> chex.Array:
        rng, subkey = jax.random.split(rng)
        outputs, _ = az_agent.forward.apply(variables.params, variables.state, subkey, state.observation, is_training=False)
        logits = outputs.pi
        action_mask = state.legal_action_mask
        logits_masked = jnp.where(action_mask, logits, -1e9)
        if deterministic:
            return logits_masked.argmax(axis=-1), logits_masked
        else:
            return jax.random.categorical(rng, logits_masked), logits_masked
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
        "bzero",
        "pgx",
    ]

    policies = [
        make_bzero_policy(deterministic=False),
        make_pgx_policy(deterministic=False),
    ]

    eval_func = jax.jit(partial(evaluate_pvp, policy1=policies[0], policy2=policies[1], batch_size=1024))

    rng = jax.random.key(0)

    wandb.init(project="wdzd")

    num_games = 0

    try:
        while True:
            rng, subkey = jax.random.split(rng)
            results, dones, records = eval_func(subkey)

            game_results = []
            for result, done in zip(results, dones):
                if done:
                    game_results.append(result.astype(jnp.int32).item())

            num_games += len(game_results)

            with open(f"wdzd_results/game_results-{num_games}.pkl", "wb") as f:
                pickle.dump(game_results, f)

            with open(f"wdzd_results/records-{num_games}.pkl", "wb") as f:
                pickle.dump(jax.device_get(records), f)
    except KeyboardInterrupt:
        pass
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
