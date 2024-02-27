import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


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


def elo_update(elo_a, elo_b, result, k=32):
    """
    Update the Elo ratings based on the results of a series of games

    :param elo_a: The previous Elo rating for player A
    :param elo_b: The previous Elo rating for player B
    :param result: (-1, 0, 1) for (loss, draw, win) for the game
    """
    actual = result / 2 + 0.5
    exp = expected(elo_a, elo_b)
    new_a = elo(elo_a, exp, actual, k)
    new_b = elo(elo_b, 1 - exp, 1 - actual, k)
    return new_a, new_b


if __name__ == '__main__':
    with open('game_history/bzero-checkpoints.pkl', 'rb') as f:
        game_history = pickle.load(f)

    num_players = 11
    player_names = [
        "bzero-100",
        "bzero-200",
        "bzero-300",
        "bzero-400",
        "bzero-500",
        "bzero-600",
        "bzero-700",
        "bzero-800",
        "bzero-900",
        "bzero-1000",
        "bzero-1100",
    ]

    rng = np.random.default_rng(0)
    perm_elo_ratings = []
    nperm = 200
    for _ in tqdm(range(nperm)):
        elo_ratings = np.ones(num_players) * 1000
        for p0, p1, result in rng.permutation(game_history):
            elo_ratings[p0], elo_ratings[p1] = elo_update(elo_ratings[p0], elo_ratings[p1], result, 1)
        perm_elo_ratings.append(elo_ratings)
    perm_elo_ratings = np.array(perm_elo_ratings)

    df = pd.DataFrame({
        "player": player_names,
        "mean_elo": np.mean(perm_elo_ratings, axis=0),
        "std_elo": np.std(perm_elo_ratings, axis=0),
    })
    df.to_csv('research_out/bzero_elo.csv', index=False)

    plt.figure()
    plt.boxplot(perm_elo_ratings)
    plt.xticks(range(1, num_players + 1), player_names, rotation=45)
    plt.savefig('research_out/bzero_elo_boxplot.png')

    plt.figure()
    colors = {
        player: f'C{i}' for i, player in enumerate(player_names)
    }
    rng = np.random.default_rng(0)
    for _ in range(3):
        elo_rating = [np.ones(num_players) * 1000]
        for p0, p1, result in rng.permutation(game_history):
            e = elo_rating[-1].copy()
            e[p0], e[p1] = elo_update(e[p0], e[p1], result, 1)
            elo_rating.append(e)
        elo_rating = np.array(elo_rating)
        for i in range(num_players):
            plt.plot(elo_rating[:, i], color=colors[player_names[i]], alpha=0.5)
    plt.legend(player_names)
    plt.savefig('research_out/bzero_elo_history.png')
