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


def elo(old, exp, score, k):
    """
    Calculate the new Elo rating for a player

    :param old: The previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match
    :param k: The k-factor for Elo (default: 32)
    """
    return old + k * (score - exp)


def elo_update(elo_a, elo_b, result, k):
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


if __name__ == "__main__":
    with open("game_history/random-mcts32-mcts128-dds-bzero-v2.pkl", "rb") as f:
        game_history = pickle.load(f)

    K = 4

    num_players = 6
    player_names = [
        "random",
        "mcts-32",
        "mcts-128",
        "dds-aggressive",
        "dds-peaceful",
        "bzero",
    ]

    rng = np.random.default_rng(0)
    perm_elo_ratings = []
    nperm = 200
    for _ in tqdm(range(nperm)):
        elo_ratings = np.ones(num_players) * 1000
        for p0, p1, result in rng.permutation(game_history):
            elo_ratings[p0], elo_ratings[p1] = elo_update(
                elo_ratings[p0], elo_ratings[p1], result, K
            )
        perm_elo_ratings.append(elo_ratings)
    perm_elo_ratings = np.array(perm_elo_ratings)

    df = pd.DataFrame(
        {
            "player": player_names,
            "mean_elo": np.mean(perm_elo_ratings, axis=0),
            "std_elo": np.std(perm_elo_ratings, axis=0),
        }
    )
    df.to_csv("research_out/calc_elo.csv", index=False)

    colors = {
        "random": "C0",
        "mcts-32": "C1",
        "mcts-128": "C2",
        "dds-aggressive": "C3",
        "dds-peaceful": "C4",
        "bzero": "C5",
    }

    fig, ax = plt.subplots()
    bplot = ax.boxplot(
        perm_elo_ratings,
        patch_artist=True,
        labels=player_names,
    )
    for patch, color in zip(bplot["boxes"], [colors[p] for p in player_names]):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)

    plt.savefig("research_out/calc_elo_boxplot.png")

    plt.figure()

    rng = np.random.default_rng(0)
    for _ in range(3):
        elo_rating = [np.ones(num_players) * 1000]
        for p0, p1, result in rng.permutation(game_history):
            e = elo_rating[-1].copy()
            e[p0], e[p1] = elo_update(e[p0], e[p1], result, K)
            elo_rating.append(e)
        elo_rating = np.array(elo_rating)
        for i in range(num_players):
            plt.plot(elo_rating[:, i], color=colors[player_names[i]], alpha=0.5)
    plt.legend(player_names)
    plt.savefig("research_out/calc_elo_history.png")
