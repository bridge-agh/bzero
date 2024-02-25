import pickle
import numpy as np

player_names = [
    "random",
    "mcts-32",
    "mcts-128",
    "dds",
    "bzero",
]
history_file_name = 'game_history/random-mcts32-mcts128-dds-bzero.pkl'


def min_num_games(num_players):
    return int(num_players * (num_players - 1) / 2)


def get_winning_scores(history, num_players):
    scores = np.zeros((num_players, num_players), int)
    for result in history:
        if result[2] == 1:
            scores[result[0], result[1]] += 1
        elif result[2] == -1:
            scores[result[1], result[0]] += 1
    return scores


if __name__ == "__main__":

    num_players = len(player_names)

    history = []
    with open(history_file_name, 'rb') as f:
        history = pickle.load(f)

    scores = get_winning_scores(history, num_players)

    num_all_games = len(history)
    num_games = num_all_games / min_num_games(num_players)

    for i in range(num_players):
        for j in range(num_players):
            if i == j:
                continue
            print(f"{player_names[i]} vs {player_names[j]}: {100 * (scores[i][j] / num_games):.2f}%")



