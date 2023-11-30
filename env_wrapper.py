import chex
import pgx
from bridge_bidding_2p import BridgeBidding2P, State
from type_aliases import Observation, Reward, Done, Action


# env = BridgeBidding2P()
# observation_shape = (480,)
# num_actions = 38
# max_steps = 32

env = pgx.make("othello")
observation_shape = (8, 8, 2)
num_actions = 65
max_steps = 64


def reset(rng: chex.PRNGKey) -> tuple[State, Observation]:
    state = env.init(rng)
    return state, state.observation


def step(state: State, action: Action) -> tuple[State, Observation, Reward, Done]:
    new_state = env.step(state, action)
    observation = new_state.observation
    reward = new_state.rewards[state.current_player]
    terminated = new_state.terminated
    truncated = new_state.truncated
    done = terminated | truncated
    return new_state, observation, reward, done
