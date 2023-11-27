import pgx
from pgx import State
from chex import PRNGKey
from type_aliases import Observation, Reward, Done, Action


env = pgx.make("go_9x9")
num_actions = env.num_actions
max_steps = env.max_termination_steps


def reset(rng: PRNGKey) -> tuple[State, Observation]:
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
