import jax.numpy as jnp
import chex
import pgx
from pgx.bridge_bidding import BridgeBidding


@chex.dataclass(frozen=True)
class State:
    current_player: chex.Array
    observation: chex.Array
    rewards: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    legal_action_mask: chex.Array
    
    _env_state: pgx.State
    _init_rng: chex.PRNGKey

    def _repr_html_(self) -> str:
        return self._env_state._repr_html_()


def id_to_pair(id):
    chex.assert_shape(id, [])
    return id // 2


def make_pair_rewards(rewards):
    chex.assert_shape(rewards, [4])
    p0 = rewards[0] + rewards[1]
    p1 = rewards[2] + rewards[3]
    r0 = jnp.where(
        p0 > p1,
        1.0,
        jnp.where(
            p0 < p1,
            -1.0,
            0.0,
        ),
    )
    r1 = -r0
    return jnp.array([r0, r1])


class BridgeBidding2P:
    def __init__(self):
        self.env = BridgeBidding()

    def init(self, key):
        env_state = self.env.init(key).replace(
            _vul_NS=jnp.array(False),
            _vul_EW=jnp.array(False),
        )
        return State(
            current_player=id_to_pair(env_state.current_player),
            observation=env_state.observation,
            rewards=make_pair_rewards(env_state.rewards),
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            legal_action_mask=env_state.legal_action_mask,
            _env_state=env_state,
            _init_rng=key,
        )

    def step(self, state, action):
        env_state = self.env.step(state._env_state, action)
        return State(
            current_player=id_to_pair(env_state.current_player),
            observation=env_state.observation,
            rewards=make_pair_rewards(env_state.rewards),
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            legal_action_mask=env_state.legal_action_mask,
            _env_state=env_state,
            _init_rng=state._init_rng,
        )
