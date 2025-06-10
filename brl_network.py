import jax
import chex
import haiku as hk


@chex.dataclass(frozen=True)
class NetworkVariables:
    params: hk.Params
    state: hk.State


@chex.dataclass(frozen=True)
class NetworkOutputs:
    pi: chex.Array
    v: chex.Array


class BRLNetwork(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x, is_training):
        for dim in [1024, 1024, 1024, 1024]:
            x = hk.Linear(dim)(x)
            x = jax.nn.relu(x)

        pi = hk.Linear(38)(x)
        v = hk.Linear(1)(x).squeeze(-1)

        return NetworkOutputs(pi=pi, v=v)
