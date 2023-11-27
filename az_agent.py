import jax
import jax.numpy as jnp
from pgx import State
from chex import PRNGKey
import haiku as hk
import mctx

from functools import partial

from type_aliases import Action, Observation
from az_network import AlphaZeroNetwork, NetworkVariables, NetworkOutputs, DiscreteActionHead
import env_wrapper as env


INITIALIZE_ENVS = 512


def act_randomly(rng: PRNGKey, state: State) -> Action:
    logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
    return jax.random.categorical(rng, logits)


@hk.transform_with_state
def forward(observation: Observation, is_training: bool) -> NetworkOutputs:
    net = AlphaZeroNetwork(action_head=DiscreteActionHead(num_actions=env.num_actions))
    return net(observation.astype(jnp.float32), is_training=is_training)


def initial_variables(rng: PRNGKey) -> NetworkVariables:
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)

    def single_move(state: State, rng: PRNGKey) -> tuple[State, Observation]:
        action = act_randomly(rng, state)
        new_state, observation, _, _ = step(state, action)
        return new_state, observation

    rng, subkey = jax.random.split(rng)
    state, initial_observation = reset(jax.random.split(subkey, INITIALIZE_ENVS))

    rng, subkey = jax.random.split(rng)
    _, batch = jax.lax.scan(single_move, state, jax.random.split(subkey, env.max_steps))

    batch = batch.reshape(-1, *batch.shape[2:])
    batch = jnp.concatenate([initial_observation, batch], axis=0)

    params, state = forward.init(rng, batch, is_training=True)
    return NetworkVariables(params=params, state=state)


def batched_root_fn(variables: NetworkVariables, rng: PRNGKey, state: State, observation: Observation) -> mctx.RootFnOutput:
    outputs, _ = forward.apply(variables.params, variables.state, rng, observation, is_training=False)
    return mctx.RootFnOutput(
        prior_logits=outputs.pi,
        value=outputs.v,
        embedding=state,
    )


def batched_recurrent_fn(variables: NetworkVariables, rng: PRNGKey, action: Action, state: State):
    new_state, observation, reward, done = jax.vmap(env.step)(state, action)
    outputs, _ = forward.apply(variables.params, variables.state, rng, observation, is_training=False)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0).astype(jnp.float32),
        prior_logits=outputs.pi,
        value=jnp.where(done, 0.0, outputs.v).astype(jnp.float32),
    )
    return recurrent_fn_output, new_state


def batched_compute_policy(variables: NetworkVariables, rng: PRNGKey, state: State, observation: Observation, num_simulations: int) -> mctx.PolicyOutput:
    policy_rng, root_rng = jax.random.split(rng, 2)
    policy_output = mctx.muzero_policy(
        params=variables,
        rng_key=policy_rng,
        root=batched_root_fn(variables, root_rng, state, observation),
        recurrent_fn=batched_recurrent_fn,
        num_simulations=num_simulations,
        max_depth=env.max_steps,
        qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        invalid_actions=1.0 - state.legal_action_mask.astype(jnp.float32),
    )
    return policy_output
