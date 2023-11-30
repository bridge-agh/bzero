import jax
import jax.numpy as jnp
from pgx import State
import chex
from chex import PRNGKey
import haiku as hk
import optax

from functools import partial
from rich.progress import track
from pydantic import BaseModel
from omegaconf import OmegaConf
import wandb
import pickle

from type_aliases import Reward, Observation, Done
import env_wrapper as env
from az_network import NetworkVariables
import az_agent


class Config(BaseModel):
    seed: int = 0

    self_play_iterations: int = 8
    self_play_batch_size: int = 256

    train_iterations: int = 16
    train_batch_size: int = 8192

    experience_buffer_size: int = 1_000_000

    mcts_simulations: int = 32

    class Config:
        extra = "forbid"


if __name__ == "__main__":
    conf_dict = OmegaConf.from_cli()
    config = Config(**conf_dict)
else:
    config = Config()


@chex.dataclass(frozen=True)
class MoveOutput:
    # observation before the move
    observation: Observation
    # predicted policy
    action_weights: chex.Array
    # reward after the move
    reward: Reward
    # whether the new state is terminal
    done: Done


@chex.dataclass(frozen=True)
class TrainingExample:
    observation: chex.Array
    value_tgt: chex.Array
    policy_tgt: chex.Array
    value_mask: chex.Array


def prepare_training_data(trajectory: MoveOutput) -> TrainingExample:
    chex.assert_shape(trajectory.observation, [env.max_steps, config.self_play_batch_size, *env.observation_shape])
    chex.assert_shape(trajectory.action_weights, [env.max_steps, config.self_play_batch_size, env.num_actions])
    chex.assert_shape(trajectory.reward, [env.max_steps, config.self_play_batch_size])
    chex.assert_shape(trajectory.done, [env.max_steps, config.self_play_batch_size])

    batch_size = trajectory.done.shape[1]

    # every MoveOutput up until the last one with done=True is valid
    value_mask = (jnp.cumsum(trajectory.done[::-1, :], axis=0)[::-1, :] >= 1).astype(jnp.float32)
    chex.assert_shape(value_mask, [env.max_steps, batch_size])

    def body_fn(carry, i):
        # when processing a terminal state, discard the values of future states
        discount = jnp.where(trajectory.done[i], 0.0, -1.0)
        v = trajectory.reward[i] + discount * carry
        return v, v

    # scan from the last state to the first accumulating the value
    _, value_tgt = jax.lax.scan(
        body_fn,
        # value accumulator
        jnp.zeros(batch_size),
        # step index
        env.max_steps - jnp.arange(env.max_steps) - 1,
    )
    # re-reverse
    value_tgt = value_tgt[::-1, :]
    chex.assert_shape(value_tgt, [env.max_steps, batch_size])

    return TrainingExample(
        observation=trajectory.observation,
        policy_tgt=trajectory.action_weights,
        value_tgt=value_tgt,
        value_mask=value_mask,
    )


@partial(jax.jit, static_argnames=('batch_size', 'num_simulations'))
def batched_self_play(variables: NetworkVariables, rng: PRNGKey, batch_size: int, num_simulations: int) -> TrainingExample:
    def batched_single_move(prev: tuple[State, Observation, Done], rng: PRNGKey) -> tuple[tuple[State, Observation, Done], MoveOutput]:
        state, observation, done = prev
        policy = az_agent.batched_compute_policy(variables, rng, state, observation, num_simulations)
        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, policy.action)
        return (new_state, new_observation, new_done), MoveOutput(
            observation=observation,
            action_weights=policy.action_weights,
            reward=new_reward,
            done=new_done,
        )
    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state, observation, jnp.zeros(batch_size, dtype=jnp.bool_)
    _, trajectory = jax.lax.scan(batched_single_move, first, jax.random.split(rng, env.max_steps))
    jax.tree_util.tree_map(lambda x: chex.assert_shape(x, [env.max_steps, batch_size, *x.shape[2:]]), trajectory)
    return prepare_training_data(trajectory)


def collect_self_play_data(
    variables: NetworkVariables,
    rng: PRNGKey,
    iterations: int,
    batch_size: int,
) -> list[TrainingExample]:
    buffer: list[TrainingExample] = []
    for _ in track(range(iterations), description="Self-play"):
        rng, subkey = jax.random.split(rng)
        batch = batched_self_play(variables, subkey, batch_size, config.mcts_simulations)
        batch = jax.device_get(batch)
        examples = [jax.tree_util.tree_map(lambda x: x[step, game], batch) for step in range(env.max_steps) for game in range(batch_size)]
        buffer.extend(examples)
    return buffer


def loss_fn(params: hk.Params, state: hk.Params, rng: PRNGKey, batch: TrainingExample):
    outputs, new_state = az_agent.forward.apply(params, state, rng, batch.observation, is_training=True)

    chex.assert_equal_shape([outputs.pi, batch.policy_tgt])
    chex.assert_equal_shape([outputs.v, batch.value_tgt])

    value_loss = optax.l2_loss(outputs.v, batch.value_tgt)
    value_loss = jnp.mean(value_loss * batch.value_mask)

    target_pr = batch.policy_tgt
    target_pr = jnp.where(target_pr > 0.0, target_pr, 1e-9)
    action_logits = jax.nn.log_softmax(outputs.pi, axis=-1)
    policy_loss = jnp.sum(target_pr * (jnp.log(target_pr) - action_logits), axis=-1)
    policy_loss = jnp.mean(policy_loss)

    return value_loss + policy_loss, new_state


def make_train_step(opt: optax.GradientTransformation):
    @jax.jit
    def train_step(variables: NetworkVariables, rng: PRNGKey, opt_state: optax.OptState, batch: TrainingExample):
        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables.params, variables.state, rng, batch)
        updates, new_opt_state = opt.update(grads, opt_state, variables.params)
        new_params = optax.apply_updates(variables.params, updates)
        new_variables = NetworkVariables(params=new_params, state=new_state)
        return new_variables, new_opt_state, loss
    return train_step


@partial(jax.jit, static_argnames=('batch_size',))
def evaluate_net_v_baseline(variables: NetworkVariables, rng: PRNGKey, batch_size: int):
    def single_move(prev: tuple[State, Observation], rng: PRNGKey) -> tuple[tuple[State, Observation], Reward]:
        state, observation = prev
        rng0a, rng0b, rng1 = jax.random.split(rng, 3)

        action_mask = state.legal_action_mask

        logits0 = az_agent.forward.apply(variables.params, variables.state, rng0a, observation, is_training=False)[0].pi
        logits0_masked = jnp.where(action_mask, logits0, -1e9)
        action0 = jax.random.categorical(rng0b, logits0_masked)

        logits1 = jnp.zeros((batch_size, env.num_actions))
        logits1_masked = jnp.where(action_mask, logits1, -1e9)
        action1 = jax.random.categorical(rng1, logits1_masked)

        action = jnp.where(state.current_player == 0, action0, action1)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)
        return (new_state, new_observation), (new_state.rewards, new_done)

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state, observation
    _, out = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    rewards, done = out
    chex.assert_shape(rewards, [env.max_steps, batch_size, 2])
    chex.assert_shape(done, [env.max_steps, batch_size])
    net_rewards = rewards[:, :, 0].sum(axis=0)
    episode_done = done.any(axis=0)
    num_episodes = episode_done.sum()
    wins = (net_rewards > 0 & episode_done).sum() / num_episodes
    draws = (net_rewards == 0 & episode_done).sum() / num_episodes
    losses = (net_rewards < 0 & episode_done).sum() / num_episodes
    return wins, draws, losses


def run():
    wandb.init(project="bzero", config=config.model_dump())

    rng = jax.random.PRNGKey(config.seed)

    rng, subkey = jax.random.split(rng)
    variables = az_agent.initial_variables(subkey)

    optimizer = optax.chain(
        optax.add_decayed_weights(1e-4),
        optax.adam(1e-3),
    )
    opt_state = optimizer.init(variables.params)

    train_step = make_train_step(optimizer)

    experience_buffer = []

    log = {
        'iteration': 0,
        'self_play/frames': 0,
        'train/frames': 0,
        'train/iteration': 0,
    }

    try:
        while True:
            rng, subkey = jax.random.split(rng)
            examples = collect_self_play_data(variables, subkey, config.self_play_iterations, config.self_play_batch_size)
            print(f'Collected {len(examples)} examples')
            log['self_play/frames'] += len(examples)

            experience_buffer.extend(examples)
            experience_buffer = experience_buffer[-config.experience_buffer_size:]
            log.update({'experience_buffer_size': len(experience_buffer)})

            for _ in track(range(config.train_iterations), description="Training"):
                rng, subkey = jax.random.split(rng)
                idx = jax.random.choice(subkey, len(experience_buffer), [config.train_batch_size], replace=False)
                examples = [experience_buffer[i] for i in idx]
                batch = jax.tree_util.tree_map(lambda *x: jnp.array(x), *examples)

                rng, subkey = jax.random.split(rng)
                variables, opt_state, loss = train_step(variables, subkey, opt_state, batch)

                log['train/loss'] = loss
                log['train/frames'] += len(examples)
                log['train/iteration'] += 1

                wandb.log(log)

            print('Evaluating...')
            rng, subkey = jax.random.split(rng)
            wins, draws, losses = evaluate_net_v_baseline(variables, subkey, config.self_play_batch_size)
            print(f'Wins: {wins:.2f}, Draws: {draws:.2f}, Losses: {losses:.2f}')
            log['eval/wins'] = wins
            log['eval/draws'] = draws
            log['eval/losses'] = losses

            log['iteration'] += 1

            wandb.log(log)

    except KeyboardInterrupt:
        pass

    with open('model.pkl', 'wb') as f:
        pickle.dump(variables, f)


if __name__ == '__main__':
    run()
