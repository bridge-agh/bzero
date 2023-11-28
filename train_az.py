import jax
import jax.numpy as jnp
import pgx
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

    self_play_iterations: int = 1
    self_play_batch_size: int = 256

    train_iterations: int = 2
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
class Trajectory:
    invalid: chex.Array
    observation: Observation
    action_weights: chex.Array
    reward: Reward


@partial(jax.jit, static_argnames=('batch_size', 'num_simulations'))
def batched_self_play(variables: NetworkVariables, rng: PRNGKey, batch_size: int, num_simulations: int) -> Trajectory:
    def single_move(prev: tuple[State, Observation, Done], rng: PRNGKey) -> tuple[tuple[State, Observation, Done], Trajectory]:
        state, observation, done = prev
        policy = az_agent.batched_compute_policy(variables, rng, state, observation, num_simulations)
        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, policy.action)
        return (new_state, new_observation, new_done), Trajectory(
            invalid=done,
            observation=observation,
            action_weights=policy.action_weights,
            reward=new_reward,
        )
    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state, observation, jnp.zeros(batch_size, dtype=jnp.bool_)
    _, trajectory = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    trajectory = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajectory)
    jax.tree_util.tree_map(lambda x: chex.assert_shape(x, [batch_size, env.max_steps, *x.shape[2:]]), trajectory)
    return trajectory


@chex.dataclass(frozen=True)
class TrainingExample:
    observation: chex.Array
    value: chex.Array
    action_weights: chex.Array


def prepare_training_data(trajectory: Trajectory) -> list[TrainingExample]:
    trajectory = jax.device_get(trajectory)
    buffer: list[TrainingExample] = []
    num_games = trajectory.invalid.shape[0]
    for game in range(num_games):
        observation = trajectory.observation[game]
        invalid = trajectory.invalid[game]
        action_weights = trajectory.action_weights[game]
        reward = trajectory.reward[game]
        num_steps = invalid.shape[0]
        value: chex.Array | None = None
        for step in reversed(range(num_steps)):
            if invalid[step]:
                continue
            if value is None:
                value = reward[step]
            else:
                value = -value
            buffer.append(TrainingExample(
                observation=observation[step],
                value=value,
                action_weights=action_weights[step],
            ))
    return buffer


def collect_self_play_data(
    variables: NetworkVariables,
    rng: PRNGKey,
    iterations: int,
    batch_size: int,
) -> list[TrainingExample]:
    buffer: list[TrainingExample] = []
    for _ in track(range(iterations), description="Self-play"):
        rng, subkey = jax.random.split(rng)
        trajectory = batched_self_play(variables, subkey, batch_size, config.mcts_simulations)
        buffer.extend(prepare_training_data(jax.device_get(trajectory)))
    return buffer


def loss_fn(params: hk.Params, state: hk.Params, rng: PRNGKey, batch: TrainingExample):
    outputs, new_state = az_agent.forward.apply(params, state, rng, batch.observation, is_training=True)

    chex.assert_equal_shape([outputs.pi, batch.action_weights])
    chex.assert_equal_shape([outputs.v, batch.value])

    value_loss = optax.l2_loss(outputs.v, batch.value)
    value_loss = jnp.mean(value_loss)

    target_pr = batch.action_weights
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
    baseline = pgx.make_baseline_model('othello_v0')
    def single_move(prev: tuple[State, Observation], rng: PRNGKey) -> tuple[tuple[State, Observation], Reward]:
        state, observation = prev

        policy1 = az_agent.batched_compute_policy(variables, rng, state, observation, config.mcts_simulations)
        action1 = policy1.action_weights.argmax(axis=-1)

        logits2, _ = baseline(observation)
        action2 = logits2.argmax(axis=-1)

        action = jnp.where(state.current_player == 0, action1, action2)

        new_state, new_observation, new_reward, new_done = jax.vmap(env.step)(state, action)
        return (new_state, new_observation), new_state.rewards

    rng, subkey = jax.random.split(rng)
    state, observation = jax.vmap(env.reset)(jax.random.split(subkey, batch_size))
    first = state, observation
    _, rewards = jax.lax.scan(single_move, first, jax.random.split(rng, env.max_steps))
    chex.assert_shape(rewards, [env.max_steps, batch_size, 2])
    net_rewards = rewards[:, :, 0].sum(axis=0)
    wins = (net_rewards > 0).sum() / batch_size
    draws = (net_rewards == 0).sum() / batch_size
    losses = (net_rewards < 0).sum() / batch_size
    return wins, draws, losses


def run():
    wandb.init(project="othello-zero", config=config.model_dump())

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
