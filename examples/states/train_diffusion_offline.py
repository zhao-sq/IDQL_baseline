import gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from flax.training import checkpoints
from jaxrl5.agents import BCLearner, IQLLearner, DDPMIQLLearner #, DDPMIQLBase
from jaxrl5.data.d4rl_datasets import D4RLDataset
from jaxrl5.data.custom_dataset import CustomDataset
from jaxrl5.evaluation import evaluate, implicit_evaluate
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
import gym
# from gymnasium import spaces


@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis=0)

    return frozen_dict.freeze(merge)


def call_main(details):
    wandb.init(project=details['project'], name=details['group'])
    wandb.config.update(details)
    print(wandb.run.settings.mode)

    task_name = details.get('task', None)
    if not task_name:
        env = gym.make(details['env_name'])
        env = wrap_gym(env)
        if details['save_video']:
            env = WANDBVideo(env)

        if "binary" in details['env_name']:
            ds = BinaryDataset(env)
        else:
            ds = D4RLDataset(env)
    else:
        ds = CustomDataset(details['task'])

    if details['take_top'] is not None or details['filter_threshold'] is not None:
        ds.filter(take_top=details['take_top'],
                  threshold=details['filter_threshold'])

    config_dict = details['rl_config']

    model_cls = config_dict.pop("model_cls")

    if "BC" in model_cls:
        agent = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    else:
        if not task_name:
            agent = globals()[model_cls].create(
                details['seed'], env.observation_space, env.action_space, **config_dict
            )
            keys = None

            if "antmaze" in details['env_name']:
                ds.dataset_dict["rewards"] -= 1.0
            elif (
                "halfcheetah" in details['env_name']
                or "walker2d" in details['env_name']
                or "hopper" in details['env_name']
            ) and details['normalize_returns']:
                ds.normalize_returns()
        else:
            uses_q_former_obs = any(
                config_dict.get(key) == 'Q_former'
                for key in ('actor_architecture', 'critic_architecture', 'value_architecture')
            )
            observation_shape = ds.get_obs_shape() if uses_q_former_obs else (ds.get_obs_dim(),)
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=observation_shape,
                dtype=np.float32
            )
            action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(ds.get_action_dim(),),
                dtype=np.float32
            )
            agent = globals()[model_cls].create(
                details['seed'], observation_space, action_space, **config_dict
            )
            keys = None
            if details['normalize_returns']:
                ds.normalize_returns()

    ds, ds_val = ds.split(0.95)
    # sample = ds.sample_jax(details['batch_size'], keys=keys)
    sample = ds.sample_jax_batch(details['batch_size'], keys=keys)
    cur_loss = 1e9

    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        # sample = ds.sample_jax(details['batch_size'], keys=keys)
        sample = ds.sample_jax_batch(details['batch_size'], keys=keys)
        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            val_sample = ds_val.sample(details['batch_size'], keys=keys)
            _, val_info = agent.update(val_sample)
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
            wandb.log({f"val/{k}": v for k, v in val_info.items()}, step=i)
            # TODO: Save BC Actor weights for lowest validation loss
            if info['actor_loss']<cur_loss:
                checkpoints.save_checkpoint(
                    ckpt_dir=details["ckpt_dir"],
                    target=agent,
                    step=i,
                    prefix="best_",
                    overwrite=True,
                )
                cur_loss = info['actor_loss']
        
        if i % details['eval_interval'] == 0 and i > 0:
            checkpoints.save_checkpoint(
                    ckpt_dir=details["ckpt_dir"],
                    target=agent,
                    step=i,
                    overwrite=False,
                    keep=5
                )

    # if details['online_max_steps'] > 0:
    #     online_replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
    #                                 details['online_max_steps'])

    #     online_replay_buffer.seed(details['seed'] + 1241)

    #     observation, done = env.reset(), False
    #     for i in tqdm(range(1, details['online_max_steps']),
    #                     smoothing=0.1):
    #         action, agent = agent.eval_actions(observation)
    #         next_observation, reward, done, info = env.step(action)

    #         if not done or 'TimeLimit.truncated' in info:
    #             mask = 1.0
    #         else:
    #             mask = 0.0

    #         if "antmaze" in details['env_name']:
    #             reward -= 1.0

    #         online_replay_buffer.insert(
    #             dict(observations=observation,
    #                 actions=action,
    #                 rewards=reward,
    #                 masks=mask,
    #                 dones=done,
    #                 next_observations=next_observation))
    #         observation = next_observation

    #         if done:
    #             observation, done = env.reset(), False

    #         if i > details['online_start_training']:
    #             online_batch = online_replay_buffer.sample(128)
    #             offline_batch = ds.sample(128)
    #             batch = merge_batch(online_batch, offline_batch)
    #             agent, info = agent.critic_update(batch)

    #             if i % details['log_interval'] == 0:
    #                 info = jax.device_get(info)
    #                 wandb.log({f'online_train/{k}': v for k, v in info.items()}, step= i + details['max_steps'])

    #             if i % details['online_eval_interval'] == 0:
    #                 for inference_params in details['inference_variants']:
    #                     agent = agent.replace(**inference_params)
    #                     eval_info = evaluate(
    #                         agent, env, details['online_eval_episodes'], save_video=details['save_video']
    #                     )
    #                     if 'binary' not in details['env_name']:
    #                         eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
    #                     wandb.log({f"online_eval/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i + details['max_steps'])

    #                 agent.replace(**details['training_time_inference_params'])
