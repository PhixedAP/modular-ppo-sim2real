"""
Using separate networks for policy and value
"""

from omni.isaac.lab.app import AppLauncher

import getpass
# run settings
headless = False
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app


import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry
import torch.nn.functional as F

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory, Memory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model, CategoricalMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

from torch.optim.lr_scheduler import StepLR

class Policy(CategoricalMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True, num_observations=32,
                 size_agent_observation=20, size_agent_selection=6):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        print("POLICY self.num_actions: ", self.num_actions)
        print("POLICY action_space: ", action_space)
        print("Policy num_observations: ", num_observations)
        print("Policy size_agent_observation: ", size_agent_observation)
        print("Policy size_agent_selection: ", size_agent_selection)

        self.num_observations = num_observations
        self.size_agent_observation = size_agent_observation
        self.size_agent_selection = size_agent_selection

        self.fc_1 = nn.Linear(self.num_observations, self.num_observations*3)
        # self.fc_2 = nn.Linear(self.num_observations*3, self.num_observations*3)
        self.fc_2_2 = nn.Linear(self.num_observations*3, self.num_observations)
        self.fc_3 = nn.Linear(self.num_observations, self.num_actions)  # this layer outputs the logits

        self.debug_prints = False

    def compute(self, inputs, role):
        self.compute_counter += 1

        if self.debug_prints and self.compute_counter < 100:
            print("SKRL Policy input shape: ", inputs["states"].shape)

        x = self.fc_1(inputs["states"])
        x = F.relu(x)
        # x = self.fc_2(x)
        # x = F.relu(x)
        x = self.fc_2_2(x)
        x = F.relu(x)
        x = self.fc_3(x)


        return x, {}


class Value(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, num_observations=32,
                 size_agent_observation=20, size_agent_selection=6):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_observations = num_observations
        self.size_agent_observation = size_agent_observation
        self.size_agent_selection = size_agent_selection

        self.fc_1 = nn.Linear(self.num_observations, self.num_observations * 3)
        # self.fc_2 = nn.Linear(self.num_observations * 3, self.num_observations * 3)
        self.fc_2_2 = nn.Linear(self.num_observations * 3, self.num_observations)
        self.fc_3 = nn.Linear(self.num_observations, 1)

    def compute(self, inputs, role):
        self.compute_counter += 1

        x = self.fc_1(inputs["states"])
        x = F.elu(x)
        # x = self.fc_2(x)
        # x = F.relu(x)
        x = self.fc_2_2(x)
        x = F.elu(x)
        x = self.fc_3(x)

        return x, {}


print("-----------Hma-Cube-Exp-v2 SKRL-------------")
env_cfg = load_cfg_from_registry("Hma-Cube-Exp-v2", "env_cfg_entry_point")
env = gym.make("Hma-Cube-Exp-v2", cfg=env_cfg, asynchronous=False)
env = SkrlVecEnvWrapper(env)

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
device = env.device

print("---------------------------------------------------------")
print("env numenvs: ", env.num_envs)

# TODO: explanation below true?
# reward converges quickly because the action space is discrete
# can be accounted for by using a high mini_batches value and low learning_epcohs
memory_size = 16

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
print("Creating models")
print("observation space: ", env.observation_space)
print("action space: ", env.action_space)
models = {}

num_observations = env.num_observations  # env.unwrapped.cfg.num_observations
size_agent_observation = env.unwrapped.cfg.ma_observation_history_size * env.unwrapped.cfg.single_observation_size
size_agent_selection = env.unwrapped.cfg.agent_selection_history_size
print("num observations: ", num_observations)

models["policy"] = Policy(env.observation_space, env.action_space, device, unnormalized_log_prob=True,
                          num_observations=num_observations,
                          size_agent_observation=size_agent_observation,
                          size_agent_selection=size_agent_selection)
models["value"] = Value(env.observation_space, env.action_space, device,
                        num_observations=num_observations,
                        size_agent_observation=size_agent_observation,
                        size_agent_selection=size_agent_selection)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
print("Creating agent")
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 64
cfg["learning_rate"] = 5e-4

# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}  # , "lr_factor": 1.2

#### OLD VALUES, JUST MADE ISSUES

# cfg["learning_epochs"] = 2
# # 16 ? 512 ? 128: num envs
# # old: 16 * 512 / 128
# # (16 * 512) / env.num_envs
# cfg["mini_batches"] = mini_batches  # 48  # (memory_size * env.num_envs) // 16
# # Note to batch_size:
# #   _isaac_sim/kit/python/lib/python3.10/site-packages/skrl/memories/torch/base.py
# #       line 378
# #   batch_size=len(indexes) // mini_batches
# #   indexes = np.arange(self.memory_size * self.num_envs)
# cfg["discount_factor"] = 0.99  # gamma
#
# # GAE Lambda (lambda in advantage estimation) controls the smoothness of advantage estimation.
# # A high value (e.g., 0.95–0.99) smooths out noisy advantages, which can help prevent erratic actions.
# cfg["lambda"] = 0.95
#
#
# cfg["learning_rate"] = 5e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}  # , "lr_factor": 1.2
#
#
# # cfg["learning_rate"] = 1e-3
# # cfg["learning_rate_scheduler"] = StepLR
# # cfg["learning_rate_scheduler_kwargs"] = {"step_size": 250, "gamma": 0.9}
#
# # Note: was not that good... but maybe try learning starts again at some point
# cfg["random_timesteps"] = 0
cfg["learning_starts"] = env.unwrapped.cfg.random_steps_for_value_collection_for_normalization if env.unwrapped.cfg.random_steps_for_value_collection_for_normalization >= 0 else 0
# cfg["grad_norm_clip"] = 0.5
# cfg["ratio_clip"] = 0.2
# cfg["value_clip"] = 0.2
# cfg["clip_predicted_values"] = False
# cfg["entropy_loss_scale"] = 0.01  # 0.01  # control balance between exploration and exploitation
# cfg["value_loss_scale"] = 0.5
# cfg["kl_threshold"] = 0.0
# cfg["rewards_shaper"] = None
# cfg["time_limit_bootstrap"] = False
#
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
#
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Hma-Cube-Exp-v2"
cfg["experiment"]["store_separately"] = False

agent = PPO(models=models,
            memory=memory,  # Note: memory is required for PPO because from here the value for the value function is loaded
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
evaluation_timesteps = 20000 if env.num_envs == 6 else 10000
training = not env.unwrapped.cfg.evaluation_mode
cfg_trainer = {"timesteps": 150000 if training else evaluation_timesteps, "headless": headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)  # or use SkrlSequentialLogTrainer, not sure whats the different with SequentialTrainer

# start training
# print("Starting training")
if training:
    trainer.train()
else:
    print("starting evaluation")
    state_inputs_version = env.unwrapped.cfg.state_inputs_version
    if state_inputs_version == "value_and_ce_value":
        # value_and_ce_value:
        # A: 24-12-15_05-36-42-203108_PPO: /home/phil/Documents/24-12-15_05-36-42-203108_PPO/agent_31000.pt
        path = "/home/phil/Documents/24-12-15_15-28-19-961914_PPO/agent_3000.pt"
        # update_value_normalization_min_max_till_timestep = 1000
    elif state_inputs_version == "value_and_ce_value_and_ce_logprobs_min":
        # value_and_ce_value_and_ce_logprobs_min: /home/phil/Documents/24-12-15_02-34-40-391707_PPO/agent_51000.pt
        path = "/home/phil/Documents/24-12-15_02-34-40-391707_PPO/agent_51000.pt"
    elif state_inputs_version == "value_and_ce_value_and_logprobs_and_ce_logprobs_each_and_ce_policy_ratio":
        # path = "/home/phil/Documents/24-12-14_14-34-27-822094_PPO/agent_125000.pt"
        path = "/home/phil/Documents/24-12-14_14-34-27-822094_PPO/agent_35000.pt"
    else:
        path = ""

    agent.load(path)
    trainer.eval()
env.close()
