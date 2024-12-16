"""
Using separate networks for policy and value
"""

from omni.isaac.lab.app import AppLauncher
import getpass
# run settings
headless = True
training = True
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app


import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry


import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory, Memory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from torchvision.utils import save_image, make_grid
import os
from finetune.cube_detection.object_detector import ObjectDetector


class Policy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", cube_detector_neurons=6):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(38, 76)
        self.history_fc2 = nn.Linear(76, 32)

        self.fc2 = nn.Linear(2 * self.cube_detector_neurons + 32, self.cube_detector_neurons + 16)
        self.fc3 = nn.Linear(self.cube_detector_neurons + 16, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        self.compute_counter += 1

        if self.compute_counter < 100 and getpass.getuser() == 'phil':
            print("Policy: ", inputs["states"].shape)

        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = torch.tanh(x)
        x = self.fc11(x)
        x = torch.tanh(x)

        y = self.history_fc1(history_data)
        y = torch.tanh(y)
        y = self.history_fc2(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return torch.tanh(x), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, cube_detector_neurons=1000):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(38, 76)
        self.history_fc2 = nn.Linear(76, 32)

        self.fc2 = nn.Linear(2*self.cube_detector_neurons + 32, self.cube_detector_neurons+16)
        self.fc3 = nn.Linear(self.cube_detector_neurons+16, 1)

    def compute(self, inputs, role):
        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = torch.tanh(x)
        x = self.fc11(x)
        x = torch.tanh(x)

        y = self.history_fc1(history_data)
        y = torch.tanh(y)
        y = self.history_fc2(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x, {}





print("Cubechaser skrl camera")
# create base environment
print("Creating base environment")
env_cfg = load_cfg_from_registry("CubeChaser-Camera-skrl-cube-detector-v0", "env_cfg_entry_point")
env = gym.make("CubeChaser-Camera-skrl-cube-detector-v0", cfg=env_cfg, asynchronous=False)
# wrap environment to enforce that reset is called before step
# env = gym.wrappers.OrderEnforcing(env)
env = SkrlVecEnvWrapper(env)

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
device = env.device

print("---------------------------------------------------------")
print("env numenvs: ", env.num_envs)
print("cube_detector_neurons: ", env.unwrapped.cfg.cube_detector_neurons)

mini_batches = 4  #  2  # 8
memory_size = 16  # env.num_envs * mini_batches  # 64

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
# memory = Memory(memory_size=memory_size, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
print("Creating models")
print("observation space: ", env.observation_space)
print("action space: ", env.action_space)
print("num observations: ", env.num_observations)
cd_neurons = env.unwrapped.cfg.cube_detector_neurons * env.unwrapped.cfg.size_cube_detector_history
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True, cube_detector_neurons=cd_neurons)
models["value"] = Value(env.observation_space, env.action_space, device, cube_detector_neurons=cd_neurons)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
print("Creating agent aa")
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 16
# 16 ? 512 ? 128: num envs
# old: 16 * 512 / 128
# (16 * 512) / env.num_envs
cfg["mini_batches"] = mini_batches  # 48  # (memory_size * env.num_envs) // 16
# Note to batch_size:
#   _isaac_sim/kit/python/lib/python3.10/site-packages/skrl/memories/torch/base.py
#       line 378
#   batch_size=len(indexes) // mini_batches
#   indexes = np.arange(self.memory_size * self.num_envs)
cfg["discount_factor"] = 0.98  # gamma
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-5
# TODO: maybe try again. threshold should be around 10 times lr. maybe increase lr too?
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 5e-4}

# Note: was not that good... but maybe try learning starts again at some point
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0  # control balance between exploration and exploitation
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0.0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False  # TODO: was true, trying false
# TODO: deactivated this, maybe the reason for value focus on -1 and 1?
# Probably also not necessary for image input, already in the range 0.0 to 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Cubechaser-camera-cube-detector-v0"
cfg["experiment"]["store_separately"] = False

# tensorboard --logdir=PATH_TO_RUNS_DIRECTORY
# TODO: with wandb check here: https://skrl.readthedocs.io/en/latest/intro/data.html

# stored here: /home/phil/.local/share/ov/pkg/isaac-sim-4.0.0/python_packages/IsaacLab/runs/

agent = PPO(models=models,
            memory=memory,  # Note: memory is required for PPO because from here the value for the value function is loaded
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)  # or use SkrlSequentialLogTrainer, not sure whats the different with SequentialTrainer

# start training
# print("Starting training")
if training:
    trainer.train()
else:
    print("starting evaluation")
    path = "/home/phil/.local/share/ov/pkg/isaac-sim-4.0.0/python_packages/IsaacLab/runs/torch/Isaac-Cartpole-v0/24-06-28_01-01-00-295395_PPO/checkpoints/best_agent.pt"
    agent.load(path)
    trainer.eval()
env.close()