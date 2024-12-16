from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
import omni.isaac.lab.sim as sim_utils
import numpy as np
import gymnasium as gym
import torch
from omni.isaac.lab.utils import configclass
from collections.abc import Sequence

from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg, RayCaster, RayCasterCfg, patterns, RayCasterCameraCfg, RayCasterCamera, ContactSensor, ContactSensorCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

import omni.replicator.core as rep

import time
from torchvision.utils import save_image, make_grid
import math
import matplotlib.pyplot as plt
import pickle
from torch.cuda import device

from ..utilitiesisaaclab import mecanum_direction, mecanum_controller
# from ..explorationlidar import exploration_lidar_skrl # TODO: implement later. Needs codecleanup and folder name must be without dash

@configclass
class HmaCubeExpEnvV0Cfg(DirectRLEnvCfg):
    decimation = 2
    action_scale = 1.0
    num_actions = 2
    # sim_dt = 1 / 60.0
    sim_dt = 1 / 30.0

    episode_length_factor = 1.5
    episode_length_s = sim_dt * 1200 * episode_length_factor  # = 20 seconds for sim_dt = 1/60.0
    max_steps_per_episode = 600 * episode_length_factor

    num_envs = 6

    size_value_network = 1
    size_log_odds_probabilities = 1
    scaled_image_width = 320
    scaled_image_height = 240

    '''
    exploration agent settings
    '''
        # lidar
    depth_image_slice_size = 16
    middle_slice_start = (scaled_image_height - depth_image_slice_size) // 2
    middle_slice_end = middle_slice_start + depth_image_slice_size

    depth_image_history_size = 0
    update_frequency_depth_image_steps = 10

        # ogs
    room_size = 8.0
    required_map_size = room_size * 2.0  # for robotic centered map, robot should be able to move to the end of the room
    grid_size = 256
    grid_map_resolution = required_map_size / grid_size

    l_occ = np.log(0.7 / 0.3)  # Log odds for occupied cell
    l_free = np.log(0.3 / 0.7)  # Log odds for free cell

        # rewards
    start_discovery_reward_after_timestep = 6

        # observation
    exp_size_action_history = 10
    exp_size_robot_orientation_history = 5
    exp_num_observations = grid_size * grid_size + exp_size_robot_orientation_history + exp_size_action_history + depth_image_slice_size * scaled_image_width * (
            depth_image_history_size + 1)

    '''
    cube agent settings
    '''
    cube_detector_neurons = 5
    cube_size_action_history = 20
    cube_size_cube_detector_history = 5
    cube_update_cube_detector_history_each_steps = 5
    cube_num_observations = cube_detector_neurons * cube_size_cube_detector_history + cube_size_action_history


    '''
    multi agent settings
    '''
    num_different_agents = 2
    penalty_for_touching_obstacle = -1.25  # -15.0
    required_distance_to_goal = 0.4  # TODO: change back to 0.3
    ma_observation_history_size = 3  # TODO: probably decrease?
    reward_for_goal_reached = 5.0
    reward_distance_change_factor = 3.0
    maximum_discovery_reward = 1.0
    size_agents_rating_action_other_agents = 2**num_different_agents
    single_observation_size = (size_value_network + size_log_odds_probabilities) * num_different_agents + size_agents_rating_action_other_agents
    agent_selection_history_size = 6
    num_observations = single_observation_size * ma_observation_history_size + agent_selection_history_size


    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)


    # robot
    model_name = "012_4wis_increased_chassis_size.usd"  # "013_4wis_moved_camera_to_account_for_increased_chassis.usd"
    own_model_path = "/home/phil/Documents/"
    velocity_joint_names = ["front_right_wheel_joint", "front_left_wheel_joint",
                            "rear_right_wheel_joint", "rear_left_wheel_joint"]
    position_joint_names = ["front_right_steering_joint", "front_left_steering_joint",
                            "rear_right_steering_joint", "rear_left_steering_joint"]

    fourwis_jetauto: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/fourwis",
        spawn=sim_utils.UsdFileCfg(
            usd_path=own_model_path + model_name,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1000.0,  # high value prevents robot from "flying" away. Or maybe not?...
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(3.5, 3.5, 0.01),
        ),
        actuators={
            "velocity_actuator": ImplicitActuatorCfg(
                joint_names_expr=velocity_joint_names,
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=100000.0,
            ),
            "steering_actuator": ImplicitActuatorCfg(
                joint_names_expr=position_joint_names,
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=50000,  # 50000.0,
                damping=50.0,  # 500.0,
            ),
        }
    )

    # goal
    goal_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=own_model_path + "goal_cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                # enable_gyroscopic_forces=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.5, 3.5, 0.02),
        ),
    )

    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/fourwis/chassis/camera",
        data_types=["distance_to_camera", "rgb"],
        spawn=None,
        width=scaled_image_width,
        height=scaled_image_height,
        update_period=sim_dt * decimation,
    )

    light_cfg: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        intensity=2000.0, color=(0.75, 0.75, 0.75)
    )

    distant_light_cfg: sim_utils.DistantLightCfg = sim_utils.DistantLightCfg(
        color=(0.9, 0.9, 0.9), intensity=2500.0
    )

    own_omniverse_scenes = "/home/phil/Documents/scenes/"

    # scenery
    own_wall_cfg: sim_utils.spawners.shapes_cfg.CuboidCfg = sim_utils.spawners.shapes_cfg.CuboidCfg(
        size=(7.0, 0.2, 2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True
    )

    obstacle_column_cfg: sim_utils.spawners.shapes_cfg.CylinderCfg = sim_utils.spawners.shapes_cfg.CylinderCfg(
        radius=0.5, height=1.0,
        # , color=(0.5, 0.5, 0.5)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
    )

    obstacle_cuboid_cfg: sim_utils.spawners.shapes_cfg.CuboidCfg = sim_utils.spawners.shapes_cfg.CuboidCfg(
        size=(2.0, 0.4, 1.0),
        # color=(0.5, 0.5, 0.5)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
    )

    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/chassis",
        track_air_time=False,
        debug_vis=False,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
        # "/World/envs/env_.*/obstacle_cuboid_7", "/World/envs/env_.*/obstacle_cuboid_7"]
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=room_size, replicate_physics=True)

'''
Exploration Agent
'''
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveRL
import torch.nn as nn
import torch.nn.functional as F

class ExplorationPolicy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", ogm_grid_size=256,
                 action_history_size=10, orientation_size=1, depth_image_height=16, depth_image_width=320,
                 depth_image_history_size=3):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.ogm_grid_size = ogm_grid_size
        self.action_history_size = action_history_size
        self.orientation_size = orientation_size
        self.depth_image_height = depth_image_height
        self.depth_image_width = depth_image_width
        self.depth_image_history_size = depth_image_history_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 256)

        # depth image 2 layer network
        self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, 8, kernel_size=(3, 6), stride=(1, 2), padding=1)
        self.pool_depth_1 = nn.MaxPool2d(2, 2)
        self.conv_depth_2 = nn.Conv2d(8, 16, kernel_size=(3, 6), stride=(1, 2), padding=0)
        self.pool_depth_2 = nn.MaxPool2d(2, 2)
        self.fc_depth_1 = nn.Linear(864, 256)

        self.fc_history_orientation = nn.Linear(self.action_history_size + self.orientation_size, (self.action_history_size + self.orientation_size)*2)

        self.fc_combined_1 = nn.Linear(256 + 256 + (self.action_history_size + self.orientation_size)*2, 256)
        self.fc_combined_1_2 = nn.Linear(256, 256)
        self.fc_combined_2 = nn.Linear(256, 64)
        self.fc_combined_3 = nn.Linear(64, self.num_actions)

        self.dropout = nn.Dropout(0.2)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        self.compute_counter += 1

        ogm_input = inputs["states"][:, :self.ogm_grid_size*self.ogm_grid_size]
        ogm_input = ogm_input.view(-1, 1, self.ogm_grid_size, self.ogm_grid_size)

        depth_input = inputs["states"][:, self.ogm_grid_size*self.ogm_grid_size:self.ogm_grid_size*self.ogm_grid_size + self.depth_image_height*self.depth_image_width*self.depth_image_history_size]
        depth_input = depth_input.view(-1, self.depth_image_history_size, self.depth_image_height, self.depth_image_width)

        history_orientation_input = inputs["states"][:, self.ogm_grid_size*self.ogm_grid_size + self.depth_image_height*self.depth_image_width*self.depth_image_history_size:]

        x = self.conv1(ogm_input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.tanh(x)

        depth = self.conv_depth_1(depth_input)
        depth = F.relu(depth)
        depth = self.pool_depth_1(depth)
        depth = self.conv_depth_2(depth)
        depth = F.relu(depth)
        depth = self.pool_depth_2(depth)
        depth = torch.flatten(depth, start_dim=1)
        depth = self.fc_depth_1(depth)
        depth = torch.tanh(depth)
        depth = self.dropout(depth)

        y = self.fc_history_orientation(history_orientation_input)
        y = torch.tanh(y)

        x = torch.cat((x, depth, y), dim=1)
        x = self.fc_combined_1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_1_2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_3(x)

        return torch.tanh(x), self.log_std_parameter, {}


class ExplorationValue(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, ogm_grid_size=256,
                 action_history_size=10, orientation_size=1, depth_image_height=16, depth_image_width=320,
                 depth_image_history_size=3):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.ogm_grid_size = ogm_grid_size
        self.action_history_size = action_history_size
        self.orientation_size = orientation_size
        self.depth_image_height = depth_image_height
        self.depth_image_width = depth_image_width
        self.depth_image_history_size = depth_image_history_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 256)

        # depth image 2 layer network
        self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, 8, kernel_size=(3, 6), stride=(1, 2), padding=1)
        self.pool_depth_1 = nn.MaxPool2d(2, 2)
        self.conv_depth_2 = nn.Conv2d(8, 16, kernel_size=(3, 6), stride=(1, 2), padding=0)
        self.pool_depth_2 = nn.MaxPool2d(2, 2)
        self.fc_depth_1 = nn.Linear(864, 256)

        self.fc_history_orientation = nn.Linear(self.action_history_size + self.orientation_size,
                                                (self.action_history_size + self.orientation_size) * 2)

        self.fc_combined_1 = nn.Linear(256 + 256 + (self.action_history_size + self.orientation_size) * 2, 256)
        self.fc_combined_1_2 = nn.Linear(256, 256)
        self.fc_combined_2 = nn.Linear(256, 64)
        self.fc_combined_3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)

    def compute(self, inputs, role):
        ogm_input = inputs["states"][:, :self.ogm_grid_size * self.ogm_grid_size]
        ogm_input = ogm_input.view(-1, 1, self.ogm_grid_size, self.ogm_grid_size)

        depth_input = inputs["states"][:,
                      self.ogm_grid_size * self.ogm_grid_size:self.ogm_grid_size * self.ogm_grid_size + self.depth_image_height * self.depth_image_width * self.depth_image_history_size]
        depth_input = depth_input.view(-1, self.depth_image_history_size, self.depth_image_height, self.depth_image_width)

        history_orientation_input = inputs["states"][:,
                                    self.ogm_grid_size * self.ogm_grid_size + self.depth_image_height * self.depth_image_width * self.depth_image_history_size:]

        x = self.conv1(ogm_input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.tanh(x)

        depth = self.conv_depth_1(depth_input)
        depth = F.relu(depth)
        depth = self.pool_depth_1(depth)
        depth = self.conv_depth_2(depth)
        depth = F.relu(depth)
        depth = self.pool_depth_2(depth)
        depth = torch.flatten(depth, start_dim=1)
        depth = self.fc_depth_1(depth)
        depth = torch.tanh(depth)
        depth = self.dropout(depth)

        y = self.fc_history_orientation(history_orientation_input)
        y = torch.tanh(y)

        x = torch.cat((x, depth, y), dim=1)
        x = self.fc_combined_1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_1_2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc_combined_3(x)

        return x, {}


class ExplorationAgent:
    model_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/hierarchical-multi-agent-cube-exploration/"
    model_name = "expo_24-10-23_03-29-34-024880_PPO_agent_48000.pt"
    mini_batches = 2
    num_actions = 2

    def __init__(self, cfg: HmaCubeExpEnvV0Cfg, device):
        print("INIT EXPLORATION AGENT!!!")
        self.expo_num_envs = 1  # TODO: use cfg.num_envs, but seems like loaded agent expects 1 env

        self.device = device
        self.memory_size = self.expo_num_envs * self.mini_batches
        single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(single_action_space, self.expo_num_envs)

        single_observation_space = gym.spaces.Dict()
        single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            shape=(cfg.exp_num_observations,)
        )
        self.observation_space = gym.vector.utils.batch_space(single_observation_space, self.expo_num_envs)

        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg["grad_norm_clip"] = 1.0  # needed?

        self.memory = RandomMemory(memory_size=self.memory_size, num_envs=self.expo_num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = ExplorationPolicy(self.observation_space, self.action_space, self.device, clip_actions=True,
                                                  ogm_grid_size=cfg.grid_size, action_history_size=cfg.exp_size_action_history,
                                                  orientation_size=cfg.exp_size_robot_orientation_history,
                                                  depth_image_height=cfg.depth_image_slice_size,
                                                  depth_image_width=cfg.scaled_image_width,
                                                  depth_image_history_size=cfg.depth_image_history_size+1)
        self.models["value"] = ExplorationValue(self.observation_space, self.action_space, self.device,
                                                ogm_grid_size=cfg.grid_size,
                                                action_history_size=cfg.exp_size_action_history,
                                                orientation_size=cfg.exp_size_robot_orientation_history,
                                                depth_image_height=cfg.depth_image_slice_size,
                                                depth_image_width=cfg.scaled_image_width,
                                                depth_image_history_size=cfg.depth_image_history_size+1)
        self.agent = PPO(models=self.models,
                         memory=self.memory,
                         cfg=ppo_cfg,
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         device=self.device)
        self.agent.load(self.model_path + self.model_name)
        self.agent.set_running_mode("eval")
        print("ExplorationAgent init done")

    def act(self, observations):
        # return of act:
        '''
        The first component is the action to be taken by the agent.
        The second component is the log of the probability density function for stochastic models
        or None for deterministic models. The third component is a dictionary containing extra output values
        '''

        policy = self.agent.policy.act({"states": observations}, role="policy")
        # print("policy: ", policy)
        value = self.agent.value.act({"states": observations}, role="value")
        # print("value: ", value)

        return policy[0], policy[1], value[0]

        # temp = self.agent.act(observations, 500, 10000)  # timestamp values dont matter here
        # print("temp: ", temp)
        # temp[0]: actions, shape: torch.Size([4, 2])
        # temp[1]: object with some values and "mean_actions":
        # tensor([[-1.7876],
        #         [-3.6416],
        #         [-1.8879],
        #         [-1.8039]], device='cuda:0'), {'mean_actions': tensor([[1.0000, -0.8326],
        #                                                                [1.0000, 0.1330],
        #                                                                [1.0000, -0.0999],
        #                                                                [1.0000, -0.3675]], device='cuda:0')})
        # return temp[0]

    def get_log_odds_for_action(self, actions):
        rating = self.agent.policy.distribution().log_prob(actions)
        rating = torch.sum(rating, dim=-1)  # applying reduction
        return rating.unsqueeze(1)

    def get_value(self, observations):
        return self.agent.value.act({"states": observations}, role="value")[0]


'''
Cube detector Agent
'''
class CubePolicy(GaussianMixin, Model):
    compute_counter = 0

    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", cube_detector_neurons=6):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2 * self.cube_detector_neurons)
        self.fc11 = nn.Linear(2 * self.cube_detector_neurons, 2 * self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(20, 40)
        self.history_fc2 = nn.Linear(40, 20)

        self.fc2 = nn.Linear(2 * self.cube_detector_neurons + 20, self.cube_detector_neurons + 10)
        self.fc3 = nn.Linear(self.cube_detector_neurons + 10, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        self.compute_counter += 1

        # if self.compute_counter < 100:
        #     print("Cube Policy: ", inputs["states"].shape)

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

class CubeValue(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, cube_detector_neurons=1000):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(20, 40)
        self.history_fc2 = nn.Linear(40, 20)

        self.fc2 = nn.Linear(2*self.cube_detector_neurons + 20, self.cube_detector_neurons+10)
        self.fc3 = nn.Linear(self.cube_detector_neurons+10, 1)

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


class RedCubeDetectorFasterRCNN:
    model_path = "/home/phil/Documents/"
    model_name = "fasterrcnn_cube_detector_mobilenet.pth"

    def __init__(self, device):
        print("init RedCubeDetectorFasterRCNN")
        self.device = device
        self.model = torch.load(self.model_path + self.model_name).to(self.device)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


from skrl.resources.preprocessors.torch import RunningStandardScaler

class CubeDetectorAgent:
    model_path = "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/model/CCC-skrl-cd-fastercnn-v0/24-09-24_08-34-18-330494_PPO/"
    model_name = "agent_66000.pt"
    memory_size = 16
    num_actions = 2

    size_cube_detector_history = 5
    cube_detector_neurons = 5


    def __init__(self, cfg: HmaCubeExpEnvV0Cfg, device):
        print("init CubeDetectorAgent")
        self.cube_num_envs = 1
        self.device = device

        single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(single_action_space, self.cube_num_envs)

        single_observation_space = gym.spaces.Dict()
        single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            shape=(cfg.cube_num_observations,)
        )
        self.observation_space = gym.vector.utils.batch_space(single_observation_space, self.cube_num_envs)

        self.device = device
        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg["grad_norm_clip"] = 1.0  # needed?
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": self.observation_space, "device": self.device}
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}

        self.memory = RandomMemory(memory_size=self.memory_size, num_envs=self.cube_num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = CubePolicy(self.observation_space, self.action_space, self.device, clip_actions=True,
                                       cube_detector_neurons=cfg.cube_detector_neurons * cfg.cube_size_cube_detector_history)
        self.models["value"] = CubeValue(self.observation_space, self.action_space, self.device,
                                     cube_detector_neurons=cfg.cube_detector_neurons * cfg.cube_size_cube_detector_history)
        self.agent = PPO(models=self.models,
                         memory=self.memory,
                         cfg=ppo_cfg,
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         device=self.device)
        self.agent.load(self.model_path + self.model_name)
        self.agent.set_running_mode("eval")
        print("CubeDetectorRlAgent init done")

    def act(self, observations):
        policy = self.agent.policy.act({"states": observations}, role="policy")
        value = self.agent.value.act({"states": observations}, role="value")
        return policy[0], policy[1], value[0]

    def get_log_odds_for_action(self, actions):
        rating = self.agent.policy.distribution().log_prob(actions)
        rating = torch.sum(rating, dim=-1)  # applying reduction
        return rating.unsqueeze(1)

    def get_value(self, observations):
        return self.agent.value.act({"states": observations}, role="value")[0]



class HmaCubeExpEnvV0(DirectRLEnv):
    cfg: HmaCubeExpEnvV0Cfg
    image_path = "/home/phil/university/thesis/data/images/"
    observation_counter = 0
    randomization_counter = 0
    export_images = False
    export_grid_maps = False
    images_saved_counter = 0

    def __init__(self, cfg: HmaCubeExpEnvV0Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg)
        self.action_scale = cfg.action_scale

        self.joint_pos = self._fourwis_jetauto.data.joint_pos
        self.joint_vel = self._fourwis_jetauto.data.joint_vel

        self._fourwis_velocity_joint_indexes = []
        self._fourwis_steering_joint_indexes = []
        for vel_joint_name in self.cfg.velocity_joint_names:
            temp, _ = self._fourwis_jetauto.find_joints(vel_joint_name)
            self._fourwis_velocity_joint_indexes.append(temp[0])
        for pos_joint_name in self.cfg.position_joint_names:
            temp, _ = self._fourwis_jetauto.find_joints(pos_joint_name)
            self._fourwis_steering_joint_indexes.append(temp[0])

        self.grid_maps = torch.ones((self.num_envs, int(self.cfg.grid_size), int(self.cfg.grid_size)),
                                    device=self.sim.device) * 0.5
        print("grid_maps shape: ", self.grid_maps.shape)

        self.mecanum_controller = mecanum_controller.MecanumController(self.cfg.num_envs, self.sim.device)

        self.goal_root_state = self._goal.data.default_root_state

        self.start_time_episode = torch.full((self.num_envs,), time.time(), dtype=torch.float64, device=self.sim.device)
        self.total_reward_sum = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_force_matrix_penalty = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_distance_change_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_discovery_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)

        self.total_goal_reached_counter = 0  # TODO: implement
        self.current_run_observation_counter_per_env = torch.zeros(self.cfg.num_envs, device=self.sim.device)

        self.reset_counter = 0

        self.discovery_percentages_history_env_0 = []
        self.robot_position_history_env_0 = []

        self.occupancy_grid_maps = torch.zeros((self.num_envs, self.cfg.grid_size, self.cfg.grid_size), device=self.sim.device)
        self.grid_origins_x = torch.zeros(self.num_envs, device=self.sim.device)
        self.grid_origins_y = torch.zeros(self.num_envs, device=self.sim.device)
        self.original_grid_origins = torch.zeros((self.num_envs, 2), device=self.sim.device)
        self.previous_robot_positions = torch.zeros((self.num_envs, 3), device=self.sim.device)
        self.robot_positions = torch.zeros((self.num_envs, 3), device=self.sim.device)

        self.robot_yaw_history = torch.zeros((self.num_envs, 3), device=self.sim.device)

        self.force_matrix_penalty = torch.zeros(self.num_envs, device=self.sim.device)

        self.multi_agent_observation_history = torch.zeros(self.cfg.num_envs,
                                                           self.cfg.ma_observation_history_size * self.cfg.single_observation_size,
                                                           device=self.sim.device, dtype=torch.float32)

        self.exp_action_history = torch.zeros(self.cfg.num_envs, self.cfg.exp_size_action_history, device=self.sim.device,
                                          dtype=torch.float32)

        self.last_cubedetector_labels = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.bool)

        self.cube_action_history = torch.zeros(self.cfg.num_envs, self.cfg.cube_size_action_history, device=self.sim.device, dtype=torch.float32)
        self.cube_detector_history = torch.zeros(self.cfg.num_envs, self.cfg.cube_detector_neurons * self.cfg.cube_size_cube_detector_history,
                                                 device=self.sim.device, dtype=torch.float32)

        self.cube_last_action = torch.zeros(self.cfg.num_envs, self.cfg.num_actions, device=self.sim.device, dtype=torch.float32)
        self.exp_last_action = torch.zeros(self.cfg.num_envs, self.cfg.num_actions, device=self.sim.device, dtype=torch.float32)
        self.last_applied_action = torch.zeros(self.cfg.num_envs, self.cfg.num_actions, device=self.sim.device, dtype=torch.float32)

        self.apply_action_counter = 0

        self.count_selected_which_agent = torch.zeros(self.cfg.num_envs, 2, device=self.sim.device, dtype=torch.float32)

        self.agent_selection_history = torch.zeros(self.cfg.num_envs, self.cfg.agent_selection_history_size, device=self.sim.device, dtype=torch.float32)

        # create tensor for depth image history, containing the last x depth images for each environment
        if self.cfg.depth_image_history_size:
            self.depth_image_history = torch.zeros(self.cfg.num_envs, self.cfg.depth_image_history_size, self.cfg.depth_image_slice_size * self.cfg.scaled_image_width,
                                                   device=self.sim.device, dtype=torch.float32)

        self.exp_robot_orientation_history = torch.zeros(self.cfg.num_envs, self.cfg.exp_size_robot_orientation_history, device=self.sim.device,
                                                        dtype=torch.float32)

        self.distance_to_goal = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.previous_discovery_percentages = torch.zeros(self.num_envs, device=self.sim.device)
        self.normalized_reward_distance_changed = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)

        self.discovery_reward_factor = self.cfg.maximum_discovery_reward / 10.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            for key in [
                "force_matrix_penalty",
                "reward_for_goal_reached",
                "distance_change_reward",
                "discovery_reward"
            ]
        }

        self.exploration_agent = ExplorationAgent(cfg, self.sim.device)
        self.cube_agent = CubeDetectorAgent(cfg, self.sim.device)
        self.cube_detector = RedCubeDetectorFasterRCNN(self.sim.device)


    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            shape=(self.cfg.num_observations,)
        )

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(self.num_actions,))
        # self.single_action_space = gym.spaces.Discrete(self.num_actions)

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        print("Setting up scene")
        self._fourwis_jetauto = Articulation(self.cfg.fourwis_jetauto)
        self._camera = Camera(self.cfg.camera)

        self._goal = RigidObject(self.cfg.goal_cube)

        self._own_walls = []
        wall_positions = [[3.5, 0.0, 0.0, 1.0], [-0.1, 3.5, 0.70711, 0.70711], [3.5, 7.0, 0.0, 1.0], [7.1, 3.5, 0.70711, 0.70711]]
        for index, wall_position in enumerate(wall_positions):
            _wall = self.cfg.own_wall_cfg.func(
                prim_path="/World/envs/env_.*/own_wall_{}".format(index),
                cfg=self.cfg.own_wall_cfg,
                orientation=np.array([wall_position[2], 0.0, 0.0, wall_position[3]]),
                translation=(wall_position[0], wall_position[1], 1.0),
            )
            self._own_walls.append(_wall)

        self._obstacle_column = self.cfg.obstacle_column_cfg.func(
            prim_path="/World/envs/env_.*/obstacle_column",
            cfg=self.cfg.obstacle_column_cfg,
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            translation=(6.0, 1.0, 0.5)
        )

        obstacle_cuboids_positions = [[3.5, 3.0, 1.0, 0.0], [1.0, 4.5, 0.70711, 0.70711], [3.0, 4.0, 1.0, 0.0],
                                      [3.5, 1.2, 0.92388, 0.38268], [5.0, 5.0, 0.70711, 0.70711], [3.0, 6.0, 1.0, 0.0], [6.0, 4.0, 0.70711, 0.70711]]
        for index, obstacle_cuboid_position in enumerate(obstacle_cuboids_positions):
            self.cfg.obstacle_cuboid_cfg.func(
                prim_path="/World/envs/env_.*/obstacle_cuboid_{}".format(index),
                cfg=self.cfg.obstacle_cuboid_cfg,
                orientation=np.array([obstacle_cuboid_position[2], 0.0, 0.0, obstacle_cuboid_position[3]]),
                translation=(obstacle_cuboid_position[0], obstacle_cuboid_position[1], 0.5)
            )

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)

        self.gnd_plane = spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5)))

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["fourwis"] = self._fourwis_jetauto

        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # add lights
        self.light = self.cfg.light_cfg.func("/World/Light", self.cfg.light_cfg)
        self.distant_light = self.cfg.distant_light_cfg.func("/World/DistantLight", self.cfg.distant_light_cfg)


        print("testing getting prims stuff...: ", self.sim.has_rtx_sensors())
        prims = sim_utils.get_all_matching_child_prims("/World/envs/env_0/fourwis/chassis")
        for prim in prims:
            print("prim path: ", prim.GetPrimPath())
        print("-----------------------")

        lidar_prims = []
        hydra_textures = []

        # used for drawing debug lines (blue lines in simulation)
        # self.writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud" + "Buffer")

        # annoNames: check rtx_lidar_test.py rtx_lidar.py test_rtx_sensor.py
        # also see: https://forums.developer.nvidia.com/t/manipulating-the-scan-buffer-of-a-lidar/282502
        # not working: RtxSensorCpuIsaacReadRTXLidarData
        # self.annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer",
        #                                                 init_params={"keepOnlyPositiveDistance": True,
        #                                                              "outputObjectId": False})
        self.annotators = []

        for i in range(self.num_envs):
            lidar_prims.append(sim_utils.get_all_matching_child_prims("/World/envs/env_" + str(i) + "/fourwis/chassis/Rotating"))
            hydra_textures.append(rep.create.render_product(lidar_prims[i][0].GetPath(), [1, 1], name="Isaac").path)
            # self.annotator.attach([hydra_textures[i]])
            # self.writer.attach([hydra_textures[i]])
            self.annotators.append(rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer",
                                                                        init_params={"keepOnlyPositiveDistance": True,
                                                                                     "outputObjectId": False}))
            self.annotators[i].attach([hydra_textures[i]])

        # print("rotating_lidar_prim_0 path: ", rotating_lidar_prim_0[0].GetPrimPath())


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        # self.actions = torch.clamp(self.actions, -1.0, 1.0)


    def _apply_action(self) -> None:
        self.apply_action_counter += 1
        self.apply_action_counter = self.apply_action_counter % 1000
        # TODO: should the last action supplied to each agent be its last action or the multi agents last action?
        condition = (self.actions[:, 0] > self.actions[:, 1]).unsqueeze(1)
        self.last_applied_action = torch.where(condition, self.exp_last_action, self.cube_last_action)
        velocities, positions = self.mecanum_controller.continuous_actions_into_velocities_and_positions(self.last_applied_action)
        self._fourwis_jetauto.set_joint_position_target(positions,
                                                        joint_ids=self._fourwis_steering_joint_indexes)
        self._fourwis_jetauto.set_joint_velocity_target(velocities,
                                                        joint_ids=self._fourwis_velocity_joint_indexes)

        if self.apply_action_counter % self.cfg.decimation == 0:
            self.agent_selection_history = torch.roll(self.agent_selection_history, 1, dims=1)
            self.count_selected_which_agent += torch.where(condition, torch.tensor([[1, 0]], device=self.sim.device),
                                                           torch.tensor([[0, 1]], device=self.sim.device))
        self.agent_selection_history[:, 0] = torch.where(condition[:, 0], torch.tensor([1.0], device=self.sim.device), torch.tensor([0.0], device=self.sim.device))


    def store_occupancy_grid_map_as_png(self, env_id):
        occupancy_grid_map = self.occupancy_grid_maps[env_id].cpu().numpy()
        ogm_probabilities = self.log_odds_to_probability(occupancy_grid_map)
        plt.figure(figsize=(10, 10))
        plt.imshow(ogm_probabilities, cmap='binary', origin='lower')
        plt.colorbar(label='Occupancy')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Occupancy Grid Map')
        plt.grid(True, which='both', color='0.95', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.image_path + "occupancy_grid_map_" + str(env_id) + "_" + str(self.observation_counter) + ".png")
        plt.close()

    def store_discovery_percentages_history_env_0_as_png(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.discovery_percentages_history_env_0)
        plt.xlabel('Steps')
        plt.ylabel('Discovery Percentage')
        plt.title('Discovery Percentage History')
        plt.grid(True, which='both', color='0.95', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.image_path + "discovery_percentage_history_" + str(self.observation_counter) + ".png")
        plt.close()

    def store_robot_positions_history_env_0_as_png(self):
        if len(self.robot_position_history_env_0) > 0:
            robot_positions_np = np.array(self.robot_position_history_env_0)
            plt.figure(figsize=(10, 10))
            plt.scatter(robot_positions_np[:, 0], robot_positions_np[:, 1], marker='o')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Robot Position History')
            plt.grid(True, which='both', color='0.95', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(self.image_path + "robot_position_history_" + str(self.observation_counter) + ".png")
            plt.close()

    def store_depth_image_as_png(self, image, name="depth_image_"):
        save_image(image, self.image_path + name + str(self.observation_counter) + ".png")


    def _get_observations(self) -> dict:
        self.current_run_observation_counter_per_env += 1
        self.observation_counter += 1

        camera_output = self._camera.data.output.clone()
        '''
        Data for exploration agent
        '''
        depth_data_type = "distance_to_camera"
        depth_camera_image = camera_output[depth_data_type]

        # slice depth camera image
        depth_camera_image = depth_camera_image[:, self.cfg.middle_slice_start:self.cfg.middle_slice_end, :]
        # if self.export_grid_maps and self.observation_counter % 25 == 0:
        #     self.store_depth_image_as_png(depth_camera_image[0], "depth_image_middle_slice_")

        # normalize depth camera image
        # depth_camera_image += 0.27  # TODO: moved the camera on this model, adjust depth measurement values?
        depth_camera_image = torch.clamp(depth_camera_image, 0, 10.0) / 10.0
        depth_camera_image = depth_camera_image.view(self.num_envs, -1)

        # combine all lidar anno data into tensor:
        lidar_data = []
        has_incomplete_lidar_data = False
        for i in range(self.num_envs):
            lidar_anno_data = self.annotators[i].get_data()
            lidar_data_tensor = torch.tensor(lidar_anno_data["data"], device=self.sim.device)
            if lidar_data_tensor.shape[0] == 0:
                has_incomplete_lidar_data = True
                break
            lidar_data.append(lidar_data_tensor)

        if not has_incomplete_lidar_data:
            max_length = max(tensor.shape[0] for tensor in lidar_data)
            # pad tensors
            padded_tensors = []
            for tensor in lidar_data:
                padding_length = max_length - tensor.shape[0]
                if padding_length == 0:
                    padded_tensors.append(tensor)
                    continue
                padded_tensor = torch.nn.functional.pad(tensor,
                                                        (0, 0, 0, padding_length))  # Pad at the end of the first dimension
                padded_tensors.append(padded_tensor)
            lidar_data = torch.stack(padded_tensors)

            temp_x_y_positions = self._fourwis_jetauto.data.root_pos_w[:, :2]
            self.robot_positions[:, :2] = temp_x_y_positions
            # let orientation lag by three timesteps
            self.robot_positions[:, 2] = self.robot_yaw_history[:, -3]

            # calculate change in position
            dx = self.robot_positions[:, 0] - self.previous_robot_positions[:, 0]
            dy = self.robot_positions[:, 1] - self.previous_robot_positions[:, 1]

            # shift grid to keep robot centered
            cells_x = (dx / self.cfg.grid_map_resolution).long()
            cells_y = (dy / self.cfg.grid_map_resolution).long()
            for j in range(self.num_envs):
                self.occupancy_grid_maps[j] = torch.roll(self.occupancy_grid_maps[j], (-cells_x[j].item(), -cells_y[j].item()), dims=(0, 1))
                self.previous_robot_positions[j] = torch.where((cells_x[j] != 0) | (cells_y[j] != 0), self.robot_positions[j], self.previous_robot_positions[j])
                # clear newly exposed cells
                if cells_x[j] > 0:
                    self.occupancy_grid_maps[j][:cells_x[j], :] = 0
                elif cells_x[j] < 0:
                    self.occupancy_grid_maps[j][cells_x[j]:, :] = 0
                if cells_y[j] > 0:
                    self.occupancy_grid_maps[j][:, :cells_y[j]] = 0
                elif cells_y[j] < 0:
                    self.occupancy_grid_maps[j][:, cells_y[j]:] = 0

            # let orientation lag
            temp_euler = self.euler_from_quaternions(self._fourwis_jetauto.data.root_quat_w)
            self.robot_yaw_history = self.robot_yaw_history.roll(-1, dims=1)
            self.robot_yaw_history[:, -1] = temp_euler[:, 2]

            self.grid_origins_x = self.robot_positions[:, 0] - self.cfg.grid_size * self.cfg.grid_map_resolution / 2
            self.grid_origins_y = self.robot_positions[:, 1] - self.cfg.grid_size * self.cfg.grid_map_resolution / 2

            # Robot is always in middle of grid
            robot_positions_grid_coordinates_x = torch.ones(self.num_envs, device=self.sim.device) * self.cfg.grid_size // 2
            robot_positions_grid_coordinates_y = torch.ones(self.num_envs, device=self.sim.device) * self.cfg.grid_size // 2

            if lidar_data.shape[0] > 0 and lidar_data.shape[1] > 0:
                lidar_grid_coords = self.transform_to_grid_coordinates(lidar_data, self.robot_positions, self.cfg.grid_map_resolution, self.grid_origins_x, self.grid_origins_y)
                lidar_grid_coords = lidar_grid_coords.to(self.sim.device)
                lidar_grid_coords_clipped = torch.clamp(lidar_grid_coords, 0, self.cfg.grid_size - 1)

                for env in range(self.num_envs):
                    if self.current_run_observation_counter_per_env[env].item() < (self.cfg.start_discovery_reward_after_timestep - 2):
                        # if environment was just reset, dont update lidar data, because lidar might still contain "old" data
                        continue
                    start = torch.stack([robot_positions_grid_coordinates_x[env], robot_positions_grid_coordinates_y[env]],
                                        dim=0).unsqueeze(0).expand(lidar_grid_coords_clipped.shape[1], -1).to(self.sim.device)
                    end = lidar_grid_coords_clipped[env]
                    points, masks = self.vectorized_bresenham(start, end, self.cfg.grid_size)

                    # Clamp points to be within grid bounds
                    N, max_length, _ = points.shape
                    flat_points = points.view(N * max_length, 2)
                    flat_masks = masks.view(N * max_length)

                    # Calculate linear indices
                    linear_indices = flat_points[:, 0] * self.cfg.grid_size + flat_points[:, 1]

                    # Filter indices using the masks
                    valid_indices = linear_indices[flat_masks]

                    self.occupancy_grid_maps[env].view(-1).scatter_add_(0, valid_indices,
                                                                   torch.ones_like(valid_indices, dtype=torch.float) * self.cfg.l_free)


                    # create indices matrix for lidar_grid_coords_clipped[env] as well
                    linear_indices = lidar_grid_coords_clipped[env][:, 0] * self.cfg.grid_size + lidar_grid_coords_clipped[env][:, 1]
                    self.occupancy_grid_maps[env].view(-1).scatter_add_(0, linear_indices,
                                                                   torch.ones_like(linear_indices, dtype=torch.float) * self.cfg.l_occ * 2)  # *2 because bresenham added l_free to same endpoint...

                # clamp cell values of occupancy grid maps within -10 and 10
                self.occupancy_grid_maps = torch.clamp(self.occupancy_grid_maps, -10, 10)

                # if self.export_grid_maps and self.observation_counter < 200:
                if self.export_grid_maps and self.observation_counter % 50 == 0:
                    self.store_occupancy_grid_map_as_png(0)

            occupancy_grid_map_observation = self.occupancy_grid_maps.clone().view(self.num_envs, -1)

            ogm_probabilities = self.log_odds_to_probability_tensor(occupancy_grid_map_observation)

            if self.observation_counter % 3 == 0:
                self.exp_robot_orientation_history = torch.roll(self.exp_robot_orientation_history, shifts=-1, dims=1)  # Shift entries lefts
            self.exp_robot_orientation_history[:, -1] = temp_euler[:, 2] / (2 * math.pi)  # normalize to [0, 1]

            explo_agent_input = torch.cat((ogm_probabilities, depth_camera_image), dim=1)

            if self.cfg.depth_image_history_size > 0:
                explo_agent_input = torch.cat((explo_agent_input, self.depth_image_history.view(self.num_envs, -1)), dim=1)
            index_of_expo_action_history = explo_agent_input.shape[-1]
            explo_agent_input = torch.cat((explo_agent_input, self.exp_action_history), dim=1)
            explo_agent_input = torch.cat((explo_agent_input, self.exp_robot_orientation_history), dim=1)

            if self.cfg.depth_image_history_size > 0 and self.observation_counter % self.cfg.update_frequency_depth_image_steps == 0:
                self.depth_image_history = torch.roll(self.depth_image_history, shifts=-1, dims=1)  # Shift entries left
                self.depth_image_history[:, -1] = depth_camera_image

            if self.observation_counter % 5 == 0:
                self.robot_position_history_env_0.append(self.robot_positions[0].cpu().numpy())
        else:
            explo_agent_input = torch.zeros((self.num_envs, self.cfg.exp_num_observations), device=self.sim.device)
            index_of_expo_action_history = -1

        explo_agent_action, explo_agent_log_odds, explo_agent_value = self.exploration_agent.act(explo_agent_input)

        # any value inside action tensor between -0.05 and 0.05 set to zero
        mask = (explo_agent_action >= -0.05) & (explo_agent_action <= 0.05)
        explo_agent_action[mask] = 0.0
        explo_agent_action = torch.clamp(explo_agent_action, -1.0, 1.0)
        self.exp_action_history = torch.roll(self.exp_action_history, shifts=-2, dims=1)  # Shift entries left
        # self.exp_action_history[:, -2:] = explo_agent_action
        # Note: for now supplying last applied action to agents
        self.exp_action_history[:, -2:] = self.last_applied_action
        self.exp_last_action = explo_agent_action

        '''
        Data for cube detector agent
        '''
        cube_data_type = "rgb"
        temp = camera_output[cube_data_type][:, :, :, :-1].clone() / 255.0
        temp = torch.swapaxes(temp, 1, 3).clone()  # batches, channels, width, height

        predictions = self.cube_detector.forward(temp)

        # Assuming predictions is a list of dictionaries
        boxes = torch.stack(
            [pred['boxes'][0] if pred['boxes'].shape[0] > 0 else torch.zeros(4, device=self.sim.device) for pred in
             predictions])
        labels = torch.tensor([pred['scores'][0] if pred['boxes'].shape[0] > 0 else 0 for pred in predictions],
                              device=self.sim.device)

        cube_detector_results = torch.zeros((self.cfg.num_envs, self.cfg.cube_detector_neurons), device=self.sim.device)
        cube_detector_results[:, :4] = boxes
        cube_detector_results[:, 4] = labels

        self.last_cubedetector_labels = labels.bool()

        # Normalize the box coordinates
        cube_detector_results[:, [0, 2]] /= temp.shape[3]
        cube_detector_results[:, [1, 3]] /= temp.shape[2]

        if (self.cfg.cube_update_cube_detector_history_each_steps > 0 and self.current_run_observation_counter_per_env[0] % self.cfg.cube_update_cube_detector_history_each_steps == 0):
            # dont always roll results!
            self.cube_detector_history = torch.roll(self.cube_detector_history, shifts=-self.cfg.cube_detector_neurons,
                                                    dims=1)

        # last 5 values store current cube detector results
        self.cube_detector_history[:, -self.cfg.cube_detector_neurons:] = cube_detector_results

        cube_obs = torch.cat((self.cube_detector_history, self.cube_action_history), dim=1)

        cube_agent_action, cube_agent_log_odds, cube_agent_value = self.cube_agent.act(cube_obs)
        mask = (cube_agent_action >= -0.05) & (cube_agent_action <= 0.05)
        cube_agent_action[mask] = 0.0
        cube_agent_action = torch.clamp(cube_agent_action, -1.0, 1.0)
        self.cube_action_history = torch.roll(self.cube_action_history, shifts=-2, dims=1)
        # self.cube_action_history[:, -2:] = cube_agent_action
        # Note: for now supplying last applied action to agents
        self.cube_action_history[:, -2:] = self.last_applied_action
        self.cube_last_action = cube_agent_action

        # rate the actions of the other agents
        explo_rating_cube_action = self.exploration_agent.get_log_odds_for_action(cube_agent_action)
        if index_of_expo_action_history > 0:
            temp_explo_action_history = self.exp_action_history.clone()
            # replace last two values with cube_agent_action
            temp_explo_action_history[:, -2:] = cube_agent_action
            explo_agent_input[:, index_of_expo_action_history:index_of_expo_action_history+self.cfg.exp_size_action_history] = temp_explo_action_history
        explo_value_with_cube_action = self.exploration_agent.get_value(explo_agent_input)


        cube_rating_explo_action = self.cube_agent.get_log_odds_for_action(explo_agent_action)
        temp_cube_action_history = self.cube_action_history.clone()
        temp_cube_action_history[:, -2:] = explo_agent_action
        cube_obs[:, -self.cfg.cube_size_action_history:] = temp_cube_action_history
        cube_value_with_explo_action = self.cube_agent.get_value(cube_obs)

        '''
        Multi agent
        '''
        if self.observation_counter < 10:
            # print all shapes
            print("explo_agent_log_odds shape: ", explo_agent_log_odds.shape)
            print("explo_agent_value shape: ", explo_agent_value.shape)
            print("cube_agent_log_odds shape: ", cube_agent_log_odds.shape)
            print("cube_agent_value shape: ", cube_agent_value.shape)
            print("explo_rating_cube_action shape: ", explo_rating_cube_action.shape)
            print("explo_value_with_cube_action shape: ", explo_value_with_cube_action.shape)
            print("cube_rating_explo_action shape: ", cube_rating_explo_action.shape)
            print("cube_value_with_explo_action shape: ", cube_value_with_explo_action.shape)

        multi_agent_observation = torch.cat((explo_agent_log_odds, explo_agent_value, cube_agent_log_odds,
                                             cube_agent_value, explo_rating_cube_action, explo_value_with_cube_action,
                                             cube_rating_explo_action, cube_value_with_explo_action), dim=1)

        self.multi_agent_observation_history = torch.roll(self.multi_agent_observation_history, shifts=-self.cfg.single_observation_size, dims=1)
        self.multi_agent_observation_history[:, -self.cfg.single_observation_size:] = multi_agent_observation

        observation = torch.cat((self.multi_agent_observation_history, self.agent_selection_history), dim=1)

        if self.observation_counter < 50:
            print("multi_agent_observation_history shape: ", self.multi_agent_observation_history.shape)
            print("multi_agent_observation_history[0]: ", self.multi_agent_observation_history[0])
            print("observation shape: ", observation.shape)

        return {"policy": observation}

    def quaternion_to_direction(self, quaternion_tensor):
        """
        Converts a tensor of quaternions to a tensor of forward direction vectors.

        Args:
            quaternion_tensor: A tensor of shape (num_instances, 4) representing quaternions
                               for multiple instances.

        Returns:
            A tensor of shape (num_instances, 3) representing the forward direction vectors
            for each instance.
        """

        w = quaternion_tensor[:, 0]
        x = quaternion_tensor[:, 1]
        y = quaternion_tensor[:, 2]
        z = quaternion_tensor[:, 3]

        # Extract forward direction from the rotation matrix for each instance
        # Note: robot oriented along the positive x axis, with z pointing upwards
        fx = 1 - 2 * (y * y + z * z)
        fy = 2 * (x * y + w * z)
        fz = 2 * (x * z - w * y)

        return torch.stack([fx, fy, fz], dim=1)  # Stack along the second dimension to create (num_instances, 3) tensor

    def _get_rewards(self) -> torch.Tensor:

        forward_directions = self.quaternion_to_direction(self._fourwis_jetauto.data.root_quat_w)
        front_point_positions = self._fourwis_jetauto.data.root_pos_w + (0.25 / 2) * forward_directions
        self.distance_to_goal = torch.linalg.norm(self._goal.data.root_pos_w - front_point_positions, dim=1,
                                             dtype=torch.float32)

        if self.observation_counter > 1:
            # prevent wrong reset and therefore resulting reward
            reward_for_goal_reached = torch.where(
                (self.distance_to_goal < self.cfg.required_distance_to_goal) & (self.previous_distance_to_goal >= self.cfg.required_distance_to_goal),
                torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.reward_for_goal_reached,
                torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32))
            # only give distance change reward if robot can actually see the cube (based on cube detector results)
            distance_change_reward = torch.where(self.last_cubedetector_labels,
                                                 (self.previous_distance_to_goal - self.distance_to_goal) * self.normalized_reward_distance_changed,
                                                 torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32))
        else:
            reward_for_goal_reached = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
            distance_change_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)

        self.previous_distance_to_goal = self.distance_to_goal.clone()

        discovery_percentages = 100.0 - self.calculate_percentage_of_cells_equal_zero(self.occupancy_grid_maps)
        if self.observation_counter % 10 == 0:
            self.discovery_percentages_history_env_0.append(discovery_percentages[0].item())

        discovery_reward = torch.where(
            self.current_run_observation_counter_per_env > self.cfg.start_discovery_reward_after_timestep,
            (discovery_percentages - self.previous_discovery_percentages) * self.discovery_reward_factor,
            torch.zeros_like(discovery_percentages))
        # only give positive discovery reward to ignore "map jitter"
        discovery_reward = torch.where(discovery_percentages > self.previous_discovery_percentages, discovery_reward,
                                       torch.zeros_like(discovery_reward))

        # check if robot touched anything, if yes, give penalty
        non_zero_measures = (self._contact_sensor.data.force_matrix_w != 0).any(dim=-1)
        active_sensors = non_zero_measures.any(dim=(1, 2))
        # make sure self.force_matrix_penalty is zero before applying, else penalty is triggered twice
        # penalty for crashing early makes no sense, agent doesnt know what early means ( - ((self.cfg.max_steps_per_episode / self.current_run_observation_counter_per_env.float())))
        self.force_matrix_penalty = torch.where(active_sensors & (self.force_matrix_penalty == 0.0),
                                                torch.ones(self.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.penalty_for_touching_obstacle,
                                                torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32))

        rewards = {
            "force_matrix_penalty": self.force_matrix_penalty,
            "reward_for_goal_reached": reward_for_goal_reached,
            "distance_change_reward": distance_change_reward,
            "discovery_reward": discovery_reward
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        self.total_reward_sum += reward.clone()
        self.total_force_matrix_penalty += self.force_matrix_penalty
        self.total_distance_change_reward += distance_change_reward
        self.total_discovery_reward += discovery_reward

        self.previous_discovery_percentages = torch.max(discovery_percentages, self.previous_discovery_percentages)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_episodes_reached = self.episode_length_buf >= self.max_episode_length - 1

        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim.device)
        died = zeros

        died += torch.where(self.force_matrix_penalty != 0.0, ones, zeros)
        # use previous_distance_to_goal
        died += torch.where(self.previous_distance_to_goal < self.cfg.required_distance_to_goal, ones, zeros)

        return died, max_episodes_reached

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._fourwis_jetauto._ALL_INDICES

        super()._reset_idx(env_ids)

        # if self.observation_counter > 2000:
        #     self.export_grid_maps = True

        if self.export_grid_maps and 0 in env_ids:
            self.store_occupancy_grid_map_as_png(0)
            self.store_discovery_percentages_history_env_0_as_png()
            self.store_robot_positions_history_env_0_as_png()
            self.discovery_percentages_history_env_0 = []
            self.robot_position_history_env_0 = []

        num_resets = len(env_ids)
        self.reset_counter += num_resets

        temp = torch.where(self.previous_distance_to_goal[env_ids] < self.cfg.required_distance_to_goal, torch.ones_like(self.previous_distance_to_goal[env_ids]), torch.zeros_like(self.previous_distance_to_goal[env_ids]))
        self.total_goal_reached_counter += (temp > 0).sum().item()
        rounded_reward_sums_per_env = [round(val.item(), 2) for val in self.total_reward_sum[env_ids]]
        rounded_total_force_matrix_penalty_per_env = [round(val.item(), 2) for val in self.total_force_matrix_penalty[env_ids]]
        rounded_distance_to_goal = [round(val.item(), 2) for val in self.distance_to_goal[env_ids]]
        rounded_total_distance_change_reward = [round(val.item(), 2) for val in self.total_distance_change_reward[env_ids]]
        rounded_total_discovery_reward = [round(val.item(), 3) for val in self.total_discovery_reward[env_ids]]
        count_selected_which_agent_as_list = [val.tolist() for val in self.count_selected_which_agent[env_ids]]

        print(f"------resetting environments: {env_ids} - total_goal_reached_counter: {self.total_goal_reached_counter} - reset_counter: {self.reset_counter} - obs counter: {self.current_run_observation_counter_per_env[env_ids]}")
        print(f"total_reward_sum: {rounded_reward_sums_per_env} - total_force_matrix_penalty: {rounded_total_force_matrix_penalty_per_env} "
              f"- rounded_distance_to_goal: {rounded_distance_to_goal} - total_distance_change_reward: {rounded_total_distance_change_reward}")
        print(f"rounded_total_discovery_reward: {rounded_total_discovery_reward} - count_selected_which_agent [exp, cube]: {count_selected_which_agent_as_list}")

        joint_pos = self._fourwis_jetauto.data.default_joint_pos[env_ids]
        joint_vel = self._fourwis_jetauto.data.default_joint_vel[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        fw_default_root_state = self._fourwis_jetauto.data.default_root_state[env_ids]
        # randomize z orientation into -1, 1
        # fw_default_root_state[:, 6] = torch.rand((len(env_ids)), device=self.sim.device) * 2 - 1
        fw_default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._fourwis_jetauto.write_root_pose_to_sim(fw_default_root_state[:, :7], env_ids)
        self._fourwis_jetauto.write_root_velocity_to_sim(fw_default_root_state[:, 7:], env_ids)
        self._fourwis_jetauto.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # set goal position
        # random_cube_positions = torch.tensor([[5.0, 2.0, 0.0], [5.5, 5.0, 0.0], [2.0, 2.0, 0.0], [2.0, 5.0, 0.0]], device=self.sim.device)
        random_cube_positions = torch.tensor([[5.0, 1.0, 0.0], [5.5, 4.0, 0.0], [5.5, 2.0, 0.0]],
                                             device=self.sim.device)
        random_int = torch.randint(0, len(random_cube_positions), (len(env_ids),), device=self.sim.device)
        goal_position = self.scene.env_origins[env_ids] + random_cube_positions[random_int]
        default_root_state = self.goal_root_state[env_ids]
        default_root_state[:, :3] = goal_position
        self._goal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        self.multi_agent_observation_history[env_ids] = torch.zeros(num_resets, self.cfg.ma_observation_history_size * self.cfg.single_observation_size, device=self.sim.device, dtype=torch.float32)

        temp_x_y_position =  self._fourwis_jetauto.data.root_pos_w[env_ids, :2]
        self.robot_positions[env_ids, :2] = temp_x_y_position
        temp_euler = self.euler_from_quaternions(self._fourwis_jetauto.data.root_quat_w[env_ids])
        self.robot_positions[env_ids, 2] = temp_euler[:, 2]
        self.previous_robot_positions[env_ids] = self.robot_positions[env_ids].clone()
        self.robot_yaw_history[env_ids] = torch.zeros((num_resets, 3), device=self.sim.device)
        self.robot_yaw_history[env_ids, 2] = temp_euler[:, 2]

        self.grid_origins_x[env_ids] = torch.floor(temp_x_y_position[:, 0] - self.cfg.grid_size * self.cfg.grid_map_resolution / 2)
        self.grid_origins_y[env_ids] = torch.floor(temp_x_y_position[:, 1] - self.cfg.grid_size * self.cfg.grid_map_resolution / 2)
        self.original_grid_origins[env_ids] = torch.stack((self.grid_origins_x[env_ids], self.grid_origins_y[env_ids]), dim=1)

        self.occupancy_grid_maps[env_ids] = torch.zeros((num_resets, self.cfg.grid_size, self.cfg.grid_size), device=self.sim.device)

        self.exp_action_history[env_ids] = torch.zeros(num_resets, self.cfg.exp_size_action_history, device=self.sim.device,
                                                   dtype=torch.float32)
        self.exp_robot_orientation_history[env_ids] = torch.zeros(num_resets, self.cfg.exp_size_robot_orientation_history, device=self.sim.device,
                                                   dtype=torch.float32)

        self.last_cubedetector_labels[env_ids] = torch.zeros(num_resets, device=self.sim.device, dtype=torch.bool)

        self.distance_to_goal[env_ids] = torch.linalg.norm(default_root_state - fw_default_root_state, dim=1,
                                                                    dtype=torch.float32)
        self.previous_distance_to_goal[env_ids] = self.distance_to_goal[env_ids].clone()

        self.normalized_reward_distance_changed[env_ids] = self.cfg.reward_distance_change_factor / (self.distance_to_goal[env_ids] - self.cfg.required_distance_to_goal)

        self.cube_last_action[env_ids] = torch.zeros(num_resets, self.num_actions, device=self.sim.device)
        self.exp_last_action[env_ids] = torch.zeros(num_resets, self.num_actions, device=self.sim.device)
        self.last_applied_action[env_ids] = torch.zeros(num_resets, self.num_actions, device=self.sim.device)

        self.count_selected_which_agent[env_ids] = torch.zeros(num_resets, 2, device=self.sim.device)
        self.agent_selection_history[env_ids] = torch.zeros(num_resets, self.cfg.agent_selection_history_size, device=self.sim.device)

        self.cube_action_history[env_ids] = torch.zeros(num_resets, self.cfg.cube_size_action_history, device=self.sim.device, dtype=torch.float32)
        self.cube_detector_history[env_ids] = torch.zeros(num_resets, self.cfg.cube_detector_neurons * self.cfg.cube_size_cube_detector_history, device=self.sim.device, dtype=torch.float32)

        if self.cfg.depth_image_history_size > 0:
            self.depth_image_history[env_ids] = torch.zeros(num_resets, self.cfg.depth_image_history_size, self.cfg.depth_image_slice_size * self.cfg.scaled_image_width, device=self.sim.device,
                                                       dtype=torch.float32)

        self.force_matrix_penalty[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.current_run_observation_counter_per_env[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.total_reward_sum[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_force_matrix_penalty[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_distance_change_reward[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_discovery_reward[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.previous_discovery_percentages[env_ids] = torch.zeros(num_resets, device=self.sim.device)


    def euler_from_quaternions(self, quaternion):
        """
        Converts a batch of quaternions to euler roll, pitch, yaw
        """
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]

        # Not needed at the moment
        # sinr_cosp = 2 * (w * x + y * z)
        # cosr_cosp = 1 - 2 * (x * x + y * y)
        # roll = torch.arctan2(sinr_cosp, cosr_cosp)
        #
        # sinp = 2 * (w * y - z * x)
        # pitch = torch.asin(sinp)
        roll = torch.zeros(quaternion.shape[0], device=quaternion.device)
        pitch = torch.zeros(quaternion.shape[0], device=quaternion.device)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.arctan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=1)

    def transform_to_grid_coordinates(self, lidar_points, robot_positions, resolution, grid_origin_x, grid_origin_y):
        robot_x, robot_y, robot_theta = robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2]
        # robot_theta = -robot_theta

        cos_theta = torch.cos(robot_theta)
        sin_theta = torch.sin(robot_theta)
        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2).to(lidar_points.device)

        # filter out points which are to far away
        # lidar_points = lidar_points[np.linalg.norm(lidar_points[:, :2], axis=1) < 10]
        # filter out points which are too close
        # lidar_points = lidar_points[np.linalg.norm(lidar_points[:, :2], axis=1) > 0.1]
        # filter out points where z is equal to ground
        # lidar_points = lidar_points[lidar_points[:, 2] > 0.1]

        points_xy = lidar_points[:, :, :2]

        # rotate points
        points_xy = torch.einsum('bji,bni->bnj', rotation_matrices, points_xy)

        # translate points
        robot_xy = torch.stack([robot_x, robot_y], dim=1).unsqueeze(1).to(lidar_points.device)
        points_xy += robot_xy

        # convert to grid coordinates
        grid_origins = torch.stack([grid_origin_x, grid_origin_y], dim=-1).unsqueeze(1).to(lidar_points.device)
        points_xy = torch.floor((points_xy - grid_origins) / resolution).long()

        # clip coordinates
        # points_xy = np.clip(points_xy, 0, grid_size - 1)

        # add z coordinate
        # points_xyz = np.hstack((points_xy, lidar_points[:, 2].reshape(-1, 1)))

        return points_xy

    def vectorized_bresenham(self, start, end, max_length):
        """
        Vectorized Bresenham's line algorithm for multiple start and end points.

        Args:
        - start: tensor of shape (N, 2) containing start points (x, y)
        - end: tensor of shape (N, 2) containing end points (x, y)
        - max_length: maximum number of points to generate for each line

        Returns:
        - points: tensor of shape (N, max_length, 2) containing points along each line
        - masks: tensor of shape (N, max_length) indicating valid points
        """
        device = start.device
        N = start.shape[0]

        # Initialize points and masks
        points = torch.zeros((N, max_length, 2), dtype=torch.long, device=device)
        masks = torch.zeros((N, max_length), dtype=torch.bool, device=device)

        # Calculate deltas and steps
        delta = (end - start).abs()
        step = (end - start).sign()

        # Initialize error and current position
        err = delta[:, 0] - delta[:, 1]
        current = start.clone()

        finished = torch.zeros(N, dtype=torch.bool, device=device)

        # Generate points
        for i in range(max_length):
            points[:, i] = current
            masks[:, i] = ~finished  # Only mask unfinished lines

            # Check if we've reached the end points
            finished |= torch.all(current == end, dim=1)
            if torch.all(finished):
                break

            # Update positions ONLY for unfinished lines
            e2 = 2 * err
            x_step = (e2 >= -delta[:, 1]) & ~finished
            y_step = (e2 <= delta[:, 0]) & ~finished

            current[:, 0] += step[:, 0] * x_step
            err -= delta[:, 1] * x_step
            current[:, 1] += step[:, 1] * y_step
            err += delta[:, 0] * y_step

        return points, masks

    def log_odds_to_probability(self, x):
        # grid = np.maximum(grid, -100)
        # return np.where(x >= 0,
        #                 1 - 1 / (1 + np.exp(x)),
        #                 np.exp(-x) / (1 + np.exp(-x)))
        x = np.clip(x, -100, 100)
        return 1 - 1 / (1 + np.exp(x))

    def log_odds_to_probability_tensor(self, x):
        x = torch.clamp(x, -100, 100)
        return 1 - 1 / (1 + torch.exp(x))


    def calculate_percentage_of_cells_equal_zero(self, occupancy_grid_maps):
        # more efficient
        zero_counts = (occupancy_grid_maps == 0).sum(dim=(1, 2)).float()
        total_cells = occupancy_grid_maps.shape[1] * occupancy_grid_maps.shape[2]
        zero_percentages = (zero_counts / total_cells) * 100

        return zero_percentages