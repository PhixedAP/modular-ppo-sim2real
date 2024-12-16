from __future__ import annotations
import getpass
import torch
import gymnasium as gym
import numpy as np
from collections.abc import Sequence

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, Camera, CameraCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg

from omni.isaac.lab.utils import configclass
from torchvision.utils import save_image, make_grid

from .fourwis_jetauto import FourwisJetauto

import math
import time
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.sim.utils import clone, safe_set_attribute_on_usd_prim
# from pxr import Gf
import random
import cv2

import sys
import os

from ..utilitiesisaaclab import mecanum_direction, mecanum_controller
import socket


@configclass
class CCCEnvCBFRCNNCfgv2(DirectRLEnvCfg):
    # env

    evaluation_mode = True  # used for creating evaluatin prints, used for thesis

    decimation = 4  # environment steps till new action
    sim_dt = 1 / 60.0

    episode_length_factor = 1.0
    calc_episode_length_s = sim_dt * 600 * episode_length_factor * decimation / 2
    episode_length_s = calc_episode_length_s if evaluation_mode else calc_episode_length_s  # steps: math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
    action_scale = 1.0
    num_actions = 2
    max_steps_per_episode = 600 * episode_length_factor

    num_channels = 3


    cube_detector_neurons = 5

    # Note: for resnet use square image
    scaled_image_width = 320
    scaled_image_height = 240

    size_action_history = 8  # last 4 actions

    size_cube_detector_history = 6
    update_cube_detector_history_each_steps = 2

    num_observations = cube_detector_neurons * size_cube_detector_history + size_action_history  #  + size_robot_history  # 1000 for resnet
    num_states = 0

    num_envs = 3 if socket.gethostname() == "phil-SCHENKER-MEDIA-E23" else 32 # 32 8  # 12 seems to be max with images on my laptop!, 8 for scenery

    required_distance_to_goal = 0.25

    reward_for_goal_reached = 3.0
    distance_change_reward = 2.0
    reward_for_driving_forward_per_step = 0.005  # 2.0 / max_steps_per_episode  # 1.5
    penalty_per_action_change_over_threshold = -0.005 #-0.04
    threshold_for_penalty_action_change = 0.3



    # simulation
    # decimation: make sure that not multiple render calls will happen during each environment step
    # use_fabric false: https://github.com/isaac-sim/IsaacLab/issues/745 , should let tiled camera stick to robot
    #   but results in robot not moving...
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # robot
    model_name = "014_4wis_added_front_bumper.usd"  # 013_4wis_moved_camera_to_account_for_increased_chassis.usd
    # own_model_path = "omniverse://localhost/Users/phil/own_models/"
    own_model_path = "/home/phil/Documents/"
    velocity_joint_names = ["front_right_wheel_joint", "front_left_wheel_joint",
                            "rear_right_wheel_joint", "rear_left_wheel_joint"]
    position_joint_names = ["front_right_steering_joint", "front_left_steering_joint",
                            "rear_right_steering_joint", "rear_left_steering_joint"]

    # Position Control: For position controlled joints, set a high stiffness and relatively low or zero damping.
    # Velocity Control: For velocity controller joints, set a high damping and zero stiffness.

    fourwis_jetauto: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/fourwis",
        spawn=sim_utils.UsdFileCfg(
            usd_path=own_model_path + model_name,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1000.0,  # high value prevents robot from "flying" away. Or maybe not?...
                disable_gravity=False,
                # enable_gyroscopic_forces=True,
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
            pos=(3.5, 3.5, 0.2),
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

    use_tiled_camera = False  # supposedly more efficient but it does not "stick" to the robot, but stays where it is...

    # camera
    if use_tiled_camera:
        tiled_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/fourwis/chassis/tiledcamera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.12, 0.0, 2.4), rot=(0.59966, 0.37471, -0.37471, -0.59966),
                                            convention="world"),
            data_types=["rgb"],
            # spawn=None,
            # None: assumes asset exists already  # sim_utils.PinholeCameraCfg(# focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
            spawn=sim_utils.PinholeCameraCfg(focal_length=3.7, focus_distance=0.0,
                                             horizontal_aperture=6.4, clipping_range=(0.1, 3.5), lock_camera=False),
            width=scaled_image_width,
            height=scaled_image_height,
            return_latest_camera_pose=True
        )
    else:
        camera: CameraCfg = CameraCfg(
            prim_path="/World/envs/env_.*/fourwis/chassis/camera",
            data_types=["rgb"],
            spawn=None,
            width=scaled_image_width,
            height=scaled_image_height,
            update_period=sim_dt * decimation,
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

    light_randomization_each_steps = 500

    light_cfg: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        intensity=2000.0, color=(0.75, 0.75, 0.75)
    )

    distant_light_cfg: sim_utils.DistantLightCfg = sim_utils.DistantLightCfg(
        color=(0.9, 0.9, 0.9), intensity=2500.0
    )

    # elements for scenery
    if getpass.getuser() == 'phil':
        # own_omniverse_scenes = "/home/phil/university/thesis/isaac/OmniIsaacGymEnvs/omniisaacgymenvs/scenes/"
        own_omniverse_scenes = "/home/phil/Documents/scenes/"
    else:
        own_omniverse_scenes = "/home/lab/phil/OmniIsaacGymEnvs/omniisaacgymenvs/scenes/"

    add_scenery = True

    crestwood_sofa_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=own_omniverse_scenes + "Crestwood_Sofa.usd",
        scale=np.array([0.01, 0.01, 0.01]),
    )

    wall_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=own_omniverse_scenes + "Geo_M2_BaseWallSide.usd",
        scale=np.array([1.0, 1.0, 1.0]),
    )

    dining_table_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=own_omniverse_scenes + "Dellwood_RoundDiningTable.usd",
        scale=np.array([0.01, 0.01, 0.01]),
    )

    round_table_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=own_omniverse_scenes + "Roxana_RoundEndTable.usd",
        scale=np.array([0.01, 0.01, 0.01]),
    )

    chair_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=own_omniverse_scenes + "Chair_Array.usd",
        scale=np.array([0.01, 0.01, 0.01]),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=7.0, replicate_physics=True)






import torchvision
import torchvision.transforms.functional as F

class RedCubeDetectorFasterRCNN:
    model_path = "/home/phil/Documents/"

    def __init__(self, device):
        self.device = device
        self.model = torch.load(self.model_path + "fasterrcnn_cube_detector_mobilenet.pth").to(self.device)
        self.model.eval()

        # Note: no transforms needed with FasterRCNN
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        # self.transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Lambda(lambda t: F.resize(t, 224)),  # Resize the tensor
        #     torchvision.transforms.Lambda(lambda t: F.center_crop(t, 224)),  # Center crop the tensor
        #     torchvision.transforms.Normalize(mean=MEAN, std=STD)  # Normalize the tensor
        # ])

    def forward(self, x):
        # transform image
        # x = self.transforms(x).to(self.device)  # requires B, C, H, W
        return self.model(x)


class CCCEnvCBFRCNNCv2(DirectRLEnv):
    cfg: CCCEnvCBFRCNNCfgv2
    if getpass.getuser() == 'phil':
        image_path = "/home/phil/university/thesis/data/images/"
    else:
        image_path = "/home/lab/phil/OmniIsaacGymEnvs/omniisaacgymenvs/data/images/"
    observation_counter = 0
    randomization_counter = 0
    export_images = False
    images_saved_counter = 0

    def __init__(self, cfg: CCCEnvCBFRCNNCfgv2, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Initializing CubeChaserCameraContinuousEnv")

        self.action_scale = self.cfg.action_scale
        self.use_tiled_camera = self.cfg.use_tiled_camera

        self.joint_pos = self._fourwis_jetauto.data.joint_pos
        self.joint_vel = self._fourwis_jetauto.data.joint_vel

        # Note: order gets messed up if not collecting one by one!
        self._fourwis_velocity_joint_indexes = []
        self._fourwis_steering_joint_indexes = []
        for vel_joint_name in self.cfg.velocity_joint_names:
            temp, _ = self._fourwis_jetauto.find_joints(vel_joint_name)
            self._fourwis_velocity_joint_indexes.append(temp[0])
        for pos_joint_name in self.cfg.position_joint_names:
            temp, _ = self._fourwis_jetauto.find_joints(pos_joint_name)
            self._fourwis_steering_joint_indexes.append(temp[0])
        print("_fourwis_velocity_joint_indexes: ", self._fourwis_velocity_joint_indexes)
        print("_fourwis_steering_joint_indexes: ", self._fourwis_steering_joint_indexes)


        self.cube_detector = RedCubeDetectorFasterRCNN(self.sim.device)
        self.mecanum_controller = mecanum_controller.MecanumController(self.cfg.num_envs, self.sim.device)

        self.goal_root_state = self._goal.data.default_root_state

        self.start_time_episode = torch.full((self.num_envs,), time.time(), dtype=torch.float64, device=self.sim.device)

        self.previous_distance_to_goal = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.initial_distance_to_goal = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_reward_sum = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_penalty_for_too_many_action_changes = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_reward_for_driving_forward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_goal_reached_counter = torch.zeros(1, device=self.sim.device, dtype=torch.int32)

        # evaluation measures for thesis
        self.total_episode_counter = -self.cfg.num_envs  # subtract initial resets
        self.total_task_successes = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.summed_up_initial_distance_to_goal = torch.zeros(1, device=self.sim.device, dtype=torch.float32)
        self.total_timesteps = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.total_timesteps_if_goal_reached = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.total_action_changes_above_threshold_025 = torch.zeros(1, device=self.sim.device, dtype=torch.float32)
        self.total_action_changes_above_threshold_05 = torch.zeros(1, device=self.sim.device, dtype=torch.float32)
        self.total_action_changes_above_threshold_075 = torch.zeros(1, device=self.sim.device, dtype=torch.float32)

        self.observation_counter_per_env = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.int32)

        self.action_history = torch.zeros(self.cfg.num_envs, self.cfg.size_action_history, device=self.sim.device,
                                          dtype=torch.float32)
        # self.robot_history = torch.zeros(self.cfg.num_envs, self.cfg.size_robot_history, device=self.sim.device,
        #                                  dtype=torch.float32)
        self.cube_detector_history = torch.zeros(self.cfg.num_envs, self.cfg.size_cube_detector_history * self.cfg.cube_detector_neurons,
                                                    device=self.sim.device, dtype=torch.float32)

        # reward if reaching goal is approx 6, max of 600 steps, so -0.005 per step
        self.penalty_per_step = torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32) * -0.005
        print("self.goal_root_state: ", self.goal_root_state)
        print("test: ", self.goal_root_state[:, :3])

        print("max_episode_length: ", self.max_episode_length)


        # position of some body parts
        print("joint names: ", self._fourwis_jetauto.data.joint_names)
        print("body_names: ", self._fourwis_jetauto.data.body_names)

        self.front_bumper_index = self._fourwis_jetauto.find_bodies("front_bumper")[0][0]
        print("front_bumper_index: ", self.front_bumper_index)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            for key in [
                "distance_change",
                "reward_for_goal_reached",
                "penalty_per_step",
                "penalty_for_too_many_action_changes",
                "reward_for_driving_forward"
            ]
        }

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()
        # used to be dtype=np.uint8, but changed for now
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            # shape=(self.cfg.scaled_image_height, self.cfg.scaled_image_width, self.cfg.num_channels)
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

        if self.cfg.use_tiled_camera:
            print("Using tiled camera")
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        else:
            self._camera = Camera(self.cfg.camera)

        self._goal = RigidObject(self.cfg.goal_cube)

        # x, y, rotation
        wall_positions = [[3.5, 0.0, 0.0, 1.0], [7.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70711, 0.70711],
                          [0.0, 3.5, 0.70711, 0.70711], [0.0, 7.0, 1.0, 0.0], [3.5, 7.0, 1.0, 0.0],
                          [7.0, 7.0, 0.70711, -0.70711], [7.0, 3.5, 0.70711, -0.70711]]
        self._walls = []
        for index, wall_position in enumerate(wall_positions):
            _wall = self.cfg.wall_cfg.func(
                prim_path="/World/envs/env_.*/Geo_M2_BaseWallSide_{}".format(index),
                cfg=self.cfg.wall_cfg,
                orientation=np.array([wall_position[2], 0.0, 0.0, wall_position[3]]),
                translation=(wall_position[0], wall_position[1], 0.0),
            )
            self._walls.append(_wall)

        # Scenery elements
        if self.cfg.add_scenery:
            self._crestwood_sofa = self.cfg.crestwood_sofa_cfg.func(
                prim_path="/World/envs/env_.*/Crestwood_Sofa",
                cfg=self.cfg.crestwood_sofa_cfg,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                translation=(3.0, 0.7, 0.0),
            )

            self._dining_table = self.cfg.dining_table_cfg.func(
                prim_path="/World/envs/env_.*/Dellwood_RoundDiningTable",
                cfg=self.cfg.dining_table_cfg,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                translation=(0.8, 3.5, 0.0),
            )

            self._round_table = self.cfg.round_table_cfg.func(
                prim_path="/World/envs/env_.*/Dellwood_RoundTable",
                cfg=self.cfg.round_table_cfg,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                translation=(6.0, 2.5, 0.0),
            )

            chair_positions = [[6.3, 6.3, 0.0], [6.3, 1.0, 0.0], [4.0, 6.3, 0.0], [1.0, 6.3, 0.0]]
            self._chairs = []
            for index, chair_position in enumerate(chair_positions):
                _chair = self.cfg.chair_cfg.func(
                    prim_path="/World/envs/env_.*/Chair_Array_{}".format(index),
                    cfg=self.cfg.chair_cfg,
                    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                    translation=np.array(chair_position),
                )
                self._chairs.append(_chair)

        self.gnd_plane = spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5)))
        # self.gnd_plane.cfg.color = (0.5, 0.5, 0.5)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["fourwis"] = self._fourwis_jetauto

        if self.cfg.use_tiled_camera:
            self.scene.sensors["tiled_camera"] = self._tiled_camera
        else:
            self.scene.sensors["camera"] = self._camera

        # add lights
        self.light = self.cfg.light_cfg.func("/World/Light", self.cfg.light_cfg)
        self.distant_light = self.cfg.distant_light_cfg.func("/World/DistantLight", self.cfg.distant_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # action will be -1.0 to 1.0
        self.actions = self.action_scale * actions.clone()
        # print("actions for env 0: ", self.actions[0])
        # clip actions to -1.0 to 1.0
        self.actions = torch.clamp(self.actions, -1.0, 1.0)

        # any value inside action tensor between -0.05 and 0.05 set to zero
        mask = (self.actions >= -0.05) & (self.actions <= 0.05)
        self.actions[mask] = 0.0

        self.action_history = torch.roll(self.action_history, shifts=-2, dims=1)  # Shift entries left
        self.action_history[:, -2:] = self.actions


    def _apply_action(self) -> None:
        if self.num_actions == 2:
            velocities, positions = self.mecanum_controller.continuous_actions_into_velocities_and_positions(
                self.actions)
            if False and self.observation_counter < 2000:
                print("actions: ", self.actions[0], "positions: ", positions[0], "velocities: ", velocities[0])
        elif self.num_actions == 4:
            velocities, positions = self.mecanum_controller.actions_softmax_into_velocities_and_positions(self.actions,
                                                                                                          self.observation_counter)
        else:
            raise ValueError("Invalid number of actions")
        # velocity and position for env 0
        # print("velocities for env 0: ", velocities[0])
        # print("positions for env 0: ", positions[0])
        self._fourwis_jetauto.set_joint_position_target(positions,
                                                        joint_ids=self._fourwis_steering_joint_indexes)
        self._fourwis_jetauto.set_joint_velocity_target(velocities,
                                                        joint_ids=self._fourwis_velocity_joint_indexes)
        pass

    def _get_observations(self) -> dict:
        self.observation_counter_per_env += 1
        if self.observation_counter < 5:
            print("get_observations")
        if self.observation_counter < 5000000:
            self.observation_counter += 1

        data_type = "rgb"
        temp = self._camera.data.output[data_type][:, :, :, :-1].clone() / 255.0  # batches, height, width, channels
        temp = torch.swapaxes(temp, 1, 3).clone()  # batches, channels, width, height
        # temp = torch.swapaxes(temp, 2, 3).clone()  # batches, channels, height, width
        if self.observation_counter < 5:
            print("got camera output, shape: ", temp.shape)
        # save_image(make_grid(temp, nrow=2),
        #            self.image_path + 'f_cam_finetuned_export_' + str(self.observation_counter) + '.png')

        # look like this for 2 environments: [{'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0')}, {'boxes': tensor([[183.0554, 230.1332, 201.2283, 245.8121]], device='cuda:0', size=(0, 4)), 'labels': tensor([1], device='cuda:0', dtype=torch.int64), 'scores': tensor([0.9993], device='cuda:0')}]
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

        # Normalize the box coordinates
        cube_detector_results[:, [0, 2]] /= temp.shape[3]
        cube_detector_results[:, [1, 3]] /= temp.shape[2]

        # cube_detector_output = torch.cat((boxPreds, classPreds), dim=1)
        if (self.cfg.update_cube_detector_history_each_steps > 0 and
                self.observation_counter_per_env[0] % self.cfg.update_cube_detector_history_each_steps == 0):
            # dont always roll results!
            self.cube_detector_history = torch.roll(self.cube_detector_history, shifts=-self.cfg.cube_detector_neurons,
                                                    dims=1)

            # just store scores in history
            # self.cube_detector_history = torch.roll(self.cube_detector_history, shifts=-1, dims=1)
            # self.cube_detector_history[:, -6] = cube_detector_results[:, 4]

        # last 5 values store current cube detector results
        self.cube_detector_history[:, -self.cfg.cube_detector_neurons:] = cube_detector_results

        action_history_normalized = (self.action_history.clone() + 1.0) / 2.0


        res = torch.cat((self.cube_detector_history, action_history_normalized), dim=1)

        # Note: deactivated robot history for now! real robot has angular velocity and linear acceleration, here
        # we seem to get angular and linear velocity.
        # self.robot_history = torch.roll(self.robot_history, shifts=-6, dims=1)
        # self.robot_history[:, -6:] = self._fourwis_jetauto.data.root_vel_w

        # res = torch.cat((res, self.robot_history), dim=1)

        if False and self.observation_counter < 50:
            print("predictions: ", predictions)
            print("cube_detector_results: ", cube_detector_results)
            print("temp shape: ", temp.shape)
            print("cube_detector_output shape: ", cube_detector_results.shape)
            print("self.action_history shape: ", self.action_history.shape)
            # print("self.robot_history shape: ", self.robot_history.shape)
            print("cube detector history shape: ", self.cube_detector_history.shape)
            print("res shape: ", res.shape)
            print("res: ", res)

        observations = {"policy": res}

        if self.export_images and self.observation_counter % 5 == 0 and self.images_saved_counter < 50:  # 50000
            print("checking image..., images_saved_counter: ", self.images_saved_counter)
            temp_orig = temp[self.images_saved_counter % self.cfg.num_envs].clone()
            print("shape temp_orig: ", temp_orig.shape)

            # change from c,w,h to w,h,c
            temp_img = torch.swapaxes(temp_orig, 0, 2).clone()
            temp_img = torch.swapaxes(temp_img, 0, 1).clone()
            print("temp_img shape: ", temp_img.shape)

            boxPreds = predictions[self.images_saved_counter % self.cfg.num_envs]['boxes']
            # print("boxPreds: ", boxPreds)
            if boxPreds.shape[0] > 0:
                print("boxPreds: ", boxPreds)
                startX = int(boxPreds[0][0].item())
                startY = int(boxPreds[0][1].item())
                endX = int(boxPreds[0][2].item())
                endY = int(boxPreds[0][3].item())
                image_cv2 = temp_img.cpu().numpy()
                image = cv2.cvtColor(image_cv2 * 255, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.imwrite(self.image_path + 'cartpole_export_' + str(self.observation_counter) + '_b.png', image)
                self.images_saved_counter += 1


        # Note: only need to randomize light while collecting data for the cube detector
        if False and self.total_goal_reached_counter > 50:  # 500
            self.randomization_counter += 1
            self.randomization_counter = self.randomization_counter % 10000  # prevent of getting to big
            if self.randomization_counter % self.cfg.light_randomization_each_steps == 0:
                random_color = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
                self.light.GetAttribute("inputs:color").Set(random_color)
                self.distant_light.GetAttribute("inputs:color").Set(random_color)

        # randomizing textures: https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_isaac_randomizers.html

        return observations

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
        front_bumper_positions = self._fourwis_jetauto.data.body_pos_w[:, self.front_bumper_index]
        distance_to_goal = torch.linalg.norm(self._goal.data.root_pos_w - front_bumper_positions, dim=1,
                                                dtype=torch.float32)



        # Note: dont go below 0.2!!!! even if right at the cube the distance is 0.11 something
        if self.observation_counter > 1:
            # prevent wrong reset and therefore resulting reward
            reward_for_goal_reached = torch.where(
                distance_to_goal < self.cfg.required_distance_to_goal,
                torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.reward_for_goal_reached,
                torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32))
            distance_change_reward = (self.previous_distance_to_goal - distance_to_goal) * self.cfg.distance_change_reward
        else:
            reward_for_goal_reached = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
            distance_change_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        # print("reward_for_goal_reached: ", reward_for_goal_reached)
        # print("distance_change_reward: ", distance_change_reward)

        # make sure distance_to_goal does not exceed -/+50.0, sometimes a bug happened and distance was a huge number
        distance_to_goal = torch.clamp(distance_to_goal, -50.0, 50.0)
        self.previous_distance_to_goal = distance_to_goal

        penalty_action_changes = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        if self.observation_counter > 1:
            # prevent wrong reset and therefore resulting reward
            penalty_action_changes = torch.where(
                torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) > self.cfg.threshold_for_penalty_action_change,
                # torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.penalty_per_action_change_over_threshold,  # fixed penalty if too much change
                (torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) - self.cfg.threshold_for_penalty_action_change) * self.cfg.penalty_per_action_change_over_threshold,  # penalty based on size of change
                torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32))
            if self.observation_counter < 50:
                print("action history: ", self.action_history[:, -4:])
                print("penalty_action_changes: ", penalty_action_changes)

            if self.cfg.evaluation_mode:
                self.total_action_changes_above_threshold_025 += torch.sum(
                    torch.where(torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) > 0.25,
                                torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32),
                                torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)))

                self.total_action_changes_above_threshold_05 += torch.sum(
                    torch.where(
                        torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) > 0.5,
                        torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32),
                        torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)))

                self.total_action_changes_above_threshold_075 += torch.sum(
                    torch.where(
                        torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) > 0.75,
                        torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32),
                        torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)))

        # reward forward, penalty backwards
        reward_for_driving_forward = (self.actions[:, 0]) * self.cfg.reward_for_driving_forward_per_step
        # double penalty for driving backwards
        reward_for_driving_forward = torch.where(reward_for_driving_forward < 0.0, reward_for_driving_forward*2, reward_for_driving_forward)


        rewards = {
            "distance_change": distance_change_reward,
            "reward_for_goal_reached": reward_for_goal_reached,
            "penalty_per_step": self.penalty_per_step.clone(),
            "penalty_for_too_many_action_changes": penalty_action_changes,
            "reward_for_driving_forward": reward_for_driving_forward
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # if self.observation_counter % 500 == 0 or self.observation_counter % 500 == 1:
        #     print("distance_to_goal: ", distance_to_goal)
        #     print("reward: ", reward)

        self.total_reward_sum += reward.clone()
        self.total_penalty_for_too_many_action_changes += penalty_action_changes
        self.total_reward_for_driving_forward += reward_for_driving_forward
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_episodes_reached = self.episode_length_buf >= self.max_episode_length - 1
        zeros = torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.sim.device)
        ones = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=self.sim.device)
        died = zeros

        died += torch.where(self.previous_distance_to_goal < self.cfg.required_distance_to_goal, ones, zeros)
        # abort early if driving wrong direction!
        died += torch.where(self.previous_distance_to_goal > (self.initial_distance_to_goal * 2), ones, zeros)

        return died, max_episodes_reached

    def adjust_alpha_tensor(self, alpha_tensor, threshold=0.3):
        # we dont want any values to close to pi/2 to make sure the robot just doesnt drive in one direction
        pi_over_2 = torch.tensor(math.pi / 2, device=alpha_tensor.device)
        distance_to_pi_over_2 = torch.abs(alpha_tensor - pi_over_2)

        # Find values too close and replace with alternatives
        too_close_mask = distance_to_pi_over_2 < threshold
        alternative_values = torch.rand(too_close_mask.sum(),
                                        device=alpha_tensor.device) * math.pi  # Random from 0 to pi

        # Create a copy and replace only the values that need adjusting
        adjusted_tensor = alpha_tensor.clone()
        adjusted_tensor[too_close_mask] = alternative_values

        return adjusted_tensor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._fourwis_jetauto._ALL_INDICES
        super()._reset_idx(env_ids)
        print("Resetting index: ", env_ids)
        if self.observation_counter > 1:
            self.total_goal_reached_counter += torch.sum(torch.where(
                self.previous_distance_to_goal[env_ids] < self.cfg.required_distance_to_goal,
                torch.ones(len(env_ids), dtype=torch.bool, device=self.sim.device),
                torch.zeros(len(env_ids), dtype=torch.bool, device=self.sim.device)))

        joint_pos = self._fourwis_jetauto.data.default_joint_pos[env_ids]
        joint_vel = self._fourwis_jetauto.data.default_joint_vel[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        fw_default_root_state = self._fourwis_jetauto.data.default_root_state[env_ids]

        if self.total_goal_reached_counter >= 0:
            # randomize z orientation into -1, 1
            fw_default_root_state[:, 6] = torch.rand((len(env_ids)), device=self.sim.device) * 2 - 1
        fw_default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._fourwis_jetauto.write_root_pose_to_sim(fw_default_root_state[:, :7], env_ids)
        self._fourwis_jetauto.write_root_velocity_to_sim(fw_default_root_state[:, 7:], env_ids)
        self._fourwis_jetauto.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # randomize goal position
        if False:
            # slowly increases the distance to the goal. But causes some issues after tgrc > 10.000
            goal_reached_counter_based_additional_distance = self.total_goal_reached_counter * 0.2 / 1000
        else:
            goal_reached_counter_based_additional_distance = 0.4
        num_resets = len(env_ids)

        alpha_tensor = torch.rand((num_resets, 1), device=self.sim.device) * torch.tensor([2 * math.pi],
                                                                                          device=self.sim.device)
        alpha_tensor = self.adjust_alpha_tensor(alpha_tensor)

        # for debugging, spawn cube always in front of robot
        # alpha_tensor = torch.ones((num_resets, 1), device=self.sim.device) * torch.tensor([0.5 * math.pi], device=self.sim.device)


        r_tensor = torch.rand((num_resets, 1), device=self.sim.device) * torch.tensor([1.0 + goal_reached_counter_based_additional_distance], device=self.sim.device) + 0.8
        z_tensor = torch.full((num_resets, 1), 0.02, device=self.sim.device)

        goal_pos_tensor = torch.cat((torch.sin(alpha_tensor) * r_tensor, torch.cos(alpha_tensor) * r_tensor, z_tensor),
                                    dim=1)
        temp_list = torch.tensor([3.5, 3.5, 0], device=self.sim.device)  # center of room
        # print("goal_pos_tensor: ", goal_pos_tensor)
        goal_new_root_state = goal_pos_tensor + torch.stack([temp_list] * num_resets) + self.scene.env_origins[env_ids]

        # print("goal_new_root_state: ", goal_new_root_state)
        default_root_state = self.goal_root_state[env_ids]
        default_root_state[:, :3] = goal_new_root_state
        # print("default_root_state: ", default_root_state)

        self._goal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # print some stuff
        rounded_reward_sums_per_env = [round(val.item(), 2) for val in self.total_reward_sum[env_ids]]
        prev_distance_to_goal = [round(val.item(), 2) for val in self.previous_distance_to_goal[env_ids]]
        init_distance_to_goal = [round(val.item(), 2) for val in self.initial_distance_to_goal[env_ids]]
        obs_counter_per_env = [val.item() for val in self.observation_counter_per_env[env_ids]]
        rounded_time_per_episode = [round(time.time() - val.item(), 2) for val in self.start_time_episode[env_ids]]
        allocated_memory = torch.cuda.memory_allocated(self.sim.device)
        rounded_total_penalty_for_too_many_action_changes = [round(val.item(), 2) for val in self.total_penalty_for_too_many_action_changes[env_ids]]
        rounded_total_reward_for_driving_forward = [round(val.item(), 2) for val in self.total_reward_for_driving_forward[env_ids]]


        print(
            f"allocated_memory: {allocated_memory / 1024 ** 2:.2f} MB, total reward sum: {rounded_reward_sums_per_env}, prev_distance_to_goal: {prev_distance_to_goal}, "
            f"init_distance_to_goal: {init_distance_to_goal}")
        print(
            f"total_goal_reached_counter: {self.total_goal_reached_counter}, obs_counter_per_env: {obs_counter_per_env}, rounded_time_per_episode: {rounded_time_per_episode}")
        print(f"total_penalty_for_too_many_action_changes: {rounded_total_penalty_for_too_many_action_changes}, total_reward_for_driving_forward: {rounded_total_reward_for_driving_forward}")

        # measures for final evaluation
        if self.cfg.evaluation_mode:
            self.total_episode_counter += num_resets
            self.summed_up_initial_distance_to_goal += torch.sum(self.initial_distance_to_goal[env_ids])
            self.total_timesteps += torch.sum(self.observation_counter_per_env[env_ids])
            self.total_timesteps_if_goal_reached += torch.sum(self.observation_counter_per_env[env_ids] *
                                                              (self.previous_distance_to_goal[env_ids] < self.cfg.required_distance_to_goal))
            self.total_task_successes = self.total_goal_reached_counter

            print("FINAL EVALUATION MEASURES: ")
            if self.total_episode_counter > 0 and self.total_task_successes.item() > 0:
                print("total_episode_counter: ", self.total_episode_counter)
                print("total_task_successes: ", self.total_task_successes.item())
                print("summed_up_initial_distance_to_goal: ", self.summed_up_initial_distance_to_goal.item())
                print("total_timesteps: ", self.total_timesteps.item())
                print("total_timesteps_if_goal_reached: ", self.total_timesteps_if_goal_reached.item())
                print("total_action_changes_above_threshold_025: ", self.total_action_changes_above_threshold_025.item())
                print("total_action_changes_above_threshold_05: ", self.total_action_changes_above_threshold_05.item())
                print("total_action_changes_above_threshold_075: ", self.total_action_changes_above_threshold_075.item())


                print(f"success percentage: {self.total_task_successes.item() / self.total_episode_counter * 100:.2f}%")
                print(f"average_initial_distance_to_goal: {self.summed_up_initial_distance_to_goal.item() / self.total_episode_counter:.2f}")
                print(f"average_timesteps_per_episode: {self.total_timesteps.item() / self.total_episode_counter:.2f}")
                print(f"average_timesteps_per_episode_if_goal_reached: {self.total_timesteps_if_goal_reached.item() / self.total_task_successes.item():.2f}")


        self.total_reward_sum[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_penalty_for_too_many_action_changes[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_reward_for_driving_forward[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.start_time_episode[env_ids] = torch.full((num_resets,), time.time(), dtype=torch.float64,
                                                      device=self.sim.device)

        self.previous_distance_to_goal[env_ids] = torch.linalg.norm(default_root_state - fw_default_root_state, dim=1,
                                                                    dtype=torch.float32)
        self.initial_distance_to_goal[env_ids] = self.previous_distance_to_goal[env_ids].clone()
        self.observation_counter_per_env[env_ids] = 0

        self.action_history[env_ids] = torch.zeros(num_resets, self.cfg.size_action_history, device=self.sim.device,
                                                   dtype=torch.float32)
        # self.robot_history[env_ids] = torch.zeros(num_resets, self.cfg.size_robot_history, device=self.sim.device,
        #                                           dtype=torch.float32)
        self.cube_detector_history[env_ids] = torch.zeros(num_resets,
                                                            self.cfg.size_cube_detector_history * self.cfg.cube_detector_neurons,
                                                            device=self.sim.device, dtype=torch.float32)
        print("")
