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
from omni.isaac.lab.sensors.ray_caster.patterns import patterns_cfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")
# enable_extension("omni.kit.window.viewport")  # Note: no difference in headless and rtx lidar stuff
import omni.replicator.core as rep
from torch.cuda import device

from ..utilitiesisaaclab import mecanum_direction, mecanum_controller
import time
from torchvision.utils import save_image, make_grid
import math
import random
import matplotlib.pyplot as plt
import pickle
import socket

@configclass
class ExplorationLidarEnvCfgV1(DirectRLEnvCfg):
    evaluation_mode = True  # used for creating evaluatin prints, used for thesis
    add_patches_and_noise_to_depth_image = False  # TODO: activate again

    decimation = 4
    action_scale = 1.0
    num_actions = 2
    # sim_dt = 1 / 60.0
    sim_dt = 1 / 60.0  # 1/20: didnt seem that good, just drove into the wall. 1/30 was surprisingly good, but probably better to use same value everywhere...


    # evaluation:
    # factor 1.5: # max_discovery_percentage: 18.93310546875
    # factor 1.0: #
    # factor 0.75: # max_discovery_percentage: 15.66
    episode_length_factor = 1.0 if evaluation_mode else 1.5
    episode_length_s = sim_dt * 1200 * episode_length_factor * decimation / 2

    # with new memory settings, laptop cannot handle 12  #Workstation: max 16, annotator doesnt allow more on isaacs side
    # laptop: 9 works
    num_envs = 6 if socket.gethostname() == "phil-SCHENKER-MEDIA-E23" else 12

    scaled_image_width = 320  # 35
    scaled_image_height = 240 # 30

    depth_image_max_distance = 3.0
    depth_image_slice_size = 16
    middle_slice_start = (scaled_image_height - depth_image_slice_size) // 2
    middle_slice_end = middle_slice_start + depth_image_slice_size

    depth_image_history_size = 0
    update_frequency_depth_image_steps = 5

    # if viewport.hydra_engine == "rtx" and viewport.render_mode == "RaytracedLighting":
    #     render_mode = "PathTracing"
    # else:
    #     render_mode = "RaytracedLighting"

    # render_mode = "PathTracing"


    # map settings
    room_size = 8.0
    required_map_size = room_size * 2.0  # for robotic centered map, robot should be able to move to the end of the room
    grid_size = 256
    grid_map_resolution = required_map_size / grid_size

    l_occ = np.log(0.7 / 0.3)  # Log odds for occupied cell
    l_free = np.log(0.3 / 0.7)  # Log odds for free cell

    start_discovery_reward_after_timestep = 15
    maximum_discovery_reward = 3.5  # 25.0  # 50.0  # Note: increased from 10.0  # reward per 10 percentage discovered

    penalty_for_touching_obstacle = -1.25  # -15.0
    penalty_for_requiring_all_steps = 0.0  # 10.0  # -15.0  # -10.0  # 4.0  # -10.0: crashes into the wall pretty early  # 10.0 resulted in robot just moving at same spot  # TODO: normally -4.0, for now give a reward for "suriving" all steps
    max_steps_per_episode = 600 * episode_length_factor

    penalty_for_too_many_action_changes_per_step = -1.5 / max_steps_per_episode

    penalty_for_driving_backwards_for_too_long = 0.0  # -1.0/450.0
    driving_backwards_penalty_after_steps = 10

    reward_for_driving_forward_per_step = 3.0 / max_steps_per_episode  # 1.5

    size_action_history = 8
    size_robot_orientation_history = 5

    robot_positions_w_lookback = 50
    robot_positions_w_min_required_distance_after_lookback = 0.1  # TODO: maybe even .15, but not higher
    robot_staying_at_same_position_penalty_per_step = -2.5 / max_steps_per_episode

    num_observations = grid_size * grid_size + size_robot_orientation_history + size_action_history + depth_image_slice_size * scaled_image_width * (depth_image_history_size + 1)


    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # robot
    model_name = "014_4wis_added_front_bumper.usd"
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
            activate_contact_sensors=True,
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
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),  # TODO: check out
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

    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/fourwis/chassis/camera",
        data_types=["distance_to_camera"],
        spawn=None,
        width=scaled_image_width,
        height=scaled_image_height,
        update_period=sim_dt * decimation,
    )



    # Note: RayCaster is not really usable! Main issue is that it can only cast agains a mesh prim paths,
    # which only supports one mesh at the moment! Also it is turning in a strange way
    # lidar: RayCasterCfg = RayCasterCfg(
    #     # prim_path="/World/envs/env_.*/fourwis/chassis/Lidar",
    #     prim_path="/World/envs/env_.*/fourwis/chassis",  # connect to physics body only because virtual sensor
    #     # prim_path="/World/envs/env_.*/fourwis/lidar_wrapper/Rotating",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=(0.5, 0.5, -0.5, -0.5)),
    #     attach_yaw_only=False,
    #     mesh_prim_paths=["/World/envs"],  # "/World/ground",  # Note: only supports one mesh primitive path
    #     pattern_cfg=patterns_cfg.LidarPatternCfg(  # TODO: patterns_cfg or patterns? somehow wrongly links to 4.0.0
    #         channels=1, vertical_fov_range=(-5.0, 5.0),
    #         horizontal_fov_range=(-180.0, 180.0),
    #         horizontal_res=1.0
    #     ),
    #     debug_vis=True,  # whether to visualize the sensor. Default false
    #     max_distance=5
    # )

    # Note: Not really usable as well, same issue with mesh and is actually a camera where the output
    # is a distance to the image plane
    # ray_caster_camera: RayCasterCameraCfg = RayCasterCameraCfg(
    #     prim_path="/World/envs/env_.*/fourwis/chassis",
    #     offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
    #     data_types=["distance_to_image_plane"],
    #     mesh_prim_paths=["/World/envs"],
    #     pattern_cfg=patterns_cfg.LidarPatternCfg(  # Note: wrong, requried pinhole camera cfg
    #         channels=1, vertical_fov_range=(-5.0, 5.0),
    #         horizontal_fov_range=(-180.0, 180.0),
    #         horizontal_res=1.0
    #     ),
    #     debug_vis=True,
    #     max_distance=5
    # )

    # lidar_data shape:  torch.Size([4, 179, 3])

    light_cfg: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        intensity=2000.0, color=(0.75, 0.75, 0.75)
    )

    distant_light_cfg: sim_utils.DistantLightCfg = sim_utils.DistantLightCfg(
        color=(0.9, 0.9, 0.9), intensity=2500.0
    )

    own_omniverse_scenes = "/home/phil/Documents/scenes/"

    # scenery
    # replaced wall with some rectangles! Else contact sensor does not work
    # wall_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
    #     usd_path=own_omniverse_scenes + "Geo_M2_BaseWallSide.usd",
    #     scale=np.array([1.0, 1.0, 1.0]),
    #     # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #     #     disable_gravity=False,
    #     # ),
    #     # collision_props=sim_utils.CollisionPropertiesCfg(
    #     #     collision_enabled=True,
    #     # ),
    #     # activate_contact_sensors=True,
    # )

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
        size=(2.0, 0.2, 1.0),
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
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6",
                                "/World/ground/GroundPlane/CollisionPlane"]
    )

    contact_sensor_front_left_wheel_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/front_left_wheel",
        track_air_time=False,
        debug_vis=False,
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
    )

    contact_sensor_front_right_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/front_right_wheel",
        track_air_time=False,
        debug_vis=False,
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
    )
    
    contact_sensor_rear_left_wheel_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/rear_left_wheel",
        track_air_time=False,
        debug_vis=False,
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
    )

    contact_sensor_rear_right_wheel_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/rear_right_wheel",
        track_air_time=False,
        debug_vis=False,
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
    )

    contact_sensor_front_bumper_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/fourwis/front_bumper",
        track_air_time=False,
        debug_vis=False,
        history_length=1,
        filter_prim_paths_expr=["/World/envs/env_.*/obstacle_column_0", "/World/envs/env_.*/obstacle_column_1",
                                "/World/envs/env_.*/own_wall_0", "/World/envs/env_.*/own_wall_1",
                                "/World/envs/env_.*/own_wall_2", "/World/envs/env_.*/own_wall_3",
                                "/World/envs/env_.*/obstacle_cuboid_0", "/World/envs/env_.*/obstacle_cuboid_1",
                                "/World/envs/env_.*/obstacle_cuboid_2", "/World/envs/env_.*/obstacle_cuboid_3",
                                "/World/envs/env_.*/obstacle_cuboid_4", "/World/envs/env_.*/obstacle_cuboid_5",
                                "/World/envs/env_.*/obstacle_cuboid_6"]
    )



    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=room_size, replicate_physics=True)


class ExplorationLidarEnvV1(DirectRLEnv):
    cfg: ExplorationLidarEnvCfgV1
    image_path = "/home/phil/university/thesis/data/images/"
    observation_counter = 0
    randomization_counter = 0
    export_images = False
    export_grid_maps = True

    def __init__(self, cfg: ExplorationLidarEnvCfgV1, render_mode: str | None = None, **kwargs):
        print("ExplorationLidarEnv render_mode: ", render_mode)
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

        self.mecanum_controller = mecanum_controller.MecanumController(self.cfg.num_envs, self.sim.device)  # , default_velocity=20.0
        self.depth_camera_intrinsics = {
            "width": self.cfg.scaled_image_width, "height":self.cfg.scaled_image_height,
            "fx":3.7, "fy":3.7,  # Focal length, estimation, just using single focal length value
            "cx":self.cfg.scaled_image_width / 2.0, "cy":self.cfg.scaled_image_height / 2.0  # Principal point, estimation
        }

        self.start_time_episode = torch.full((self.num_envs,), time.time(), dtype=torch.float64, device=self.sim.device)
        self.total_reward_sum = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_discovery_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_penalty_per_step = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_penalty_for_too_many_action_changes = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_force_matrix_penalty = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_backwards_driving_penalty = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_forward_driving_reward = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.total_penalty_for_not_moving = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        self.exploration_reward_counter = torch.zeros(4, device=self.sim.device, dtype=torch.long)

        self.current_run_observation_counter_per_env = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.int32)

        self.reset_counter = 0

        # evaluation measures for thesis
        self.total_episode_counter = -self.cfg.num_envs  # subtract initial resets
        self.max_discovery_percentage = 0.0
        self.total_timesteps = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.count_collisions = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.summed_timesteps_when_collision_happened = torch.zeros(1, device=self.sim.device, dtype=torch.int32)
        self.total_discovery_percentage = torch.zeros(1, device=self.sim.device, dtype=torch.float32)
        self.total_discovery_percentage_when_not_colliding = torch.zeros(1, device=self.sim.device, dtype=torch.float32)

        self.discovery_percentages_history_env_0 = []
        self.robot_position_history_env_0 = []

        self.occupancy_grid_maps = torch.zeros((self.num_envs, self.cfg.grid_size, self.cfg.grid_size), device=self.sim.device)
        self.grid_origins_x = torch.zeros(self.num_envs, device=self.sim.device)
        self.grid_origins_y = torch.zeros(self.num_envs, device=self.sim.device)
        self.original_grid_origins = torch.zeros((self.num_envs, 2), device=self.sim.device)
        self.previous_robot_positions = torch.zeros((self.num_envs, 3), device=self.sim.device)
        self.robot_positions = torch.zeros((self.num_envs, 3), device=self.sim.device)

        self.robot_yaw_history = torch.zeros((self.num_envs, 3), device=self.sim.device)

        self.previous_discovery_percentages = torch.zeros(self.num_envs, device=self.sim.device)
        self.discovery_reward_factor = self.cfg.maximum_discovery_reward / 10.0
        self.force_matrix_penalty = torch.zeros(self.num_envs, device=self.sim.device)

        # reward if reaching goal is approx 6, max of 600 steps, so -0.005 per step
        self.penalty_per_step = torch.ones(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32) * (self.cfg.penalty_for_requiring_all_steps / self.cfg.max_steps_per_episode)
        self.penalty_for_too_many_action_changes = torch.ones(self.cfg.num_envs, device=self.sim.device,
                                                             dtype=torch.float32) * self.cfg.penalty_for_too_many_action_changes_per_step

        self.action_history = torch.zeros(self.cfg.num_envs, self.cfg.size_action_history, device=self.sim.device,
                                          dtype=torch.float32)

        # create tensor for depth image history, containing the last x depth images for each environment
        if self.cfg.depth_image_history_size:
            self.depth_image_history = torch.zeros(self.cfg.num_envs, self.cfg.depth_image_history_size, self.cfg.depth_image_slice_size * self.cfg.scaled_image_width,
                                                   device=self.sim.device, dtype=torch.float32)

        self.robot_orientation_history = torch.zeros(self.cfg.num_envs, self.cfg.size_robot_orientation_history, device=self.sim.device,
                                                        dtype=torch.float32)

        self.counter_driving_backwards_per_env = torch.zeros(self.num_envs, device=self.sim.device)

        self.robot_positions_w_history = torch.zeros(self.num_envs, self.cfg.robot_positions_w_lookback, 3, device=self.sim.device)


        self.reflectance_patches_num = []
        self.reflectance_patches_h = []
        self.reflectance_patches_w = []
        self.reflectance_patches_y = []
        self.reflectance_patches_x = []
        self.reflectance_value = 0.0

        self.temp_obs_counter = 0
        self.temp_pre_physic_counter = 0
        self.temp_apply_action_counter = 0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            for key in [
                "discovery_reward",
                "full_exploration",
                "penalty_per_step",
                "penalty_for_too_many_action_changes",
                "force_matrix_penalty",
                "penalty_for_backwards_driving",
                "reward_for_driving_forward",
                "penalty_for_not_moving"
            ]
        }

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            # shape=(self.cfg.num_observations,)
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

        # wall_positions = [[3.5, 0.0, 0.0, 1.0], [7.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70711, 0.70711],
        #                   [0.0, 3.5, 0.70711, 0.70711], [0.0, 7.0, 1.0, 0.0], [3.5, 7.0, 1.0, 0.0],
        #                   [7.0, 7.0, 0.70711, -0.70711], [7.0, 3.5, 0.70711, -0.70711]]
        # self._walls = []
        # for index, wall_position in enumerate(wall_positions):
        #     _wall = self.cfg.wall_cfg.func(
        #         prim_path="/World/envs/env_.*/Geo_M2_BaseWallSide_{}".format(index),
        #         cfg=self.cfg.wall_cfg,
        #         orientation=np.array([wall_position[2], 0.0, 0.0, wall_position[3]]),
        #         translation=(wall_position[0], wall_position[1], 0.0),
        #     )
        #     self._walls.append(_wall)

        self._own_walls = []
        wall_positions = [[3.5, 0.0, 0.0, 1.0], [-0.1, 3.5, 0.70711, 0.70711], [3.5, 7.0, 0.0, 1.0],
                          [7.1, 3.5, 0.70711, 0.70711]]
        for index, wall_position in enumerate(wall_positions):
            _wall = self.cfg.own_wall_cfg.func(
                prim_path="/World/envs/env_.*/own_wall_{}".format(index),
                cfg=self.cfg.own_wall_cfg,
                orientation=np.array([wall_position[2], 0.0, 0.0, wall_position[3]]),
                translation=(wall_position[0], wall_position[1], 1.0),
            )
            self._own_walls.append(_wall)

        column_positions = [[5.75, 1.25], [1.2, 1.7]]
        for index, column_position in enumerate(column_positions):
            self.cfg.obstacle_column_cfg.func(
                prim_path="/World/envs/env_.*/obstacle_column_{}".format(index),
                cfg=self.cfg.obstacle_column_cfg,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                translation=(column_position[0], column_position[1], 0.5)
            )

        # obstacle_cuboids_positions = [[4.5, 3.0, 1.0, 0.0], [2.5, 3.0, 1.0, 0.0], [1.0, 3.5, 0.70711, 0.70711], [1.0, 5.5, 0.70711, 0.70711], [3.0, 4.0, 1.0, 0.0],
        #                               [3.5, 1.5, 0.92388, 0.38268], [5.0, 5.5, 0.70711, 0.70711], [3.0, 6.0, 1.0, 0.0], [6.0, 4.0, 0.70711, 0.70711]]
        obstacle_cuboids_positions = [[3.5, 3.0, 1.0, 0.0], [1.0, 4.5, 0.70711, 0.70711], [3.0, 4.25, 1.0, 0.0],
                                      [3.5, 1.4, 0.96593, 0.25882], [5.0, 5.0, 0.70711, 0.70711], [3.0, 5.8, 1.0, 0.0], [6.0, 4.0, 0.70711, 0.70711]]
        for index, obstacle_cuboid_position in enumerate(obstacle_cuboids_positions):
            self.cfg.obstacle_cuboid_cfg.func(
                prim_path="/World/envs/env_.*/obstacle_cuboid_{}".format(index),
                cfg=self.cfg.obstacle_cuboid_cfg,
                orientation=np.array([obstacle_cuboid_position[2], 0.0, 0.0, obstacle_cuboid_position[3]]),
                translation=(obstacle_cuboid_position[0], obstacle_cuboid_position[1], 0.5)
            )
        # self._obstacle_cuboid = self.cfg.obstacle_cuboid_cfg.func(
        #     prim_path="/World/envs/env_.*/obstacle_cuboid",
        #     cfg=self.cfg.obstacle_cuboid_cfg,
        #     orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        #     translation=(2.5, 1.5, 0.5)
        # )

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self._contact_sensor_front_left_wheel = ContactSensor(self.cfg.contact_sensor_front_left_wheel_cfg)
        self._contact_sensor_front_right_wheel = ContactSensor(self.cfg.contact_sensor_front_right_cfg)
        self._contact_sensor_rear_left_wheel = ContactSensor(self.cfg.contact_sensor_rear_left_wheel_cfg)
        self._contact_sensor_rear_right_wheel = ContactSensor(self.cfg.contact_sensor_rear_right_wheel_cfg)
        self._contact_sensor_front_bumper = ContactSensor(self.cfg.contact_sensor_front_bumper_cfg)

        self.gnd_plane = spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5)))

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["fourwis"] = self._fourwis_jetauto

        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["contact_sensor_front_left_wheel"] = self._contact_sensor_front_left_wheel
        self.scene.sensors["contact_sensor_front_right_wheel"] = self._contact_sensor_front_right_wheel
        self.scene.sensors["contact_sensor_rear_left_wheel"] = self._contact_sensor_rear_left_wheel
        self.scene.sensors["contact_sensor_rear_right_wheel"] = self._contact_sensor_rear_right_wheel
        self.scene.sensors["contact_sensor_front_bumper"] = self._contact_sensor_front_bumper

        # self.scene.sensors["ray_caster"] = self._lidar
        # self.scene.sensors["ray_caster_camera"] = self._ray_caster_camera

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
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        # any value inside action tensor between -0.05 and 0.05 set to zero
        mask = (self.actions >= -0.05) & (self.actions <= 0.05)
        self.actions[mask] = 0.0
        self.action_history = torch.roll(self.action_history, shifts=-2, dims=1)  # Shift entries left
        self.action_history[:, -2:] = self.actions

        self.counter_driving_backwards_per_env = torch.where(self.actions[:, 0] < 0, torch.ones(self.num_envs, device=self.sim.device) + self.counter_driving_backwards_per_env, torch.zeros(self.num_envs, device=self.sim.device))

        self.temp_pre_physic_counter += 1


    def _apply_action(self) -> None:
        velocities, positions = self.mecanum_controller.continuous_actions_into_velocities_and_positions(
            self.actions, only_allow_driving_forward=False)  # TODO: trying to only allow forward driving
        # Note: only_allow_driving_forward didnt work that well...

        self._fourwis_jetauto.set_joint_position_target(positions,
                                                        joint_ids=self._fourwis_steering_joint_indexes)
        self._fourwis_jetauto.set_joint_velocity_target(velocities,
                                                        joint_ids=self._fourwis_velocity_joint_indexes)
        self.temp_apply_action_counter += 1


    def depth_to_point_cloud(self, depth_images, intrinsics):
        """
        Converts a batch of depth images to a batch of point clouds.

        Args:
            depth_images: A PyTorch tensor of shape (batch_size, 1, height, width) representing depth images.
            intrinsics: A dictionary containing camera intrinsics as PyTorch tensors:
                - 'fx': Focal length x
                - 'fy': Focal length y
                - 'cx': Principal point x
                - 'cy': Principal point y

        Returns:
            A PyTorch tensor of shape (batch_size, num_points, 3) representing point clouds.
        """

        batch_size, _, height, width = depth_images.shape

        # Create pixel coordinates
        # v, u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        # u = u.to(depth_images.device).unsqueeze(0).expand(batch_size, -1, -1).float()
        # v = v.to(depth_images.device).unsqueeze(0).expand(batch_size, -1, -1).float()
        u = torch.arange(0, width, device=depth_images.device).unsqueeze(0).expand(batch_size, -1, -1).float()
        v = torch.arange(0, height, device=depth_images.device).unsqueeze(0).expand(batch_size, -1, -1).float()

        # Apply inverse projection
        z = depth_images.squeeze(1)
        x = (u - intrinsics['cx']) * z / intrinsics['fx']
        y = (v - intrinsics['cy']) * z / intrinsics['fy']

        # Stack coordinates to form point clouds
        point_clouds = torch.stack((x, y, z), dim=-1)

        return point_clouds

    def store_single_point_cloud_as_graph(self, point_cloud_np, name="point_cloud"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud_np[:, :, 0].flatten(), point_cloud_np[:, :, 1].flatten(),
                   point_cloud_np[:, :, 2].flatten(), c=point_cloud_np[:, :, 2].flatten(), cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(self.image_path + name + str(self.observation_counter) + ".png")

    def point_clouds_to_grid_map(self, point_clouds, robot_positions, robot_quaternion_orientations):
        # update self.grid_maps using point_clouds
        # point_clouds: (batch_size, 320, 320, 3)
        batch_size, height, width, _ = point_clouds.shape
        for b in range(batch_size):
            w, x, y, z = robot_quaternion_orientations[b]
            rotation_matrix = torch.tensor([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w],
                                            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2]], device=self.sim.device)

            transformed_point_clouds = torch.matmul(point_clouds[b, :, :, :2].view(-1, 2), rotation_matrix.t()).view(height, width, 2)
            print("transformed_point_clouds shape: ", transformed_point_clouds.shape)
            for row in range(transformed_point_clouds.shape[0]):
                for col in range(transformed_point_clouds.shape[1]):
                    point = transformed_point_clouds[row, col]
                    x = int(point[0].item() / self.cfg.grid_map_resolution)
                    y = int(point[1].item() / self.cfg.grid_map_resolution)
                    if b == 0 and row == 50 and col > 30 and col < 40:
                        print("point: ", point)
                        print("x, y: ", x, y)
                    if 0 <= x < self.cfg.grid_size and 0 <= y < self.cfg.grid_size:
                        self.grid_maps[b, y, x] = 1.0


    def filter_out_floor_points_from_point_clouds(self, point_clouds):
        camera_angle = 26.0
        camera_height = 0.45
        batch_size, height, width, _ = point_clouds.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=point_clouds.device).repeat(height, 1)
        y_coords = torch.arange(height, dtype=torch.float32, device=point_clouds.device).repeat(width, 1).t()
        # Reshape coordinates to match point_cloud shape and expand for batch dimension
        x_coords = x_coords.reshape(1, height, width, 1).expand(batch_size, -1, -1, -1)
        y_coords = y_coords.reshape(1, height, width, 1).expand(batch_size, -1, -1, -1)
        print("x_coords shape: ", x_coords.shape)
        # Calculate the expected z-coordinate of the floor
        floor_z = camera_height - math.tan(camera_angle) * torch.sqrt(x_coords ** 2 + y_coords ** 2)
        print("floor_z[0]: ", floor_z[0])
        print("floor_z shape: ", floor_z.shape)

        # Create a mask to filter out floor points
        mask = point_clouds[:, :, :, 2].unsqueeze(-1) > floor_z

        # Apply the mask to the point cloud
        filtered_point_cloud = point_clouds * mask

        return filtered_point_cloud

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        w = quaternion[0]
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


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

            # also save as pickle
            with open(self.image_path + "robot_position_history_" + str(self.observation_counter) + ".pkl", 'wb') as f:
                pickle.dump(robot_positions_np, f)

    def store_depth_image_as_png(self, image, name="depth_image_"):
        save_image(image, self.image_path + name + str(self.observation_counter) + ".png")


    def add_depth_noise(self, depth_tensor, noise_std=0.01, min_depth=0.0, distance_factor=0.0001):
        """
        Add Gaussian noise to a depth tensor.

        Args:
            depth_tensor (torch.Tensor): Input depth tensor
            noise_std (float): Standard deviation of the noise (default: 0.01)
            min_depth (float): Minimum depth value to avoid negative depths

        Returns:
            torch.Tensor: Depth tensor with added noise
        """

        distance_dependent_noise = noise_std + (depth_tensor * distance_factor)
        # Generate noise with the same shape as the input tensor
        noise = torch.randn_like(depth_tensor) * distance_dependent_noise
        noisy_depth = depth_tensor + noise

        # Ensure no negative depth values
        noisy_depth = torch.clamp(noisy_depth, min=min_depth)

        return noisy_depth


    def calculate_random_reflectance_patches(self, depth_tensor, num_patches_range=(2, 6),
                                                patch_size_w_range=(5, 20), patch_size_h_range=(30, 100)):
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        # Get tensor dimensions
        B, H, W = depth_tensor.shape

        self.reflectance_patches_num = []
        self.reflectance_patches_h = []
        self.reflectance_patches_w = []
        self.reflectance_patches_y = []
        self.reflectance_patches_x = []
        # random value between 0.0 and 0.2
        self.reflectance_value = random.uniform(0.0, 0.2)

        # Generate random number of patches for each image in batch
        # Add patches to each image in batch
        for b in range(B):
            # Generate random number of patches
            num_patches = torch.randint(low=num_patches_range[0],
                                        high=num_patches_range[1] + 1,
                                        size=(1,)).item()
            self.reflectance_patches_num.append(num_patches)
            for _ in range(num_patches):
                # Random patch size
                h = torch.randint(patch_size_h_range[0], patch_size_h_range[1], (1,))
                w = torch.randint(patch_size_w_range[0], patch_size_w_range[1], (1,))

                # Random patch location (ensure within bounds)
                y = torch.randint(0, max(1, H - h), (1,)).item()
                x = torch.randint(0, max(1, W - w), (1,)).item()

                self.reflectance_patches_h.append(h)
                self.reflectance_patches_w.append(w)
                self.reflectance_patches_y.append(y)
                self.reflectance_patches_x.append(x)


    def apply_reflectance_patches(self, depth_tensor):
        # Add patches to each image in batch
        B, H, W = depth_tensor.shape
        # Create a copy of the input tensor
        result = depth_tensor.clone()
        top_counter = 0
        counter = 0
        if len(self.reflectance_patches_num) > 0:
            for b in range(B):
                for _ in range(self.reflectance_patches_num[top_counter]):
                    h = self.reflectance_patches_h[counter]
                    w = self.reflectance_patches_w[counter]

                    y = self.reflectance_patches_y[counter]
                    x = self.reflectance_patches_x[counter]

                    # Apply the patch (simple rectangular patch)
                    result[b, y:y + h, x:x + w] = self.reflectance_value
                    counter += 1
                top_counter += 1

        return result.squeeze() if depth_tensor.shape[0] == 1 else result


    def _get_observations(self) -> dict:
        self.temp_obs_counter += 1
        self.current_run_observation_counter_per_env += 1
        if self.observation_counter < 5000000:
            self.observation_counter += 1

        camera_data_type = self.cfg.camera.data_types[0]
        depth_camera_image = self._camera.data.output[camera_data_type].clone()

        if self.export_grid_maps and self.observation_counter % 25 == 0:
            self.store_depth_image_as_png(depth_camera_image[0])

        if self.cfg.add_patches_and_noise_to_depth_image:
            # add reflectance patches to the depth_camera_image
            if self.observation_counter % 5 == 0:
                # to make sure patches dont "move to much" and get easily filtered out by lstm
                self.calculate_random_reflectance_patches(depth_camera_image)

            depth_camera_image = self.apply_reflectance_patches(depth_camera_image)
            if self.export_grid_maps and self.observation_counter % 25 == 0:
                self.store_depth_image_as_png(depth_camera_image[0], "depth_image_reflectance_")

            # add some noise to the depth_camera_image
            depth_camera_image = self.add_depth_noise(depth_camera_image, noise_std=0.02, min_depth=0.0,
                                                      distance_factor=0.0005)

            if self.export_grid_maps and self.observation_counter % 25 == 0:
                self.store_depth_image_as_png(depth_camera_image[0], "depth_image_noisy_")



        # print("depth_camera_image shape: ", depth_camera_image.shape)  # torch.Size([4, 240, 320])
        # camera_pickle_store = {"depth_image": depth_camera_image.cpu()}
        # with open("/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/exploration-lidar/pickles/" + "depth_image_" + str(self.observation_counter) + ".pkl", "wb") as f:
        #     pickle.dump(camera_pickle_store, f)



        # slice depth camera image
        depth_camera_image = depth_camera_image[:, self.cfg.middle_slice_start:self.cfg.middle_slice_end, :]
        if self.export_grid_maps and self.observation_counter % 25 == 0:
            self.store_depth_image_as_png(depth_camera_image[0], "depth_image_middle_slice_")

        # normalize depth camera image
        depth_camera_image = torch.clamp(depth_camera_image, 0, self.cfg.depth_image_max_distance) / self.cfg.depth_image_max_distance
        depth_camera_image = depth_camera_image.view(self.num_envs, -1)


        # contact_sensor_data = self._contact_sensor.data
        # print("contact_sensor_data: ", contact_sensor_data)
        # print("net_forces_w: ", contact_sensor_data.net_forces_w[0])
        # print("force_matrix_w: ", contact_sensor_data.force_matrix_w.shape)
        # print("force_matrix_w: ", contact_sensor_data.force_matrix_w)
        # print("force_matrix_w: ", contact_sensor_data.force_matrix_w.shape)

        # print("last_contact_time: ", contact_sensor_data.last_contact_time)
        # print("current_contact_time: ", contact_sensor_data.current_contact_time)
        # print("chassis: ", self._contact_sensor.find_bodies("chassis"))
        # print("sensor body names:", self._contact_sensor.body_names)
        # print("sensor num bodies:", self._contact_sensor.num_bodies)


        # print("_get_observations")
        # lidar_anno_data_1 = self.annotators[0].get_data()
        # lidar_anno_data_2 = self.annotators[1].get_data()
        # print("lidar_anno_data_1 : ", lidar_anno_data_1)
        # print("lidar_anno_data_1 len: ", len(lidar_anno_data_1["data"]))
        # print("lidar_anno_data_2 len: ", len(lidar_anno_data_2["data"]))
        # # combine lidar anno data into tensor:
        # min_len = min(len(lidar_anno_data_1["data"]), len(lidar_anno_data_2["data"]))
        # combined_lidar_data = torch.tensor([lidar_anno_data_1["data"][:min_len], lidar_anno_data_2["data"][:min_len]])
        # temp_lidar_pickle = {}
        # temp_lidar_pickle["lidar_data"] = combined_lidar_data.cpu()
        # temp_lidar_pickle["robot_position"] = (self._fourwis_jetauto.data.root_pos_w[:, :3][0:2]).cpu()  # TODO:  - self.scene.env_origins[0] ???
        # # temp_lidar_pickle["robot_orientation"] = self.euler_from_quaternion(self._fourwis_jetauto.data.root_quat_w[0].cpu().detach().numpy())
        # temp_lidar_pickle["robot_orientation_quat"] = self._fourwis_jetauto.data.root_quat_w[0:2].cpu()
        #
        # # temp_lidar_pickle["lidar_anno_data_2"] = lidar_anno_data_2
        # with open("/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/exploration-lidar/pickles/" + "pickle_" + str(self.observation_counter) + ".pkl", "wb") as f:
        #     pickle.dump(temp_lidar_pickle, f)

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
        if has_incomplete_lidar_data:
            # create observation with zeros only
            print("!!!! has_incomplete_lidar_data, resulting in empty observation !!!!")
            empty_observation = torch.zeros((self.num_envs, self.cfg.num_observations), device=self.sim.device)
            return {"policy": empty_observation}
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


        # store data in pickle
        # temp_lidar_pickle = {}
        # temp_lidar_pickle["lidar_data"] = lidar_data.clone().cpu()
        # temp_lidar_pickle["robot_positions"] = self.robot_positions.clone().cpu()
        # temp_lidar_pickle["robot_orientation_quat"] = self._fourwis_jetauto.data.root_quat_w.clone().cpu()
        # temp_lidar_pickle["current_run_observation_counter_per_env"] = self.current_run_observation_counter_per_env.clone().cpu()
        #
        # # temp_lidar_pickle["lidar_anno_data_2"] = lidar_anno_data_2
        # with open("/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/exploration-lidar/pickles/" + "pickle_" + str(self.observation_counter) + ".pkl", "wb") as f:
        #     pickle.dump(temp_lidar_pickle, f)

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
                if self.current_run_observation_counter_per_env[env].item() < (self.cfg.start_discovery_reward_after_timestep - 10):
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

            if self.export_grid_maps and self.observation_counter % 50 == 0:
                self.store_occupancy_grid_map_as_png(0)

                # second environment
                # occupancy_grid_map = self.occupancy_grid_maps[1].cpu().numpy()
                # ogm_probabilities = self.log_odds_to_probability(occupancy_grid_map)
                # plt.figure(figsize=(10, 10))
                # plt.imshow(ogm_probabilities, cmap='binary', origin='lower')
                # plt.colorbar(label='Occupancy')
                #
                # plt.xlabel('X')
                # plt.ylabel('Y')
                # plt.title('Occupancy Grid Map')
                # plt.grid(True, which='both', color='0.95', linestyle='-', linewidth=0.5)
                # plt.tight_layout()
                # plt.savefig(self.image_path + "occupancy_grid_map_1_" + str(self.observation_counter) + ".png")
                # plt.close()

        occupancy_grid_map_observation = self.occupancy_grid_maps.clone().view(self.num_envs, -1)

        ogm_probabilities = self.log_odds_to_probability_tensor(occupancy_grid_map_observation)

        if self.observation_counter % 3 == 0:
            self.robot_orientation_history = torch.roll(self.robot_orientation_history, shifts=-1, dims=1)  # Shift entries lefts
        self.robot_orientation_history[:, -1] = temp_euler[:, 2] / (2 * math.pi)  # normalize to [0, 1]


        res = torch.cat((ogm_probabilities, depth_camera_image), dim=1)
        if self.cfg.depth_image_history_size > 0:
            res = torch.cat((res, self.depth_image_history.view(self.num_envs, -1)), dim=1)

        action_history_normalized = (self.action_history.clone() + 1.0) / 2.0
        res = torch.cat((res, action_history_normalized), dim=1)
        res = torch.cat((res, self.robot_orientation_history), dim=1)

        if self.cfg.depth_image_history_size > 0 and self.observation_counter % self.cfg.update_frequency_depth_image_steps == 0:
            self.depth_image_history = torch.roll(self.depth_image_history, shifts=-1, dims=1)  # Shift entries left
            self.depth_image_history[:, -1] = depth_camera_image

        if self.observation_counter % 5 == 0:
            self.robot_position_history_env_0.append(self.robot_positions[0].cpu().numpy())



        # if self.observation_counter < 50:
        #     print("res shape: ", res.shape)



        # data_type = self.cfg.camera.data_types[0]
        # temp = self._camera.data.output[data_type][:, :, :, :-1].clone() / 255.0
        # temp = self._camera.data.output[data_type].clone()
        # save temp as pickle
        # pickle_store = {"depth_image": temp, "robot_position": self._fourwis_jetauto.data.root_pos_w[:, :3],
        #                "robot_orientation": self._fourwis_jetauto.data.root_quat_w,
        #                 "lidar_data": lidar_data}
        # with open("/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/exploration-depth-camera/" + "pickle_" + str(self.observation_counter) + ".pkl", "wb") as f:
        #     pickle.dump(pickle_store, f)


        # print("temp some values: ", temp[0])
        # # Normalize each depth image in the batch independently
        # depth_image_normalized = (temp - temp.min(dim=1, keepdim=True)[0]) / (
        #         temp.max(dim=1, keepdim=True)[0] - temp.min(dim=1, keepdim=True)[0]
        # )
        # # depth_image_normalized = temp
        # print("depth_image_normalized: ", depth_image_normalized[0])
        # depth_image_3ch = depth_image_normalized.unsqueeze(1).repeat(1, 3, 1, 1)
        # print("depth_image_3ch shape: ", depth_image_3ch.shape)
        # if self.observation_counter < 5:
        #     save_image(make_grid(depth_image_3ch, nrow=2), self.image_path + "depth_image_3ch_" + str(self.observation_counter) + ".png")
        #
        #
        # # create point cloud
        # point_clouds = self.depth_to_point_cloud(temp.unsqueeze(1), self.depth_camera_intrinsics)
        # print("point_clouds shape: ", point_clouds.shape)
        # if self.observation_counter < 5:
        #     # store one pointcloud
        #     self.store_single_point_cloud_as_graph(point_clouds[0].cpu().numpy())
        #
        # # TODO: filter out floor points based on camera height and angle
        # point_clouds = self.filter_out_floor_points_from_point_clouds(point_clouds)
        # if self.observation_counter < 5:
        #     # store one pointcloud
        #     self.store_single_point_cloud_as_graph(point_clouds[0].cpu().numpy(), name="point_cloud_filtered")
        # print("point_clouds shape after filtering: ", point_clouds.shape)
        # # TODO: update grid map
        # print("self._fourwis_jetauto.data.root_pos_w[:, :3]: ", self._fourwis_jetauto.data.root_pos_w[:, :3])
        # self.point_clouds_to_grid_map(point_clouds, self._fourwis_jetauto.data.root_pos_w,
        #                               self._fourwis_jetauto.data.root_quat_w)
        #
        # # store first grid map
        # if self.observation_counter < 5:
        #     og = self.grid_maps[0].unsqueeze(0)
        #     print("og shape: ", og.shape)
        #     save_image(og, self.image_path + "grid_map_" + str(self.observation_counter) + ".png")

        # create observation
        # temp = temp.view(self.num_envs, -1)  # unpack


        return {"policy": res}

    def _get_rewards(self) -> torch.Tensor:
        # TODO: implement some reward for goal reached / some finish stuff --> full_exploration

        # TODO: plot discovery percentages


        # discovery_percentages = self.calculate_discovery_percentage(self.occupancy_grid_maps)
        discovery_percentages = 100.0 - self.calculate_percentage_of_cells_equal_zero(self.occupancy_grid_maps)

        if self.observation_counter % self.cfg.start_discovery_reward_after_timestep == 0:
            self.discovery_percentages_history_env_0.append(discovery_percentages[0].item())

        # calculate a reward factor. For now assume, after initial lidar scan, 10% of the map can still be discovered
        # Note: for now using fixed reward factor...
        # self.discovery_reward_factor = torch.where(self.current_run_observation_counter_per_env == self.cfg.start_discovery_reward_after_timestep,
        #                                                   self.cfg.maximum_discovery_reward / 10.0,
        #                                                   self.discovery_reward_factor)

        discovery_reward = torch.where(self.current_run_observation_counter_per_env > self.cfg.start_discovery_reward_after_timestep,
                                       (discovery_percentages - self.previous_discovery_percentages) * self.discovery_reward_factor,
                                       torch.zeros_like(discovery_percentages))

        # only give positive discovery reward to ignore "map jitter"
        discovery_reward = torch.where(discovery_percentages > self.previous_discovery_percentages, discovery_reward, torch.zeros_like(discovery_reward))

        # check if robot touched anything, if yes, give penalty
        non_zero_measures = (self._contact_sensor.data.force_matrix_w != 0).any(dim=-1)
        active_sensors_chassis = non_zero_measures.any(dim=(1, 2))
        active_sensors_flw = (self._contact_sensor_front_left_wheel.data.force_matrix_w != 0).any(dim=-1).any(dim=(1, 2))
        active_sensors_frw = (self._contact_sensor_front_right_wheel.data.force_matrix_w != 0).any(dim=-1).any(dim=(1, 2))
        active_sensors_rlw = (self._contact_sensor_rear_left_wheel.data.force_matrix_w != 0).any(dim=-1).any(dim=(1, 2))
        active_sensors_rrw = (self._contact_sensor_rear_right_wheel.data.force_matrix_w != 0).any(dim=-1).any(dim=(1, 2))
        active_sensors_fb = (self._contact_sensor_front_bumper.data.force_matrix_w != 0).any(dim=-1).any(dim=(1, 2))
        # make sure self.force_matrix_penalty is zero before applying, else penalty is triggered twice
        # TODO: not working correctly, penalty still happens twice...
        # penalty for crashing early makes no sense, agent doesnt know what early means ( - ((self.cfg.max_steps_per_episode / self.current_run_observation_counter_per_env.float())))
        self.force_matrix_penalty = torch.where((active_sensors_chassis | active_sensors_flw | active_sensors_frw | active_sensors_rlw | active_sensors_rrw | active_sensors_fb) & (self.force_matrix_penalty == 0.0),
                                                torch.ones(self.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.penalty_for_touching_obstacle,
                                                torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32))


        # if self.observation_counter < 200:
        #     print("discovery_percentages: ", discovery_percentages)
        #     print("discovery_reward: ", discovery_reward)
        #     print("active_sensors: ", active_sensors)
        #     print("self.force_matrix_penalty: ", self.force_matrix_penalty)

        penalty_action_changes = torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32)
        if self.observation_counter > 1:
            # prevent wrong reset and therefore resulting reward
            penalty_action_changes = torch.where(
                torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1) > 0.7,
                self.penalty_for_too_many_action_changes,
                torch.zeros(self.cfg.num_envs, device=self.sim.device, dtype=torch.float32))

        penalty_for_backwards_driving = torch.where(self.counter_driving_backwards_per_env > self.cfg.driving_backwards_penalty_after_steps,
                                                    torch.ones(self.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.penalty_for_driving_backwards_for_too_long,
                                                    torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32))

        # reward forward, penalty backwards
        # reward_for_driving_forward = (self.actions[:, 0]) * self.cfg.reward_for_driving_forward_per_step

        # only reward for driving forward
        reward_for_driving_forward = torch.where(self.actions[:, 0] > 0.0,
                                                 self.actions[:, 0] * self.cfg.reward_for_driving_forward_per_step,
                                                 torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32))


        # calculate distance from previous position
        distance_to_lookback_position = torch.linalg.norm(self.robot_positions_w_history[:, 0, :] - self._fourwis_jetauto.data.root_pos_w, dim=1, dtype=torch.float32)
        # if self.observation_counter % 20:
        #     print("distance_to_lookback_position: ", distance_to_lookback_position)
        penalty_for_not_moving = torch.where((self.current_run_observation_counter_per_env > (self.cfg.robot_positions_w_lookback+1)) & (distance_to_lookback_position < self.cfg.robot_positions_w_min_required_distance_after_lookback),
                                             torch.ones(self.num_envs, device=self.sim.device, dtype=torch.float32) * self.cfg.robot_staying_at_same_position_penalty_per_step,
                                                torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32))
        self.robot_positions_w_history = torch.roll(self.robot_positions_w_history, shifts=-1, dims=1)  # Shift entries left
        self.robot_positions_w_history[:, -1, :] = self._fourwis_jetauto.data.root_pos_w

        rewards = {
            "discovery_reward": discovery_reward,
            "full_exploration": torch.zeros(self.num_envs, device=self.sim.device),
            "penalty_per_step": self.penalty_per_step.clone(),
            "penalty_for_too_many_action_changes": penalty_action_changes,
            "force_matrix_penalty": self.force_matrix_penalty,
            "penalty_for_backwards_driving": penalty_for_backwards_driving,
            "reward_for_driving_forward": reward_for_driving_forward,
            "penalty_for_not_moving": penalty_for_not_moving
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        self.total_reward_sum += reward.clone()
        self.total_discovery_reward += discovery_reward
        self.total_penalty_per_step += self.penalty_per_step
        self.total_penalty_for_too_many_action_changes += penalty_action_changes
        self.total_force_matrix_penalty += self.force_matrix_penalty
        self.total_backwards_driving_penalty += penalty_for_backwards_driving
        self.total_forward_driving_reward += reward_for_driving_forward
        self.total_penalty_for_not_moving += penalty_for_not_moving

        for key, value in rewards.items():
            self._episode_sums[key] += value

        self.previous_discovery_percentages = torch.max(discovery_percentages, self.previous_discovery_percentages)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_episodes_reached = self.episode_length_buf >= self.max_episode_length - 1
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim.device)
        died = zeros

        died += torch.where(self.force_matrix_penalty != 0.0, ones, zeros)
        return died, max_episodes_reached

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._fourwis_jetauto._ALL_INDICES

        super()._reset_idx(env_ids)

        if not self.cfg.evaluation_mode and self.observation_counter > 2000:
            self.export_grid_maps = False

        if self.export_grid_maps and 0 in env_ids:
            self.store_occupancy_grid_map_as_png(0)
            self.store_discovery_percentages_history_env_0_as_png()
            self.store_robot_positions_history_env_0_as_png()
            self.discovery_percentages_history_env_0 = []
            self.robot_position_history_env_0 = []

        num_resets = len(env_ids)
        self.reset_counter += num_resets

        self.exploration_reward_counter[0] += torch.sum(self.previous_discovery_percentages[env_ids] > 4.0).item()
        self.exploration_reward_counter[1] += torch.sum(self.previous_discovery_percentages[env_ids] > 8.0).item()
        self.exploration_reward_counter[2] += torch.sum(self.previous_discovery_percentages[env_ids] > 12.0).item()
        self.exploration_reward_counter[3] += torch.sum(self.previous_discovery_percentages[env_ids] > 16.0).item()


        rounded_reward_sums_per_env = [round(val.item(), 2) for val in self.total_reward_sum[env_ids]]
        rounded_total_discovery_reward_per_env = [round(val.item(), 2) for val in self.total_discovery_reward[env_ids]]
        rounded_total_penalty_per_step_per_env = [round(val.item(), 2) for val in self.total_penalty_per_step[env_ids]]
        rounded_total_penalty_for_too_many_action_changes_per_env = [round(val.item(), 2) for val in self.total_penalty_for_too_many_action_changes[env_ids]]
        rounded_total_force_matrix_penalty_per_env = [round(val.item(), 2) for val in self.total_force_matrix_penalty[env_ids]]
        rounded_total_backwards_driving_penalty = [round(val.item(), 4) for val in self.total_backwards_driving_penalty[env_ids]]
        rounded_total_forward_driving_reward = [round(val.item(), 4) for val in self.total_forward_driving_reward[env_ids]]
        rounded_total_penalty_for_not_moving = [round(val.item(), 2) for val in self.total_penalty_for_not_moving[env_ids]]
        rounded_previous_discovery_percentages = [round(val.item(), 2) for val in self.previous_discovery_percentages[env_ids]]
        exploration_reward_counter_list = [val.item() for val in self.exploration_reward_counter]

        print(f"------resetting environments: {env_ids} - reset_counter: {self.reset_counter} - obs counter: {self.current_run_observation_counter_per_env[env_ids]}")
        print(f"total_reward_sum: {rounded_reward_sums_per_env} - total_discovery_reward: {rounded_total_discovery_reward_per_env} - total_penalty_per_step: {rounded_total_penalty_per_step_per_env} - total_penalty_for_too_many_action_changes: {rounded_total_penalty_for_too_many_action_changes_per_env} - total_force_matrix_penalty: {rounded_total_force_matrix_penalty_per_env} - total_penalty_for_not_moving: {rounded_total_penalty_for_not_moving}")
        print(f"total_backwards_driving_penalty: {rounded_total_backwards_driving_penalty}, rounded_total_forward_driving_reward: {rounded_total_forward_driving_reward} - last_discovery_percentage: {rounded_previous_discovery_percentages}")
        print(f"erc: {exploration_reward_counter_list}")
        # print(self.discovery_percentages_history_env_0)
        # measures for final evaluation
        if self.cfg.evaluation_mode:
            self.total_episode_counter += num_resets
            self.total_timesteps += torch.sum(self.current_run_observation_counter_per_env[env_ids])
            tempzeros = torch.zeros(num_resets, dtype=torch.bool, device=self.sim.device)
            tempones = torch.ones(num_resets, dtype=torch.bool, device=self.sim.device)
            self.count_collisions += torch.sum(self.total_force_matrix_penalty[env_ids] < 0.0)
            self.total_discovery_percentage += torch.sum(self.previous_discovery_percentages[env_ids]).item()
            self.total_discovery_percentage_when_not_colliding += torch.sum(self.previous_discovery_percentages[env_ids] * (self.total_force_matrix_penalty[env_ids] >= 0.0))
            self.summed_timesteps_when_collision_happened += torch.sum(self.current_run_observation_counter_per_env[env_ids] * (self.total_force_matrix_penalty[env_ids] < 0.0))
            self.max_discovery_percentage = max(self.max_discovery_percentage, torch.max(self.previous_discovery_percentages[env_ids]).item())
            if self.total_episode_counter > 0 and self.count_collisions.item() > 0 and (self.total_episode_counter - self.count_collisions.item()) > 0:
                print("---------------------------")
                print(f"max_discovery_percentage: {self.max_discovery_percentage} - total_episode_counter: {self.total_episode_counter} - total_timesteps: {self.total_timesteps.item()} - count_collisions: {self.count_collisions.item()} - total_discovery_percentage: {self.total_discovery_percentage.item()} - summed_timesteps_when_collision_happened: {self.summed_timesteps_when_collision_happened.item()}")
                print(f"success rate: {(self.total_episode_counter - self.count_collisions.item()) / self.total_episode_counter*100:.2f}% - average timesteps: {self.total_timesteps.item() / self.total_episode_counter} - average discovery percentage: {self.total_discovery_percentage.item() / self.total_episode_counter} - average discovery percentage when not colliding: {self.total_discovery_percentage_when_not_colliding.item() / (self.total_episode_counter - self.count_collisions.item()) } - average timesteps when collision happened: {self.summed_timesteps_when_collision_happened.item() / self.count_collisions.item()}")
                print(f"temp_obs_counter: {self.temp_obs_counter} - temp_pre_physic_counter: {self.temp_pre_physic_counter} - temp_apply_action_counter: {self.temp_apply_action_counter}")
                print("----------------------------")


        joint_pos = self._fourwis_jetauto.data.default_joint_pos[env_ids]
        joint_vel = self._fourwis_jetauto.data.default_joint_vel[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # z: 0.01
        if self.cfg.evaluation_mode:
            random_robot_start_positions = torch.tensor(
                [[3.5, 3.5, 1.0, 0.0]], device=self.sim.device)
            random_int = torch.zeros(len(env_ids), device=self.sim.device, dtype=torch.int64)
        else:
            random_robot_start_positions = torch.tensor([[3.5, 3.5, 1.0, 0.0], [3.5, 3.5, 0.0, 1.0], [3.5, 5.0, 1.0, 0.0],
                                                         [2.5, 5.0, 0.0, 1.0], [0.7, 6.2, 1.0, 0.0], [0.65, 1.0, 0.88295, -0.46947],
                                                         [4.35, 0.66, 1.0, 0.0], [5.9, 2.3, 0.55919, 0.55919],
                                                         [5.9, 6.1, 0.55919, 0.55919]], device=self.sim.device)
            random_int = torch.randint(0, len(random_robot_start_positions), (len(env_ids),), device=self.sim.device)

        fw_default_root_state = self._fourwis_jetauto.data.default_root_state[env_ids]
        fw_default_root_state[:, :2] = random_robot_start_positions[random_int, :2]
        fw_default_root_state[:, 3] = random_robot_start_positions[random_int, 2]
        fw_default_root_state[:, 6] = random_robot_start_positions[random_int, 3]
        # randomize z orientation into -1, 1
        # fw_default_root_state[:, 6] = torch.rand((len(env_ids)), device=self.sim.device) * 2 - 1
        fw_default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._fourwis_jetauto.write_root_pose_to_sim(fw_default_root_state[:, :7], env_ids)
        self._fourwis_jetauto.write_root_velocity_to_sim(fw_default_root_state[:, 7:], env_ids)
        self._fourwis_jetauto.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.robot_positions_w_history[env_ids, -1, :] = fw_default_root_state[:, :3]

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

        self.previous_discovery_percentages[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        # self.discovery_reward_factor[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.action_history[env_ids] = torch.zeros(num_resets, self.cfg.size_action_history, device=self.sim.device,
                                                   dtype=torch.float32)
        self.robot_orientation_history[env_ids] = torch.zeros(num_resets, self.cfg.size_robot_orientation_history, device=self.sim.device,
                                                   dtype=torch.float32)

        if self.cfg.depth_image_history_size > 0:
            self.depth_image_history[env_ids] = torch.zeros(num_resets, self.cfg.depth_image_history_size, self.cfg.depth_image_slice_size * self.cfg.scaled_image_width, device=self.sim.device,
                                                       dtype=torch.float32)

        self.force_matrix_penalty[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.current_run_observation_counter_per_env[env_ids] = torch.zeros(num_resets, device=self.sim.device, dtype=torch.int32)

        self.total_reward_sum[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_discovery_reward[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_penalty_per_step[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_penalty_for_too_many_action_changes[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_force_matrix_penalty[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_backwards_driving_penalty[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_forward_driving_reward[env_ids] = torch.zeros(num_resets, device=self.sim.device)
        self.total_penalty_for_not_moving[env_ids] = torch.zeros(num_resets, device=self.sim.device)

        self.counter_driving_backwards_per_env[env_ids] = torch.zeros(num_resets, device=self.sim.device)


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


    def calculate_discovery_percentage(self, occupancy_grid_maps, discovery_threshold=0.1):
        # Convert log-odds to probabilities
        probabilities = 1 / (1 + torch.exp(-occupancy_grid_maps))

        # Calculate the absolute difference from 0.5 (unknown state)
        difference_from_unknown = torch.abs(probabilities - 0.5)

        # Count cells that are considered discovered
        discovered_cells = torch.sum(difference_from_unknown > discovery_threshold, dim=(1, 2))

        # Calculate the percentage
        total_cells = occupancy_grid_maps.shape[1] * occupancy_grid_maps.shape[2]
        discovery_percentages = (discovered_cells.float() / total_cells) * 100

        return discovery_percentages


    def calculate_percentage_of_cells_equal_zero(self, occupancy_grid_maps):
        # more efficient
        zero_counts = (occupancy_grid_maps == 0).sum(dim=(1, 2)).float()
        total_cells = occupancy_grid_maps.shape[1] * occupancy_grid_maps.shape[2]
        zero_percentages = (zero_counts / total_cells) * 100

        return zero_percentages


