import gymnasium as gym

from . import agents
from .cubechaser_camera_env import CubeChaserCameraEnv, CubeChaserCameraEnvCfg
from .cubechaser_camera_continuous_env import CubeChaserCameraContinuousEnv, CubeChaserCameraContinuousEnvCfg
from .cubechaser_env import CubeChaserEnv, CubeChaserEnvCfg
from .cubechaser_camera_env_2 import CubeChaserCameraEnv2, CubeChaserCameraEnv2Cfg
from .cubechaser_camera_env_hist_3 import CubeChaserCameraEnv3, CubeChaserCameraEnv3Cfg
from .cubechaser_camera_env_resnet import CubeChaserCameraEnvResnet, CubeChaserCameraEnvResnetCfg
from .cubechaser_camera_env_cube_detector import CubeChaserCameraEnvCubeDetector, CubeChaserCameraEnvCubeDetectorCfg
from .cubechaser_camera_env_cube_detector_fastercnn import CCCEnvCBFRCNNC, CCCEnvCBFRCNNCfg
from .cubechaser_camera_env_cube_detector_fastercnn_v2 import CCCEnvCBFRCNNCv2, CCCEnvCBFRCNNCfgv2

gym.register(
    id="CubeChaser-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    }
)

gym.register(
    id="CubeChaser-Camera-Continuous-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraContinuousEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraContinuousEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_continuous_ppo_cfg.yaml",
    }
)

gym.register(
    id="CubeChaser-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserEnvCfg
    }
)


gym.register(
    id="CubeChaser-Camera-skrl-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraEnv2Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg_2.yaml",
    }
)

gym.register(
    id="CubeChaser-Camera-skrl-v1",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraEnv3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraEnv3Cfg,
    }
)

gym.register(
    id="CubeChaser-Camera-skrl-resnet-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraEnvResnet",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraEnvResnetCfg,
    }
)

gym.register(
    id="CubeChaser-Camera-skrl-cube-detector-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CubeChaserCameraEnvCubeDetector",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CubeChaserCameraEnvCubeDetectorCfg,
    }
)

gym.register(
    id="CCC-skrl-cd-fastercnn-v0",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CCCEnvCBFRCNNC",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CCCEnvCBFRCNNCfg,
    }
)

gym.register(
    id="CCC-skrl-cd-fastercnn-v2",
    entry_point="omni.isaac.lab_tasks.direct.thesis-isaaclab-direct-cubechaser-camera:CCCEnvCBFRCNNCv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CCCEnvCBFRCNNCfgv2,
    }
)