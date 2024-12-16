import gymnasium as gym

from .exploration_lidar_env import ExplorationLidarEnvCfg, ExplorationLidarEnv
from .exploration_lidar_env_v1 import ExplorationLidarEnvCfgV1, ExplorationLidarEnvV1
from .exploration_lidar_env_v2 import ExplorationLidarEnvCfgV2, ExplorationLidarEnvV2
from .exploration_lidar_env_v3 import ExplorationLidarEnvCfgV3, ExplorationLidarEnvV3

gym.register(
    id="Exploration-Lidar-v0",
    entry_point="omni.isaac.lab_tasks.direct.exploration-lidar:ExplorationLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExplorationLidarEnvCfg,
    }
)

gym.register(
    id="Exploration-Lidar-v1",
    entry_point="omni.isaac.lab_tasks.direct.exploration-lidar:ExplorationLidarEnvV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExplorationLidarEnvCfgV1,
    }
)

gym.register(
    id="Exploration-Lidar-v2",
    entry_point="omni.isaac.lab_tasks.direct.exploration-lidar:ExplorationLidarEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExplorationLidarEnvCfgV2,
    }
)

gym.register(
    id="Exploration-Lidar-v3",
    entry_point="omni.isaac.lab_tasks.direct.exploration-lidar:ExplorationLidarEnvV3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExplorationLidarEnvCfgV3,
    }
)