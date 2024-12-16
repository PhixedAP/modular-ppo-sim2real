import rospy
import logging

logging.basicConfig(level=logging.WARNING) 
import sys
import gymnasium as gym
import numpy as np
import torch
import time
import torch.nn as nn
from PIL import Image as pil_image
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torchvision.transforms as T
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud, PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
import math

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveRL
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


class Policy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", cube_detector_neurons=6,
                 size_action_history=20):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.cube_detector_neurons = cube_detector_neurons
        self.size_action_history = size_action_history

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(self.size_action_history, 2*self.size_action_history)
        self.history_fc2 = nn.Linear(2*self.size_action_history, self.size_action_history)

        self.fc2 = nn.Linear(2 * self.cube_detector_neurons + self.size_action_history, self.cube_detector_neurons + self.size_action_history//2)
        self.fc3 = nn.Linear(self.cube_detector_neurons + self.size_action_history//2, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        self.compute_counter += 1

        if self.compute_counter < 100:
            print("Policy: ", inputs["states"].shape)

        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = F.relu(x)
        x = self.fc11(x)
        x = F.relu(x)

        y = self.history_fc1(history_data)
        y = F.relu(y)
        y = self.history_fc2(y)
        y = F.relu(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, cube_detector_neurons=6,
                 size_action_history=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cube_detector_neurons = cube_detector_neurons
        self.size_action_history = size_action_history

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(self.size_action_history, 2 * self.size_action_history)
        self.history_fc2 = nn.Linear(2 * self.size_action_history, self.size_action_history)

        self.fc2 = nn.Linear(2 * self.cube_detector_neurons + self.size_action_history,
                             self.cube_detector_neurons + self.size_action_history // 2)
        self.fc3 = nn.Linear(self.cube_detector_neurons + self.size_action_history // 2, 1)

    def compute(self, inputs, role):
        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = F.relu(x)
        x = self.fc11(x)
        x = F.relu(x)

        y = self.history_fc1(history_data)
        y = F.relu(y)
        y = self.history_fc2(y)
        y = F.relu(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x, {}


class CubeDetectorModel:
    # model_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/thesis-isaaclab-direct-cubechaser-camera/finetune/cube_detection_faster_rcnn/"
    model_path = "/home/phil/Documents/"
    # model_name = "fasterrcnn_cube_detector.pth"
    model_name = "fasterrcnn_cube_detector_mobilenet.pth"

    transforms = T.Compose([
        T.ToTensor()
    ])
    image_target_width = 320
    image_target_height = 240

    def __init__(self, device):
        print("device:", device)
        self.device = device
        if device == torch.device("cpu"):
            # sometimes issues with GPU, this way model loads correctly
            self.model = torch.load(self.model_path + self.model_name, map_location=torch.device('cpu')).to(self.device)
        else:
            self.model = torch.load(self.model_path + self.model_name).to(self.device)
        self.model.eval()

    def predict_image(self, image):
        # Notes:
        # - using resnet 50 as backbone, this takes up to 0.2 seconds on cpu
        # - using mobilenet as backbone, this takes up to 0.04 seconds on cpu
        # - mobilenet with cuda takes 0.01-0.02 seconds
        # print("predict_image...: ", image.size)
        # return torch.zeros((1, 5), device=self.device), torch.zeros((1, 4), device=self.device)

        transformed_image = self.transforms(image).unsqueeze(0).to(self.device)
        # print("apply model to iamge shape:", transformed_image.shape)
        start = time.time()
        predictions = self.model(transformed_image)
        print(f"model prediction took: {time.time() - start}")
        print("predictions:", predictions)
        # print("predictions['scores']:", predictions[0]['scores'])
        index_highest_prediction = torch.argmax(predictions[0]['scores']) if predictions[0]['scores'].shape[0] > 0 else 0
        print("index_highest_prediction:", index_highest_prediction)

        boxes = torch.stack(
            [pred['boxes'][index_highest_prediction] if pred['boxes'].shape[index_highest_prediction] > 0 else torch.zeros(4, device=self.device) for pred in
             predictions])
        labels = torch.tensor([pred['scores'][index_highest_prediction] if pred['boxes'].shape[index_highest_prediction] > 0 else 0 for pred in predictions],
                              device=self.device)
        print("boxes:", boxes)
        print("labels:", labels)
        cube_detector_results = torch.zeros((1, 5), device=self.device)
        cube_detector_results[:, :4] = boxes
        cube_detector_results[:, 4] = labels
        cube_detector_results[:, [0, 2]] /= self.image_target_height
        cube_detector_results[:, [1, 3]] /= self.image_target_width
        return cube_detector_results, boxes


class CubeDetectorRlAgent:
    # commits of this model:
    # https://gitlab.com/phil-masterthesis/isaaclab-direct-cubechaser-camera/-/blob/0e494eace7dc9676aa6921bb7742c85784cbd448/cubechaser_camera_env_cube_detector.py
    # https://gitlab.com/phil-masterthesis/isaaclab-direct-cubechaser-camera/-/blob/0e494eace7dc9676aa6921bb7742c85784cbd448/cubechaser_skrl_cnn_func_cube_detector.py
    # "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/model/CCC-skrl-cd-fastercnn-v0/24-09-24_08-34-18-330494_PPO/agent_66000.pt"
    # /home/phil/Documents/24-12-08_05-12-47-861514_PPO/agent_69000.pt
    model_path = "/home/phil/Documents/24-12-08_05-12-47-861514_PPO/"
    model_name = "agent_69000.pt"
    memory_size = 16
    num_envs = 1
    num_actions = 2

    observation_counter = 0

    size_action_history = 8
    size_cube_detector_history = 6
    cube_detector_neurons = 5
    cd_neurons = cube_detector_neurons * size_cube_detector_history
    num_observations = cd_neurons + size_action_history

    update_cube_detector_history_each_steps = 2

    single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(num_actions,))
    action_space = gym.vector.utils.batch_space(single_action_space, num_envs)

    single_observation_space = gym.spaces.Dict()
    single_observation_space["policy"] = gym.spaces.Box(
        low=0.0, high=1.0, dtype=np.float32,
        shape=(num_observations,)
    )
    observation_space = gym.vector.utils.batch_space(single_observation_space, num_envs)

    change_greater_025 = 0
    change_greater_05 = 0
    change_greater_075 = 0

    def __init__(self, device):
        self.action_history = torch.zeros(1, self.size_action_history, device=device, dtype=torch.float32)
        self.cube_detector_history = torch.zeros(1, self.size_cube_detector_history * self.cube_detector_neurons, device=device, dtype=torch.float32)

        self.cube_detector_model = CubeDetectorModel(device)

        self.device = device
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["grad_norm_clip"] = 0.5  # needed?

        self.memory = RandomMemory(memory_size=self.memory_size, num_envs=self.num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = Policy(self.observation_space, self.action_space, self.device, clip_actions=True, cube_detector_neurons=self.cd_neurons, size_action_history=self.size_action_history)
        self.models["value"] = Value(self.observation_space, self.action_space, self.device, cube_detector_neurons=self.cd_neurons,size_action_history=self.size_action_history)
        self.agent = PPO(models=self.models,
                    memory=self.memory,
                    cfg=cfg,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    device=self.device)
        self.agent.load(self.model_path + self.model_name)
        print("CubeDetectorRlAgent init done")
        time.sleep(1)
        self.agent.set_running_mode("eval")

    def get_observation(self, image):
        print("get_observation...")
        self.observation_counter += 1
        cube_detector_output, boxes = self.cube_detector_model.predict_image(image)

        if self.observation_counter > 0 and self.observation_counter % self.update_cube_detector_history_each_steps == 0:
            self.cube_detector_history = torch.roll(self.cube_detector_history, shifts=-self.cube_detector_neurons, dims=1)
        self.cube_detector_history[:, -self.cube_detector_neurons:] = cube_detector_output

        action_history_normalized = (self.action_history.clone() + 1.0) / 2.0

        obs = torch.cat((self.cube_detector_history, action_history_normalized), dim=1)

        return obs, boxes

    def get_action(self, observation):
        action = self.agent.act(observation, 500, 10000)[0]
        self.action_history = torch.roll(self.action_history, shifts=-self.num_actions, dims=1)
        self.action_history[:, -self.num_actions:] = action
        change_in_action_value = torch.sum(torch.abs(self.action_history[:, -2:] - self.action_history[:, -4:-2]), dim=1)
        if change_in_action_value > 0.25:
            self.change_greater_025 += 1
        if change_in_action_value > 0.5:
            self.change_greater_05 += 1
        if change_in_action_value > 0.75:
            self.change_greater_075 += 1
        return action
    


class ExplorationPolicy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", ogm_grid_size=256,
                 action_history_size=10, orientation_size=1, depth_image_height=16, depth_image_width=320,
                 depth_image_history_size=3, depth_network_selector="res_small", include_ogm_data=True,
                 num_envs=1, num_layers=2, hidden_size=256, sequence_length=48):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        print("ExplorationPolicy init, num_envs: ", num_envs)

        self.ogm_grid_size = ogm_grid_size
        self.action_history_size = action_history_size
        self.orientation_size = orientation_size
        self.depth_image_height = depth_image_height
        self.depth_image_width = depth_image_width
        self.depth_image_history_size = depth_image_history_size

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.depth_network_selector = depth_network_selector
        self.include_ogm_data = include_ogm_data

    
        ### OGM
        # original
        if self.include_ogm_data:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
            self.batch_norm_1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=3)
            self.batch_norm_2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.batch_norm_3 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(1024, 256)


        '''
        depth image network
        '''

        if self.depth_network_selector == "res_small":
            # ResDepthCNN for depth image network
            # small version
            self.depth_initial_filters = 16
            self.depth_res_small_conv1 = nn.Sequential(
                nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU()
            )

            self.depth_res_small_maxpool = nn.MaxPool2d(2, 2)

            # Residual blocks
            self.depth_res_small_res1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(initial_filters)
            )

            self.depth_res_small_res2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=(3, 3),
                          stride=(2, 2), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters * 2),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 2, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )

            self.depth_res_small_downsample1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=1,
                          stride=2),
                # nn.BatchNorm2d(initial_filters * 2)
            )

            # Final layers
            self.depth_res_small_fc = nn.Sequential(
                nn.Linear(640, self.hidden_size),
                nn.ReLU(),
                # nn.Dropout(0.2),
                # nn.Linear(320, self.hidden_size),
                # nn.ReLU()
            )
        elif self.depth_network_selector == "res_big":
            ### big version
            self.depth_initial_filters = 32
            self.depth_res_conv_1 = nn.Sequential(
                nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=7, stride=2, padding=3,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU()
            )

            self.depth_res_res_1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=3, stride=1, padding=1,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=3, stride=1, padding=1,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters)
            )
            self.depth_res_res_2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 2, kernel_size=(3, 3), stride=(1, 1),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )
            self.depth_res_res_3 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 4, self.depth_initial_filters * 4, kernel_size=(3, 3), stride=(1, 1),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4)
            )

            self.depth_res_downsample2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=1, stride=2, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )
            self.depth_res_downsample3 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=1, stride=2,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4)
            )

            self.depth_res_fc = nn.Sequential(
                nn.Linear((320 // 8) * (16 // 8) * (self.depth_initial_filters * 4), 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, self.hidden_size),
                nn.ReLU()
            )
        else:
            # depth_original
            self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, 8, kernel_size=(3, 6), stride=(1, 2),
                                          padding=1)
            '''
            Note: BatchNorm2D is not suitable for RL usage, because: (Claude)
                Data is highly correlated (sequential states)
                Training/evaluation boundaries are blurred
                Small/single batch updates are common
                The input distribution shifts as the policy improves
            '''
            self.depth_batch_norm_1 = nn.BatchNorm2d(8)
            self.pool_depth_1 = nn.MaxPool2d(2, 2)
            self.conv_depth_2 = nn.Conv2d(8, 16, kernel_size=(3, 6), stride=(1, 2), padding=0)
            self.depth_batch_norm_2 = nn.BatchNorm2d(16)
            self.pool_depth_2 = nn.MaxPool2d(2, 2)
            self.fc_depth_1 = nn.Linear(864, self.hidden_size)


        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc_history_orientation = nn.Linear(self.action_history_size + self.orientation_size, (self.action_history_size + self.orientation_size)*2)
        # self.fc_history_bn = nn.BatchNorm1d((self.action_history_size + self.orientation_size)*2)

        if self.include_ogm_data:
            self.fc_combined_1 = nn.Linear(256 + self.hidden_size + (self.action_history_size + self.orientation_size) * 2, 256)
            self.fc_combined_2 = nn.Linear(256, 64)
        else:
            self.fc_combined_1 = nn.Linear(self.hidden_size + (self.action_history_size + self.orientation_size) * 2, self.hidden_size * 2)
            self.fc_combined_2 = nn.Linear(self.hidden_size * 2, 64)
        self.fc_combined_3 = nn.Linear(64, self.num_actions)

        self.dropout = nn.Dropout(0.2)
        self.dropout_depth_fc = nn.Dropout(0.1)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.debug_print = False
        self.export_images = False

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                  # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs,
                                   self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        self.compute_counter += 1

        # x = inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 2, 1)
        ogm_input = inputs["states"][:, :self.ogm_grid_size*self.ogm_grid_size]
        ogm_input = ogm_input.view(-1, 1, self.ogm_grid_size, self.ogm_grid_size)

        depth_input = inputs["states"][:, self.ogm_grid_size*self.ogm_grid_size:self.ogm_grid_size*self.ogm_grid_size + self.depth_image_height*self.depth_image_width*self.depth_image_history_size]
        depth_input = depth_input.view(-1, self.depth_image_history_size, self.depth_image_height, self.depth_image_width)

        history_orientation_input = inputs["states"][:, self.ogm_grid_size*self.ogm_grid_size + self.depth_image_height*self.depth_image_width*self.depth_image_history_size:]

        if self.export_images and (self.compute_counter < 550):  #  or (1500 < self.compute_counter < 2500)
            depth_as_image = depth_input.view(-1, 1, self.depth_image_height, self.depth_image_width).clone()
            print("depth_as_image shape: ", depth_as_image.shape)
            save_image(make_grid(depth_as_image, nrow=1, padding=10),
                       '/home/phil/university/thesis/data/images/policy_depth_image_export_' + str(self.compute_counter) + '.png')

            ogm_as_image = ogm_input.view(-1, 1, self.ogm_grid_size, self.ogm_grid_size).clone()
            print("ogm_as_image shape: ", ogm_as_image.shape)
            save_image(make_grid(ogm_as_image, nrow=2, padding=10),
                          '/home/phil/university/thesis/data/images/policy_ogm_image_export_' + str(self.compute_counter) + '.png')

        if self.debug_print and self.compute_counter < 100:
            print("Policy: ", inputs["states"].shape)
            print("self.observation_space.shape: ", self.observation_space.shape)
            print("x: ", inputs["states"].shape)
            print("ogm_input: ", ogm_input.shape)
            print(f"ogm max: {torch.max(ogm_input)}, min: {torch.min(ogm_input)}")
            print("depth_input: ", depth_input.shape)
            print(f"depth max: {torch.max(depth_input)}, min: {torch.min(depth_input)}")
            print("history_orientation_input: ", history_orientation_input.shape)
            print(f"history_orientation max: {torch.max(history_orientation_input)}, min: {torch.min(history_orientation_input)}")

        if self.include_ogm_data:
            x = self.conv1(ogm_input)
            # x = self.batch_norm_1(x)
            x = F.relu(x)
            x = self.conv2(x)
            # x = self.batch_norm_2(x)
            x = F.relu(x)
            x = self.conv3(x)
            # x = self.batch_norm_3(x)
            x = F.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            # x = torch.tanh(x)
            x = F.relu(x)


        if self.depth_network_selector == "res_small":
            depth = self.depth_res_small_conv1(depth_input)
            depth = self.depth_res_small_maxpool(depth)
            depth_identity = depth.clone()
            depth = self.depth_res_small_res1(depth)
            depth += depth_identity
            depth = F.relu(depth)
            depth = self.depth_res_small_maxpool(depth)
            depth_identity = self.depth_res_small_downsample1(depth)
            depth = self.depth_res_small_res2(depth)
            depth += depth_identity
            depth = F.relu(depth)
            depth = self.depth_res_small_maxpool(depth)
            depth = torch.flatten(depth, start_dim=1)
            depth = self.depth_res_small_fc(depth)
        elif self.depth_network_selector == "res_big":
            depth = self.depth_res_conv_1(depth_input)
            # First residual block
            depth = self.depth_res_res_1(depth)
            depth = F.relu(depth)
            initial_depth = depth
            depth = self.depth_res_res_2(depth)
            depth += self.depth_res_downsample2(initial_depth)
            depth = F.relu(depth)

            initial_depth = depth
            depth = self.depth_res_res_3(depth)
            depth += self.depth_res_downsample3(initial_depth)
            depth = F.relu(depth)

            depth = torch.flatten(depth, start_dim=1)
            depth = self.depth_res_fc(depth)
        else:
            # depth_original
            depth = self.conv_depth_1(depth_input)
            depth = F.relu(depth)
            depth = self.pool_depth_1(depth)
            depth = self.conv_depth_2(depth)
            depth = F.relu(depth)
            depth = self.pool_depth_2(depth)
            depth = torch.flatten(depth, start_dim=1)
            depth = self.fc_depth_1(depth)
            depth = F.relu(depth)


        states = depth
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.debug_print:
            print("---Policy before lstm, states shape: ", states.shape)
            print("hidden_states: ", hidden_states.shape)
            print("cell_states: ", cell_states.shape)
            for i in range(states.shape[0]):
                # print first 5 values
                print(f"states[{i}]: {states[i, :5]}")
        if self.training:
            rnn_input = states.view(-1, self.sequence_length,
                                    states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            if self.debug_print:
                print("training rnn_input: ", rnn_input.shape)
                # print first 5 values of each rnn_input
                for i in range(rnn_input.shape[0]):
                    for j in range(rnn_input.shape[1]):
                        print(f"rnn_input[{i}, {j}]: {rnn_input[i, j, :5]}")

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length,
                                               hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length,
                                           cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [
                    self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:, i0:i1, :],
                                                                         (hidden_states, cell_states))
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            if self.debug_print:
                print("not training, rnn_input: ", rnn_input.shape)
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        if self.debug_print:
            print("")

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        y = self.fc_history_orientation(history_orientation_input)
        y = F.relu(y)

        if self.include_ogm_data:
            x = torch.cat((x, rnn_output, y), dim=1)
        else:
            x = torch.cat((rnn_output, y), dim=1)
        x = self.fc_combined_1(x)
        x = F.relu(x)
        x = self.fc_combined_2(x)
        x = F.relu(x)
        x = self.fc_combined_3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class ExplorationValue(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, ogm_grid_size=256,
                 action_history_size=10, orientation_size=1, depth_image_height=16, depth_image_width=320,
                 depth_image_history_size=3, depth_network_selector="res_small", include_ogm_data=True,
                 num_envs=1, num_layers=2, hidden_size=256, sequence_length=48):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.ogm_grid_size = ogm_grid_size
        self.action_history_size = action_history_size
        self.orientation_size = orientation_size
        self.depth_image_height = depth_image_height
        self.depth_image_width = depth_image_width
        self.depth_image_history_size = depth_image_history_size

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.depth_network_selector = depth_network_selector
        self.include_ogm_data = include_ogm_data

        '''
        OGM
        '''
        # original
        if self.include_ogm_data:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
            self.batch_norm_1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=3)
            self.batch_norm_2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.batch_norm_3 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(1024, 256)


        '''
        depth image network
        '''

        if self.depth_network_selector == "res_small":
            # ResDepthCNN for depth image network
            # small version
            self.depth_initial_filters = 16
            self.depth_res_small_conv1 = nn.Sequential(
                nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU()
            )

            self.depth_res_small_maxpool = nn.MaxPool2d(2, 2)

            # Residual blocks
            self.depth_res_small_res1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(initial_filters)
            )

            self.depth_res_small_res2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=(3, 3),
                          stride=(2, 2), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters * 2),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 2, kernel_size=(3, 3),
                          stride=(1, 1), padding=1),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )

            self.depth_res_small_downsample1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=1,
                          stride=2),
                # nn.BatchNorm2d(initial_filters * 2)
            )

            # Final layers
            self.depth_res_small_fc = nn.Sequential(
                nn.Linear(640, self.hidden_size),
                nn.ReLU(),
                # nn.Dropout(0.2),
                # nn.Linear(320, self.hidden_size),
                # nn.ReLU()
            )
        elif self.depth_network_selector == "res_big":
            ### big version
            self.depth_initial_filters = 32
            self.depth_res_conv_1 = nn.Sequential(
                nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=7, stride=2, padding=3,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU()
            )

            self.depth_res_res_1 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=3, stride=1, padding=1,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters, kernel_size=3, stride=1, padding=1,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters)
            )
            self.depth_res_res_2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 2, kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )
            self.depth_res_res_3 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=(3, 3),
                          stride=(2, 2),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4),
                nn.ReLU(),
                nn.Conv2d(self.depth_initial_filters * 4, self.depth_initial_filters * 4, kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1, bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4)
            )

            self.depth_res_downsample2 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=1, stride=2,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 2)
            )
            self.depth_res_downsample3 = nn.Sequential(
                nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=1, stride=2,
                          bias=True),
                # nn.BatchNorm2d(self.depth_initial_filters * 4)
            )
            self.depth_res_fc = nn.Sequential(
                nn.Linear((320 // 8) * (16 // 8) * (self.depth_initial_filters * 4), 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, self.hidden_size),
                nn.ReLU()
            )

        else:
            # depth_original
            self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, 8, kernel_size=(3, 6), stride=(1, 2),
                                          padding=1)
            self.depth_batch_norm_1 = nn.BatchNorm2d(8)
            self.pool_depth_1 = nn.MaxPool2d(2, 2)
            self.conv_depth_2 = nn.Conv2d(8, 16, kernel_size=(3, 6), stride=(1, 2), padding=0)
            self.depth_batch_norm_2 = nn.BatchNorm2d(16)
            self.pool_depth_2 = nn.MaxPool2d(2, 2)
            self.fc_depth_1 = nn.Linear(864, self.hidden_size)


        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc_history_orientation = nn.Linear(self.action_history_size + self.orientation_size,
                                                (self.action_history_size + self.orientation_size)*2)
        # self.fc_history_bn = nn.BatchNorm1d((self.action_history_size + self.orientation_size)*2)

        if self.include_ogm_data:
            self.fc_combined_1 = nn.Linear(256 + self.hidden_size + (self.action_history_size + self.orientation_size) * 2, 256)
            self.fc_combined_2 = nn.Linear(256, 64)
        else:
            self.fc_combined_1 = nn.Linear(self.hidden_size + (self.action_history_size + self.orientation_size) * 2, self.hidden_size * 2)
            self.fc_combined_2 = nn.Linear(self.hidden_size * 2, 64)
        self.fc_combined_3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.dropout_depth_fc = nn.Dropout(0.1)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                  # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs,
                                   self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        ogm_input = inputs["states"][:, :self.ogm_grid_size * self.ogm_grid_size]
        ogm_input = ogm_input.view(-1, 1, self.ogm_grid_size, self.ogm_grid_size)

        depth_input = inputs["states"][:,
                      self.ogm_grid_size * self.ogm_grid_size:self.ogm_grid_size * self.ogm_grid_size + self.depth_image_height * self.depth_image_width * self.depth_image_history_size]
        depth_input = depth_input.view(-1, self.depth_image_history_size, self.depth_image_height, self.depth_image_width)

        history_orientation_input = inputs["states"][:,
                                    self.ogm_grid_size * self.ogm_grid_size + self.depth_image_height * self.depth_image_width * self.depth_image_history_size:]

        if self.include_ogm_data:
            x = self.conv1(ogm_input)
            # x = self.batch_norm_1(x)
            x = F.relu(x)
            x = self.conv2(x)
            # x = self.batch_norm_2(x)
            x = F.relu(x)
            x = self.conv3(x)
            # x = self.batch_norm_3(x)
            x = F.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            # x = torch.tanh(x)
            x = F.relu(x)
            # x = self.dropout(x)

        if self.depth_network_selector == "res_small":
            depth = self.depth_res_small_conv1(depth_input)
            depth = self.depth_res_small_maxpool(depth)
            depth_identity = depth.clone()
            depth = self.depth_res_small_res1(depth)
            depth += depth_identity
            depth = F.relu(depth)
            depth = self.depth_res_small_maxpool(depth)
            depth_identity = self.depth_res_small_downsample1(depth)
            depth = self.depth_res_small_res2(depth)
            depth += depth_identity
            depth = F.relu(depth)
            depth = self.depth_res_small_maxpool(depth)
            depth = torch.flatten(depth, start_dim=1)
            depth = self.depth_res_small_fc(depth)
        elif self.depth_network_selector == "res_big":
            depth = self.depth_res_conv_1(depth_input)
            # First residual block
            depth = self.depth_res_res_1(depth)
            depth = F.relu(depth)
            initial_depth = depth
            depth = self.depth_res_res_2(depth)
            depth += self.depth_res_downsample2(initial_depth)
            depth = F.relu(depth)

            initial_depth = depth
            depth = self.depth_res_res_3(depth)
            depth += self.depth_res_downsample3(initial_depth)
            depth = F.relu(depth)

            depth = torch.flatten(depth, start_dim=1)
            depth = self.depth_res_fc(depth)
        else:
            # depth_original
            depth = self.conv_depth_1(depth_input)
            # depth = self.depth_batch_norm_1(depth)
            depth = F.relu(depth)
            depth = self.pool_depth_1(depth)
            depth = self.conv_depth_2(depth)
            # depth = self.depth_batch_norm_2(depth)
            depth = F.relu(depth)
            depth = self.pool_depth_2(depth)
            depth = torch.flatten(depth, start_dim=1)
            depth = self.fc_depth_1(depth)
            # depth = torch.tanh(depth)
            depth = F.relu(depth)
            # depth = self.dropout(depth)


        states = depth
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]


        if self.training:
            print("SHOULD NOT HAPPEN???!!!")
            # print("self.training")
            rnn_input = states.view(-1, self.sequence_length,
                                    states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            # print("rnn_input: ", rnn_input.shape)

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length,
                                               hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length,
                                           cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [
                    self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:, i0:i1, :],
                                                                         (hidden_states, cell_states))
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            # print("not training")
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            # print("rnn_input: ", rnn_input.shape)
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # print("")

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        y = self.fc_history_orientation(history_orientation_input)
        y = F.relu(y)

        # ReLU is generally better than tanh for hidden layers because it helps avoid vanishing gradients and provides faster training
        if self.include_ogm_data:
            x = torch.cat((x, rnn_output, y), dim=1)
        else:
            x = torch.cat((rnn_output, y), dim=1)
        x = self.fc_combined_1(x)
        x = F.relu(x)
        x = self.fc_combined_2(x)
        x = F.relu(x)
        x = self.fc_combined_3(x)

        return x, {"rnn": [rnn_states[0], rnn_states[1]]}


class ExplorationAgent:
    model_path = "/home/phil/Documents/24-12-12_21-57-51-400216_PPO_RNN/"
    model_name = "agent_40000.pt"
    mini_batches = 32
    num_actions = 2
    sequence_length = 15
    lstm_num_layers = 2
    lstm_hidden_size = 128
    include_ogm_data = False
    depth_network_selector = "res_small"

    def __init__(self, cfg, device):
        print("INIT EXPLORATION AGENT!!!")
        self.expo_num_envs = 1  # TODO: use cfg.num_envs? but seems like loaded agent expects 1 env
        self.sim_num_envs = cfg["num_envs"]  # use for policy and value, because lstm needs correct batch size

        self.device = device
        self.memory_size = self.sequence_length * self.mini_batches

        single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(single_action_space, self.expo_num_envs)

        single_observation_space = gym.spaces.Dict()
        single_observation_space["policy"] = gym.spaces.Box(
            low=0.0, high=1.0, dtype=np.float32,
            shape=(cfg["exp_num_observations"],)
        )
        self.observation_space = gym.vector.utils.batch_space(single_observation_space, self.expo_num_envs)

        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg["grad_norm_clip"] = 0.5  # needed?
        ppo_cfg["rollouts"] = self.memory_size

        self.memory = RandomMemory(memory_size=self.memory_size, num_envs=self.expo_num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = ExplorationPolicy(self.observation_space, self.action_space, self.device, clip_actions=True,
                                                  ogm_grid_size=cfg["ogm_grid_size"], action_history_size=cfg["exp_size_action_history"],
                                                  orientation_size=cfg["exp_size_robot_orientation_history"],
                                                  depth_image_height=cfg["depth_image_slice_size"],
                                                  depth_image_width=cfg["image_target_width"],
                                                  depth_image_history_size=cfg["depth_image_history_size"]+1,
                                                  sequence_length=self.sequence_length, num_envs=self.sim_num_envs,
                                                  num_layers=self.lstm_num_layers, hidden_size=self.lstm_hidden_size,
                                                  depth_network_selector=self.depth_network_selector,
                                                  include_ogm_data=self.include_ogm_data
                                                  )
        self.models["value"] = ExplorationValue(self.observation_space, self.action_space, self.device,
                                                ogm_grid_size=cfg["ogm_grid_size"],
                                                action_history_size=cfg["exp_size_action_history"],
                                                orientation_size=cfg["exp_size_robot_orientation_history"],
                                                depth_image_height=cfg["depth_image_slice_size"],
                                                depth_image_width=cfg["image_target_width"],
                                                depth_image_history_size=cfg["depth_image_history_size"]+1,
                                                sequence_length=self.sequence_length, num_envs=self.sim_num_envs,
                                                num_layers=self.lstm_num_layers, hidden_size=self.lstm_hidden_size,
                                                depth_network_selector=self.depth_network_selector,
                                                include_ogm_data=self.include_ogm_data
                                                )
        self.agent = PPO_RNN(models=self.models,
                             memory=self.memory,
                             cfg=ppo_cfg,
                             observation_space=self.observation_space,
                             action_space=self.action_space,
                             device=self.device
                             )
        self.agent.load(self.model_path + self.model_name)
        self.agent.set_running_mode("eval")

        # initialize rnn states
        self.rnn_states = {"policy": [], "value": []}
        for i, size in enumerate(self.models["policy"].get_specification().get("rnn", {}).get("sizes", [])):
            self.rnn_states["policy"].append(torch.zeros(size).to(self.device))
        for i, size in enumerate(self.models["value"].get_specification().get("rnn", {}).get("sizes", [])):
            self.rnn_states["value"].append(torch.zeros(size).to(self.device))
        # TODO: maybe have to reset states if episode has ended! see ppo_rnn -> record_transition
        print("ExplorationAgent init done")

    def reset_rnn_states(self, env_id):
        for rnn_state in self.rnn_states["policy"]:
            rnn_state[:, env_id] = 0
        for rnn_state in self.rnn_states["value"]:
            rnn_state[:, env_id] = 0

    def act(self, observations):
        # return of act:
        '''
        The first component is the action to be taken by the agent.
        The second component is the log of the probability density function for stochastic models
        or None for deterministic models. The third component is a dictionary containing extra output values
        '''

        policy_rnn = {"rnn": self.rnn_states["policy"]}
        policy = self.agent.policy.act({"states": observations, **policy_rnn}, role="policy")
        self.rnn_states["policy"] = policy[2].get("rnn", [])
        print("shape rnn_states: ", len(self.rnn_states["policy"]))

        # Note: value network not needed for exploration agent only!
        # value_rnn = {"rnn": self.rnn_states["value"]}
        # value = self.agent.value.act({"states": observations, **value_rnn}, role="value")
        # self.rnn_states["value"] = value[2].get("rnn", [])

        mean_actions = policy[2].get("mean_actions", policy[0])  # TODO: check if wanna use mean actions or not

        return mean_actions, policy[1], None, policy[0]  # value[0]

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
        value_rnn = {"rnn": self.rnn_states["value"]}
        return self.agent.value.act({"states": observations, **value_rnn}, role="value")[0]
    

class ArbitrationAgentNode:
    def __init__(self, device):
        print("ArbitrationAgentNode init")

        # EXPLO AGENT
        self.image_slice_level = 1/5  # 2/5  (top to bottom)

        self.num_envs = 1
        self.device = device
        self.agent_config = {
            "ogm_grid_size": 256,
            "image_target_width": 320,
            "image_target_height": 240,
            "depth_image_slice_size": 16,
            "middle_slice_start": None,
            "middle_slice_end": None,
            "depth_image_max_distance": 3.0,
            "depth_image_history_size": 0,
            "exp_size_action_history": 8,
            "exp_size_robot_orientation_history": 5,
            "num_envs": self.num_envs,
            "exp_num_observations": None
        }
        self.agent_config["middle_slice_start"] = int(self.agent_config["image_target_height"]*self.image_slice_level)  # 1/5
        self.agent_config["middle_slice_end"] = self.agent_config["middle_slice_start"] + self.agent_config["depth_image_slice_size"]
        self.agent_config["exp_num_observations"] = self.agent_config["ogm_grid_size"] * self.agent_config["ogm_grid_size"] + self.agent_config["exp_size_robot_orientation_history"] + self.agent_config["exp_size_action_history"] + self.agent_config["depth_image_slice_size"] * self.agent_config["image_target_width"] * (self.agent_config["depth_image_history_size"] + 1)

        self.exploration_agent = ExplorationAgent(self.agent_config, device)
        self.cd_agent = CubeDetectorRlAgent(device)


        rospy.init_node('ArbitrationAgentNode')
        rospy.on_shutdown(self.shutdown)
        temp_twist = Twist()
        temp_twist.linear.x = 0.0
        self.global_vel_pub = rospy.Publisher('/jetauto_controller/cmd_vel', Twist, queue_size=1)  # for some reason needs to be global...

        self.global_vel_pub.publish(temp_twist)
        time.sleep(1)

        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback_guard, queue_size=1)
        self.image_callback_running = False

        
        rospy.Subscriber('/camera/depth/image', Image, self.depth_image_callback_guard)
        rospy.Subscriber('/imu_data', Imu, self.imu_callback)
        self.initial_imu_data = None  # use to reset imu to zero orientation  # TODO: what is zero orientation for isaac sim?!
        self.last_yaw_orientation = None
        self.robot_yaw_history = torch.zeros((self.num_envs, 3), device=self.device)  # this is only used for lidar maps
        self.robot_orientation_history = torch.zeros(self.agent_config["num_envs"], self.agent_config["exp_size_robot_orientation_history"], device=self.device, dtype=torch.float32)
        self.occupancy_grid_maps = torch.zeros((self.num_envs, self.agent_config["ogm_grid_size"], self.agent_config["ogm_grid_size"]), device=self.device)

        self.action_history = torch.zeros(self.num_envs, self.agent_config["exp_size_action_history"], device=self.device, dtype=torch.float32)

        self.depth_image_callback_running = False
        self.shutting_down = False

        self.image_counter = 0
        self.observation_counter = 0
        self.store_depth_images = True

        self.total_action_count = 0
        self.total_time_between_image_and_action = 0
        self.total_run_time = None

        self.image_path = "/home/phil/university/thesis/ros-sim-2-real-v2/local_ws/src/fourwis_cubechaser_sb3/scripts/exploration_agent/images/"
        print("init done!")

        self.last_explo_action = None
        self.last_explo_value_estimate = None
        self.last_explo_log_odds = None

        self.last_cube_action = None
        self.last_cube_value_estimate = None
        self.last_cube_log_odds = None

        self.has_new_cube_actions = False
        self.has_new_explo_actions = False

        self.agent_selection_history_size = 5
        self.ma_observation_history_size = 3

        self.multi_agent_observation_history = torch.ones(1,
                                                           self.ma_observation_history_size * self.single_observation_size,
                                                           device=self.sim.device, dtype=torch.float32) * -1.0

    def get_arbitration_action(self):
        if not self.has_new_cube_actions or not self.has_new_explo_actions:
            return None
        
        single_observation = torch.cat((self.last_explo_value_estimate, self.last_cube_value_estimate), dim=1)


        self.has_new_cube_actions = False
        self.has_new_explo_actions = False



    def explo_get_action_from_agent(self, depth_image_tensor):
        if self.last_yaw_orientation is None:
            return None
        self.observation_counter += 1
        explo_agent_input = torch.cat((self.occupancy_grid_maps.view(self.num_envs, -1), depth_image_tensor), dim=1)
        action_history_normalized = (self.action_history + 1.0) / 2.0  # normalize to [0, 1]
        explo_agent_input = torch.cat((explo_agent_input, action_history_normalized), dim=1)
        explo_agent_input = torch.cat((explo_agent_input, self.robot_orientation_history), dim=1)

        explo_agent_mean_action, explo_agent_log_odds, explo_agent_value, explo_agent_action = self.exploration_agent.act(explo_agent_input)
        self.last_explo_action = explo_agent_mean_action
        self.last_explo_value_estimate = explo_agent_value
        self.last_explo_log_odds = explo_agent_log_odds
        self.has_new_explo_actions = True
        return explo_agent_mean_action, explo_agent_action
    
    def cube_get_action_from_agent(self, observation):
        cube_agent_mean_action, cube_agent_log_odds, cube_agent_value, cube_agent_action = self.cd_agent.act(observation)
        self.last_cube_action = cube_agent_mean_action
        self.last_cube_value_estimate = cube_agent_value
        self.last_cube_log_odds = cube_agent_log_odds
        self.has_new_cube_actions = True
        return cube_agent_mean_action


    def imu_callback(self, msg):
        # print("imu: ", msg)
        orientation = msg.orientation
        tensor_orientation_wxyz = torch.tensor([[orientation.w, orientation.x, orientation.y, orientation.z]], device=self.device)
        if self.initial_imu_data is None:
            self.initial_imu_data = tensor_orientation_wxyz
        tensor_orientation_wxyz = tensor_orientation_wxyz - self.initial_imu_data
        euler = self.euler_from_quaternions(tensor_orientation_wxyz)
        self.last_yaw_orientation = euler[:, 2]
        # print("imu: ", self.last_yaw_orientation)
        self.robot_yaw_history = self.robot_yaw_history.roll(-1, dims=1)
        self.robot_yaw_history[:, -1] = self.last_yaw_orientation

        if self.observation_counter % 3 == 0:
            self.robot_orientation_history = torch.roll(self.robot_orientation_history, shifts=-1, dims=1)
        self.robot_orientation_history[:, -1] = self.last_yaw_orientation / (2 * math.pi)  # normalize to [0, 1]

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
    
    def image_callback_guard(self, msg):
        if self.shutting_down:
            return
        if not self.image_callback_running:
            self.image_callback(msg)

    def depth_image_callback_guard(self, msg):
        if self.shutting_down:
            return
        if not self.depth_image_callback_running:
            self.depth_image_callback(msg)

    def image_callback(self, msg):
        if self.image_callback_running:
            return
        if self.total_run_time is None:
            self.total_run_time = time.time()
        msg = self.most_recent_image
        self.image_counter += 1
        start_time = time.time()
        self.image_callback_running = True
        print("")
        print("self.image_counter:", self.image_counter)
        print(f'image received, width: {msg.width}, height: {msg.height}, encoding: {msg.encoding}, step: {msg.step}')
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # print("image size:", cv_image.shape)
        cv_image_resized = cv2.resize(cv_image, (self.cd_agent.cube_detector_model.image_target_width, self.cd_agent.cube_detector_model.image_target_height))
        cv_image_resized = cv2.rotate(cv_image_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imwrite(
        #     "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/" + "test.jpg",
        #     cv_image_resized)
        # print("image size after rotation:", cv_image_resized.shape)
        observation, boxes = self.cd_agent.get_observation(
            pil_image.fromarray(cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2RGB)))

        # print("observation:", observation)
        print("boxes:", boxes)
        # save image to drive
        startX = int(boxes[0][0])
        startY = int(boxes[0][1])
        endX = int(boxes[0][2])
        endY = int(boxes[0][3])
        cv2.rectangle(cv_image_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # rotate image by 90 degrees
        cv_image_resized = cv2.rotate(cv_image_resized, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imwrite(
        #     "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/sample_images/" + "robot_cam_with_pred_" + str(self.image_counter) + ".jpg",
        #     cv_image_resized)

        action = self.cube_get_action_from_agent(observation)
        self.image_callback_running = False
        print(f"image callback took: {time.time() - start_time}")
        self.get_arbitration_action()

    def store_depth_image_as_png(self, image, name="depth_image_tensor_"):
        save_image(image, self.image_path + name + ".png")  # + str(self.observation_counter)

    def depth_image_callback(self, msg):
        if self.depth_image_callback_running:
            return
        print("")
        if self.total_run_time is None:
            self.total_run_time = time.time()
        self.depth_image_callback_running = True
        start_time = time.time()
        self.image_counter += 1
        # depth image received, width: 640, height: 480, encoding: 32FC1, step: 2560
        # print(f'depth image received, width: {msg.width}, height: {msg.height}, encoding: {msg.encoding}, step: {msg.step}')
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")  # /image, image_rect: 32FC1 , /image_raw: 16UC1
        ### for saving the image
        # Normalize the depth image to the range 0-255
        if self.store_depth_images:
            normalized_image = cv2.normalize(cv_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(
                self.image_path + "depth_image" + str(self.image_counter) + ".jpg",
                normalized_image)
        # print("image size:", normalized_image.shape)  # image size: (480, 640)

        # first get max value:
        temp_np = np.array(cv_image)
        temp_np = np.nan_to_num(temp_np, nan=0)
        initial_max_value = temp_np.max()

        # remove all black pixels
        cv_normalized = cv2.normalize(cv_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv_image_np = np.array(cv_normalized)
        print(f"cv_image_np shape: {cv_image_np.shape}")
        for y in range(0, cv_image_np.shape[0]):
            for x in range(cv_image_np.shape[1]//2, 0, -1):
                if cv_image_np[y, x] <= 5:
                    cv_image_np[y, x] = cv_image_np[y, x+1]
            for x in range(cv_image_np.shape[1]//2, cv_image_np.shape[1]):
                if cv_image_np[y, x] <= 5:
                    cv_image_np[y, x] = cv_image_np[y, x-1]
        print(f"cv_image_np min max: {cv_image_np.min()}, {cv_image_np.max()}")
        cv_image_cleaned = cv_image_np/255.0

        # Note: the outer part of the depth image contains a lot of nan values due to the camera. We should crop 50 pixels on each side
        pixels_to_crop_pct = 0.05
        pixels_to_crop_w = int(pixels_to_crop_pct * msg.width)
        pixels_to_crop_h = int(pixels_to_crop_pct * msg.height)
        new_width = msg.width - 2 * pixels_to_crop_w
        new_height = msg.height - 2 * pixels_to_crop_h
        cv_image_cropped = cv_image_cleaned[pixels_to_crop_h:new_height + pixels_to_crop_h, pixels_to_crop_w:new_width + pixels_to_crop_w]
        cv_image_resized = cv2.resize(cv_image_cropped, (self.agent_config["image_target_width"], self.agent_config["image_target_height"]))

        transform = T.ToTensor()
        depth_tensor = transform(cv_image_resized).to(self.device, dtype=torch.float32)

        # set nan to 0.0
        depth_tensor[torch.isnan(depth_tensor)] = 0.0
        depth_tensor = depth_tensor * initial_max_value


        print(f"depth tensor min max: {depth_tensor.min()}, {depth_tensor.max()}")
        depth_tensor = torch.clamp(depth_tensor, 0, self.agent_config["depth_image_max_distance"]) / self.agent_config["depth_image_max_distance"]

        # print(f"depth image tensor: {depth_tensor.shape}")  # torch.Size([1, 240, 320])

        if self.store_depth_images:
            self.store_depth_image_as_png(depth_tensor[0], "depth_image_tensor_" + str(self.image_counter))

        depth_tensor = depth_tensor[:, self.agent_config["middle_slice_start"]:self.agent_config["middle_slice_end"], :]    
        if self.store_depth_images:
            self.store_depth_image_as_png(depth_tensor[0], "depth_image_middle_slice_" + str(self.image_counter))

        # depth_tensor = torch.clamp(depth_tensor, 0, self.agent_config["depth_image_max_distance"]) / self.agent_config["depth_image_max_distance"]
        # if self.store_depth_images:
        #     self.store_depth_image_as_png(depth_tensor[0], "depth_image_middle_slice_clamped_" + str(self.image_counter))
        depth_tensor = depth_tensor.view(self.num_envs, -1)


        # print("depth_tensor shape: ", depth_tensor.shape)  # torch.Size([1, 5120])

        mean_action, action = self.explo_get_action_from_agent(depth_tensor)
        self.total_action_count += 1
        if mean_action is not None:
            print(f"mean_action: {mean_action} - orignal action: {action}")
            mean_action = torch.clamp(mean_action, -1.0, 1.0)
            mask = (mean_action >= -0.05) & (mean_action <= 0.05)
            mean_action[mask] = 0.0
            self.action_history = torch.roll(self.action_history, shifts=-2, dims=1)
            self.action_history[:, -2:] = mean_action

        self.depth_image_callback_running = False
        print(f"depth image callback took: {time.time() - start_time}")
        self.total_time_between_image_and_action += time.time() - start_time
        self.get_arbitration_action()


    def send_action(self, action):
        # print(f'sending action: {action}')
        twist_msg = Twist()
        if abs(action[0][1].item()) < 0.75:  # TODO: why again? I think to prevent driving forward if robot turns too much
            twist_msg.linear.x = action[0][0].item() * 0.1  # 0.2
        # else:
        #     twist_msg.linear.x = action[0][0].item() * 0.05
        twist_msg.angular.z = action[0][1].item() * 0.4
        
        print(f"twist_msg: x: {twist_msg.linear.x}, z: {twist_msg.angular.z}")
        self.global_vel_pub.publish(twist_msg)

    def shutdown(self):
        print("shutting down")
        self.total_run_time = time.time() - self.total_run_time
        self.shutting_down = True
        rospy.sleep(1)
        self.global_vel_pub.publish(Twist())
        rospy.sleep(1)
        print("----------------")
        print(f"total_action_count: {self.total_action_count} - avg total_time_between_image_and_action: {self.total_time_between_image_and_action / self.total_action_count}")
        print(f"total_run_time: {self.total_run_time}")