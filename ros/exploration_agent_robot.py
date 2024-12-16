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
    # "/home/phil/Documents/24-11-06_08-34-59-950245_PPO_RNN/", "agent_66000.pt": best agent
    # /home/phil/Documents/24-11-21_01-21-35-155916_PPO_RNN/  : first agent with patches, training not done yet
    #   agent_21000: somewhat okay
    #   agent_26000: not so good
    #   TODO: try agent agent_60000, best agent with new patches
    model_path = "/home/phil/Documents/24-12-13_15-54-22-158312_PPO_RNN/"
    model_name = "agent_180000.pt"
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


'''
ToDos
- get depth image working
- try out lidar, but not too important right now
    - check lidar type: os.environ.get('LIDAR_TYPE')
- check positioning and rotation information, probably an issue?

LIDAR:   G4
CAMERA:  AstraProPlus

'''


class ExploAgentNode:
    def __init__(self, device):
        print("ExploAgentNode init")

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
        # default configuration, but a bit too low on the real robot
        # self.agent_config["middle_slice_start"] = (self.agent_config["image_target_height"] - self.agent_config["depth_image_slice_size"]) // 2
        self.agent_config["middle_slice_start"] = int(self.agent_config["image_target_height"]*self.image_slice_level)  # 1/5
        self.agent_config["middle_slice_end"] = self.agent_config["middle_slice_start"] + self.agent_config["depth_image_slice_size"]
        # grid_size * grid_size + exp_size_robot_orientation_history + exp_size_action_history + depth_image_slice_size * scaled_image_width * (depth_image_history_size + 1)
        self.agent_config["exp_num_observations"] = self.agent_config["ogm_grid_size"] * self.agent_config["ogm_grid_size"] + self.agent_config["exp_size_robot_orientation_history"] + self.agent_config["exp_size_action_history"] + self.agent_config["depth_image_slice_size"] * self.agent_config["image_target_width"] * (self.agent_config["depth_image_history_size"] + 1)

        self.exploration_agent = ExplorationAgent(self.agent_config, device)


        rospy.init_node('ExploAgentNode')
        rospy.on_shutdown(self.shutdown)
        temp_twist = Twist()
        temp_twist.linear.x = 0.0
        print("publishing temp_twist")
        self.global_vel_pub = rospy.Publisher('/jetauto_controller/cmd_vel', Twist, queue_size=1)  # for some reason needs to be global...

        self.global_vel_pub.publish(temp_twist)
        time.sleep(1)

        print("subscribe camera")
        # /camera/depth/image_raw -> useless
        # /camera/depth/image
        # /camera/depth/image_rect -> not much better
        rospy.Subscriber('/camera/depth/image', Image, self.depth_image_callback_guard)
        # rospy.Subscriber('/camera/depth/points', PointCloud2, self.depth_points_callback)

        ### IMU ###
        # imu_corrected: only has ancular_velocity and linear_acceleration, orientation is always zero
        # imu_data: also contains orientation! in x,y,z,w
        # imu_raw: same as imu_corrected
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html
        rospy.Subscriber('/imu_data', Imu, self.imu_callback)
        self.initial_imu_data = None  # use to reset imu to zero orientation  # TODO: what is zero orientation for isaac sim?!
        self.last_yaw_orientation = None
        self.robot_yaw_history = torch.zeros((self.num_envs, 3), device=self.device)  # this is only used for lidar maps
        self.robot_orientation_history = torch.zeros(self.agent_config["num_envs"], self.agent_config["exp_size_robot_orientation_history"], device=self.device, dtype=torch.float32)


        ### LIDAR ###
        # point cloud works and gives x,y,z coordinates. scan contains just empty ranges. But point cloud is good anyway
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
        # rospy.Subscriber('/scan', LaserScan, self.lidar_callback)  # not working
        # rospy.Subscriber('/point_cloud', PointCloud, self.point_cloud_callback)  # works
        self.occupancy_grid_maps = torch.zeros((self.num_envs, self.agent_config["ogm_grid_size"], self.agent_config["ogm_grid_size"]), device=self.device)



        self.action_history = torch.zeros(self.num_envs, self.agent_config["exp_size_action_history"], device=self.device, dtype=torch.float32)

        self.depth_image_callback_running = False
        self.shutting_down = False

        self.image_counter = 0
        self.observation_counter = 0
        self.store_depth_images = True

        self.min_action_value = [1.0, 1.0, 1.0, 1.0]
        self.max_action_value = [-1.0, -1.0, -1.0, -1.0]

        self.image_path = "/home/phil/university/thesis/ros-sim-2-real-v2/local_ws/src/fourwis_cubechaser_sb3/scripts/exploration_agent/images/"
        print("init done!")


    def get_action_from_agent(self, depth_image_tensor):
        if self.last_yaw_orientation is None:
            return None
        self.observation_counter += 1
        # print("self.occupancy_grid_maps shape: ", self.occupancy_grid_maps.view(self.num_envs, -1).shape)
        # print("depth_image_tensor shape: ", depth_image_tensor.shape)
        # print("self.action_history shape: ", self.action_history.shape)
        # print("self.robot_orientation_history shape: ", self.robot_orientation_history.shape)
        # print("self.action_history: ", self.action_history)
        explo_agent_input = torch.cat((self.occupancy_grid_maps.view(self.num_envs, -1), depth_image_tensor), dim=1)
        action_history_normalized = (self.action_history + 1.0) / 2.0  # normalize to [0, 1]
        explo_agent_input = torch.cat((explo_agent_input, action_history_normalized), dim=1)
        explo_agent_input = torch.cat((explo_agent_input, self.robot_orientation_history), dim=1)
        # Note: in simulation: torch.Size([4, 70681])
        # print("explo_agent_input shape: ", explo_agent_input.shape)  # torch.Size([1, 70681])

        explo_agent_mean_action, explo_agent_log_odds, explo_agent_value, explo_agent_action = self.exploration_agent.act(explo_agent_input)
        return explo_agent_mean_action, explo_agent_action


    def point_cloud_callback(self, msg):
        print(f"Point cloud, points: {msg.points[90:95]}")

    def lidar_callback(self, msg):
        # print("lidar: ", msg)
        print(f"lidar: angle_min: {msg.angle_min}, angle_max: {msg.angle_max}, angle_increment: {msg.angle_increment}")
        print(f"lidar: range_min: {msg.range_min}, range_max: {msg.range_max}")
        # first 10 range data values
        print(f"lidar: ranges: {msg.ranges[90:120]}")
        print(f"lidar: intensities: {msg.intensities[90:120]}")

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
    
    def depth_points_callback(self, pointcloud_msg):
        '''
        Useless result...
        '''
        print(f" Point cloud width: {pointcloud_msg.width}, height: {pointcloud_msg.height}, point_step: {pointcloud_msg.point_step}, row_step: {pointcloud_msg.row_step}, is_dense: {pointcloud_msg.is_dense}")

        # Extract points from the PointCloud2 message
        # points = np.array(list(pc2.read_points(pointcloud_msg, skip_nans=True)))

        # # Assuming points are in (x, y, z) format
        # x = points[:, 0]
        # y = points[:, 1]
        # z = points[:, 2]

        # # Create an image from the z values (depth)
        # # You might need to adjust scaling and data type based on your needs
        # depth_image = (z - np.min(z)) / (np.max(z) - np.min(z)) * 255
        # depth_image = depth_image.astype(np.uint8)
        # depth_image = cv2.resize(depth_image, (pointcloud_msg.width, pointcloud_msg.height))

        # if self.store_depth_images:
        #     cv2.imwrite(
        #         self.image_path + "depth_image_from_pointcloud" + str(self.image_counter) + ".jpg",
        #         depth_image)

        pc_array = []
        for point in pc2.read_points(pointcloud_msg, skip_nans=True):
            pc_array.append([point[0], point[1], point[2]])  # x, y, z coordinates
        
        pc_array = np.array(pc_array)
        
        # Create empty image
        image = np.zeros((pointcloud_msg.height, pointcloud_msg.width), dtype=np.uint8)
        
        if len(pc_array) > 0:
            # Normalize coordinates to image space
            x_normalized = ((pc_array[:, 0] - pc_array[:, 0].min()) / 
                          (pc_array[:, 0].max() - pc_array[:, 0].min()) * 
                          (pointcloud_msg.width - 1))
            y_normalized = ((pc_array[:, 1] - pc_array[:, 1].min()) / 
                          (pc_array[:, 1].max() - pc_array[:, 1].min()) * 
                          (pointcloud_msg.height - 1))
            
            # Convert to integer coordinates
            x_pixels = x_normalized.astype(np.int32)
            y_pixels = y_normalized.astype(np.int32)
            
            # Use depth (z-coordinate) for pixel intensity
            z_normalized = ((pc_array[:, 2] - pc_array[:, 2].min()) / 
                          (pc_array[:, 2].max() - pc_array[:, 2].min()) * 255)
            
            # Set pixel values
            for i in range(len(x_pixels)):
                if 0 <= x_pixels[i] < pointcloud_msg.width and 0 <= y_pixels[i] < pointcloud_msg.height:
                    image[y_pixels[i], x_pixels[i]] = int(z_normalized[i])

        # Save image
        if self.store_depth_images:
            cv2.imwrite(
                self.image_path + "depth_image_from_pointcloud" + str(self.image_counter) + ".jpg",
                image)

    def depth_image_callback_guard(self, msg):
        if self.shutting_down:
            return
        if not self.depth_image_callback_running:
            self.depth_image_callback(msg)

    def store_depth_image_as_png(self, image, name="depth_image_tensor_"):
        save_image(image, self.image_path + name + ".png")  # + str(self.observation_counter)

    def fill_nan_with_neighbors(self, depth_image, kernel_size=9):
        """Fills NaN values in a depth image with the average of their valid neighbors.

        Args:
            depth_image: A torch.Tensor representing the depth image.

        Returns:
            A torch.Tensor with NaN values replaced.
        """
        print("depth_image shape: ", depth_image.shape)

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        # Get indices of NaN values
        nan_indices = torch.isnan(depth_image)

        # Create a kernel for neighbor averaging (adjust size as needed)
        kernel = torch.ones((kernel_size, kernel_size), device=depth_image.device)
        kernel[kernel_size // 2, kernel_size // 2] = 0  # Exclude the central pixel

        padding = kernel_size // 2

        # Perform convolution to calculate the sum of valid neighbors
        padded_image = torch.nn.functional.pad(depth_image, (padding, padding, padding, padding), mode='reflect')
        neighbor_sums = torch.nn.functional.conv2d(
            padded_image, # .unsqueeze(0).unsqueeze(0)
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=0
        ).squeeze()

        # Count the number of valid neighbors
        valid_neighbor_counts = torch.nn.functional.conv2d(
            (~nan_indices).float(),  # .unsqueeze(0).unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=0
        ).squeeze()

        # Calculate the average of valid neighbors
        print("neighbor_sums shape: ", neighbor_sums.shape)
        print("valid_neighbor_counts shape: ", valid_neighbor_counts.shape)
        neighbor_averages = neighbor_sums / valid_neighbor_counts

        # Replace NaN values with the calculated averages
        depth_image[nan_indices] = neighbor_averages[nan_indices]

        print("depth_image shape: ", depth_image.shape)
        return depth_image

    def depth_image_callback(self, msg):
        if self.depth_image_callback_running:
            return
        print("")
        self.depth_image_callback_running = True
        start_time = time.time()
        self.image_counter += 1
        # depth image received, width: 640, height: 480, encoding: 32FC1, step: 2560
        # print(f'depth image received, width: {msg.width}, height: {msg.height}, encoding: {msg.encoding}, step: {msg.step}')
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")  # /image, image_rect: 32FC1 , /image_raw: 16UC1
        ### for saving the image
        # Normalize the depth image to the range 0-255
        # if self.store_depth_images:
        #     normalized_image = cv2.normalize(cv_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
        #     cv2.imwrite(
        #         self.image_path + "depth_image" + str(self.image_counter) + ".jpg",
        #         normalized_image)
        # print("image size:", normalized_image.shape)  # image size: (480, 640)

        # Note: the outer part of the depth image contains a lot of nan values due to the camera. We should crop 50 pixels on each side
        pixels_to_crop_pct = 0.05
        pixels_to_crop_w = int(pixels_to_crop_pct * msg.width)
        pixels_to_crop_h = int(pixels_to_crop_pct * msg.height)
        new_width = msg.width - 2 * pixels_to_crop_w
        new_height = msg.height - 2 * pixels_to_crop_h
        cv_image_cropped = cv_image[pixels_to_crop_h:new_height + pixels_to_crop_h, pixels_to_crop_w:new_width + pixels_to_crop_w]
        cv_image_resized = cv2.resize(cv_image_cropped, (self.agent_config["image_target_width"], self.agent_config["image_target_height"]))

        # Note: somewhat working but replaced values are "not the best choice" and some weird "raster"
        # replace nan values with average of neighbors using cv2, run twice to get better results
        # mask = np.isnan(cv_image_resized).astype(np.uint8)
        # cv_image_resized = cv2.inpaint(cv_image_resized.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)  # INPAINT_TELEA  INPAINT_NS
        # mask = np.isnan(cv_image_resized).astype(np.uint8)
        # cv_image_resized = cv2.inpaint(cv_image_resized.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)

        transform = T.ToTensor()
        depth_tensor = transform(cv_image_resized).to(self.device)

        # Note_ Not really working... maybe try again
        # valid_mask = ~torch.isnan(depth_tensor)
        # grid_y, grid_x = torch.meshgrid(torch.arange(depth_tensor.shape[1]), 
        #                            torch.arange(depth_tensor.shape[2]), indexing='ij')

        # grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float().to(self.device)
        # print("grid shape: ", grid.shape)
        # depth_input = depth_tensor.unsqueeze(0)
        # # Interpolate using nearest valid neighbors
        # filled_tensor = torch.nn.functional.grid_sample(
        #     depth_input, 
        #     grid, 
        #     mode='nearest', 
        #     padding_mode='border'
        # ).squeeze()
        # depth_tensor = torch.where(valid_mask, depth_tensor, filled_tensor)


        # first replace nan values, then we get max value to use for previous nan values
        # works but kinda destroys a lot of depth information...
        # nan_mask = torch.isnan(depth_tensor)
        # depth_tensor[nan_mask] = 0.0
        # max_value = torch.max(depth_tensor)
        # depth_tensor[nan_mask] = max_value

        # set nan to 0.0
        depth_tensor[torch.isnan(depth_tensor)] = 0.0


        # depth_tensor[torch.isnan(depth_tensor)] =  self.agent_config["depth_image_max_distance"]  # TODO: try 0.0  # Note: maybe also good enough?
        # depth_tensor = self.fill_nan_with_neighbors(depth_tensor)  # fill nan values with average of neighbors  # TODO: try out

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

        mean_action, action = self.get_action_from_agent(depth_tensor)
        if mean_action is not None:
            print(f"mean_action: {mean_action} - orignal action: {action}")
            mean_action = torch.clamp(mean_action, -1.0, 1.0)
            mask = (mean_action >= -0.05) & (mean_action <= 0.05)
            mean_action[mask] = 0.0
            self.action_history = torch.roll(self.action_history, shifts=-2, dims=1)
            self.action_history[:, -2:] = mean_action

            self.min_action_value[0] = min(self.min_action_value[0], mean_action[0][0].item())
            self.min_action_value[1] = min(self.min_action_value[1], mean_action[0][1].item())
            self.min_action_value[2] = min(self.min_action_value[2], action[0][0].item())
            self.min_action_value[3] = min(self.min_action_value[3], action[0][1].item())
            self.max_action_value[0] = max(self.max_action_value[0], mean_action[0][0].item())
            self.max_action_value[1] = max(self.max_action_value[1], mean_action[0][1].item())
            self.max_action_value[2] = max(self.max_action_value[2], action[0][0].item())
            self.max_action_value[3] = max(self.max_action_value[3], action[0][1].item())

            # self.send_action(action)  

        self.depth_image_callback_running = False
        print(f"depth image callback took: {time.time() - start_time}")


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
        self.shutting_down = True
        rospy.sleep(1)
        self.global_vel_pub.publish(Twist())
        rospy.sleep(1)
        print("----------------")
        print(f"min action value: {self.min_action_value} - max action value: {self.max_action_value}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

explo_agent_node = ExploAgentNode(device)


print("exploration agent rl robot started")
# rospy.init_node('hoge')
# global_vel_pub2 = rospy.Publisher('/jetauto_controller/cmd_vel', Twist, queue_size=10)
# temp_twist = Twist()
# temp_twist.linear.x = 0.1
rate = rospy.Rate(15)  # Refresh rate in hertz  # TODO: increase?

while not rospy.is_shutdown():
    # global_vel_pub2.publish(temp_twist)
    rate.sleep()