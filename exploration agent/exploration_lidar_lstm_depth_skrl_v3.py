"""
Using separate networks for policy and value
"""

from omni.isaac.lab.app import AppLauncher

import socket

import getpass
# run settings
headless = socket.gethostname() != "phil-SCHENKER-MEDIA-E23"
training = True
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app


import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry
import torch.nn.functional as F

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory, Memory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import StepLR

class Policy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", ogm_grid_size=256,
                 action_history_size=10, orientation_size=1, depth_image_height=16, depth_image_width=320,
                 depth_image_history_size=3, depth_network_selector="res_small", include_ogm_data=True,
                 num_envs=1, num_layers=2, hidden_size=256, sequence_length=48):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

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

        # self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=1)
        # self.pool1 = nn.MaxPool2d(3, 2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        # self.pool2 = nn.MaxPool2d(3, 2)
        # self.ogm_fc = nn.Linear(3136, 256)

        '''
        depth image network
        '''
        # self.depth_initial_filters = 16
        # self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_1 = nn.BatchNorm2d(self.depth_initial_filters)
        # self.pool_depth_1 = nn.MaxPool2d(2, 2)
        #
        # self.conv_depth_2 = nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_2 = nn.BatchNorm2d(self.depth_initial_filters * 2)
        # self.pool_depth_2 = nn.MaxPool2d(2, 2)
        #
        # self.conv_depth_3 = nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_3 = nn.BatchNorm2d(self.depth_initial_filters * 4)
        # self.pool_depth_3 = nn.MaxPool2d(2, 2)
        #
        # self.fc_depth_1 = nn.Linear((320 // 8) * (16 // 8) * (self.depth_initial_filters * 4), 512)

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
        self.export_ogm = False

        self.collection_depth_as_image = torch.zeros((15, 14, self.depth_image_width), dtype=torch.float32, device=device)  # self.depth_image_height

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

        if self.export_images:  #  or (1500 < self.compute_counter < 2500)  # self.compute_counter % 25 == 0
            self.collection_depth_as_image = torch.roll(self.collection_depth_as_image, shifts=1, dims=0)
            temp_depth = depth_input.view(-1, 1, self.depth_image_height, self.depth_image_width)[0, 0, 1:15, :].clone()
            self.collection_depth_as_image[-1] = temp_depth
            save_image(make_grid(self.collection_depth_as_image.unsqueeze(1), nrow=1, padding=2),
                       '/home/phil/university/thesis/data/images/policy_depth_image_export_' + str(
                           self.compute_counter) + '.png')

            # depth_as_image = depth_input.view(-1, 1, self.depth_image_height, self.depth_image_width).clone()
            # print("depth_as_image shape: ", depth_as_image.shape)
            # save_image(make_grid(depth_as_image, nrow=1, padding=10),
            #            '/home/phil/university/thesis/data/images/policy_depth_image_export_' + str(self.compute_counter) + '.png')

        if self.export_ogm and (self.compute_counter < 20):
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
            # x = self.dropout(x)

            # x = self.conv1(ogm_input)
            # x = F.relu(x)
            # x = self.pool1(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.pool2(x)
            # x = torch.flatten(x, start_dim=1)
            # x = self.ogm_fc(x)
            # x = torch.tanh(x)

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

        # BatchNorm2D: instability and reduced performance if using small batch sizes...
        # depth = self.conv_depth_1(depth_input)
        # # depth = self.depth_batch_norm_1(depth)
        # depth = F.relu(depth)
        # depth = self.pool_depth_1(depth)
        #
        # depth = self.conv_depth_2(depth)
        # # depth = self.depth_batch_norm_2(depth)
        # depth = F.relu(depth)
        # depth = self.pool_depth_2(depth)
        #
        # depth = self.conv_depth_3(depth)
        # # depth = self.depth_batch_norm_3(depth)
        # depth = F.relu(depth)
        # depth = self.pool_depth_3(depth)
        #
        # depth = torch.flatten(depth, start_dim=1)
        # depth = self.fc_depth_1(depth)
        # depth = F.relu(depth)  # RELU preferred in CNN to prevent vanish gradient...
        # depth = self.dropout_depth_fc(depth)




        states = depth
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # print("policy states: ", states.shape)

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
        # y = self.fc_history_bn(y)
        # y = torch.tanh(y)
        y = F.relu(y)

        if self.include_ogm_data:
            x = torch.cat((x, rnn_output, y), dim=1)
        else:
            x = torch.cat((rnn_output, y), dim=1)
        x = self.fc_combined_1(x)
        x = F.relu(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.fc_combined_2(x)
        x = F.relu(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.fc_combined_3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class Value(DeterministicMixin, Model):
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

        # self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=1)
        # self.pool1 = nn.MaxPool2d(3, 2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        # self.pool2 = nn.MaxPool2d(3, 2)
        # self.ogm_fc = nn.Linear(3136, 256)

        '''
        depth image network
        '''
        # self.depth_initial_filters = 16
        # self.conv_depth_1 = nn.Conv2d(self.depth_image_history_size, self.depth_initial_filters, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_1 = nn.BatchNorm2d(self.depth_initial_filters)
        # self.pool_depth_1 = nn.MaxPool2d(2, 2)
        #
        # self.conv_depth_2 = nn.Conv2d(self.depth_initial_filters, self.depth_initial_filters * 2, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_2 = nn.BatchNorm2d(self.depth_initial_filters * 2)
        # self.pool_depth_2 = nn.MaxPool2d(2, 2)
        #
        # self.conv_depth_3 = nn.Conv2d(self.depth_initial_filters * 2, self.depth_initial_filters * 4, kernel_size=3,
        #                               stride=1, padding=1)
        # self.depth_batch_norm_3 = nn.BatchNorm2d(self.depth_initial_filters * 4)
        # self.pool_depth_3 = nn.MaxPool2d(2, 2)
        #
        # self.fc_depth_1 = nn.Linear((320 // 8) * (16 // 8) * (self.depth_initial_filters * 4), 512)

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

            # x = self.conv1(ogm_input)
            # x = F.relu(x)
            # x = self.pool1(x)
            # x = self.conv2(x)
            # x = F.relu(x)
            # x = self.pool2(x)
            # x = torch.flatten(x, start_dim=1)
            # x = self.ogm_fc(x)
            # x = torch.tanh(x)

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

        # training
        # print("---Value before lstm, states shape: ", states.shape)
        # print("hidden_states: ", hidden_states.shape)
        # print("cell_states: ", cell_states.shape)
        # for i in range(states.shape[0]):
        #     # print first 5 values
        #     print(f"states[{i}]: {states[i, :5]}")
        if self.training:
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
        # y = self.fc_history_bn(y)
        # y = torch.tanh(y)
        y = F.relu(y)

        # ReLU is generally better than tanh for hidden layers because it helps avoid vanishing gradients and provides faster training
        if self.include_ogm_data:
            x = torch.cat((x, rnn_output, y), dim=1)
        else:
            x = torch.cat((rnn_output, y), dim=1)
        x = self.fc_combined_1(x)
        x = F.relu(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.fc_combined_2(x)
        x = F.relu(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.fc_combined_3(x)

        return x, {"rnn": [rnn_states[0], rnn_states[1]]}


print("Exploration lidar SKRL")
env_cfg = load_cfg_from_registry("Exploration-Lidar-v3", "env_cfg_entry_point")
env = gym.make("Exploration-Lidar-v3", cfg=env_cfg, asynchronous=False)  # try order_enforce? Probably not: If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions

# Note: doesnt work with isaac lab and multi-environment implementation
# env = gym.vector.make("Exploration-Lidar-v1", cfg=env_cfg, asynchronous=False)  # gymnasium.vector.make(...)` is deprecated and will be replaced by `gymnasium.make_vec(...)` in v1.0"
# env = gym.make_vec("Exploration-Lidar-v1", asynchronous=False, num_envs=env_cfg.scene.num_envs, cfg=env_cfg, vectorization_mode="sync")

# wrap environment to enforce that reset is called before step
# env = gym.wrappers.OrderEnforcing(env)
env = SkrlVecEnvWrapper(env)  # internally calls wrap_env from skrl, Environment wrapper: Isaac Lab

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
device = env.device

print("---------------------------------------------------------")
print("env numenvs: ", env.num_envs)


'''
Notes to memory
batches = BatchSampler(self.all_sequence_indexes, batch_size=len(self.all_sequence_indexes) // mini_batches, drop_last=True)
self.all_sequence_indexes = np.concatenate([np.arange(i, memory_size * num_envs + i, num_envs) for i in range(num_envs)])

'''
# batch size and sequence length: Due to memory implementation, sequence length should be equal to memory_size
# Careful: mini_batches = 1 causes lot of issues / instability with learning. Just does not really learn at all...
sequence_length = 15
mini_batches = 32
memory_size = sequence_length * mini_batches
# batch_size = (memory_size * env.num_envs) // mini_batches


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
models = {}

grid_size = env.unwrapped.cfg.grid_size
action_history_size = env.unwrapped.cfg.size_action_history
orientation_size = env.unwrapped.cfg.size_robot_orientation_history
depth_image_height = env.unwrapped.cfg.depth_image_slice_size
depth_image_width = env.unwrapped.cfg.scaled_image_width
depth_image_history_size = env.unwrapped.cfg.depth_image_history_size + 1

depth_network_selector = "res_small"  # res_small "depth_original"  # "res_big
lstm_num_layers = 1
lstm_hidden_size = 128
include_ogm_data = False

models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True,
                          ogm_grid_size=grid_size, action_history_size=action_history_size, orientation_size=orientation_size,
                          depth_image_height=depth_image_height, depth_image_width=depth_image_width,
                          depth_image_history_size=depth_image_history_size, sequence_length=sequence_length, num_envs=env.num_envs,
                          num_layers=lstm_num_layers, hidden_size=lstm_hidden_size, depth_network_selector=depth_network_selector,
                          include_ogm_data=include_ogm_data)
models["value"] = Value(env.observation_space, env.action_space, device,
                        ogm_grid_size=grid_size, action_history_size=action_history_size, orientation_size=orientation_size,
                        depth_image_height=depth_image_height, depth_image_width=depth_image_width,
                        depth_image_history_size=depth_image_history_size, sequence_length=sequence_length, num_envs=env.num_envs,
                        num_layers=lstm_num_layers, hidden_size=lstm_hidden_size, depth_network_selector=depth_network_selector,
                        include_ogm_data=include_ogm_data)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
print("Creating agent")
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 4  # reduced from 8, presumably more stable
# 16 ? 512 ? 128: num envs
# old: 16 * 512 / 128
# (16 * 512) / env.num_envs
cfg["mini_batches"] = mini_batches  # 48  # (memory_size * env.num_envs) // 16
# Note to batch_size:
#   _isaac_sim/kit/python/lib/python3.10/site-packages/skrl/memories/torch/base.py
#       line 378
#   batch_size=len(indexes) // mini_batches
#   indexes = np.arange(self.memory_size * self.num_envs)
cfg["discount_factor"] = 0.99  # gamma
cfg["lambda"] = 0.95  # 0.95

# Note: bigger LR, just drives in circles??
# cfg["learning_rate"] = 5e-4  # 5e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 5e-3}  # default: 1e-3  # , "min_lr": 1e-5 , "min_lr": 5e-6

cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}  # , "lr_factor": 1.2


# cfg["learning_rate"] = 1e-3
# cfg["learning_rate_scheduler"] = StepLR
# cfg["learning_rate_scheduler_kwargs"] = {"step_size": 250, "gamma": 0.9}

# Note: was not that good... but maybe try learning starts again at some point
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
# helps stabilize training by preventing exploding gradients
# lower value can be combined with higher learning rate
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0  # 0.01  # control balance between exploration and exploitation
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0.0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False

# Note: testing without
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
#
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Exploration-Lidar-Lstm-Depth-v3"
cfg["experiment"]["store_separately"] = False

agent = PPO_RNN(models=models,
            memory=memory,  # Note: memory is required for PPO because from here the value for the value function is loaded
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 300000 if training else 100000, "headless": headless}  # 1000000
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])  # or use SkrlSequentialLogTrainer, not sure whats the different with SequentialTrainer

# start training
# print("Starting training")
if training:
    trainer.train()
else:
    print("starting evaluation")
    # agent_22000.pt  - also left oriented and very slow driving, barely makes any distance! DONT USE
    # agent_50000.pt - not looking too good...
    # agent_82000.pt - same shit
    # agent_151000.pt
    # agent_180000.pt - likes to just do left turns...
    path = "/home/phil/Documents/24-12-13_15-54-22-158312_PPO_RNN/agent_82000.pt"
    agent.load(path)
    trainer.eval()
env.close()
