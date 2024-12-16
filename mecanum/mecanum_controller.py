import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import mecanum_direction
import torch
import math
import numpy as np


class MecanumController:
    def __init__(self, num_envs: int, device, default_velocity=10.0):
        self._wheel_radius = 0.04
        self._wheel_base = 0.2
        self.track_width = 0.22
        self.num_envs = num_envs
        self.previous_joint_positions = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_envs)]
        self._default_velocity = default_velocity  # 6.0
        self.ninety_degree_in_radians = 1.5708
        self.fortyfive_degree_in_radians = 0.7854
        self.turn_on_spot_angle = math.atan(self._wheel_base / self.track_width)
        self.max_change_joint_position_per_step = self.fortyfive_degree_in_radians
        self._device = device

        self.last_velo_and_angle = [[0.0, 0.0] for _ in range(self.num_envs)]

        self.max_velo_change = self._default_velocity * 0.4
        self.max_angle_change = self.turn_on_spot_angle * 0.1

    def continuous_to_mecanum_direction(self, actions: torch.Tensor):
        # TODO: is this correct??? nope, dont think so...
        thresholds = torch.tensor([-0.8, -0.33, 0, 0.33, 0.8, 1.0], device=self._device)

        direction_indices = torch.bucketize(actions, thresholds)

        # Convert indices to MecanumDirection enum values
        # return torch.tensor([MecanumDirection(i.item()) for i in direction_indices.view(-1)])
        # return torch.tensor([i.item() for i in direction_indices.view(-1)], dtype=torch.int32)
        return np.array([i.item() for i in direction_indices.view(-1)])

    def drive_command_into_velocities_and_positions(self, index, direction: mecanum_direction.MecanumDirection):
        joint_positions = [0.0, 0.0, 0.0, 0.0]
        joint_velocities = [0.0, 0.0, 0.0, 0.0]
        if direction == mecanum_direction.MecanumDirection.FORWARD:
            joint_velocities = [self._default_velocity, self._default_velocity, self._default_velocity,
                                self._default_velocity]
        elif direction == mecanum_direction.MecanumDirection.BACKWARD:
            joint_velocities = [-self._default_velocity, -self._default_velocity, -self._default_velocity,
                                -self._default_velocity]
        elif direction == mecanum_direction.MecanumDirection.SIDEWAYS_LEFT:
            joint_positions = [self.ninety_degree_in_radians, self.ninety_degree_in_radians,
                               self.ninety_degree_in_radians, self.ninety_degree_in_radians]
            joint_velocities = [self._default_velocity, self._default_velocity, self._default_velocity,
                                self._default_velocity]
        elif direction == mecanum_direction.MecanumDirection.SIDEWAYS_RIGHT:
            joint_positions = [-self.ninety_degree_in_radians, -self.ninety_degree_in_radians,
                               -self.ninety_degree_in_radians, -self.ninety_degree_in_radians]
            joint_velocities = [self._default_velocity, self._default_velocity, self._default_velocity,
                                self._default_velocity]
        elif direction == mecanum_direction.MecanumDirection.ROTATE_CW:
            joint_positions = [self.turn_on_spot_angle, -self.turn_on_spot_angle, -self.turn_on_spot_angle,
                               self.turn_on_spot_angle]
            joint_velocities = [self._default_velocity, -self._default_velocity, self._default_velocity,
                                -self._default_velocity]
        elif direction == mecanum_direction.MecanumDirection.ROTATE_CCW:
            joint_positions = [self.turn_on_spot_angle, -self.turn_on_spot_angle, -self.turn_on_spot_angle,
                               self.turn_on_spot_angle]
            joint_velocities = [-self._default_velocity, self._default_velocity, -self._default_velocity,
                                self._default_velocity]
        else:
            print("!!! Error: Invalid direction: ", direction)

        # if position changed set velocity to zero
        # position_changed = False
        # for i in range(len(joint_positions)):
        #     if joint_positions[i] != self.previous_joint_positions[index][i]:
        #         position_changed = True
        #         break
        # if position_changed:
        #     joint_velocities = [0.0, 0.0, 0.0, 0.0]

        # make sure joint positions dont change to much else robot "flies away"
        for i in range(len(joint_positions)):
            previous_joint_position = self.previous_joint_positions[index][i]
            goal_joint_position = joint_positions[i]
            if abs(goal_joint_position - previous_joint_position) > self.max_change_joint_position_per_step:
                if goal_joint_position > previous_joint_position:
                    joint_positions[i] = previous_joint_position + self.max_change_joint_position_per_step
                else:
                    joint_positions[i] = previous_joint_position - self.max_change_joint_position_per_step
        self.previous_joint_positions[index] = joint_positions

        return torch.tensor(joint_velocities, device=self._device), torch.tensor(joint_positions, device=self._device)

    def actions_softmax_into_velocities_and_positions(self, actions: torch.Tensor, obs_counter) -> tuple[
        torch.Tensor, torch.Tensor]:
        # actions is a tensor of shape (num_envs, 4) with value range -1.0 to 1.0
        # first turn actions into discrete mecanum actions
        all_joint_positions = []
        all_joint_velocities = []
        mecanum_actions = torch.argmax(actions, dim=1)
        if obs_counter < 500000 and obs_counter % 1000 == 0:
            print("actions: ", actions)
            print("mecanum_actions: ", mecanum_actions)
        for index, action in enumerate(mecanum_actions):
            joint_velocities, joint_positions = self.drive_command_into_velocities_and_positions(
                index, mecanum_direction.MecanumDirection(action.item()))
            all_joint_velocities.append(joint_velocities)
            all_joint_positions.append(joint_positions)

        return torch.stack(all_joint_velocities, dim=0), torch.stack(all_joint_positions, dim=0)

    def continuous_actions_into_velocities_and_positions(self, actions: torch.Tensor, only_allow_driving_forward=False, limit_max_change=True):
        # first value: multiply with velocity * 10
        # second value: use for position, multiply with turn_on_spot_angle
        # rear_left and rear_right have same sign, front_left and front_right have opposite sign
        all_joint_positions = []
        all_joint_velocities = []

        # print("actions: ", actions)
        for index, action in enumerate(actions):
            # print("action: ", action)

            velo_action = action[0].item()
            if only_allow_driving_forward:
                velo_action = velo_action + 1.0

            temp_velo = self._default_velocity * velo_action  # 5.0
            if limit_max_change:
                # let temp_velo change be maximum 10% of default velo based on self.last_velo_and_angle
                velo_change = temp_velo - self.last_velo_and_angle[index][0]
                if velo_change > self.max_velo_change:
                    temp_velo = self.last_velo_and_angle[index][0] + self.max_velo_change
                elif velo_change < -self.max_velo_change:
                    temp_velo = self.last_velo_and_angle[index][0] - self.max_velo_change

            temp_velo = max(-self._default_velocity, min(self._default_velocity, temp_velo))
            joint_velocities = [temp_velo] * 4
            self.last_velo_and_angle[index][0] = temp_velo

            temp_angle = self.turn_on_spot_angle * action[1].item()
            if limit_max_change:
                angle_change = temp_angle - self.last_velo_and_angle[index][1]
                # let temp_angle change be maximum 10% of default angle based on self.last_velo_and_angle
                if angle_change > self.max_angle_change:
                    temp_angle = self.last_velo_and_angle[index][1] + self.max_angle_change
                elif angle_change < -self.max_angle_change:
                    temp_angle = self.last_velo_and_angle[index][1] - self.max_angle_change

            temp_angle = max(-self.turn_on_spot_angle, min(self.turn_on_spot_angle, temp_angle))
            joint_positions = [temp_angle, temp_angle, -temp_angle, -temp_angle]
            self.last_velo_and_angle[index][1] = temp_angle

            joint_velocities, joint_positions = torch.tensor(joint_velocities, device=self._device), torch.tensor(
                joint_positions, device=self._device)
            all_joint_velocities.append(joint_velocities)
            all_joint_positions.append(joint_positions)
        # print("all_joint_velocities: ", all_joint_velocities)
        # print("all_joint_positions: ", all_joint_positions)
        return torch.stack(all_joint_velocities, dim=0), torch.stack(all_joint_positions, dim=0)