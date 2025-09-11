from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

class A1DreamWaQ(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    # def _compute_torques(self, actions):
    #     # Choose between pd controller and actuator network
    #     if self.cfg.control.use_actuator_network:
    #         with torch.inference_mode():
    #             self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
    #             self.sea_input[:, 0, 1] = self.dof_vel.flatten()
    #             torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
    #         return torques
    #     else:
    #         # pd controller
    #         return super()._compute_torques(actions)
        
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel,
                                    self.projected_gravity,
                                    self.commands,
                                    self.dof_pos,
                                    self.dof_vel,
                                    self.last_actions
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        return super()._reward_tracking_lin_vel()
    
    def _reward_tracking_ang_vel(self):
        return super()._reward_tracking_ang_vel()
    
    def _reward_base_lin_vel_z(self):
        return super()._reward_lin_vel_z()
    
    def _reward_ang_vel_xy(self):
        return super()._reward_ang_vel_xy()
    
    def _reward_orientation(self):
        return super()._reward_orientation()
    
    def _reward_dof_acc(self):
        return super()._reward_dof_acc()
    
    def _reward_dof_power(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=-1)

    def _reward_base_height(self):
        return super()._reward_base_height()
    
    # def _reward_foot_clearance(self):
    #     in_contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
    #     in_swing = torch.logical_not(in_contact)

    #     foot_states = self.rigid_body_state[:, self.feet_indices, :]

    #     foot_height = foot_states[:, :, 2]
    #     foot_velocity_xy = foot_states[:, :, 7:9]

    #     foot_height_error = torch.square(self.cfg.rewards.foot_height_target - foot_height)

    #     return torch.sum(torch.square(self.cfg.rewards.foot_height_target - foot_height), dim=-1) * foot_velocity_xy

    def _reward_action_rate(self):
        return super()._reward_action_rate()
    
    def _reward_smootheness(self):
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.second_last_actions), dim=-1)

    def _reward_power_distribution(self):
        power_terms = self.torques * self.dof_vel

        return torch.var(power_terms, dim=-1, unbiased=False)
