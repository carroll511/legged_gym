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

from legged_gym.envs.a1_dreamwaq.a1_dreamwaq_config import A1DreamWaQCfg

class A1DreamWaQ(LeggedRobot):
    def __init__(self, cfg: A1DreamWaQCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.history_len = cfg.env.history_len # Added for DreamWaQ -> input for encoder

        self.obs_history_buf = torch.zeros(self.num_envs, cfg.env.history_len * self.num_obs, device=self.device, dtype=torch.float) # Added for DreamWaQ -> input for encoder
        self.velocity_truth_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.second_last_actions[env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()

        self.second_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

    def get_observations_history(self):
        return self.obs_history_buf
    
    def get_velocity_truth(self):
        return self.base_lin_vel
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
        current_obs = torch.cat((  self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.last_actions
                                    ),dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = current_obs

        self.obs_history_buf = torch.cat([self.obs_history_buf[:, self.num_obs:], self.obs_buf], dim=1)
        self.obs_history_buf[:, -self.num_obs:] = self.obs_buf

        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)

        disturbance_force = self.contact_forces[:, 0, :] * self.obs_scales.disturbance
        current_obs = torch.cat((current_obs, disturbance_force), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.privileged_obs_buf = current_obs
        
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, history_obs, velocity_truth, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, history_obs, velocity_truth

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf = torch.cat([self.obs_history_buf[:, self.num_obs:], self.obs_buf], dim=1)
        self.obs_history_buf[:, -self.num_obs:] = self.obs_buf

        return self.obs_buf, self.privileged_obs_buf, self.obs_history_buf, self.velocity_truth_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.velocity_truth_buf[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # observation compute and append to history
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.second_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

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
    
    # # def _reward_foot_clearance(self):
    # #     in_contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
    # #     in_swing = torch.logical_not(in_contact)

    # #     foot_states = self.rigid_body_state[:, self.feet_indices, :]

    # #     foot_height = foot_states[:, :, 2]
    # #     foot_velocity_xy = foot_states[:, :, 7:9]

    # #     foot_height_error = torch.square(self.cfg.rewards.foot_height_target - foot_height)

    # #     return torch.sum(torch.square(self.cfg.rewards.foot_height_target - foot_height), dim=-1) * foot_velocity_xy

    def _reward_action_rate(self):
        return super()._reward_action_rate()
    
    # def _reward_smoothness(self):
    #     return torch.sum(torch.square(self.actions - 2*self.last_actions + self.second_last_actions), dim=-1)

    # def _reward_power_distribution(self):
    #     power_terms = self.torques * self.dof_vel

    #     return torch.var(power_terms, dim=-1, unbiased=False)
