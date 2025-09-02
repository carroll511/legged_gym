from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class A1(LeggedRobot):
    def _reward_forward(self):
        return torch.clamp(self.base_lin_vel[:, 0], min=0.35)

    def _reward_lateral_movement_and_rotation(self):
        return -torch.square(torch.norm(self.base_lin_vel[:, 1])) - torch.square(torch.norm(self.base_ang_vel[:, 2]))

    def _reward_work(self):
        return -torch.abs((self.torques * (self.dof_vel - self.last_dof_vel)).sum(dim=1))

    def _reward_ground_impact(self):
        return -torch.square(torch.norm(self.contact_forces[:, self.feet_indices, :]))

    def _reward_smoothness(self):
        cur_torque = self.torques
        prev_torque = self._compute_torques(self.last_actions)
        return -torch.square(torch.norm(cur_torque - prev_torque))

    def _reward_action_magnitude(self):
        return -torch.square(torch.norm(self.actions))
    
    def _reward_joint_speed(self):
        return -torch.square(torch.norm(self.dof_vel))
    
    def _reward_orientation(self):
        return -torch.square(torch.norm(self.base_quat[:, 1:4]))

    def _reward_z_acceleration(self):
        return -torch.square(torch.norm(self.base_lin_vel[:, 2]))

    # def _reward_foot_slip(self):