# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1DreamWaQCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45 # ang_vel(3), gravity(3), vel_cmd(3), joint_pos(12), joint_vel(12), last_action(12)
        num_privileged_obs = 138 # observation(45), body_lin_vel(3), disturbance(3), height_map(187)
        num_actions = 12
        env_spacing = 3.
        spend_timeouts = True
        episode_length_s = 20
        history_len = 5 # number of previous observations

    # class terrain( LeggedRobotCfg.terrain):
    #     mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    #     horizontal_scale = 0.1 # [m]
    #     vertical_scale = 0.005 # [m]
    #     border_size = 25 # [m]
    #     curriculum = True
    #     static_friction = 1.0
    #     dynamic_friction = 1.0
    #     restitution = 0.
    #     # rough terrain only:
    #     measure_heights = True # 17 * 11
    #     measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
    #     measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    #     selected = False # select a unique terrain type and pass all arguments
    #     terrain_kwargs = None # Dict of arguments for selected terrain
    #     max_init_terrain_level = 5 # starting curriculum state
    #     terrain_length = 8.
    #     terrain_width = 8.
    #     num_rows= 10 # number of terrain rows (levels)
    #     num_cols = 20 # number of terrain cols (types)
    #     # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    #     terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    #     # trimesh only:
    #     slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # lin_vel_x, lin_vel_y, ang_vel_yaw
        resampling_time = 10.
        heading_command = False # ang_vel_yaw command
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad] -> P gain
        damping = {'joint': 0.7}     # [N*m*s/rad] -> D gain
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    # class normalization:
    #     class obs( LeggedRobotCfg.normalization.obs ):
    #         clip_obs = 10.0
    #         lin_vel_scale = 1.0
    #         ang_vel_scale = 1.0
    #         height_measurements_scale = 1.0
    #     class actions( LeggedRobotCfg.normalization.actions ):
    #         clip_actions = 1.0

class A1DreamWaQCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticAsymmetric'
        algorithm_class_name = 'PPODreamWaQ'
        run_name = ''
        experiment_name = 'a1_dreamwaq'

  