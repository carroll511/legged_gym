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
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45 # ang_vel(3), gravity(3), vel_cmd(3), joint_pos(12), joint_vel(12), last_action(12)
        num_privileged_obs = 238 # observations(45), height_map(187), disturbance(3) body_vel(3)
        num_actions = 12
        history_len = 5

    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane'
        measure_heights = True
        slope_threshold = 0.38

    # class commands:
    #     curriculum = False
    #     max_curriculum = 1.
    #     num_commands = 3 # lin_vel_x, lin_vel_y, ang_vel_yaw
    #     resampling_time = 10.
    #     heading_command = False # ang_vel_yaw command
    #     class ranges:
    #         lin_vel_x = [-1.0, 1.0]
    #         lin_vel_y = [-1.0, 1.0]
    #         ang_vel_yaw = [-1, 1]

    class commands:
        curriculum_adaptive = "grid"  # box or grid
        curriculum_x = True
        curriculum_yaw = True
        max_curriculum_x = 1.0
        max_curriculum_y = 0.4
        max_curriculum_yaw = 0.6

        # Passing grade should be in [0,1]
        passing_grade_x = 0.9
        passing_grade_yaw = 0.4
        inc_curriculum = 0.05

        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.0, 0.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.0, 0.0]

        use_zero_command = False
        if use_zero_command:
            zero_command_prob = 0.1

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
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
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

    class domain_rand:
        init_body_lin_vels = [-1.0, 1.0]
        init_body_ang_vels = [-1.5, 1.5]

        randomize_base_mass = False
        added_mass_range = [-1.0, 2.0]

        randomize_kp = True
        kp_factor_range = [0.9, 1.1]

        randomize_kd = True
        kd_factor_range = [0.9, 1.1]

        randomize_motor_strength = True
        motor_strength_factor = [0.8, 1.2]

        randomize_base_com = True
        com_shift_range = [-0.05, 0.05]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_latency = True
        latency_range = [0.0, 0.015]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        max_push_vel_z = 1.0


    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.


    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4 # 0.25 -> 0.4
        foot_height_target = 0.1 # 0.07 -> 0.1
        soft_dof_vel_limit = 0.65 # 1. -> 0.65
        soft_dof_pos_limit = 0.90 # 1. -> 0.90
        max_contact_force = 200. # 100. -> 200.

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            orientation = -0.2
            dof_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            smoothness = -0.01
            power_distribution = -1e-5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

class A1DreamWaQCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunnerDreamWaQ'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticDreamWaQ'
        algorithm_class_name = 'PPODreamWaQ'

        max_iterations = 1000
        run_name = ''
        experiment_name = 'a1_dreamwaq'

  