# # # # walking straight on flat ground
# # # from legged_gym import LEGGED_GYM_ROOT_DIR
# # # import os
# # # import isaacgym
# # # from legged_gym.envs import *
# # # from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

# # # import numpy as np
# # # import torch
# # # import time


# # # def play(args):
# # #     # === Load configs ===
# # #     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

# # #     # --- Environment overrides for testing ---
# # #     env_cfg.env.num_envs = 1
# # #     env_cfg.terrain.num_rows = 1
# # #     env_cfg.terrain.num_cols = 1
# # #     env_cfg.terrain.curriculum = False
# # #     env_cfg.noise.add_noise = False
# # #     env_cfg.domain_rand.randomize_friction = False
# # #     env_cfg.domain_rand.push_robots = False

# # #     # --- 🐛 수정: 평지(plane) 테스트 설정 명확화 ---
# # #     # 평지 테스트이므로 'plane'만 설정합니다.
# # #     env_cfg.terrain.mesh_type = 'plane'

# # #     # --- Adjust camera for testing ---
# # #     # (카메라 위치는 필요에 따라 조정하세요)
# # #     env_cfg.viewer.pos = [2.0, 0.0, 1.0]
# # #     env_cfg.viewer.lookat = [0.0, 0.0, 0.0]

# # #     # === Prepare environment ===
# # #     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
# # #     obs = env.get_observations()
# # #     history_obs = env.get_history_observations()

# # #     # === Load trained policy ===
# # #     train_cfg.runner.resume = True
# # #     ppo_runner, train_cfg = task_registry.make_alg_runner(
# # #         env=env, name=args.task, args=args, train_cfg=train_cfg
# # #     )
# # #     policy = ppo_runner.get_inference_policy(device=env.device)

# # #     # === Export JIT policy (optional) ===
# # #     if EXPORT_POLICY:
# # #         path = os.path.join(
# # #             LEGGED_GYM_ROOT_DIR, "logs",
# # #             train_cfg.runner.experiment_name,
# # #             "exported", "policies"
# # #         )
# # #         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
# # #         print("✅ Exported policy as JIT script to:", path)

# # #     # === Initialize logger ===
# # #     logger = Logger(env.dt)
# # #     robot_index = 0
# # #     joint_index = 1
# # #     stop_state_log = 100
# # #     stop_rew_log = env.max_episode_length + 1
# # #     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
# # #     camera_vel = np.array([1., 1., 0.])
# # #     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
# # #     img_idx = 0

# # #     print("🚀 Starting flat ground (plane) test...")

# # #     # === Main loop ===
# # #     for i in range(10 * int(env.max_episode_length)):
# # #         loop_t0 = time.perf_counter()

# # #         # --- Forward walking command (gradually increase speed) ---
# # #         # (치우침/속도 문제를 해결하기 위한 사용자 지정 명령)
# # #         forward_speed = min(0.4, i * 0.002)  # accelerate gradually to 0.4 m/s
# # #         env.commands[:] = torch.tensor([[forward_speed, 0.0, 0.0]], device=env.device)

# # #         # --- Policy inference and environment step ---
# # #         actions = policy(obs.detach(), history_obs.detach())
        
# # #         # --- 🐛 중요 버그 수정 ---
# # #         # A1DreamWaQ의 step은 (obs, _, history_obs, rews, dones, infos) 6개 반환
# # #         # history_obs를 매 스텝 갱신해야 합니다.
# # #         obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

# # #         # --- (Optional) Save frames for video ---
# # #         if RECORD_FRAMES and i % 2:
# # #             filename = os.path.join(
# # #                 LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
# # #                 "exported", "frames", f"{img_idx}.png"
# # #             )
# # #             env.gym.write_viewer_image_to_file(env.viewer, filename)
# # #             img_idx += 1

# # #         # --- (Optional) Move camera ---
# # #         if MOVE_CAMERA:
# # #             camera_position += camera_vel * env.dt
# # #             env.set_camera(camera_position, camera_position + camera_direction)

# # #         # --- Log states ---
# # #         if i < stop_state_log:
# # #             logger.log_states({
# # #                 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
# # #                 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
# # #                 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
# # #                 'dof_torque': env.torques[robot_index, joint_index].item(),
# # #                 'command_x': env.commands[robot_index, 0].item(),
# # #                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
# # #                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
# # #                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
# # #             })
# # #         elif i == stop_state_log:
# # #             logger.plot_states()

# # #         # --- Log rewards ---
# # #         if 0 < i < stop_rew_log:
# # #             if infos["episode"]:
# # #                 num_episodes = torch.sum(env.reset_buf).item()
# # #                 if num_episodes > 0:
# # #                     logger.log_rewards(infos["episode"], num_episodes)
# # #         elif i == stop_rew_log:
# # #             logger.print_rewards()

# # #         # --- Maintain real-time playback ---
# # #         desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
# # #         elapsed = time.perf_counter() - loop_t0
# # #         sleep_time = desired_step_time - elapsed
# # #         if sleep_time > 0:
# # #             time.sleep(sleep_time)


# # # if __name__ == "__main__":
# # #     EXPORT_POLICY = True
# # #     RECORD_FRAMES = False
# # #     MOVE_CAMERA = False
# # #     REALTIME_FACTOR = 1.0

# # #     args = get_args()
# # #     play(args)

# Stairs up
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time


def play(args):
    # === Load configs ===
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # --- Environment overrides for testing ---
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # --- 🚀 수정: 계단(stairs) 지형 설정 ---
    # 평지('plane') 대신 계단('trimesh')을 생성하도록 설정합니다.
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

    # --- Adjust camera for stairs ---
    # (카메라 위치를 계단이 잘 보이도록 조정)
    env_cfg.viewer.pos = [2.5, 0.0, 1.0]
    env_cfg.viewer.lookat = [1.0, 0.0, 0.0]

    # === Prepare environment ===
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    history_obs = env.get_history_observations()

    # === Load trained policy ===
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # === Export JIT policy (optional) ===
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs",
            train_cfg.runner.experiment_name,
            "exported", "policies"
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("✅ Exported policy as JIT script to:", path)

    # === Initialize logger ===
    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    print("🚀 Starting stair climbing test...")

    # === Main loop ===
    for i in range(10 * int(env.max_episode_length)):
        loop_t0 = time.perf_counter()

        # --- Forward walking command (gradually increase speed) ---
        # (계단을 오르기 위해 직진 명령을 유지)
        forward_speed = min(1.0, i * 0.02)  # accelerate gradually to 1.0 m/s
        env.commands[:] = torch.tensor([[forward_speed, 0.0, 0.0]], device=env.device)

        # --- Policy inference and environment step ---
        actions = policy(obs.detach(), history_obs.detach())
        
        # --- (버그 수정 유지) history_obs를 올바르게 갱신합니다 ---
        obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

        # --- (Optional) Save frames for video ---
        if RECORD_FRAMES and i % 2:
            filename = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
                "exported", "frames", f"{img_idx}.png"
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        # --- (Optional) Move camera ---
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # --- Log states ---
        if i < stop_state_log:
            logger.log_states({
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            })
        elif i == stop_state_log:
            logger.plot_states()

        # --- Log rewards ---
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # --- Maintain real-time playback ---
        desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
        elapsed = time.perf_counter() - loop_t0
        sleep_time = desired_step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    REALTIME_FACTOR = 1.0

    args = get_args()
    play(args)

# Stairs up and down
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time


def play(args):
    # === Load configs ===
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # --- Environment overrides for testing ---
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # --- 🚀 수정: 계단(stairs) 지형 설정 ---
    # 평지('plane') 대신 계단('trimesh')을 생성하도록 설정합니다.
    env_cfg.terrain.mesh_type = 'trimesh'
    # 'stairs_up'과 'stairs_down'이 모두 포함되도록 비율 조정 (필요시)
    # 여기서는 'trimesh' 기본값이 계단을 포함한다고 가정합니다.
    # 만약 'stairs_up'만 생성된다면, 'stairs_down' 지형도 추가해야 할 수 있습니다.
    # 하지만 이 데모에서는 올라갔던 계단을 다시 내려오는 것이 목표입니다.
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

    # --- Adjust camera for stairs ---
    # (카메라 위치를 계단이 잘 보이도록 조정)
    env_cfg.viewer.pos = [2.5, 0.0, 1.0]
    env_cfg.viewer.lookat = [1.0, 0.0, 0.0]

    # === Prepare environment ===
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    history_obs = env.get_history_observations()

    # === Load trained policy ===
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # === Export JIT policy (optional) ===
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs",
            train_cfg.runner.experiment_name,
            "exported", "policies"
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("✅ Exported policy as JIT script to:", path)

    # === Initialize logger ===
    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # --- 🚀 [신규] 상태 머신(State Machine) 변수 ---
    PHASE_CLIMB_UP = 1
    PHASE_TURN = 2
    PHASE_CLIMB_DOWN = 3
    current_phase = PHASE_CLIMB_UP

    # --- 🚀 [신규] 상태 전환을 위한 임계값 ---
    # (이 값들은 환경이나 로봇에 맞게 조정해야 할 수 있습니다)
    PHASE_UP_X_THRESHOLD = 2.5  # (m) 이 X-좌표를 넘으면 회전 시작
    PHASE_TURN_YAW_THRESHOLD = -3.0 # (rad) 이 Yaw 각도(약 -172도)를 넘으면 하강 시작

    # --- 🚀 [신규] 로봇 속도 변수 ---
    current_forward_speed = 0.0
    
    print(f"🚀 Starting test... Phase {current_phase}: CLIMB UP")

    # === Main loop ===
    for i in range(10 * int(env.max_episode_length)):
        loop_t0 = time.perf_counter()

        # --- 🚀 [수정] 상태 머신(State Machine) 기반의 명령 생성 ---
        if current_phase == PHASE_CLIMB_UP:
            # Phase 1: 계단 오르기 (서서히 가속)
            current_forward_speed = min(1.0, current_forward_speed + 0.005)
            env.commands[:] = torch.tensor([[current_forward_speed, 0.0, 0.0]], device=env.device)

            # 상태 전환 확인: X-좌표가 임계값을 넘었는지 확인
            current_x_pos = env.root_states[robot_index, 0].item()
            if current_x_pos > PHASE_UP_X_THRESHOLD:
                current_phase = PHASE_TURN
                current_forward_speed = 0.0  # 회전 전 정지
                print(f"✅ Reached top (X={current_x_pos:.2f}m). Phase {current_phase}: TURN")

        elif current_phase == PHASE_TURN:
            # Phase 2: 180도 시계방향 회전
            turn_speed = -0.8  # (rad/s) 시계방향(음수) 회전
            env.commands[:] = torch.tensor([[0.0, 0.0, turn_speed]], device=env.device)

            # 상태 전환 확인: Yaw 각도가 임계값(-180도)에 근접했는지 확인
            # env.base_euler_xyz는 [roll, pitch, yaw]를 담고 있습니다.
            current_yaw = env.base_euler_xyz[robot_index, 2].item()
            if current_yaw < PHASE_TURN_YAW_THRESHOLD:
                current_phase = PHASE_CLIMB_DOWN
                current_forward_speed = 0.0  # 하강 전 속도 리셋
                print(f"✅ Turn complete (Yaw={current_yaw:.2f} rad). Phase {current_phase}: CLIMB DOWN")

        elif current_phase == PHASE_CLIMB_DOWN:
            # Phase 3: 계단 내려오기 (서서히 가속)
            current_forward_speed = min(0.7, current_forward_speed + 0.005) # 하강 시 속도 (조금 느리게)
            env.commands[:] = torch.tensor([[current_forward_speed, 0.0, 0.0]], device=env.device)

        # --- Policy inference and environment step ---
        actions = policy(obs.detach(), history_obs.detach())
        
        # --- (버그 수정 유지) history_obs를 올바르게 갱신합니다 ---
        obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

        # --- 🚀 [신규] 환경 리셋 시 상태 머신 초기화 ---
        # 로봇이 넘어지거나 에피소드가 끝나면, 상태를 다시 '오르기'로 리셋
        if dones[robot_index]:
            print("🔄 Environment reset. Returning to Phase 1: CLIMB UP")
            current_phase = PHASE_CLIMB_UP
            current_forward_speed = 0.0

        # --- (Optional) Save frames for video ---
        if RECORD_FRAMES and i % 2:
            filename = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
                "exported", "frames", f"{img_idx}.png"
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        # --- (Optional) Move camera ---
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # --- Log states ---
        if i < stop_state_log:
            logger.log_states({
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_yaw': env.commands[robot_index, 2].item(), # 🚀 [수정] Yaw 명령 로깅
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'base_pos_x': env.root_states[robot_index, 0].item(), # 🚀 [신규] X-좌표 로깅
                'base_yaw': env.base_euler_xyz[robot_index, 2].item(), # 🚀 [신규] Yaw 각도 로깅
            })
        elif i == stop_state_log:
            logger.plot_states()

        # --- Log rewards ---
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # --- Maintain real-time playback ---
        desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
        elapsed = time.perf_counter() - loop_t0
        sleep_time = desired_step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    REALTIME_FACTOR = 1.0  # 실시간 재생 속도 (1.0 = 1배속)

    args = get_args()
    play(args)