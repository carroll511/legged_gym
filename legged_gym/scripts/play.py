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
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # --- 🚀 수정: 계단(stairs) 지형 설정 ---
    # 평지('plane') 대신 계단('trimesh')을 생성하도록 설정합니다.
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [1.0, 0, 0.0, 0, 0]
    env_cfg.terrain.terrain_length = 8.0
    env_cfg.terrain_width = 8.0
    env_cfg.terrain.border_size = 0.0

    center_x = (env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length) / 2.0
    center_y = (env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width) / 2.0
    env_cfg.init_state.pos = [center_x + np.random.uniform(-2.0, 2.0),
                              center_y + np.random.uniform(-2.0, 2.0),
                              0.42]
    env_cfg.init_state.rpy = [0.0, 0.0, np.random.uniform(-np.pi, np.pi)]  # yaw 랜덤

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
        forward_speed = min(3.0, i * 0.02)  # accelerate gradually to 2.5 m/s
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

# # Stairs up -> Curved Turn -> Stairs down (Full Corrected Code v10: Const 0.5 speed)
# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os
# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

# import numpy as np
# import torch
# import time


# # 쿼터니언을 Yaw 각도로 변환하는 유틸리티 함수
# def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
#     """
#     (N, 4) 또는 (4,) 쿼터니언(x, y, z, w) 텐서를 Yaw (Z축 회전) 텐서로 변환합니다.
#     """
#     if quat.dim() == 1:
#         quat = quat.unsqueeze(0)

#     qx = quat[:, 0]
#     qy = quat[:, 1]
#     qz = quat[:, 2]
#     qw = quat[:, 3]

#     t0 = 2.0 * (qw * qz + qx * qy)
#     t1 = 1.0 - 2.0 * (qy * qy + qz * qz)
#     yaw = torch.atan2(t0, t1)
    
#     return yaw.squeeze()


# def play(args):
#     # === Load configs ===
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

#     # --- Environment overrides for testing ---
#     env_cfg.env.num_envs = 1
#     env_cfg.terrain.num_rows = 1
#     env_cfg.terrain.num_cols = 1
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     # --- [v8] 지형 설정: 'stairs_up'만 생성 ---
#     env_cfg.terrain.mesh_type = 'trimesh'
#     env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

#     # --- Adjust camera for stairs ---
#     env_cfg.viewer.pos = [2.5, 0.0, 1.0]
#     env_cfg.viewer.lookat = [1.0, 0.0, 0.0]

#     # === Prepare environment ===
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     obs = env.get_observations()
#     history_obs = env.get_history_observations()
    
#     # --- [v8] 시간 기반 상태 전환을 위한 설정 ---
#     dt = env.dt
#     DURATION_SECONDS = 3.0
#     DURATION_STEPS = int(DURATION_SECONDS / dt)
#     print(f"🚀 Simulation dt: {dt:.4f}s. 5 seconds = {DURATION_STEPS} steps.")
    
#     # --- 🚀 [신규] 고정 속도 설정 ---
#     CONST_FORWARD_SPEED = 1.0
#     TURN_YAW_SPEED = -0.5 # (rad/s) 시계방향 회전

#     # === Load trained policy ===
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(
#         env=env, name=args.task, args=args, train_cfg=train_cfg
#     )
#     policy = ppo_runner.get_inference_policy(device=env.device)

#     # === Export JIT policy (optional) ===
#     if EXPORT_POLICY:
#         path = os.path.join(
#             LEGGED_GYM_ROOT_DIR, "logs",
#             train_cfg.runner.experiment_name,
#             "exported", "policies"
#         )
#         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#         print("✅ Exported policy as JIT script to:", path)

#     # === Initialize logger ===
#     logger = Logger(env.dt)
#     robot_index = 0
#     joint_index = 1
#     stop_state_log = 100
#     stop_rew_log = env.max_episode_length + 1
#     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
#     camera_vel = np.array([1., 1., 0.])
#     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
#     img_idx = 0

#     # --- [v8] 상태 머신(State Machine) 변수 (3단계) ---
#     PHASE_CLIMB_UP = 1
#     PHASE_TURN = 2
#     PHASE_CLIMB_DOWN = 3
#     current_phase = PHASE_CLIMB_UP

#     # --- [v8] 상태 전환을 위한 임계값 ---
#     PHASE_TURN_YAW_THRESHOLD = -3.14 # (rad) 이 Yaw 각도(약 -172도)를 넘으면 하강
    
#     phase_step_counter = 0

#     print(f"🚀 Starting test... Phase {current_phase}: CLIMB UP (Speed: {CONST_FORWARD_SPEED} m/s)")

#     # === Main loop ===
#     for i in range(10 * int(env.max_episode_length)):
#         loop_t0 = time.perf_counter()

#         # --- 🚀 [수정됨] 고정 속도 0.5 기반 3단계 상태 머신 ---
        
#         # 기본 명령: 항상 0.5 m/s로 전진
#         current_command = [CONST_FORWARD_SPEED, 0.0, 0.0]

#         if current_phase == PHASE_CLIMB_UP:
#             # Phase 1: 계단 오르기 (명령 수정 불필요)
            
#             # 상태 전환 확인: 5초(DURATION_STEPS)가 지났는지 확인
#             if phase_step_counter > DURATION_STEPS:
#                 current_phase = PHASE_TURN
#                 phase_step_counter = 0
#                 print(f"✅ 5s UP complete. Phase {current_phase}: TURN (Speed: {CONST_FORWARD_SPEED} m/s)")

#         elif current_phase == PHASE_TURN:
#             # Phase 2: 걸으면서(0.5) 180도 회전
#             current_command[2] = TURN_YAW_SPEED # Yaw 명령 추가
            
#             # 상태 전환 확인: 180도(-3.0 rad) 회전했는지 확인
#             quat = env.root_states[robot_index, 3:7]
#             current_yaw = quat_to_yaw(quat).item()
            
#             if current_yaw < PHASE_TURN_YAW_THRESHOLD:
#                 current_phase = PHASE_CLIMB_DOWN
#                 phase_step_counter = 0
#                 print(f"✅ Turn complete (Yaw={current_yaw:.2f} rad). Phase {current_phase}: CLIMB DOWN (Speed: {CONST_FORWARD_SPEED} m/s)")

#         elif current_phase == PHASE_CLIMB_DOWN:
#             # Phase 3: 계단 내려오기 (명령 수정 불필요)
            
#             # 상태 전환 확인: 5초(DURATION_STEPS)가 지났는지 확인
#             if phase_step_counter > DURATION_STEPS:
#                 current_phase = PHASE_CLIMB_UP # Phase 1로 복귀 (반복)
#                 phase_step_counter = 0
#                 print(f"✅ 5s DOWN complete. Phase {current_phase}: CLIMB UP (Speed: {CONST_FORWARD_SPEED} m/s)")

#         # --- [신규] 계산된 명령을 환경에 적용 ---
#         env.commands[:] = torch.tensor([current_command], device=env.device)

#         # --- Policy inference and environment step ---
#         actions = policy(obs.detach(), history_obs.detach())
        
#         obs, _, history_obs, rews, dones, infos = env.step(actions.detach())
        
#         # --- [v8] 현재 Phase의 스텝 카운터 증가 ---
#         phase_step_counter += 1

#         # --- [v8] 리셋 시 상태 머신 초기화 ---
#         if dones[robot_index]:
#             print("🔄 Environment reset. Returning to Phase 1: CLIMB UP")
#             current_phase = PHASE_CLIMB_UP
#             phase_step_counter = 0 

#         # --- (Optional) Save frames for video ---
#         if RECORD_FRAMES and i % 2:
#             filename = os.path.join(
#                 LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
#                 "exported", "frames", f"{img_idx}.png"
#             )
#             env.gym.write_viewer_image_to_file(env.viewer, filename)
#             img_idx += 1

#         # --- (Optional) Move camera ---
#         if MOVE_CAMERA:
#             camera_position += camera_vel * env.dt
#             env.set_camera(camera_position, camera_position + camera_direction)

#         # --- Log states ---
#         if i < stop_state_log:
            
#             log_quat = env.root_states[robot_index, 3:7]
#             log_current_yaw = quat_to_yaw(log_quat).item()
            
#             logger.log_states({
#                 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
#                 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
#                 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
#                 'dof_torque': env.torques[robot_index, joint_index].item(),
#                 'command_x': env.commands[robot_index, 0].item(),
#                 'command_yaw': env.commands[robot_index, 2].item(),
#                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
#                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
#                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
#                 'base_pos_x': env.root_states[robot_index, 0].item(),
#                 'base_yaw': log_current_yaw,
#                 'phase_step_counter': phase_step_counter,
#             })
#         elif i == stop_state_log:
#             logger.plot_states()

#         # --- Log rewards ---
#         if 0 < i < stop_rew_log:
#             if infos["episode"]:
#                 num_episodes = torch.sum(env.reset_buf).item()
#                 if num_episodes > 0:
#                     logger.log_rewards(infos["episode"], num_episodes)
#         elif i == stop_rew_log:
#             logger.print_rewards()

#         # --- Maintain real-time playback ---
#         desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
#         elapsed = time.perf_counter() - loop_t0
#         sleep_time = desired_step_time - elapsed
#         if sleep_time > 0:
#             time.sleep(sleep_time)


# if __name__ == "__main__":
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     REALTIME_FACTOR = 1.0

#     args = get_args()
#     play(args)