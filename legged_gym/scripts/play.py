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

# # # Stairs up
# # # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# # # SPDX-License-Identifier: BSD-3-Clause
# # #
# # # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# # from legged_gym import LEGGED_GYM_ROOT_DIR
# # import os
# # import isaacgym
# # from legged_gym.envs import *
# # from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

# # import numpy as np
# # import torch
# # import time


# # def play(args):
# #     # === Load configs ===
# #     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

# #     # --- Environment overrides for testing ---
# #     env_cfg.env.num_envs = 1
# #     env_cfg.terrain.num_rows = 1
# #     env_cfg.terrain.num_cols = 1
# #     env_cfg.terrain.curriculum = False
# #     env_cfg.noise.add_noise = False
# #     env_cfg.domain_rand.randomize_friction = False
# #     env_cfg.domain_rand.push_robots = False

# #     # --- 🚀 수정: 계단(stairs) 지형 설정 ---
# #     # 평지('plane') 대신 계단('trimesh')을 생성하도록 설정합니다.
# #     env_cfg.terrain.mesh_type = 'trimesh'
# #     env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

# #     # --- Adjust camera for stairs ---
# #     # (카메라 위치를 계단이 잘 보이도록 조정)
# #     env_cfg.viewer.pos = [2.5, 0.0, 1.0]
# #     env_cfg.viewer.lookat = [1.0, 0.0, 0.0]

# #     # === Prepare environment ===
# #     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
# #     obs = env.get_observations()
# #     history_obs = env.get_history_observations()

# #     # === Load trained policy ===
# #     train_cfg.runner.resume = True
# #     ppo_runner, train_cfg = task_registry.make_alg_runner(
# #         env=env, name=args.task, args=args, train_cfg=train_cfg
# #     )
# #     policy = ppo_runner.get_inference_policy(device=env.device)

# #     # === Export JIT policy (optional) ===
# #     if EXPORT_POLICY:
# #         path = os.path.join(
# #             LEGGED_GYM_ROOT_DIR, "logs",
# #             train_cfg.runner.experiment_name,
# #             "exported", "policies"
# #         )
# #         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
# #         print("✅ Exported policy as JIT script to:", path)

# #     # === Initialize logger ===
# #     logger = Logger(env.dt)
# #     robot_index = 0
# #     joint_index = 1
# #     stop_state_log = 100
# #     stop_rew_log = env.max_episode_length + 1
# #     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
# #     camera_vel = np.array([1., 1., 0.])
# #     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
# #     img_idx = 0

# #     print("🚀 Starting stair climbing test...")

# #     # === Main loop ===
# #     for i in range(10 * int(env.max_episode_length)):
# #         loop_t0 = time.perf_counter()

# #         # --- Forward walking command (gradually increase speed) ---
# #         # (계단을 오르기 위해 직진 명령을 유지)
# #         forward_speed = min(1.0, i * 0.02)  # accelerate gradually to 1.0 m/s
# #         env.commands[:] = torch.tensor([[forward_speed, 0.0, 0.0]], device=env.device)

# #         # --- Policy inference and environment step ---
# #         actions = policy(obs.detach(), history_obs.detach())
        
# #         # --- (버그 수정 유지) history_obs를 올바르게 갱신합니다 ---
# #         obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

# #         # --- (Optional) Save frames for video ---
# #         if RECORD_FRAMES and i % 2:
# #             filename = os.path.join(
# #                 LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
# #                 "exported", "frames", f"{img_idx}.png"
# #             )
# #             env.gym.write_viewer_image_to_file(env.viewer, filename)
# #             img_idx += 1

# #         # --- (Optional) Move camera ---
# #         if MOVE_CAMERA:
# #             camera_position += camera_vel * env.dt
# #             env.set_camera(camera_position, camera_position + camera_direction)

# #         # --- Log states ---
# #         if i < stop_state_log:
# #             logger.log_states({
# #                 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
# #                 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
# #                 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
# #                 'dof_torque': env.torques[robot_index, joint_index].item(),
# #                 'command_x': env.commands[robot_index, 0].item(),
# #                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
# #                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
# #                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
# #             })
# #         elif i == stop_state_log:
# #             logger.plot_states()

# #         # --- Log rewards ---
# #         if 0 < i < stop_rew_log:
# #             if infos["episode"]:
# #                 num_episodes = torch.sum(env.reset_buf).item()
# #                 if num_episodes > 0:
# #                     logger.log_rewards(infos["episode"], num_episodes)
# #         elif i == stop_rew_log:
# #             logger.print_rewards()

# #         # --- Maintain real-time playback ---
# #         desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
# #         elapsed = time.perf_counter() - loop_t0
# #         sleep_time = desired_step_time - elapsed
# #         if sleep_time > 0:
# #             time.sleep(sleep_time)


# # if __name__ == "__main__":
# #     EXPORT_POLICY = True
# #     RECORD_FRAMES = False
# #     MOVE_CAMERA = False
# #     REALTIME_FACTOR = 1.0

# #     args = get_args()
# #     play(args)

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

# def play(args, terrain_name, mesh_type, proportions):
#     """
#     지정된 지형 설정으로 시뮬레이션을 실행하는 함수
#     """
#     print(f"\n--- 🚀 Initializing test for: {terrain_name} ---")

#     # === Load configs ===
#     # 매번 새로운 설정을 위해 cfg를 다시 불러옵니다.
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

#     # --- Environment overrides for testing ---
#     env_cfg.env.num_envs = 1
#     env_cfg.terrain.num_rows = 1
#     env_cfg.terrain.num_cols = 1
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     # --- 🚀 수정: 전달받은 지형 설정 적용 ---
#     print(f"Applying terrain: {terrain_name} (Type: {mesh_type}, Props: {proportions})")
#     env_cfg.terrain.mesh_type = mesh_type
#     if mesh_type == 'trimesh':
#         env_cfg.terrain.terrain_proportions = proportions
    
#     # --- Adjust camera based on terrain ---
#     if terrain_name == "plane":
#         env_cfg.viewer.pos = [2.0, 0.0, 1.0] # 평지용 카메라
#         env_cfg.viewer.lookat = [0.0, 0.0, 0.0]
#     else:
#         env_cfg.viewer.pos = [2.5, 0.0, 1.0] # trimesh용 카메라
#         env_cfg.viewer.lookat = [1.0, 0.0, 0.0]

#     # === Prepare environment ===
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     obs = env.get_observations()
#     history_obs = env.get_history_observations()

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
    
#     # --- 🚀 수정: 테스트 시간 설정 ---
#     # 각 지형을 적당한 시간(예: 2 * max_episode_length) 동안 테스트합니다.
#     TEST_DURATION_STEPS = int(env.max_episode_length * 2) 
#     print(f"🚀 Starting simulation for {terrain_name} ({TEST_DURATION_STEPS} steps)...")

#     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
#     camera_vel = np.array([1., 1., 0.])
#     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
#     img_idx = 0

#     # === Main loop ===
#     for i in range(TEST_DURATION_STEPS):
#         loop_t0 = time.perf_counter()

#         # --- Forward walking command ---
#         # (지형에 상관없이 일정한 속도로 전진 명령)
#         forward_speed = 0.5  # 0.5 m/s
#         env.commands[:] = torch.tensor([[forward_speed, 0.0, 0.0]], device=env.device)

#         # --- Policy inference and environment step ---
#         actions = policy(obs.detach(), history_obs.detach())
        
#         # --- (버그 수정 유지) history_obs를 올바르게 갱신합니다 ---
#         obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

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

#         # --- Log states (루프 내내 기록) ---
#         logger.log_states({
#             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
#             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
#             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
#             'dof_torque': env.torques[robot_index, joint_index].item(),
#             'command_x': env.commands[robot_index, 0].item(),
#             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
#             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
#             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
#         })

#         # --- Log rewards (에피소드 종료 시 기록) ---
#         if infos["episode"]:
#             num_episodes = torch.sum(env.reset_buf).item()
#             if num_episodes > 0:
#                 logger.log_rewards(infos["episode"], num_episodes)

#         # --- Maintain real-time playback ---
#         desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
#         elapsed = time.perf_counter() - loop_t0
#         sleep_time = desired_step_time - elapsed
#         if sleep_time > 0:
#             time.sleep(sleep_time)
            
#     # --- 🚀 수정: 루프 종료 후 로그 플로팅 ---
#     print("... Simulation finished. Plotting logs ...")
#     logger.plot_states()
#     logger.print_rewards()

#     # --- 🧹 중요: 다음 테스트를 위해 IsaacGym 환경 정리 ---
#     print(f"Cleaning up test for: {terrain_name} ...")
#     env.gym.destroy_viewer(env.viewer)
#     env.gym.destroy_sim(env.sim)
#     print(f"✅ Cleaned up test for: {terrain_name} ---")


# if __name__ == "__main__":
#     # --- 전역 설정 ---
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     REALTIME_FACTOR = 1.0  # 1.0 = 실시간. 테스트를 빨리 보려면 2.0 이상으로 설정

#     # --- 🚀 테스트할 지형 목록 정의 ---
#     # legged_gym의 기본 trimesh 비율 순서:
#     # [slopes, waves, stairs, random_uniform, discrete_obstacles]
#     TERRAIN_DEFINITIONS = {
#         # 이름: (mesh_type, proportions_list)
#         "plane":     ("plane",   [0.0, 0.0, 0.0, 0.0, 0.0]),
#         "slopes":    ("trimesh", [1.0, 0.0, 0.0, 0.0, 0.0]),
#         "waves":     ("trimesh", [0.0, 1.0, 0.0, 0.0, 0.0]),
#         "stairs":    ("trimesh", [0.0, 0.0, 1.0, 0.0, 0.0]),
#         "uniform":   ("trimesh", [0.0, 0.0, 0.0, 1.0, 0.0]),
#         "obstacles": ("trimesh", [0.0, 0.0, 0.0, 0.0, 1.0])
#         # "mix_all":   ("trimesh", [0.2, 0.2, 0.2, 0.2, 0.2]) # 믹스 테스트도 가능
#     }

#     # legged_gym 기본 인자 한 번만 파싱
#     args = get_args()

#     # --- 🚀 정의된 각 지형에 대해 'play' 함수 순차적 호출 ---
#     for name, (mesh, props) in TERRAIN_DEFINITIONS.items():
#         play(args, 
#              terrain_name=name, 
#              mesh_type=mesh, 
#              proportions=props)
        
#         # (선택적) IsaacGym이 완전히 종료되고 다음 시뮬레이션을
#         # 시작하기 전에 잠시 대기합니다. (충돌 방지에 도움)
#         time.sleep(2) 

#     print("\n🎉 All terrain tests complete.")

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import matplotlib.pyplot as plt # 💡 Boxplot을 위해 matplotlib 추가

def test_tracking(args):
    # === Load configs ===
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # --- Environment overrides for tracking test ---
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 💡 평지(plane)에서 테스트합니다.
    env_cfg.terrain.mesh_type = 'plane'
    
    env_cfg.viewer.pos = [2.0, 0.0, 1.0]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.0]

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

    print("🚀 Starting Command Tracking Test (Figure 4)...")

    # === 💡 실험 설정 (논문 기반) ===
    # 논문: 10분 테스트  (10 * 60 / env.dt 스텝)
    # 논문: 10초마다 명령 변경  (10 / env.dt 스텝)
    
    # (테스트 시간을 줄여서 설정)
    TOTAL_TEST_DURATION_SEC = 60  # 60초 (논문은 10분)
    COMMAND_CHANGE_INTERVAL_SEC = 5 # 5초 (논문은 10초)

    TOTAL_STEPS = int(TOTAL_TEST_DURATION_SEC / env.dt)
    COMMAND_CHANGE_INTERVAL_STEPS = int(COMMAND_CHANGE_INTERVAL_SEC / env.dt)

    # 에러를 저장할 리스트
    errors_vx = [] # Forward velocity error (v_x^e)
    errors_vy = [] # Lateral velocity error (v_y^e)
    errors_vyaw = [] # Yaw rate error (omega_z^e)

    current_command = torch.zeros(1, 3, device=env.device)

    # === Main loop ===
    for i in range(TOTAL_STEPS):
        loop_t0 = time.perf_counter()

        # --- 💡 10초(설정값)마다 임의의 명령 생성 ---
        if i % COMMAND_CHANGE_INTERVAL_STEPS == 0:
            # 논문: [-1.0, 1.0] m/s, [-1.0, 1.0] rad/s 
            # (명령 범위를 확인하세요. env_cfg.commands.ranges...)
            cmd_x = np.random.uniform(-1.0, 1.0)
            cmd_y = np.random.uniform(-1.0, 1.0)
            cmd_yaw = np.random.uniform(-1.0, 1.0)
            
            # DreamWaQ는 x, y, yaw 3가지 명령을 받습니다.
            current_command[:] = torch.tensor([[cmd_x, cmd_y, cmd_yaw]], device=env.device)
            print(f"Step {i}: New Command -> [x:{cmd_x:.2f}, y:{cmd_y:.2f}, yaw:{cmd_yaw:.2f}]")
        
        env.commands[:] = current_command

        # --- Policy inference and environment step ---
        actions = policy(obs.detach(), history_obs.detach())
        obs, _, history_obs, rews, dones, infos = env.step(actions.detach())

        # --- 💡 ATE (Absolute Tracking Error) 기록 ---
        # [cite: 227]
        commanded_vel = env.commands[0]
        base_lin_vel = env.base_lin_vel[0]
        base_ang_vel = env.base_ang_vel[0]

        # v_x^e
        errors_vx.append(torch.abs(commanded_vel[0] - base_lin_vel[0]).item())
        # v_y^e
        errors_vy.append(torch.abs(commanded_vel[1] - base_lin_vel[1]).item())
        # omega_z^e
        errors_vyaw.append(torch.abs(commanded_vel[2] - base_ang_vel[2]).item())

        # --- Maintain real-time playback ---
        desired_step_time = env.dt / max(REALTIME_FACTOR, 1e-3)
        elapsed = time.perf_counter() - loop_t0
        sleep_time = desired_step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # --- 💡 테스트 종료 후 Boxplot 그리기 ---
    print("... Test finished. Generating Boxplot (Fig 4) ...")
    
    # 논문과 유사하게 DreamWaQ, EstimatorNet, Baseline 등을 비교하려면
    # 각 모델에 대해 이 스크립트를 실행하고 error 리스트를 저장해야 합니다.
    # 여기서는 "DreamWaQ w/ AdaBoot" 하나만 그립니다.
    
    data_to_plot = [errors_vx, errors_vy, errors_vyaw]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    
    # Boxplot 생성
    bp = ax.boxplot(data_to_plot, 
                    labels=['$v_x^e$', '$v_y^e$', '$\omega_z^e$'], # [cite: 243, 244]
                    patch_artist=True,
                    showfliers=False) # 이상치(outlier)는 그리지 않음

    # (논문 스타일 흉내)
    colors = ['#2C7BB6', '#ABD9E9', '#FDAE61'] # 예시 색상
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_title('Command Tracking Error (Absolute)')
    ax.set_ylabel('Tracking error')
    ax.set_ylim(0, 0.25) # [cite: 230-235]
    
    # (선택) 논문처럼 여러 알고리즘 비교하기
    # data_to_plot = [dreamwaq_vx, estimatornet_vx, baseline_vx, ...]
    # 이런 식으로 데이터를 모아서 그려야 합니다.
    
    plt.savefig("command_tracking_error.png")
    print("✅ Saved plot to command_tracking_error.png")
    
    # --- 환경 정리 ---
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    REALTIME_FACTOR = 1.0 # 1.0 = 실시간, 10.0 = 10배속 (테스트 속도 향상)
    args = get_args()
    test_tracking(args)