# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 환경 점검: 지형/로봇/센서/관측 규격이 실제로 구성되는지 확인

점검 항목:
    로봇을 default_joint_pos로 유지시켜서, PD 게인이 로봇을 지탱하는지
    지형/로봇이 겹치거나 깨지지 않는지
    단계별 지형, 관측·특권 관측 차원, 보행 위상 대상 관절이 의도대로 구성되는지

    실행 명령어:
    ./isaaclab.sh -p scripts/00_test_env.py --robot-id unitree_g1 --num_envs 4
    ./isaaclab.sh -p scripts/00_test_env.py --robot-id unitree_g1 --terrain phase2 --num_envs 16
"""

import argparse
import logging
import os
import sys
import torch

from isaaclab.app import AppLauncher

# CLI 인자 정의 + Isaac Sim 부팅
parser = argparse.ArgumentParser(description="로봇·지형·관측 규격 점검.")
parser.add_argument("--robot-id", type=str, default="unitree_g1", help="스폰할 로봇 ID (ROBOT_ENV_CFGS의 키).")
parser.add_argument("--num_envs", type=int, default=4, help="스폰할 환경 개수.")
parser.add_argument("--terrain", choices=["plane", "phase1", "phase2"], default="plane",
                   help="점검할 지형. phase1/phase2는 학습 단계별 지형 생성기를 쓴다.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# scripts/00_test_env.py -> 루트: dirname 2번.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from src.sim.rl.dreamwaq_env import DreamWaQEnv
from src.sim.rl.configs import ROBOT_ENV_CFGS
from src.sim.rl.env_cfg import configure_training_phase

log = logging.getLogger(__name__)


def apply_terrain(cfg, terrain):
    """점검 대상 지형을 설정하고 난이도 커리큘럼을 끈다."""
    cfg.curriculum.terrain_levels = None
    if terrain == "plane":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.visual_material = None
        return
    configure_training_phase(cfg, int(terrain[-1]))
    cfg.scene.terrain.terrain_generator.curriculum = False


def main():
    """지정한 로봇·지형으로 환경을 만들고 구성을 로그로 남긴 뒤 기본자세를 유지하며 스텝한다."""
    # 환경 설정
    cfg = ROBOT_ENV_CFGS[args_cli.robot_id]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.sim.save_logs_to_file = False
    cfg.sim.logging_level = "INFO"
    apply_terrain(cfg, args_cli.terrain)

    # 환경 생성
    env = DreamWaQEnv(cfg=cfg)
    joint_pos_term = env.action_manager.get_term("joint_pos")
    matched_names = joint_pos_term.IO_descriptor.joint_names
    robot = env.scene["robot"]

    # 로그 출력
    log.info(f"robot_id: {args_cli.robot_id}")
    log.info(f"robot usd 경로: {cfg.scene.robot.spawn.usd_path}")
    log.info(f"지형: {args_cli.terrain}")
    log.info(f"실제로 매칭된 관절({len(matched_names)}개): {matched_names}")
    log.info(f"로봇 전체 관절({robot.num_joints}개): {robot.joint_names}")
    log.info(f"관측 차원: {env.observation_manager.group_obs_dim}")
    log.info(f"위상 주기: {cfg.gait.cycle_time_s}s, 기준 동작 관절: "
             f"{[robot.joint_names[index] for index in env.gait.tracked_joint_ids]}")
    log.info(f"보상 항목({len(env.reward_manager.active_terms)}개): {env.reward_manager.active_terms}")

    # 매 스텝 유지할 기본자세(0) 액션. ActionManager가 관리하는 전체 액션 차원 수를 사용한다.
    default_actions = torch.zeros((args_cli.num_envs, env.action_manager.total_action_dim), device=cfg.sim.device)

    # 시뮬레이션 루프
    env.reset()
    while simulation_app.is_running():
        env.step(default_actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
