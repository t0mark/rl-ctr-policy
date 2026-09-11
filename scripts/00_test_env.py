# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 파일럿 테스트: 지형/로봇/센서가 실제로 GUI에서 제대로 스폰되는지 점검

점검 항목:
    로봇을 default_joint_pos로 유지시켜서, PD 게인이 로봇을 지탱하는지
    지형/로봇이 겹치거나 깨지지 않는지

    실행 명령어:
    ./isaaclab.sh -p scripts/00_test_env.py --robot-id unitree_g1 --num_envs 4
"""

import argparse
import logging
import os
import sys
import torch

from isaaclab.app import AppLauncher

# CLI 인자 정의 + Isaac Sim 부팅
parser = argparse.ArgumentParser(description="로봇 스폰 파일럿 테스트.")
parser.add_argument("--robot-id", type=str, default="unitree_g1", help="스폰할 로봇 ID (ROBOT_ENV_CFGS의 키).")
parser.add_argument("--num_envs", type=int, default=4, help="스폰할 환경 개수.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# scripts/00_test_env.py -> 루트: dirname 2번.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from src.sim.rl.configs import ROBOT_ENV_CFGS

log = logging.getLogger(__name__)


def main():
    """--robot-id로 지정된 로봇의 EnvCfg로 ManagerBasedRLEnv를 만들고, 설정을 로그로 남긴 뒤 기본자세를 유지하며 계속 스텝한다."""
    # 환경 설정
    cfg = ROBOT_ENV_CFGS[args_cli.robot_id]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.sim.save_logs_to_file = False
    cfg.sim.logging_level = "INFO"
    # 지형 생성 설정
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.curriculum.terrain_levels = None

    # 환경 생성
    env = ManagerBasedRLEnv(cfg=cfg)
    joint_pos_term = env.action_manager.get_term("joint_pos")
    matched_names = joint_pos_term.IO_descriptor.joint_names

    # 로그 출력
    log.info(f"robot_id: {args_cli.robot_id}")
    log.info(f"robot usd 경로h: {cfg.scene.robot.spawn.usd_path}")
    log.info(f"실제로 매칭된 관절({len(matched_names)}개): {matched_names}")
    log.info(f"로봇 전체 관절({env.scene['robot'].num_joints}개): {env.scene['robot'].joint_names}")

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
