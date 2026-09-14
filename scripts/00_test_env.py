# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""시뮬레이션·로봇 스폰 점검: 평지에서 로봇이 정상적으로 스폰되고 기본자세를 유지하는지 확인한다.

점검 항목:
    시뮬레이션과 환경이 오류 없이 구성되는지
    로봇 USD와 RL 제어 관절이 의도대로 매칭되는지
    기본자세(action 0)에서 PD 게인이 로봇을 지탱하는지

실행 명령어:
    ./isaaclab.sh -p scripts/00_test_env.py --robot-id unitree_g1
"""

import argparse
import logging

import torch
from isaaclab.app import AppLauncher

from utils.app_launcher import launch_app

log = logging.getLogger(__name__)

# 스폰 점검에 사용하는 병렬 환경 수.
NUM_ENVS = 4


def main():
    """평지에 로봇을 스폰해 구성을 로그로 남긴 뒤 기본자세를 유지하며 스텝한다."""
    parser = argparse.ArgumentParser(description="시뮬레이션·로봇 스폰 점검.")
    parser.add_argument("--robot-id", default="unitree_g1", help="스폰할 로봇 ID (ROBOT_ENV_CFGS의 키).")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    with launch_app(args) as app:
        from src.sim.rl.dreamwaq_env import DreamWaQEnv
        from src.sim.rl.configs import ROBOT_ENV_CFGS

        # 환경 수를 고정하고 지형을 평면으로 두어 지형 커리큘럼을 끈다.
        cfg = ROBOT_ENV_CFGS[args.robot_id]()
        cfg.scene.num_envs = NUM_ENVS
        cfg.sim.device = args.device
        cfg.sim.save_logs_to_file = False
        cfg.sim.logging_level = "INFO"
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.visual_material = None
        cfg.curriculum.terrain_levels = None

        # 환경 생성
        env = DreamWaQEnv(cfg=cfg)
        try:
            # 로봇 USD와 RL 제어 관절 매칭 결과를 기록한다.
            robot = env.scene["robot"]
            matched_names = env.action_manager.get_term("joint_pos").joint_names
            log.info(f"robot_id: {args.robot_id}")
            log.info(f"robot usd 경로: {cfg.scene.robot.spawn.usd_path}")
            log.info(f"RL 제어 관절({len(matched_names)}개): {matched_names}")
            log.info(f"로봇 전체 관절({robot.num_joints}개): {robot.joint_names}")

            # 매 스텝 유지할 기본자세(0) 액션을 전체 액션 차원으로 만든다.
            default_actions = torch.zeros((NUM_ENVS, env.action_manager.total_action_dim), device=env.device)

            # 앱이 종료될 때까지 기본자세를 유지하며 스텝한다.
            env.reset()
            while app.is_running():
                env.step(default_actions)
        finally:
            env.close()


if __name__ == "__main__":
    main()
