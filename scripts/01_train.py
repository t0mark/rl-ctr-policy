# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습: --phase로 Two-Phase 학습 단계를 선택해 실행한다.

1단계는 쉬운 지형에서 사인파 기준 동작을 추종하는 보행을 학습한다.
--resume을 지정하면 중단된 1단계 학습 상태를 이어간다.

2단계는 어려운 지형에서 기준 동작 추종 보상 없이 적응 보행을 학습하며 --resume이 필수다.
1단계 checkpoint를 지정하면 모델 가중치·정규화 통계·명령 curriculum만 이어받아 2단계를 시작하고,
2단계 checkpoint를 지정하면 중단된 2단계 학습 상태 전체를 이어간다.

실행 명령어:
    ./isaaclab.sh -p scripts/01_train.py --phase 1 --mode pilot --robot-id unitree_g1
    ./isaaclab.sh -p scripts/01_train.py --phase 1 --mode full --robot-id unitree_g1
    ./isaaclab.sh -p scripts/01_train.py --phase 2 --mode full --robot-id unitree_g1 \
        --resume data/robot/policy/unitree_g1/phase1/<실행시각>/model_p1_3000.pt

산출물 경로:
    data/robot/policy/{robot-id}/phase{phase}/{실행시각}/model_p{phase}_{iteration}.pt
"""

import argparse
import logging
from pathlib import Path

import yaml

from utils import PROJECT_ROOT
from utils.app_launcher import add_common_arguments, launch_app, parse_arguments
from utils.run_paths import create_run_dir

# 파일럿은 로직 점검용 소수 환경, 본 학습은 논문 Table 6의 병렬 환경 수를 사용한다.
PILOT_NUM_ENVS = 16
FULL_NUM_ENVS = 4096

# 학습 설정 파일 경로.
TRAIN_CFG_PATH = PROJECT_ROOT / "configs" / "train_cfg.yaml"


def main():
    """선택한 학습 단계의 환경과 runner를 구성하고 학습 후 자원을 정리한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ/Two-Phase 학습")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True, help="실행할 학습 단계")
    add_common_arguments(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--resume", type=Path,
                        help="1단계는 이어갈 1단계 checkpoint, 2단계는 이어받을 1단계 또는 이어갈 2단계 checkpoint")
    args = parse_arguments(parser, PILOT_NUM_ENVS, FULL_NUM_ENVS)

    # 2단계는 이어받을 checkpoint 없이 시작할 수 없으므로 앱 시작 전에 확인한다.
    if args.phase == 2 and args.resume is None:
        parser.error("2단계 학습은 --resume으로 1단계 또는 2단계 checkpoint를 지정해야 합니다.")

    with launch_app(args):
        from src.sim.rl.training_session import TrainingSession

        # 학습 설정을 읽고 산출물 폴더를 만든 뒤 학습 세션을 구성한다.
        train_cfg = yaml.safe_load(TRAIN_CFG_PATH.read_text())
        run_dir = create_run_dir(args.robot_id, args.phase)
        session = TrainingSession(
            robot_id=args.robot_id, phase=args.phase, train_cfg=train_cfg,
            device=args.device, num_envs=args.num_envs, log_dir=run_dir, seed=args.seed)
        try:
            # checkpoint가 지정되면 단계에 맞춰 정책을 이어받거나 학습 상태를 재개한다.
            if args.resume is not None:
                session.runner.load(args.resume, mode="resume")

            # 학습 설정과 관측·행동 규격을 산출물 폴더에 기록하고 학습을 실행한다.
            (run_dir / "train_cfg.yaml").write_text(yaml.safe_dump(train_cfg, allow_unicode=True))
            (run_dir / "environment_spec.yaml").write_text(yaml.safe_dump(session.wrapper.specification))
            logging.info("phase=%d mode=%s run_dir=%s", args.phase, args.mode, run_dir)
            session.learn(session.iterations(args.mode, args.max_iterations))
        finally:
            session.close()


if __name__ == "__main__":
    main()
