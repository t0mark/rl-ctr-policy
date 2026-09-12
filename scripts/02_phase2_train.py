# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습 2단계: 어려운 지형에서 기준 동작 추종 보상 없이 적응 보행을 학습한다.

1단계 정책을 이어받아 시작하며, 모델 가중치와 정규화 통계·명령 curriculum만 넘겨받고
optimizer 모멘텀과 AdaBoot 통계는 새로 시작한다(보상 구성이 바뀌어 스케일이 달라진다).

실행 명령어:
    ./isaaclab.sh -p scripts/02_phase2_train.py --mode pilot --robot-id unitree_g1 \
        --transfer-from data/robot/policy/unitree_g1/phase1/<실행시각>/model_p1_3000.pt
    ./isaaclab.sh -p scripts/02_phase2_train.py --mode full --robot-id unitree_g1 \
        --transfer-from data/robot/policy/unitree_g1/phase1/<실행시각>/model_p1_3000.pt --headless

산출물 경로:
    data/robot/policy/{robot-id}/phase2/{실행시각}/model_p2_{iteration}.pt
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import yaml
from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

TRAINING_PHASE = 2


def main():
    """1단계 체크포인트를 이어받아 2단계 학습을 실행한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ/Two-Phase G1 2단계 학습")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--robot-id", default="unitree_g1")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--transfer-from", type=Path, help="이어받을 1단계 checkpoint 경로")
    parser.add_argument("--resume", type=Path, help="중단된 2단계 학습을 이어갈 checkpoint 경로")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.transfer_from is None) == (args.resume is None):
        parser.error("--transfer-from(1단계 이어받기) 또는 --resume(2단계 재개) 중 하나만 지정해야 합니다.")
    launcher = AppLauncher(args)
    session = None
    try:
        from src.sim.rl.training_session import TrainingSession, resolve_num_envs

        logging.basicConfig(level=logging.INFO)
        train_cfg = yaml.safe_load((_PROJECT_ROOT / "configs/train_cfg.yaml").read_text())
        run_dir = (_PROJECT_ROOT / "data/robot/policy" / args.robot_id
                   / f"phase{TRAINING_PHASE}" / time.strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=False)
        session = TrainingSession(
            robot_id=args.robot_id, phase=TRAINING_PHASE, train_cfg=train_cfg,
            device=args.device, num_envs=resolve_num_envs(args.mode, args.num_envs),
            log_dir=run_dir, seed=args.seed)

        # 단계 전환과 같은 단계 재개는 검증 규칙이 다르므로 명시적으로 구분한다.
        if args.transfer_from is not None:
            session.runner.load(args.transfer_from, mode="transfer")
        else:
            session.runner.load(args.resume, mode="resume")
        (run_dir / "train_cfg.yaml").write_text(yaml.safe_dump(train_cfg, allow_unicode=True))
        (run_dir / "environment_spec.yaml").write_text(yaml.safe_dump(session.wrapper.specification))
        logging.info("mode=%s run_dir=%s", args.mode, run_dir)
        session.learn(session.iterations(args.mode, args.max_iterations))
    finally:
        if session is not None:
            session.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
