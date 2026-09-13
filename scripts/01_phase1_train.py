# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습 1단계: 쉬운 지형에서 사인파 기준 동작을 추종하는 보행을 학습한다.

실행 명령어:
    ./isaaclab.sh -p scripts/01_phase1_train.py --mode pilot --robot-id unitree_g1
    ./isaaclab.sh -p scripts/01_phase1_train.py --mode full --robot-id unitree_g1 --headless

산출물 경로:
    data/robot/policy/{robot-id}/phase1/{실행시각}/model_p1_{iteration}.pt
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

TRAINING_PHASE = 1


def main():
    """1단계 환경과 runner를 구성하고 학습 후 자원을 정리한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ/Two-Phase G1 1단계 학습")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--robot-id", default="unitree_g1")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="같은 1단계 학습을 이어갈 checkpoint 경로")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
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

        # checkpoint가 지정되면 중단된 1단계 학습 상태를 이어간다.
        if args.resume is not None:
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
