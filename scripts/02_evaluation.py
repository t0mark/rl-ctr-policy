# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""평가: 학습된 checkpoint의 결정적 정책을 기본 동역학·고정 명령에서 평가한다.

실행 명령어:
    ./isaaclab.sh -p scripts/02_evaluation.py --mode pilot --phase 2 \
        --checkpoint model_p2_5000.pt
    ./isaaclab.sh -p scripts/02_evaluation.py --mode full --phase 2 --terrain flat \
        --checkpoint model_p2_5000.pt --output data/eval/phase2_flat.json

checkpoint는 data/robot/policy/{robot-id}/phase{phase}/{실행시각}/{checkpoint}에서 찾는다.
--timestamp를 생략하면 해당 파일이 있는 가장 최근 run을 쓴다.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from utils.app_launcher import add_common_arguments, launch_app, parse_arguments
from utils.run_paths import resolve_checkpoint

# 파일럿은 로직 점검용 소수 환경, 본 평가는 통계 집계용 병렬 환경 수를 사용한다.
PILOT_NUM_ENVS = 4
FULL_NUM_ENVS = 64


def main():
    """checkpoint를 적재한 평가 세션을 실행하고 보고서를 터미널과 JSON 파일에 기록한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ/Two-Phase 평가")
    add_common_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--timestamp",
                        help="phase 폴더 안에서 사용할 실행 시각. 생략하면 가장 최근 run을 쓴다.")
    parser.add_argument("--run-dir", type=Path,
                        help="기본 경로 대신 직접 지정할 checkpoint 폴더")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="checkpoint 탐색과 평가 지형에 사용할 학습 단계")
    parser.add_argument("--terrain", choices=["phase", "flat"], default="phase")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--output", type=Path, help="평가 집계 JSON 출력 경로")
    args = parse_arguments(parser, PILOT_NUM_ENVS, FULL_NUM_ENVS)

    # 앱 시작 전에 checkpoint 경로를 확정한다.
    path = resolve_checkpoint(args.robot_id, args.phase, args.checkpoint, args.timestamp, args.run_dir)

    with launch_app(args):
        from src.sim.rl.evaluation_session import EvaluationSession

        # checkpoint의 학습 설정으로 평가 세션을 구성한다.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        session = EvaluationSession(
            robot_id=args.robot_id, phase=args.phase, train_cfg=checkpoint["train_cfg"],
            device=args.device, num_envs=args.num_envs,
            command=(args.vx, args.vy, args.yaw_rate), seed=args.seed,
            flat_terrain=args.terrain == "flat", observation_noise=args.noise)
        try:
            # checkpoint의 모델과 정규화 통계를 적재하고 평가를 실행한다.
            session.runner.load(path, mode="evaluate")
            report = {
                "checkpoint": str(path), "checkpoint_phase": int(checkpoint["phase"]),
                "evaluation_phase": args.phase, "seed": args.seed, "terrain": args.terrain,
                "noise": args.noise, **session.evaluate(args.steps),
            }

            # 보고서를 터미널과 JSON 파일에 기록한다.
            logging.info("mode=%s evaluation=%s", args.mode, report)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            session.close()


if __name__ == "__main__":
    main()
