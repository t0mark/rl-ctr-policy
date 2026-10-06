# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""평가: 학습된 checkpoint의 결정적 정책을 기본 동역학·고정 명령에서 평가한다.

실행 명령어:
    ./isaaclab.sh -p scripts/rl/02_evaluation.py --mode pilot --checkpoint model_1000.pt
    ./isaaclab.sh -p scripts/rl/02_evaluation.py --mode full --terrain flat \
        --checkpoint model_1000.pt --output data/eval/flat.json

checkpoint는 data/robot/policy/{robot-id}/{실행시각}/{checkpoint}에서 찾는다.
--timestamp를 생략하면 해당 파일이 있는 가장 최근 run을 쓴다.
"""

import argparse
import json
import logging
from pathlib import Path

from common import PROJECT_ROOT
from common.app_launcher import add_common_arguments, launch_app, parse_arguments
from common.config import load_robot_config, load_yaml
from common.paths import resolve_checkpoint
from common.run_io import RunIO

EVALUATION_CONFIG = load_yaml("configs/rl/evaluation.yaml")["evaluation"]


def main():
    """checkpoint를 적재해 평가하고 보고서를 터미널과 JSON 파일에 기록한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ 평가")
    add_common_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--timestamp", help="사용할 실행 시각 폴더. 생략하면 가장 최근 run을 쓴다.")
    parser.add_argument("--run-dir", type=Path, help="기본 경로 대신 직접 지정할 checkpoint 폴더")
    parser.add_argument("--terrain", choices=["rough", "flat"], default=EVALUATION_CONFIG["terrain"])
    parser.add_argument("--steps", type=int, default=EVALUATION_CONFIG["steps"])
    parser.add_argument("--vx", type=float, default=EVALUATION_CONFIG["command"][0])
    parser.add_argument("--vy", type=float, default=EVALUATION_CONFIG["command"][1])
    parser.add_argument("--yaw_rate", type=float, default=EVALUATION_CONFIG["command"][2])
    parser.add_argument("--noise", action="store_true", default=EVALUATION_CONFIG["observation_noise"])
    parser.add_argument("--output", type=Path, help="평가 결과 JSON 출력 경로")
    args = parse_arguments(parser, EVALUATION_CONFIG["pilot_num_envs"], EVALUATION_CONFIG["full_num_envs"])
    path = resolve_checkpoint(args.robot_id, args.checkpoint, args.timestamp, args.run_dir)

    with launch_app(args):
        from isaaclab.envs import ManagerBasedRLEnv

        from src.sim.rl.envs.env_cfg import build_env_config, configure_evaluation
        from src.sim.rl.runners.on_policy_runner import OnPolicyRunner
        from src.sim.rl.wrappers.env_adapter import EnvironmentAdapter

        # checkpoint의 학습 설정으로 평가 환경과 runner를 만든다.
        checkpoint = RunIO.load_checkpoint(path, args.device)
        train_cfg = checkpoint["train_cfg"]
        command = [args.vx, args.vy, args.yaw_rate]
        cfg = build_env_config(load_robot_config(args.robot_id), PROJECT_ROOT)
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        configure_evaluation(cfg, command, args.terrain == "flat", args.noise)
        env = ManagerBasedRLEnv(cfg=cfg)
        adapter = EnvironmentAdapter(env)
        runner = OnPolicyRunner(adapter, train_cfg, device=args.device)
        try:
            # 모델을 적재하고 평가한 뒤 보고서를 터미널과 JSON 파일에 기록한다.
            runner.load_state_dict(checkpoint, resume=False)
            report = {"checkpoint": str(path), "terrain": args.terrain, "command": command,
                      "noise": args.noise, "seed": args.seed, **runner.evaluate(args.steps)}
            logging.info("evaluation=%s", report)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
