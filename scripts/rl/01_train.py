# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ 학습.

--resume을 지정하면 checkpoint의 모델·optimizer·AdaBoot·반복 횟수를 이어받아 학습을 계속한다.

실행 명령어:
    ./isaaclab.sh -p scripts/rl/01_train.py --mode pilot --robot-id unitree_g1 --headless
    ./isaaclab.sh -p scripts/rl/01_train.py --mode full --robot-id unitree_g1
    ./isaaclab.sh -p scripts/rl/01_train.py --mode full --robot-id unitree_g1 \
        --resume data/robot/policy/unitree_g1/<실행시각>/model_500.pt

산출물 경로:
    data/robot/policy/{robot-id}/{실행시각}/model_{iteration}.pt
    data/robot/policy/{robot-id}/{실행시각}/model_best.pt (평균 지형 레벨 최고 정책)
"""

import argparse
import logging
from pathlib import Path

import yaml

from common import PROJECT_ROOT
from common.app_launcher import add_common_arguments, launch_app, parse_arguments
from common.config import load_robot_config, load_yaml
from common.paths import create_run_dir
from common.run_io import RunIO

TRAIN_CONFIG = load_yaml("configs/rl/train.yaml")


def main():
    """환경과 runner를 조립해 학습한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ 학습")
    add_common_arguments(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--resume", type=Path, help="이어서 학습할 checkpoint")
    execution_cfg = TRAIN_CONFIG["execution"]
    args = parse_arguments(parser, execution_cfg["pilot_num_envs"], execution_cfg["full_num_envs"])

    with launch_app(args):
        from isaaclab.envs import ManagerBasedRLEnv

        from src.sim.rl.envs.env_cfg import build_env_config
        from src.sim.rl.runners.on_policy_runner import OnPolicyRunner
        from src.sim.rl.wrappers.env_adapter import EnvironmentAdapter

        # 로봇 YAML로 환경을 만들고 runner를 조립한다.
        cfg = build_env_config(load_robot_config(args.robot_id), PROJECT_ROOT)
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        if args.seed is not None:
            cfg.seed = args.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        adapter = EnvironmentAdapter(env)
        run_dir = create_run_dir(args.robot_id)
        run_io = RunIO(run_dir)
        runner = OnPolicyRunner(adapter, TRAIN_CONFIG, run_io=run_io, device=args.device)
        try:
            # checkpoint가 지정되면 학습 상태를 이어받는다.
            if args.resume is not None:
                runner.load_state_dict(RunIO.load_checkpoint(args.resume, args.device), resume=True)

            # 학습 설정과 관측·행동 규격을 산출물 폴더에 기록하고 학습을 실행한다.
            (run_dir / "train.yaml").write_text(yaml.safe_dump(TRAIN_CONFIG, allow_unicode=True))
            (run_dir / "environment_spec.yaml").write_text(yaml.safe_dump(adapter.specification))
            logging.info("mode=%s run_dir=%s", args.mode, run_dir)
            runner_cfg = TRAIN_CONFIG["runner"]
            iterations = args.max_iterations
            if iterations is None:
                iterations = runner_cfg["pilot_iterations" if args.mode == "pilot" else "max_iterations"]
            runner.learn(iterations)
        finally:
            run_io.close()
            env.close()


if __name__ == "__main__":
    main()
