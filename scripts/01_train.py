# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""dwaq(PPO + DreamWaQ 컨텍스트 인코더)로 G1 보행 정책을 학습한다.

--mode pilot: 적은 환경 수/반복으로 GUI에서 학습되는 모습을 직접 눈으로 확인하는 파일럿 테스트.
--mode full : 실제 학습 규모(대량 환경, headless 권장)로 전체 학습을 수행한다.
두 모드 모두 로직(보상/관측/커리큘럼 등)은 동일하며 규모(num_envs, max_iterations)만 다르다.

    실행 명령어:
    ./isaaclab.sh -p scripts/01_train.py --mode pilot --robot-id unitree_g1
    ./isaaclab.sh -p scripts/01_train.py --mode full --robot-id unitree_g1 --headless
"""

import argparse
import logging
import os
import sys
import time

import yaml
from isaaclab.app import AppLauncher

# 모드별 기본 규모. --num_envs/--max_iterations로 직접 지정하면 이 기본값을 덮어쓴다.
_MODE_DEFAULTS = {
    "pilot": {"num_envs": 16, "max_iterations": 200},
    "full": {"num_envs": 4096, "max_iterations": 3000},
}

# scripts/01_train.py -> 루트: dirname 2번.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# CLI 인자 정의 + Isaac Sim 부팅
parser = argparse.ArgumentParser(description="dwaq G1 보행 정책 학습.")
parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="pilot", help="학습 모드.")
parser.add_argument("--robot-id", type=str, default="unitree_g1", help="학습할 로봇 ID (ROBOT_ENV_CFGS의 키).")
parser.add_argument("--num_envs", type=int, default=None, help="병렬 환경 개수 (기본값: 모드별 기본값).")
parser.add_argument("--max_iterations", type=int, default=None, help="정책 업데이트 반복 횟수 (기본값: 모드별 기본값).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from src.dwaq.runners.on_policy_runner import OnPolicyRunner
from src.sim.rl.configs import ROBOT_ENV_CFGS
from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper

log = logging.getLogger(__name__)


def main():
    """--robot-id/--mode로 환경을 만들고, dwaq wrapper로 감싼 뒤 OnPolicyRunner로 학습한다."""
    mode_defaults = _MODE_DEFAULTS[args_cli.mode]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else mode_defaults["num_envs"]
    max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else mode_defaults["max_iterations"]

    # 환경 설정
    cfg = ROBOT_ENV_CFGS[args_cli.robot_id]()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args_cli.device
    cfg.sim.save_logs_to_file = False
    cfg.sim.logging_level = "INFO"

    # 환경 생성 + dwaq 계약으로 변환
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped_env = DwaqVecEnvWrapper(env)

    # 로그/체크포인트 디렉터리: data/robot/policy/{robot_id}/{타임스탬프}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_PROJECT_ROOT, "data", "robot", "policy", args_cli.robot_id, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # 로그 출력
    log.info(f"mode: {args_cli.mode}, robot_id: {args_cli.robot_id}")
    log.info(f"num_envs: {num_envs}, max_iterations: {max_iterations}")
    log.info(
        f"num_obs: {wrapped_env.num_obs}, num_obs_hist: {wrapped_env.num_obs_hist}, "
        f"num_privileged_obs: {wrapped_env.num_privileged_obs}, num_actions: {wrapped_env.num_actions}"
    )
    log.info(f"log_dir: {log_dir}")

    # 학습 실행: configs/train_cfg.yaml -> dwaq(OnPolicyRunner)가 기대하는 train_cfg 딕셔너리
    train_cfg_path = os.path.join(_PROJECT_ROOT, "configs", "train_cfg.yaml")
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=args_cli.device)
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
