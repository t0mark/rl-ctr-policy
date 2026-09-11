# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습된 dwaq 정책 체크포인트를 불러와 평가한다.

--mode pilot: GUI 렌더링으로 실제 걷는 모습을 눈으로 확인한다.
--mode full : GUI 없이(headless) 더 많은 환경으로 평가 지표(추적 오차/낙상률 등)를 집계한다.
--checkpoint는 파일명만 지정하면 data/robot/policy/{robot_id}/{가장 최근 학습 run}/{checkpoint}를 읽는다.
cmd_vel은 항상 전진(lin_vel_x 고정, lin_vel_y/ang_vel_z=0)만 나가도록 고정한다.

    실행 명령어:
    ./isaaclab.sh -p scripts/02_eval.py --mode pilot --robot_id unitree_g1 --checkpoint model_2850.pt --terrain rough
    ./isaaclab.sh -p scripts/02_eval.py --mode full --robot_id unitree_g1 --checkpoint model_2850.pt --terrain flat
"""

import argparse
import glob
import logging
import os
import sys

import torch
import yaml
from isaaclab.app import AppLauncher

# scripts/02_eval.py -> 루트: dirname 2번.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

_MODE_NUM_ENVS = {"pilot": 4, "full": 64}
_EVAL_STEPS_FULL = 1000

# CLI 인자 정의 + Isaac Sim 부팅
parser = argparse.ArgumentParser(description="dwaq G1 보행 정책 평가.")
parser.add_argument("--mode", type=str, choices=["pilot", "full"], default="pilot", help="평가 모드.")
parser.add_argument("--robot_id", type=str, default="unitree_g1", help="평가할 로봇 ID (ROBOT_ENV_CFGS의 키).")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="체크포인트 파일명(예: model_2850.pt). 경로는 자동으로 찾는다."
)
parser.add_argument("--terrain", type=str, choices=["flat", "rough"], default="rough", help="평지/험지 지형 선택.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# full: 평가만 하면 되므로 GUI 렌더링이 필요 없다. pilot: GUI로 직접 봐야 하므로 그대로 둔다.
if args_cli.mode == "full":
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from src.dwaq.runners.on_policy_runner import OnPolicyRunner
from src.sim.rl.configs import ROBOT_ENV_CFGS
from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper

log = logging.getLogger(__name__)


def _resolve_checkpoint_path(robot_id: str, checkpoint_filename: str) -> str:
    """data/robot/policy/{robot_id}/ 아래에서 가장 최근 학습 run 폴더를 찾아 체크포인트 경로를 만든다."""
    policy_root = os.path.join(_PROJECT_ROOT, "data", "robot", "policy", robot_id)
    run_dirs = sorted(glob.glob(os.path.join(policy_root, "*")))
    latest_run_dir = run_dirs[-1]
    return os.path.join(latest_run_dir, checkpoint_filename)


def main():
    """env를 만들고 dwaq wrapper로 감싼 뒤, 체크포인트를 불러와 전진 명령으로 평가한다."""
    # 환경 설정
    cfg = ROBOT_ENV_CFGS[args_cli.robot_id]()
    cfg.scene.num_envs = _MODE_NUM_ENVS[args_cli.mode]
    cfg.sim.device = args_cli.device
    cfg.sim.save_logs_to_file = False
    cfg.sim.logging_level = "INFO"

    # 지형 설정
    if args_cli.terrain == "flat":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.visual_material = None
        cfg.curriculum.terrain_levels = None
    # rough는 학습 때 쓰던 지형(ROUGH_TERRAINS_CFG) 기본값을 그대로 둔다.

    # 명령: 항상 전진만 나가도록 고정 (좌우/회전 명령 없음, 정지 명령도 없음)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # 환경 생성 + dwaq 계약으로 변환
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped_env = DwaqVecEnvWrapper(env)

    # 체크포인트 로드 (학습 때와 동일한 train_cfg 스키마가 있어야 actor_critic 구조가 맞게 만들어진다)
    train_cfg_path = os.path.join(_PROJECT_ROOT, "configs", "train_cfg.yaml")
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    checkpoint_path = _resolve_checkpoint_path(args_cli.robot_id, args_cli.checkpoint)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=args_cli.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    log.info(f"mode: {args_cli.mode}, robot_id: {args_cli.robot_id}, terrain: {args_cli.terrain}")
    log.info(f"checkpoint: {checkpoint_path}")

    # 평가 루프
    obs, obs_hist = wrapped_env.get_observations()
    if args_cli.mode == "pilot":
        with torch.inference_mode():
            while simulation_app.is_running():
                actions = policy(obs, obs_hist)
                obs, _, _, obs_hist, _, _, _ = wrapped_env.step(actions)
    else:
        # full: 지표를 집계해서 마지막에 평균을 로그로 남긴다.
        episode_infos = []
        with torch.inference_mode():
            for _ in range(_EVAL_STEPS_FULL):
                actions = policy(obs, obs_hist)
                obs, _, _, obs_hist, _, _, infos = wrapped_env.step(actions)
                if "episode" in infos:
                    episode_infos.append(infos["episode"])

        log.info(f"집계된 에피소드 종료 이벤트 수: {len(episode_infos)}")
        if episode_infos:
            for key in episode_infos[0]:
                values = [torch.as_tensor(info[key]).float() for info in episode_infos if key in info]
                mean_value = torch.stack(values).mean().item()
                log.info(f"{key}: {mean_value:.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
