# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""저장된 규격·정규화와 결정적 context로 학습된 정책을 평가한다.

실행 명령어:
    ./isaaclab.sh -p scripts/03_eval.py --mode pilot --phase 2 \
        --checkpoint model_p2_5000.pt
    ./isaaclab.sh -p scripts/03_eval.py --mode full --phase 2 --terrain flat \
        --checkpoint model_p2_5000.pt --output data/eval/phase2_flat.json

checkpoint는 data/robot/policy/{robot-id}/phase{phase}/{실행시각}/{checkpoint}에서 찾는다.
--timestamp를 생략하면 해당 파일이 있는 가장 최근 run을 쓴다.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def resolve_checkpoint(robot_id, phase, filename, timestamp=None, run_dir=None):
    """학습 단계 폴더 아래에서 checkpoint를 찾는다."""
    if run_dir is not None:
        path = Path(run_dir) / filename
    else:
        root = _PROJECT_ROOT / "data/robot/policy" / robot_id / f"phase{phase}"
        if timestamp is not None:
            path = root / timestamp / filename
        else:
            candidates = sorted(root.glob(f"*/{filename}"))
            if not candidates:
                raise FileNotFoundError(f"{root} 아래에서 {filename}을 찾지 못했습니다.")
            path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(f"{path}를 찾지 못했습니다.")
    return path


def main():
    """고정 명령·seed·지형 조건으로 평가하고 추정 오차와 종료 수를 기록한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ/Two-Phase G1 평가")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--robot-id", default="unitree_g1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--timestamp",
                       help="phase 폴더 안에서 사용할 실행 시각. 생략하면 가장 최근 run을 쓴다.")
    parser.add_argument("--run-dir", type=Path,
                       help="기본 경로 대신 직접 지정할 checkpoint 폴더")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                       help="평가 지형으로 사용할 학습 단계")
    parser.add_argument("--terrain", choices=["phase", "flat"], default="phase")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--output", type=Path, help="평가 집계 JSON 출력 경로")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "full":
        args.headless = True
    launcher = AppLauncher(args)
    env = None
    try:
        from src.sim.rl.dreamwaq_env import DreamWaQEnv
        from src.sim.rl.configs import ROBOT_ENV_CFGS
        from src.sim.rl.env_cfg import configure_training_phase
        from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper
        from src.dwaq.runners.on_policy_runner import OnPolicyRunner

        logging.basicConfig(level=logging.INFO)
        path = resolve_checkpoint(args.robot_id, args.phase, args.checkpoint,
                                  args.timestamp, args.run_dir)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ROBOT_ENV_CFGS[args.robot_id]()
        cfg.scene.num_envs = 4 if args.mode == "pilot" else 64
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.sim.save_logs_to_file = False
        cfg.sim.logging_level = "INFO"

        # 학습과 같은 단계 설정으로 지형을 만든 뒤 평가 조건을 고정한다.
        configure_training_phase(cfg, args.phase)
        cfg.observations.policy_current.enable_corruption = args.noise
        cfg.curriculum.terrain_levels = None
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.commands.base_velocity.curriculum_enabled = False
        cfg.commands.base_velocity.rel_standing_envs = 0.0
        cfg.commands.base_velocity.ranges.lin_vel_x = (args.vx, args.vx)
        cfg.commands.base_velocity.ranges.lin_vel_y = (args.vy, args.vy)
        cfg.commands.base_velocity.ranges.ang_vel_z = (args.yaw_rate, args.yaw_rate)
        # 평가 외란을 고정하며 별도 강건성 실험과 구분한다.
        cfg.events.disturbance.params["force_range"] = (0.0, 0.0)
        if args.terrain == "flat":
            cfg.scene.terrain.terrain_type = "plane"
            cfg.scene.terrain.terrain_generator = None
            cfg.scene.terrain.visual_material = None
        env = DreamWaQEnv(cfg=cfg)
        wrapper = DwaqVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapper, checkpoint["train_cfg"], device=args.device)
        runner.load(path, mode="evaluate")
        policy = runner.get_inference_policy()
        observation = wrapper.get_observations()
        squared_error = torch.zeros(3, device=args.device)
        tracking_error = torch.zeros(3, device=args.device)
        terminated_count = torch.zeros((), device=args.device)
        truncated_count = torch.zeros((), device=args.device)
        step = 0
        with torch.no_grad():
            while launcher.app.is_running() and (args.mode == "pilot" or step < args.steps):
                estimated = runner.model.estimate_velocity(observation["history"])
                squared_error += (estimated - observation["velocity"]).square().sum(0)
                command = env.command_manager.get_command("base_velocity")
                measured = torch.cat((observation["velocity"][:, :2],
                                      env.scene["robot"].data.root_ang_vel_b[:, 2:3]), -1)
                tracking_error += (measured-command).abs().sum(0)
                action = policy(observation["history"])
                observation, result = wrapper.step(action)
                terminated_count += result["terminated"].sum()
                truncated_count += (result["truncated"] & ~result["terminated"]).sum()
                step += 1
        count = max(step * wrapper.num_envs, 1)
        deaths, timeouts = int(terminated_count.item()), int(truncated_count.item())
        report = {
            "checkpoint": str(path), "checkpoint_phase": int(checkpoint["phase"]),
            "evaluation_phase": args.phase, "seed": args.seed, "terrain": args.terrain,
            "steps": step, "num_envs": wrapper.num_envs, "noise": args.noise,
            "command": [args.vx, args.vy, args.yaw_rate],
            "velocity_rmse_mps": (squared_error/count).sqrt().tolist(),
            "command_mae": (tracking_error/count).tolist(),
            "terminated_episodes": deaths, "timeout_episodes": timeouts,
            "termination_fraction_of_completed": deaths/max(deaths+timeouts, 1),
            "note": "미완료 episode는 종료 비율 분모에서 제외; 전체 보행 성공률과 다름",
        }
        logging.info("evaluation=%s", report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
