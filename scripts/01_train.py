"""논문 로직의 DreamWaQ 정책을 새 체크포인트 형식으로 학습한다."""
import argparse
import logging
from pathlib import Path
import sys
import time

import yaml
from isaaclab.app import AppLauncher

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    """환경·wrapper·runner를 구성하고 학습 후 자원을 정리한다."""
    parser = argparse.ArgumentParser(description="DreamWaQ G1 학습")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--robot-id", default="unitree_g1")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--resume", type=Path, help="새 형식의 checkpoint 경로")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    env = None
    try:
        from src.sim.rl.dreamwaq_env import DreamWaQEnv
        from src.sim.rl.configs import ROBOT_ENV_CFGS
        from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper
        from src.dwaq.runners.on_policy_runner import OnPolicyRunner

        logging.basicConfig(level=logging.INFO)
        cfg = ROBOT_ENV_CFGS[args.robot_id]()
        cfg.scene.num_envs = args.num_envs or (16 if args.mode == "pilot" else 4096)
        cfg.sim.device = args.device
        cfg.sim.save_logs_to_file = False
        cfg.sim.logging_level = "INFO"
        env = DreamWaQEnv(cfg=cfg)
        wrapper = DwaqVecEnvWrapper(env)
        train_cfg = yaml.safe_load((_PROJECT_ROOT / "configs/train_cfg.yaml").read_text())
        run_dir = _PROJECT_ROOT / "data/robot/policy" / args.robot_id / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "train_cfg.yaml").write_text(yaml.safe_dump(train_cfg, allow_unicode=True))
        (run_dir / "environment_spec.yaml").write_text(yaml.safe_dump(wrapper.specification))
        runner = OnPolicyRunner(wrapper, train_cfg, run_dir, device=args.device)
        if args.resume is not None:
            runner.load(args.resume)
        logging.info("mode=%s specification=%s", args.mode, wrapper.specification)
        iterations = args.max_iterations or (200 if args.mode == "pilot" else 3000)
        runner.learn(iterations, init_at_random_ep_len=False)
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
