"""RL YAML 설정을 프로젝트 루트 기준으로 읽는다."""

from pathlib import Path

import yaml

from common import PROJECT_ROOT


def load_yaml(relative_path):
    """프로젝트 루트 기준 YAML 파일을 읽어 딕셔너리로 반환한다."""
    path = PROJECT_ROOT / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"설정 파일을 찾지 못했습니다: {path}")
    return yaml.safe_load(path.read_text())


def load_robot_config(robot_id):
    """지원하는 로봇 ID의 YAML 설정을 읽는다."""
    paths = {
        "unitree_g1": "configs/rl/robot/legged/humanoid/unitree_g1.yaml",
        "unitree_go2": "configs/rl/robot/legged/multi-legged/unitree_go2.yaml",
        "boston_dynamics_spot": "configs/rl/robot/legged/multi-legged/boston_dynamics_spot.yaml",
    }
    if robot_id not in paths:
        raise ValueError(f"지원하지 않는 로봇 ID: {robot_id}")
    return load_yaml(paths[robot_id])
