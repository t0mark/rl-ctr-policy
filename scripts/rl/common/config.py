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
    """configs/rl/robot/ 아래에서 로봇 ID 이름의 YAML 설정을 찾아 읽는다."""
    matches = sorted((PROJECT_ROOT / "configs" / "rl" / "robot").rglob(f"{robot_id}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"로봇 ID {robot_id}의 설정 파일이 하나가 아닙니다: {matches}")
    return yaml.safe_load(matches[0].read_text())
