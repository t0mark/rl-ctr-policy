# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습 산출물 폴더 생성과 checkpoint 탐색의 경로 규칙을 담당한다."""

import time
from pathlib import Path

from utils import PROJECT_ROOT

# 로봇별·학습 단계별 정책 산출물이 저장되는 루트 폴더.
POLICY_ROOT = PROJECT_ROOT / "data" / "robot" / "policy"


def phase_dir(robot_id, phase):
    """로봇·학습 단계의 정책 산출물 폴더 경로를 반환한다."""
    return POLICY_ROOT / robot_id / f"phase{phase}"


def create_run_dir(robot_id, phase):
    """실행 시각 이름의 새 학습 산출물 폴더를 만들어 반환한다."""
    run_dir = phase_dir(robot_id, phase) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_checkpoint(robot_id, phase, filename, timestamp=None, run_dir=None):
    """학습 단계 폴더 아래에서 checkpoint를 찾는다.

    run_dir을 지정하면 그 폴더에서, timestamp를 지정하면 해당 실행 폴더에서,
    둘 다 없으면 파일이 있는 가장 최근 실행 폴더에서 찾는다.
    """
    if run_dir is not None:
        path = Path(run_dir) / filename
    elif timestamp is not None:
        path = phase_dir(robot_id, phase) / timestamp / filename
    else:
        # 실행 시각 이름순으로 정렬해 파일이 있는 가장 최근 실행 폴더를 고른다.
        root = phase_dir(robot_id, phase)
        candidates = sorted(root.glob(f"*/{filename}"))
        if not candidates:
            raise FileNotFoundError(f"{root} 아래에서 {filename}을 찾지 못했습니다.")
        path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(f"{path}를 찾지 못했습니다.")
    return path
