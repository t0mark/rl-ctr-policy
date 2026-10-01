# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""data 폴더의 학습 산출물과 checkpoint 경로 규칙을 담당한다."""

import time
from pathlib import Path

from common import PROJECT_ROOT

# 로봇별 정책 산출물이 저장되는 루트 폴더.
POLICY_ROOT = PROJECT_ROOT / "data" / "robot" / "policy"


def create_run_dir(robot_id):
    """실행 시각 이름의 새 학습 산출물 폴더를 만들어 반환한다."""
    run_dir = POLICY_ROOT / robot_id / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_checkpoint(robot_id, filename, timestamp=None, run_dir=None):
    """checkpoint 경로를 찾는다.

    run_dir을 지정하면 그 폴더에서, timestamp를 지정하면 해당 실행 폴더에서,
    둘 다 없으면 파일이 있는 가장 최근 실행 폴더에서 찾는다.
    """
    if run_dir is not None:
        return Path(run_dir) / filename
    if timestamp is not None:
        return POLICY_ROOT / robot_id / timestamp / filename
    return sorted((POLICY_ROOT / robot_id).glob(f"*/{filename}"))[-1]
