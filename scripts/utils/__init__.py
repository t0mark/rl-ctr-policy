# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""scripts/ 실행 스크립트가 공유하는 실행 준비 함수 모음.

패키지를 import하면 src 패키지를 import할 수 있도록 프로젝트 루트를 import 경로에 등록한다.
"""

import sys
from pathlib import Path

# scripts/utils/__init__.py에서 2단계 상위 폴더인 프로젝트 루트.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# src 패키지를 import할 수 있도록 프로젝트 루트를 import 경로 맨 앞에 등록한다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
