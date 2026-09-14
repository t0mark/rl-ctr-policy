# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""스크립트 공통 실행 인자 해석과 Isaac Sim 앱 시작·종료를 담당한다.

isaaclab·src.sim 모듈은 앱이 시작된 뒤에만 import할 수 있으므로,
이 모듈은 isaaclab.app과 표준 라이브러리만 import한다.
"""

import contextlib
import logging

from isaaclab.app import AppLauncher


def add_common_arguments(parser):
    """실행 모드·로봇 ID·병렬 환경 수와 AppLauncher 인자를 parser에 추가한다."""
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot은 소수 환경 점검, full은 headless 본 실행이다.")
    parser.add_argument("--robot-id", default="unitree_g1", help="로봇 ID (ROBOT_ENV_CFGS의 키).")
    parser.add_argument("--num_envs", type=int,
                        help="병렬 환경 수. 생략하면 실행 모드의 기본 환경 수를 쓴다.")
    AppLauncher.add_app_launcher_args(parser)


def parse_arguments(parser, pilot_num_envs, full_num_envs):
    """실행 인자를 해석하고 실행 모드에 맞춰 headless 여부와 병렬 환경 수를 채운다."""
    args = parser.parse_args()

    # 본 실행은 화면 없이 수행한다.
    if args.mode == "full":
        args.headless = True

    # 병렬 환경 수를 지정하지 않으면 실행 모드의 기본 환경 수를 쓴다.
    if args.num_envs is None:
        args.num_envs = pilot_num_envs if args.mode == "pilot" else full_num_envs
    return args


@contextlib.contextmanager
def launch_app(args):
    """해석한 인자로 Isaac Sim 앱을 시작하고 로그를 설정한 뒤, 블록이 끝나면 앱을 종료한다.

    isaaclab과 src.sim 모듈은 이 블록 안에서 import해야 한다.
    """
    launcher = AppLauncher(args)
    logging.basicConfig(level=logging.INFO)
    try:
        yield launcher.app
    finally:
        launcher.app.close()
