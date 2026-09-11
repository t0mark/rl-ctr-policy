# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""로봇 ID 문자열로 로봇별 환경 설정 클래스를 찾는 레지스트리.

딕셔너리로 로봇 ID -> EnvCfg 클래스를 직접 매핑한다. 
로봇을 추가할 때는 해당 로봇의 `{robot}_env_cfg.py`를 만들고 이 딕셔너리에 "로봇 ID" -> "EnvCfg 클래스"를 추가하면 된다.
"""

from src.sim.rl.configs.g1_env_cfg import G1EnvCfg

# 로봇 ID -> 해당 로봇의 ManagerBasedRLEnvCfg 클래스
ROBOT_ENV_CFGS = {
    "unitree_g1": G1EnvCfg,
}
