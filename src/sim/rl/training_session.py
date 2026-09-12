# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습 단계별 환경·wrapper·runner 구성을 한곳에서 담당한다.

단계마다 프로세스를 분리해 실행하므로 1단계·2단계 스크립트가 같은 구성 절차를
중복해서 갖지 않도록, 구성과 실행은 이 모듈이 맡고 스크립트는 인자 해석만 맡는다.
"""

import logging

from src.sim.rl.configs import ROBOT_ENV_CFGS
from src.sim.rl.dreamwaq_env import DreamWaQEnv
from src.sim.rl.env_cfg import configure_training_phase
from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper
from src.dwaq.runners.on_policy_runner import OnPolicyRunner

log = logging.getLogger(__name__)

# 파일럿은 로직 점검용 소수 환경, 본 학습은 논문 Table 6의 병렬 환경 수를 사용한다.
PILOT_NUM_ENVS = 16
FULL_NUM_ENVS = 4096


class TrainingSession:
    """한 학습 단계의 환경과 runner를 생성하고 학습을 실행한다."""

    def __init__(self, robot_id, phase, train_cfg, device, num_envs, log_dir=None, seed=None):
        """단계 설정을 적용한 환경을 만들고 runner를 연결한다."""
        self._phase = phase
        self._train_cfg = train_cfg
        cfg = ROBOT_ENV_CFGS[robot_id]()
        cfg.scene.num_envs = num_envs
        cfg.sim.device = device
        cfg.sim.save_logs_to_file = False
        cfg.sim.logging_level = "INFO"
        if seed is not None:
            cfg.seed = seed

        # 지형과 기준 동작 보상 활성 여부는 단계 설정 하나에서 결정한다.
        configure_training_phase(cfg, phase)
        self._env = DreamWaQEnv(cfg=cfg)
        self._wrapper = DwaqVecEnvWrapper(self._env)
        self._runner = OnPolicyRunner(self._wrapper, train_cfg, log_dir, device=device)
        log.info("phase=%d num_envs=%d obs=%d critic=%d actor_history=%d estimator_history=%d",
                 phase, num_envs, self._wrapper.num_obs, self._wrapper.num_privileged_obs,
                 self._wrapper.actor_history_length, self._wrapper.estimator_history_length)

    @property
    def runner(self):
        """체크포인트 적재와 학습을 수행하는 runner를 제공한다."""
        return self._runner

    @property
    def wrapper(self):
        """체크포인트 계약 기록에 사용할 환경 wrapper를 제공한다."""
        return self._wrapper

    def iterations(self, mode, override=None):
        """단계 설정에서 학습 반복 수를 읽고 인자 지정이 있으면 그 값을 쓴다."""
        if override is not None:
            return int(override)
        phase_cfg = self._train_cfg["phases"][f"phase{self._phase}"]
        return int(phase_cfg["pilot_iterations"] if mode == "pilot" else phase_cfg["iterations"])

    def learn(self, iterations):
        """현재 단계의 학습을 실행한다."""
        log.info("phase=%d iterations=%d 학습 시작", self._phase, iterations)
        self._runner.learn(iterations, init_at_random_ep_len=False)

    def close(self):
        """시뮬레이션 자원을 정리한다."""
        self._env.close()


def resolve_num_envs(mode, override=None):
    """실행 모드에 맞는 병렬 환경 수를 정한다."""
    if override is not None:
        return int(override)
    return PILOT_NUM_ENVS if mode == "pilot" else FULL_NUM_ENVS
