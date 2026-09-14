# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""평가 조건별 환경·wrapper·runner 구성을 한곳에서 담당한다.

구성과 실행은 이 모듈이 맡고, 스크립트는 인자 해석·checkpoint 적재·결과 저장만 맡는다.
평가 조건은 env_cfg.configure_evaluation(), 출발 구간을 제외한 진단값 집계는 EvaluationEnv,
평가 루프는 OnPolicyRunner.evaluate()가 담당한다.
"""

import logging

from src.sim.rl.configs import ROBOT_ENV_CFGS
from src.sim.rl.dreamwaq_env import EvaluationEnv
from src.sim.rl.env_cfg import configure_evaluation, configure_training_phase
from src.sim.rl.models.dwaq_wrapper import DwaqVecEnvWrapper
from src.dwaq.runners.on_policy_runner import OnPolicyRunner

log = logging.getLogger(__name__)


class EvaluationSession:
    """평가 조건을 적용한 환경과 runner를 생성하고 평가를 실행한다.

    checkpoint는 runner.load(path, mode="evaluate")로 적재한 뒤 evaluate()를 호출한다.
    """

    def __init__(self, robot_id, phase, train_cfg, device, num_envs, command, seed=None,
                 flat_terrain=False, observation_noise=False):
        """단계 지형과 평가 조건을 적용한 환경을 만들고 runner를 연결한다.

        command는 고정할 [vx, vy, yaw rate] 명령이다.
        """
        self._command = [float(value) for value in command]
        cfg = ROBOT_ENV_CFGS[robot_id]()
        cfg.scene.num_envs = num_envs
        cfg.sim.device = device
        cfg.sim.save_logs_to_file = False
        cfg.sim.logging_level = "INFO"
        if seed is not None:
            cfg.seed = seed

        # 단계 지형을 적용한 뒤 고정 명령·평가 지형·관측 노이즈·기본 동역학의 평가 조건을 적용한다.
        configure_training_phase(cfg, phase)
        configure_evaluation(cfg, self._command, flat_terrain, observation_noise)
        self._env = EvaluationEnv(cfg=cfg)
        self._wrapper = DwaqVecEnvWrapper(self._env)
        self._runner = OnPolicyRunner(self._wrapper, train_cfg, device=device)

        # 구성한 평가 조건과 관측 규격을 터미널에 기록한다.
        log.info("phase=%d num_envs=%d obs=%d flat_terrain=%s observation_noise=%s command=%s warmup=%.1f s",
                 phase, num_envs, self._wrapper.num_obs, flat_terrain, observation_noise, self._command,
                 self._env.warmup_steps * self._env.step_dt)

    @property
    def runner(self):
        """checkpoint 적재와 평가를 수행하는 runner를 제공한다."""
        return self._runner

    @property
    def wrapper(self):
        """평가 기록에 사용할 환경 wrapper를 제공한다."""
        return self._wrapper

    def evaluate(self, steps):
        """적재된 정책을 평가하고 runner 지표에 명령·yaw 요약·넘어짐 원인을 더한 보고서를 반환한다."""
        # 평가 시작 시 평가 스텝 수를 터미널에 기록한다.
        log.info("steps=%d 평가 시작", steps)

        report = self._runner.evaluate(steps, self._env.warmup_steps)
        summary = self._env.diagnostics.summarize(report["diagnostics"])
        report["command"] = self._command
        report["yaw"] = {"command_radps": self._command[2],
                         "rate_mean_radps": self._command[2] + summary["yaw"]["rate_bias_radps"],
                         **summary["yaw"]}
        report["falls"]["by_term"] = summary["fall_terms"]
        return report

    def close(self):
        """시뮬레이션 자원을 정리한다."""
        self._env.close()
