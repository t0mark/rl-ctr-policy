"""선속도 추적 점수에 따라 선속도 명령 범위를 단계적으로 확대한다.

yaw 명령 범위는 확대 대상이 아니며 설정값을 처음부터 그대로 사용한다.
"""
import logging
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

log = logging.getLogger(__name__)


class CurriculumVelocityCommand(UniformVelocityCommand):
    """축별 추적 점수가 임계값을 넘으면 그 축의 명령 범위만 일정량 확대한다.

    점수는 제어 스텝마다 계산한 추적 보상식의 평균이다.
    yaw 범위는 설정 한계에 고정하며 각속도 점수는 기록용으로만 남긴다.
    """

    def __init__(self, cfg, env):
        """최대 범위와 초기 저속 범위, 전체 환경의 점수 누적기를 만든다."""
        super().__init__(cfg, env)

        # curriculum 설정값의 유효 범위를 확인한다.
        if cfg.initial_limit < 0 or cfg.expansion_step <= 0 or cfg.minimum_duration_s <= 0:
            raise ValueError("명령 curriculum 범위·증분·평가 기간이 잘못되었습니다.")
        if not 0 <= cfg.success_threshold <= 1:
            raise ValueError("성공 임계값은 0~1 사이여야 합니다.")
        if cfg.curriculum_enabled and cfg.heading_command:
            raise ValueError("명령 curriculum은 yaw 속도를 직접 지정해야 합니다.")

        # [vx, vy, yaw] 순서로 설정의 최대 범위를 텐서로 보관한다.
        bounds = [cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z]
        self._limit_low = torch.tensor([value[0] for value in bounds], device=self.device)
        self._limit_high = torch.tensor([value[1] for value in bounds], device=self.device)
        if (self._limit_low > self._limit_high).any():
            raise ValueError("명령 범위의 하한이 상한보다 큽니다.")

        # 선속도 초기 범위만 ±initial_limit으로 좁히고 yaw는 설정 한계에서 시작한다.
        self._low = self._limit_low.clone()
        self._high = self._limit_high.clone()
        self._low[:2] = torch.minimum(self._limit_high[:2],
                                      self._limit_low[:2].clamp_min(-cfg.initial_limit))
        self._high[:2] = torch.maximum(self._limit_low[:2],
                                       self._limit_high[:2].clamp_max(cfg.initial_limit))

        # 판정 구간의 [선속도, 각속도] 점수 합과 스텝 수, 마지막 누적 스텝 번호를 보관한다.
        self._score = torch.zeros(2, device=self.device)
        self._steps = 0
        self._last_metrics_step = None

        # 기록용으로 마지막 판정 평균 점수를 축별로 보관한다.
        self._last_score = torch.zeros(2, device=self.device)

    def _update_metrics(self):
        """명령 갱신 전에 이번 스텝의 추적 점수를 누적한다."""
        self.capture_step_metrics()

    def capture_step_metrics(self):
        """전체 환경의 추적 점수를 제어 스텝당 한 번만 누적한다."""
        # 같은 스텝에서 reset 경로와 명령 갱신 경로가 모두 호출해도 한 번만 누적한다.
        step = self._env.common_step_counter
        if self._last_metrics_step == step:
            return
        self._last_metrics_step = step

        # 선속도·yaw 각속도 추적 보상식을 정지·실패 환경까지 포함한 전체 환경 평균으로 누적한다.
        if self.cfg.curriculum_enabled:
            velocity = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w),
                                          self.robot.data.root_lin_vel_w)
            linear = (self.command[:, :2] - velocity[:, :2]).square().sum(-1)
            angular = (self.command[:, 2] - self.robot.data.root_ang_vel_w[:, 2]).square()
            self._score[0] += torch.exp(-4 * linear).mean()
            self._score[1] += torch.exp(-4 * angular).mean()
            self._steps += 1

        # Isaac Lab 기본 명령 추적 오차 기록을 누적한다.
        super()._update_metrics()

    def _resample_command(self, env_ids):
        """명령 재샘플링 시점에 범위 확대를 판정하고 현재 범위에서 명령을 균등 샘플링한다."""
        if not self.cfg.curriculum_enabled:
            super()._resample_command(env_ids)
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        # 최소 평가 구간이 모이면 선속도 평균 점수를 임계값과 비교해 x·y 범위만 넓힌다.
        judged = self._steps and self._steps * self._env.step_dt >= self.cfg.minimum_duration_s
        if judged:
            average = self._score / self._steps
            expansion = torch.zeros(3, device=self.device)
            expansion[:2] = self.cfg.expansion_step * (average[0] > self.cfg.success_threshold)
            self._low = torch.maximum(self._limit_low, self._low - expansion)
            self._high = torch.minimum(self._limit_high, self._high + expansion)
            self._score.zero_()
            self._steps = 0

        # 현재 범위에서 명령을 균등 샘플링하고 정지 명령 환경을 다시 고른다.
        self.vel_command_b[ids] = self._low + torch.rand(len(ids), 3, device=self.device) * (self._high - self._low)
        self.is_standing_env[ids] = torch.rand(len(ids), device=self.device) < self.cfg.rel_standing_envs

        # 판정 점수를 기록용으로 보관하고 범위를 넓혔으면 축별 점수와 새 범위를 터미널에 기록한다.
        if judged:
            self._last_score.copy_(average)
            if expansion.any():
                log.info("명령 curriculum: 선속도=%.4f 각속도=%.4f 범위=%s~%s",
                         average[0].item(), average[1].item(),
                         self._low.tolist(), self._high.tolist())

    def curriculum_state(self):
        """체크포인트에 저장할 현재 명령 범위와 최대 범위를 복사한다."""
        return {"low": self._low.clone(), "high": self._high.clone(),
                "limit_low": self._limit_low.clone(), "limit_high": self._limit_high.clone()}

    def curriculum_metrics(self):
        """마지막 축별 추적 점수와 축별 현재 명령 범위를 GPU 텐서로 반환한다."""
        metrics = {"Curriculum/linear_score": self._last_score[0],
                   "Curriculum/angular_score": self._last_score[1]}
        for index, axis in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            metrics[f"Curriculum/{axis}_min"] = self._low[index]
            metrics[f"Curriculum/{axis}_max"] = self._high[index]
        return metrics

    def load_curriculum_state(self, state):
        """최대 범위가 같은 curriculum 범위를 복원하고 점수 누적기를 초기화한다."""
        # 저장된 범위의 형태와 유한성을 확인한다.
        for name in ("low", "high", "limit_low", "limit_high"):
            if name not in state or state[name].shape != self._low.shape:
                raise ValueError("명령 curriculum checkpoint 규격이 다릅니다.")
            if not torch.isfinite(state[name]).all():
                raise ValueError("명령 curriculum 범위가 유한하지 않습니다.")

        # 최대 범위가 현재 설정과 같고 저장된 범위가 그 안에 있는지 확인한다.
        if not torch.equal(state["limit_low"], self._limit_low) or not torch.equal(state["limit_high"], self._limit_high):
            raise ValueError("명령 curriculum 최대 범위가 다릅니다.")
        if ((state["low"] < self._limit_low) | (state["high"] > self._limit_high)
                | (state["low"] > state["high"])).any():
            raise ValueError("저장된 명령 범위가 최대 범위를 벗어납니다.")

        # 현재 범위를 복원하고 판정 구간을 새로 시작한다.
        self._low.copy_(state["low"])
        self._high.copy_(state["high"])

        # yaw 범위는 커리큘럼 대상이 아니므로 설정 한계로 되돌린다.
        self._low[2] = self._limit_low[2]
        self._high[2] = self._limit_high[2]
        self._score.zero_()
        self._steps = 0

    def reset(self, env_ids=None):
        """전체 환경 reset도 명시적인 환경 인덱스로 변환해 명령을 재샘플링한다."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return super().reset(env_ids)


@configclass
class CurriculumVelocityCommandCfg(UniformVelocityCommandCfg):
    """선속도 추적 점수가 0.75를 넘을 때 선속도 명령 범위를 0.05씩 확대하는 명령 설정."""

    class_type: type = CurriculumVelocityCommand
    # curriculum 사용 여부
    curriculum_enabled: bool = True
    # 초기 명령 범위의 절댓값 상한
    initial_limit: float = 0.25
    # 범위를 확대하는 평균 추적 점수 임계값
    success_threshold: float = 0.75
    # 한 번에 넓히는 범위 증분
    expansion_step: float = 0.05
    # 평균 점수를 판정하기 위한 최소 누적 시간(초)
    minimum_duration_s: float = 2.0
