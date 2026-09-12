"""평균 속도 추적 보상에 따라 명령 범위를 연속적으로 확대한다."""
import logging
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

log = logging.getLogger(__name__)


class CurriculumVelocityCommand(UniformVelocityCommand):
    """평균 추적 점수가 임계값을 넘으면 각 축 범위를 일정량 확대한다."""

    def __init__(self, cfg, env):
        """최대 범위와 초기 저속 범위, 전체 환경의 점수 누적기를 만든다."""
        super().__init__(cfg, env)
        if cfg.initial_limit < 0 or cfg.expansion_step <= 0 or cfg.minimum_duration_s <= 0:
            raise ValueError("명령 curriculum 범위·증분·평가 기간이 잘못되었습니다.")
        if not 0 <= cfg.success_threshold <= 1:
            raise ValueError("성공 임계값은 0~1 사이여야 합니다.")
        if cfg.curriculum_enabled and cfg.heading_command:
            raise ValueError("명령 curriculum은 yaw 속도를 직접 지정해야 합니다.")
        bounds = [cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z]
        self._limit_low = torch.tensor([value[0] for value in bounds], device=self.device)
        self._limit_high = torch.tensor([value[1] for value in bounds], device=self.device)
        if (self._limit_low > self._limit_high).any():
            raise ValueError("명령 범위의 하한이 상한보다 큽니다.")
        self._low = torch.minimum(self._limit_high, self._limit_low.clamp_min(-cfg.initial_limit))
        self._high = torch.maximum(self._limit_low, self._limit_high.clamp_max(cfg.initial_limit))
        self._score = torch.zeros((), device=self.device)
        self._steps = 0

    def _update_metrics(self):
        """선속도·yaw의 무가중 추적 점수를 환경과 시간에 걸쳐 평균한다."""
        super()._update_metrics()
        if not self.cfg.curriculum_enabled:
            return
        linear = (self.command[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).square().sum(-1)
        angular = (self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]).square()
        # 논문의 평균 구간 세부사항은 명시적 선택으로 두며 정지·실패 표본도 포함한다.
        self._score += (0.5 * (torch.exp(-4 * linear) + torch.exp(-4 * angular))).mean()
        self._steps += 1

    def _resample_command(self, env_ids):
        """명령 재샘플링 경계에서 범위를 한 번 확대하고 균등 샘플링한다."""
        if not self.cfg.curriculum_enabled:
            super()._resample_command(env_ids)
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        if self._steps and self._steps * self._env.step_dt >= self.cfg.minimum_duration_s:
            average = self._score / self._steps
            if average > self.cfg.success_threshold:
                self._low = torch.maximum(self._limit_low, self._low - self.cfg.expansion_step)
                self._high = torch.minimum(self._limit_high, self._high + self.cfg.expansion_step)
                log.info("명령 curriculum: 평균=%.4f 범위=%s~%s",
                         average.item(), self._low.tolist(), self._high.tolist())
            # 환경별 reset이 같은 시점에 여러 번 호출되어도 중복 확대하지 않는다.
            self._score.zero_()
            self._steps = 0
        self.vel_command_b[ids] = self._low + torch.rand(len(ids), 3, device=self.device) * (self._high - self._low)
        self.is_standing_env[ids] = torch.rand(len(ids), device=self.device) < self.cfg.rel_standing_envs

    def curriculum_state(self):
        """새 episode에서도 유지할 명령 범위와 상한을 복사한다."""
        return {"low": self._low.clone(), "high": self._high.clone(),
                "limit_low": self._limit_low.clone(), "limit_high": self._limit_high.clone()}

    def load_curriculum_state(self, state):
        """동일 최대 범위의 curriculum을 복원하고 평가 누적기는 초기화한다."""
        for name in ("low", "high", "limit_low", "limit_high"):
            if name not in state or state[name].shape != self._low.shape:
                raise ValueError("명령 curriculum checkpoint 규격이 다릅니다.")
            if not torch.isfinite(state[name]).all():
                raise ValueError("명령 curriculum 범위가 유한하지 않습니다.")
        if not torch.equal(state["limit_low"], self._limit_low) or not torch.equal(state["limit_high"], self._limit_high):
            raise ValueError("명령 curriculum 최대 범위가 다릅니다.")
        if ((state["low"] < self._limit_low) | (state["high"] > self._limit_high)
                | (state["low"] > state["high"])).any():
            raise ValueError("저장된 명령 범위가 최대 범위를 벗어납니다.")
        self._low.copy_(state["low"])
        self._high.copy_(state["high"])
        self._score.zero_()
        self._steps = 0

    def reset(self, env_ids=None):
        """전체 환경 reset도 명시적인 인덱스로 전달한다."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return super().reset(env_ids)


@configclass
class CurriculumVelocityCommandCfg(UniformVelocityCommandCfg):
    """0.75 초과 시 선속도 m/s와 각속도 rad/s 범위를 각각 0.05 확대한다."""

    class_type: type = CurriculumVelocityCommand
    curriculum_enabled: bool = True
    initial_limit: float = 0.25
    success_threshold: float = 0.75
    expansion_step: float = 0.05
    # 평균을 평가할 최소 구간은 논문에서 확정되지 않은 구현 설정이다.
    minimum_duration_s: float = 2.0
