"""PD 토크에 출력 배율과 실행 지연을 함께 적용한다."""
import math

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass


class StrengthPDActuator(IdealPDActuator):
    """배율을 적용한 PD 토크를 고정된 effort 한계로 제한하고 지연시켜 전달한다."""

    def __init__(self, *args, **kwargs):
        """환경·관절별 출력 배율과 토크 지연 상태를 생성한다."""
        super().__init__(*args, **kwargs)
        self._strength = torch.ones_like(self.stiffness)
        self._delay_ticks = torch.zeros(self.stiffness.shape[0], dtype=torch.long,
                                        device=self.stiffness.device)
        self._env_indices = torch.arange(self.stiffness.shape[0], device=self.stiffness.device)
        self._torque_queue = None
        self._cursor = 0

    def randomize_strength(self, env_ids, factor_range):
        """선택한 환경의 배율을 독립적으로 샘플링한다."""
        ids = slice(None) if env_ids is None else env_ids
        self._strength[ids] = torch.empty_like(self._strength[ids]).uniform_(*factor_range)

    def randomize_torque_delay(self, env_ids, delay_range_s, physics_dt):
        """토크 지연을 physics tick으로 양자화해 환경별로 샘플링한다."""
        if delay_range_s[0] < 0.0 or delay_range_s[1] < delay_range_s[0]:
            raise ValueError("토크 지연 범위가 잘못되었습니다.")
        max_ticks = math.ceil(delay_range_s[1] / physics_dt)

        # 지연 큐는 최대 지연 tick 수가 정해지는 첫 호출에서 만든다.
        if self._torque_queue is None or self._torque_queue.shape[0] != max_ticks + 1:
            self._torque_queue = torch.zeros(max_ticks + 1, *self.stiffness.shape,
                                             device=self.stiffness.device)
            self._cursor = 0
        ids = slice(None) if env_ids is None else env_ids
        delay = torch.empty(self._delay_ticks[ids].shape, device=self.stiffness.device)
        delay.uniform_(*delay_range_s)
        self._delay_ticks[ids] = (delay / physics_dt).round().long().clamp(0, max_ticks)

        # reset된 환경은 이전 episode의 토크를 이어받지 않는다.
        self._torque_queue[:, ids] = 0.0

    def compute(self, control_action, joint_pos, joint_vel):
        """계산된 토크를 큐에 넣고 지연된 토크를 실제 출력으로 전달한다."""
        control_action = super().compute(control_action, joint_pos, joint_vel)
        if self._torque_queue is None:
            return control_action
        self._torque_queue[self._cursor] = self.applied_effort
        indices = (self._cursor - self._delay_ticks) % self._torque_queue.shape[0]
        delayed = self._torque_queue[indices, self._env_indices]
        self._cursor = (self._cursor + 1) % self._torque_queue.shape[0]
        self.applied_effort = delayed
        control_action.joint_efforts = delayed
        return control_action

    def _clip_effort(self, effort):
        """PD gain을 보존하고 출력 토크에 배율과 포화를 적용한다."""
        return super()._clip_effort(effort * self._strength)


@configclass
class StrengthPDActuatorCfg(IdealPDActuatorCfg):
    """physics tick에서 직접 토크를 계산하는 actuator 설정."""

    class_type: type = StrengthPDActuator
