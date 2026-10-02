"""시스템 지연이 있는 관절 위치 action과 출력 배율을 적용하는 PD actuator."""
import math

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DelayedJointPositionAction(JointPositionAction):
    """환경별로 샘플링한 physics tick 수만큼 관절 목표를 늦게 전달한다."""

    def __init__(self, cfg, env):
        """지연된 목표를 꺼낼 원형 큐와 환경별 지연 tick을 만든다."""
        super().__init__(cfg, env)
        max_ticks = math.ceil(cfg.delay_range_s[1] / env.physics_dt)
        self._queue = torch.zeros(max_ticks + 1, *self._processed_actions.shape, device=self.device)
        self._cursor = 0
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        delay = torch.empty(self.num_envs, device=self.device).uniform_(*cfg.delay_range_s)
        self._delay_ticks = (delay / env.physics_dt).round().long()
        self.reset()

    @property
    def joint_names(self):
        """action 관절 순서를 반환한다."""
        return self._joint_names

    def apply_actions(self):
        """현재 목표를 큐에 넣고 지연된 목표를 physics에 전달한다. 제어하지 않는 관절은 기본 자세를 유지한다."""
        self._queue[self._cursor] = self.processed_actions
        target = self._queue[(self._cursor - self._delay_ticks) % self._queue.shape[0], self._env_indices]
        self._asset.set_joint_position_target(self._asset.data.default_joint_pos)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._cursor = (self._cursor + 1) % self._queue.shape[0]

    def reset(self, env_ids=None):
        """reset된 환경의 목표 큐를 기본 자세로 채운다."""
        super().reset(env_ids)
        if not hasattr(self, "_queue"):
            return
        ids = self._env_indices if env_ids is None else env_ids
        default = self._asset.data.default_joint_pos[ids][:, self._joint_ids]
        self._queue[:, ids] = default.unsqueeze(0)
        self._processed_actions[ids] = default


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """초 단위 시스템 지연 범위를 갖는 관절 위치 action 설정."""

    class_type: type = DelayedJointPositionAction
    delay_range_s: tuple[float, float] = (0.0, 0.0)


class StrengthPDActuator(IdealPDActuator):
    """PD 토크에 환경·관절별 출력 배율을 곱한 뒤 토크 한계로 제한한다."""

    def __init__(self, *args, **kwargs):
        """출력 배율을 1로 초기화한다."""
        super().__init__(*args, **kwargs)
        self._strength = torch.ones_like(self.stiffness)

    def randomize_strength(self, env_ids, factor_range):
        """선택한 환경의 출력 배율을 샘플링한다."""
        ids = slice(None) if env_ids is None else env_ids
        self._strength[ids] = torch.empty_like(self._strength[ids]).uniform_(*factor_range)

    def _clip_effort(self, effort):
        """출력 배율을 적용한 토크를 토크 한계로 제한한다."""
        return super()._clip_effort(effort * self._strength)


@configclass
class StrengthPDActuatorCfg(IdealPDActuatorCfg):
    """출력 배율을 적용하는 explicit PD actuator 설정."""

    class_type: type = StrengthPDActuator
