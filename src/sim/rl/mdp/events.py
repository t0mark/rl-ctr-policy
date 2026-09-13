"""링크별 world-frame 외란의 샘플링·물리 적용과, 매 physics tick 그 외란을 함께 반영해야 하는 지연 관절 action을 연결한다."""
import logging
import math

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

log = logging.getLogger(__name__)


def _resolve_body_ids(asset, asset_cfg):
    """전체 링크를 가리키는 slice 표현까지 포함해 링크 인덱스를 확정한다."""
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        return list(range(asset.num_bodies))[body_ids]
    return list(body_ids)


def randomize_disturbance(env, env_ids, asset_cfg, force_range, probability=1.0):
    """선택한 링크들에 적용할 world-frame 힘을 저장한다."""
    robot = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(robot, asset_cfg)
    target = (asset_cfg.name, tuple(body_ids))
    if not hasattr(env, "_dwaq_force_world"):
        env._dwaq_force_world = torch.zeros(env.num_envs, len(body_ids), 3, device=env.device)
        env._dwaq_disturbance_target = target
        log.info("외란 적용 대상: asset=%s links=%d", asset_cfg.name, len(body_ids))
    elif env._dwaq_disturbance_target != target:
        raise ValueError("외란 reset과 interval 이벤트는 같은 자산·링크를 사용해야 합니다.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # reset 이벤트의 영외력도 선택한 환경의 저장값에 반영한다.
    shape = (len(env_ids), len(body_ids), 3)
    force = torch.empty(shape, device=env.device).uniform_(*force_range)
    force *= torch.rand((shape[0], shape[1], 1), device=env.device) < probability
    env._dwaq_force_world[env_ids] = force


def apply_disturbance(env):
    """매 physics tick에 각 링크의 현재 자세로 변환한 외력을 설정한다."""
    force = getattr(env, "_dwaq_force_world", None)
    if force is None:
        return
    asset_name, body_ids = env._dwaq_disturbance_target
    robot = env.scene[asset_name]

    # world 방향을 유지하도록 적용 링크들의 최신 자세를 사용한다.
    quaternion = robot.data.body_quat_w[:, list(body_ids)]
    force_link = quat_apply_inverse(quaternion.reshape(-1, 4), force.reshape(-1, 3)).reshape_as(force)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=force_link, torques=torch.zeros_like(force_link),
        body_ids=list(body_ids), is_global=False,
    )


class DelayedJointPositionAction(JointPositionAction):
    """환경마다 뽑은 지연을 physics tick으로 양자화하고, 매 tick 외란도 함께 적용한다."""

    def __init__(self, cfg, env):
        """최근 목표와 지연 tick을 저장할 GPU 버퍼를 생성한다."""
        super().__init__(cfg, env)
        if cfg.delay_range_s[0] < 0 or cfg.delay_range_s[1] < cfg.delay_range_s[0]:
            raise ValueError("action 지연 범위가 잘못되었습니다.")
        self._target_clipped = torch.zeros_like(self._processed_actions, dtype=torch.bool)
        self._max_ticks = math.ceil(cfg.delay_range_s[1] / env.physics_dt)
        self._physics_dt = env.physics_dt
        self._queue = torch.as_tensor(self._offset, device=self.device).expand_as(
            self._processed_actions).unsqueeze(0).repeat(
            self._max_ticks + 1, 1, 1).clone()
        self._cursor = 0
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        # action 지연은 환경마다 한 번만 뽑아 학습 내내 유지한다.
        delay = torch.empty(self.num_envs, device=self.device).uniform_(*cfg.delay_range_s)
        self._delay_ticks = (delay / self._physics_dt).round().long().clamp(0, self._max_ticks)
        self.reset()

    @property
    def joint_names(self):
        """checkpoint에 기록할 실제 action 관절 순서를 반환한다."""
        return self._joint_names

    @property
    def target_clipped(self):
        """요청한 목표가 관절의 하드 제한을 넘었는지 반환한다."""
        return self._target_clipped

    def process_actions(self, actions):
        """정책의 원출력은 보존하고 물리 목표만 관절 제한 안으로 제한한다."""
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        bounded = self._processed_actions.clamp(min=limits[..., 0], max=limits[..., 1])
        self._target_clipped.copy_(bounded != self._processed_actions)
        if self.cfg.enforce_joint_limits:
            self._processed_actions.copy_(bounded)

    def apply_actions(self):
        """현재 목표를 큐에 넣고 지연된 목표를 physics에 전달한다."""
        # 물리 진행 전에 현재 링크 자세에 맞춰 외력을 갱신한다.
        apply_disturbance(self._env)
        self._queue[self._cursor] = self.processed_actions
        indices = (self._cursor - self._delay_ticks) % self._queue.shape[0]
        target = self._queue[indices, self._env_indices]
        # RL 비제어 관절도 기본 자세 목표를 매 physics tick 유지한다.
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
        self._target_clipped[ids] = False
        self._queue[:, ids] = default.unsqueeze(0)
        self._processed_actions[ids] = default


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """지연 범위는 초 단위이며 physics tick으로 반올림한다."""

    class_type: type = DelayedJointPositionAction
    delay_range_s: tuple[float, float] = (0.0, 0.015)
    enforce_joint_limits: bool = True


def disturbance_body(env):
    """저장된 링크별 world 외력의 합력을 현재 root 좌표계로 변환한다."""
    force = getattr(env, "_dwaq_force_world", None)
    if force is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    asset_name, _ = env._dwaq_disturbance_target
    return quat_apply_inverse(env.scene[asset_name].data.root_quat_w, force.sum(1))


def randomize_motor_strength(env, env_ids, factor_range=(0.9, 1.1), actuator_names=None):
    """gain·토크 한계를 변경하지 않고 actuator의 출력 배율만 샘플링한다."""
    from .actuators import StrengthPDActuator

    robot = env.scene["robot"]
    for name in (robot.actuators if actuator_names is None else actuator_names):
        actuator = robot.actuators[name]
        if not isinstance(actuator, StrengthPDActuator):
            raise TypeError("motor strength에는 StrengthPDActuator가 필요합니다.")
        actuator.randomize_strength(env_ids, factor_range)


def randomize_torque_delay(env, env_ids, delay_range_s=(0.0, 0.010), actuator_names=None):
    """actuator가 출력하는 토크의 실행 지연을 환경별로 샘플링한다."""
    from .actuators import StrengthPDActuator

    robot = env.scene["robot"]
    for name in (robot.actuators if actuator_names is None else actuator_names):
        actuator = robot.actuators[name]
        if not isinstance(actuator, StrengthPDActuator):
            raise TypeError("토크 지연에는 StrengthPDActuator가 필요합니다.")
        actuator.randomize_torque_delay(env_ids, delay_range_s, env.physics_dt)
