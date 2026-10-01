"""DreamWaQ Table I 보상 중 Isaac Lab 기본 항목에 없는 보상."""
import torch
from isaaclab.managers import ManagerTermBase


def terrain_height(env, positions, sensor_cfg):
    """대상 XY에 가장 가까운 높이 스캐너 ray의 지면 높이를 구한다."""
    rays = env.scene[sensor_cfg.name].data.ray_hits_w
    distance = (rays[:, None, :, :2] - positions[:, :, None, :2]).square().sum(-1)
    nearest = torch.nan_to_num(distance, nan=float("inf"), posinf=float("inf")).argmin(-1)
    return rays[:, :, 2].gather(1, nearest)


def joint_power(env, asset_cfg):
    """관절 토크와 관절 속도 크기의 곱을 합산한다."""
    robot = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return (torque.abs() * velocity.abs()).sum(-1)


def power_distribution(env, asset_cfg):
    """관절별 기계적 파워(토크·관절 속도)의 관절 간 분산을 계산한다."""
    robot = env.scene[asset_cfg.name]
    power = robot.data.applied_torque[:, asset_cfg.joint_ids] * robot.data.joint_vel[:, asset_cfg.joint_ids]
    return power.var(dim=-1, unbiased=False)


def body_height(env, target_height, sensor_cfg, asset_cfg):
    """몸통 아래 지면 대비 root 높이의 제곱 오차를 계산한다."""
    position = env.scene[asset_cfg.name].data.root_pos_w
    ground = terrain_height(env, position.unsqueeze(1), sensor_cfg).squeeze(1)
    return (target_height - (position[:, 2] - ground)).square()


def feet_clearance(env, target_height, sensor_cfg, asset_cfg):
    """발의 지면 대비 높이 오차 제곱에 발 수평 속도를 곱해 합산한다."""
    robot = env.scene[asset_cfg.name]
    position = robot.data.body_pos_w[:, asset_cfg.body_ids]
    ground = terrain_height(env, position, sensor_cfg)
    speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return ((target_height - (position[..., 2] - ground)).square() * speed).sum(-1)


class ActionSmoothness(ManagerTermBase):
    """현재와 직전 두 action의 2차 차분 제곱합을 계산한다."""

    def __init__(self, cfg, env):
        """두 스텝 전 action 버퍼를 만든다."""
        super().__init__(cfg, env)
        self._previous_previous_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None):
        """reset된 환경의 두 스텝 전 action을 0으로 되돌린다."""
        self._previous_previous_action[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env):
        """2차 차분 제곱합을 반환하고 두 스텝 전 action을 갱신한다."""
        action = env.action_manager.action
        previous_action = env.action_manager.prev_action
        smoothness = (action - 2.0 * previous_action + self._previous_previous_action).square().sum(-1)
        self._previous_previous_action.copy_(previous_action)
        return smoothness
