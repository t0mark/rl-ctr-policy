"""Two-Phase 논문 Table 4와 G1 보행 설정의 보상 항목.

위상에 의존하는 항목은 모두 env.gait 하나에서 지지·스윙 구간과 기준 관절각을 받아
관측과 같은 시점의 위상을 사용한다.
"""
import torch
from isaaclab.utils.math import quat_apply_inverse


def joint_power(env, asset_cfg):
    """Table 4의 Joint power 항목에 명시된 토크 제곱합을 계산한다."""
    return env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids].square().sum(-1)


def velocity_mismatch(env, asset_cfg):
    """수직 속도와 roll·pitch 각속도 오차의 지수 보상을 평균한다."""
    robot = env.scene[asset_cfg.name]
    vertical = torch.exp(-10.0 * robot.data.root_lin_vel_b[:, 2].square())
    angular = torch.exp(-5.0 * robot.data.root_ang_vel_b[:, :2].square().sum(-1))
    return 0.5 * (vertical + angular)


def default_joint_tracking(env, asset_cfg):
    """기본 관절 자세의 제곱 오차를 지수 보상으로 변환한다."""
    robot = env.scene[asset_cfg.name]
    error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-2.0 * error.square().sum(-1))


def feet_air_time(env, sensor_cfg):
    """착지 순간 완료된 공중 시간을 합산한다. 접촉은 첫 착지 펄스로 해석한다."""
    sensor = env.scene[sensor_cfg.name]
    contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    return (sensor.data.last_air_time[:, sensor_cfg.body_ids] * contact).sum(-1)


def action_smoothness(env):
    """현재와 직전 두 action의 2차 차분 합을 계산한다."""
    difference = env.action_manager.action - 2 * env.action_manager.prev_action + env.previous_previous_action
    return difference.square().sum(-1) * env.action_history_valid


def terrain_height(env, positions, sensor_cfg):
    """대상 XY에 가장 가까운 유효 ray의 지면 높이를 구한다."""
    rays = env.scene[sensor_cfg.name].data.ray_hits_w
    finite = torch.isfinite(rays).all(-1)
    if not finite.any(-1).all():
        raise RuntimeError("높이 보상에 필요한 유효 지면 ray가 없습니다.")
    distance = (rays[:, None, :, :2] - positions[:, :, None, :2]).square().sum(-1)
    nearest = distance.masked_fill(~finite[:, None, :], float("inf")).argmin(-1)
    return rays[:, :, 2].gather(1, nearest)


def body_height(env, target_height, sensor_cfg, asset_cfg):
    """근접 지면 대비 root 높이의 제곱 오차를 계산한다."""
    position = env.scene[asset_cfg.name].data.root_pos_w
    ground = terrain_height(env, position.unsqueeze(1), sensor_cfg).squeeze(1)
    return (position[:, 2] - ground - target_height).square()


def feet_clearance(env, target_height, sensor_cfg, asset_cfg):
    """스윙 구간에 있는 발만 지면 대비 높이 오차를 누적한다.

    asset_cfg의 발 링크 순서는 [왼발, 오른발]이어야 위상 mask와 대응된다.
    """
    robot = env.scene[asset_cfg.name]
    position = robot.data.body_pos_w[:, asset_cfg.body_ids]
    swing = env.gait.swing_mask
    if position.shape[1] != swing.shape[1]:
        raise ValueError("clearance 보상의 발 링크 수가 보행 위상의 다리 수와 다릅니다.")
    height = position[:, :, 2] - terrain_height(env, position, sensor_cfg)
    return ((height - target_height).square() * swing).sum(-1)


def joint_position_tracking(env, coefficient, asset_cfg):
    """사인파 기준 관절각과 실제 관절각의 오차를 지수 보상으로 변환한다.

    학습 1단계에서만 사용하며, 대상 관절은 env.gait가 관리하는 기준 동작 관절이다.
    """
    robot = env.scene[asset_cfg.name]
    measured = robot.data.joint_pos[:, env.gait.tracked_joint_ids]
    error = (measured - env.gait.reference_joint_pos).square().sum(-1)
    return torch.exp(-coefficient * error)


def gait_phase_contact(env, sensor_cfg, threshold=1.0):
    """스윙 위상에서 접촉한 발 개수를 세어 음의 가중치로 벌한다.

    sensor_cfg의 발 링크 순서는 [왼발, 오른발]이어야 한다.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact = forces.norm(dim=-1) > threshold
    return (contact * env.gait.swing_mask).sum(-1)


def lateral_distance(env, asset_cfg, minimum_distance, maximum_distance):
    """측방 간격과 두 목표 거리의 절대 오차에 대한 지수 보상을 평균한다.

    minimum_distance와 maximum_distance는 구간 경계가 아닌 두 목표값이다.
    asset_cfg의 발 링크 순서는 [왼발, 오른발]이어야 한다.
    """
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("간격 보상은 좌우 링크 두 개를 대상으로 합니다.")
    robot = env.scene[asset_cfg.name]
    position = robot.data.body_pos_w[:, asset_cfg.body_ids]
    offset = quat_apply_inverse(robot.data.root_quat_w, position[:, 0] - position[:, 1])
    distance = offset[:, 1].abs()
    return 0.5 * (torch.exp(-100.0 * (distance - minimum_distance).abs())
                  + torch.exp(-100.0 * (distance - maximum_distance).abs()))


def feet_orientation(env, asset_cfg):
    """발 링크 좌표계에서 본 중력의 수평 성분을 누적한다."""
    robot = env.scene[asset_cfg.name]
    quaternion = robot.data.body_quat_w[:, asset_cfg.body_ids]
    gravity = robot.data.GRAVITY_VEC_W.unsqueeze(1).expand_as(quaternion[..., :3])
    projected = quat_apply_inverse(quaternion.reshape(-1, 4), gravity.reshape(-1, 3))
    return projected.reshape(quaternion.shape[0], -1, 3)[..., :2].square().sum((-1, -2))


def root_acceleration(env, asset_cfg, coefficient=1.0e-4):
    """루트 가속도 크기의 세제곱에 대한 지수 보상을 계산한다."""
    robot = env.scene[asset_cfg.name]
    magnitude = (robot.data.body_com_lin_acc_w[:, 0].square().sum(-1)
                 + robot.data.body_com_ang_acc_w[:, 0].square().sum(-1)).sqrt()
    return torch.exp(-coefficient * magnitude.pow(3))
