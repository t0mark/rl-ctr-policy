"""Two-Phase 논문 Table 4와 G1 보행 설정의 보상 항목.

위상에 의존하는 항목은 모두 env.gait 하나에서 지지·스윙 구간과 기준 관절각을 받아
관측과 같은 시점의 위상을 사용한다.
"""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

# 접촉 기반 보상이 공유하는 접촉력 임계값(N).
CONTACT_FORCE_THRESHOLD = 1.0

# 간격 보상의 허용 범위 밖에서 보상이 감소하는 급격함.
LATERAL_DISTANCE_SHARPNESS = 100.0


def feet_contact(env, sensor_cfg):
    """접촉력 이력의 최댓값으로 발의 접지 여부를 판정한다."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return forces.norm(dim=-1).max(dim=1)[0] > CONTACT_FORCE_THRESHOLD


def joint_power(env, asset_cfg):
    """DreamWaQ Table I의 관절 파워인 토크와 관절 속도의 크기 곱을 합산한다."""
    robot = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids].abs()
    velocity = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
    return (torque * velocity).sum(-1)


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


def feet_slip(env, sensor_cfg, asset_cfg):
    """접지 중인 발의 수평 속도를 누적한다."""
    robot = env.scene[asset_cfg.name]
    velocity = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (velocity * feet_contact(env, sensor_cfg)).sum(-1)


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


def foot_sole_positions(robot, body_ids, sole_offset):
    """발 링크 원점에서 밑창 중심까지 내려간 지점의 월드 좌표를 구한다."""
    position = robot.data.body_pos_w[:, body_ids]
    quaternion = robot.data.body_quat_w[:, body_ids]
    offset = torch.zeros_like(position)
    offset[..., 2] = -sole_offset
    rotated = quat_apply(quaternion.reshape(-1, 4), offset.reshape(-1, 3))
    return position + rotated.reshape(position.shape)


def feet_clearance(env, target_height, sole_offset, left_sensor_cfg, right_sensor_cfg, asset_cfg):
    """스윙 구간에 있는 발만 밑창의 지면 대비 높이 오차를 누적한다.

    발 아래 지면 높이는 각 발에 부착된 스캐너에서 읽고,
    asset_cfg의 발 링크 순서는 [왼발, 오른발]이어야 위상 mask와 대응된다.
    """
    robot = env.scene[asset_cfg.name]
    swing = env.gait.swing_mask
    sole = foot_sole_positions(robot, asset_cfg.body_ids, sole_offset)
    if sole.shape[1] != swing.shape[1]:
        raise ValueError("clearance 보상의 발 링크 수가 보행 위상의 다리 수와 다릅니다.")
    ground = torch.cat([terrain_height(env, sole[:, index:index + 1], sensor_cfg)
                        for index, sensor_cfg in enumerate((left_sensor_cfg, right_sensor_cfg))], dim=1)
    return ((sole[:, :, 2] - ground - target_height).square() * swing).sum(-1)


def joint_position_tracking(env, coefficient, asset_cfg):
    """사인파 기준 관절각과 실제 관절각의 오차를 지수 보상으로 변환한다.

    학습 1단계에서만 사용하며, 대상 관절은 env.gait가 관리하는 기준 동작 관절이다.
    """
    robot = env.scene[asset_cfg.name]
    measured = robot.data.joint_pos[:, env.gait.tracked_joint_ids]
    error = (measured - env.gait.reference_joint_pos).square().sum(-1)
    return torch.exp(-coefficient * error)


def gait_phase_contact(env, sensor_cfg):
    """스윙 위상에서 접지한 발 개수를 세어 음의 가중치로 벌한다.

    sensor_cfg의 발 링크 순서는 [왼발, 오른발]이어야 한다.
    """
    return (feet_contact(env, sensor_cfg) * env.gait.swing_mask).sum(-1)


def lateral_distance(env, asset_cfg, minimum_distance, maximum_distance):
    """측방 간격이 허용 범위를 벗어난 양에 대한 지수 보상을 평균한다.

    범위 안에서는 두 항 모두 최대값이 되어 보상이 1이고,
    범위를 벗어나면 벗어난 쪽의 항만 지수적으로 줄어든다.
    asset_cfg의 링크 순서는 [왼쪽, 오른쪽]이어야 한다.
    """
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("간격 보상은 좌우 링크 두 개를 대상으로 합니다.")
    robot = env.scene[asset_cfg.name]
    position = robot.data.body_pos_w[:, asset_cfg.body_ids]
    offset = quat_apply_inverse(robot.data.root_quat_w, position[:, 0] - position[:, 1])
    distance = offset[:, 1].abs()
    below = (distance - minimum_distance).clamp(max=0.0).abs()
    above = (distance - maximum_distance).clamp(min=0.0)
    return 0.5 * (torch.exp(-LATERAL_DISTANCE_SHARPNESS * below)
                  + torch.exp(-LATERAL_DISTANCE_SHARPNESS * above))


def feet_orientation(env, asset_cfg):
    """발 링크 좌표계에서 본 중력의 수평 성분을 누적한다."""
    robot = env.scene[asset_cfg.name]
    quaternion = robot.data.body_quat_w[:, asset_cfg.body_ids]
    gravity = robot.data.GRAVITY_VEC_W.unsqueeze(1).expand_as(quaternion[..., :3])
    projected = quat_apply_inverse(quaternion.reshape(-1, 4), gravity.reshape(-1, 3))
    return projected.reshape(quaternion.shape[0], -1, 3)[..., :2].square().sum((-1, -2))


class RootAcceleration(ManagerTermBase):
    """제어 스텝 사이 root 속도 변화량을 지수 보상으로 변환한다.

    선속도와 각속도를 이어 붙인 6차원 속도의 변화 크기를 사용한다.
    """

    def __init__(self, cfg, env):
        """직전 제어 스텝의 root 속도를 보관할 버퍼를 만든다."""
        super().__init__(cfg, env)
        self._asset_name = cfg.params["asset_cfg"].name
        self._previous_velocity = torch.zeros(env.num_envs, 6, device=env.device)

    def reset(self, env_ids=None):
        """reset된 환경은 초기 상태를 기준으로 삼아 순간 이동을 보상에서 제외한다."""
        velocity = self._root_velocity()
        if env_ids is None:
            self._previous_velocity.copy_(velocity)
        else:
            self._previous_velocity[env_ids] = velocity[env_ids]

    def _root_velocity(self):
        """월드 좌표계의 선속도와 각속도를 6차원 벡터로 잇는다."""
        robot = self._env.scene[self._asset_name]
        return torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)

    def __call__(self, env, coefficient, asset_cfg):
        """직전 스텝 대비 속도 변화 크기로 보상을 계산하고 현재 속도를 보관한다."""
        velocity = self._root_velocity()
        change = (velocity - self._previous_velocity).norm(dim=-1)
        self._previous_velocity.copy_(velocity)
        return torch.exp(-coefficient * change)
