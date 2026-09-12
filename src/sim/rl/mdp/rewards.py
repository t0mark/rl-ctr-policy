"""DreamWaQ Table I의 power·높이·clearance·smoothness 보상."""
import torch


def joint_power(env, asset_cfg):
    """선택한 관절의 절대 기계적 power 합을 계산한다."""
    robot = env.scene[asset_cfg.name]
    return (robot.data.applied_torque[:, asset_cfg.joint_ids]
            * robot.data.joint_vel[:, asset_cfg.joint_ids]).abs().sum(-1)


def power_distribution(env, asset_cfg):
    """관절별 부호 있는 power의 분산 제곱을 계산한다."""
    robot = env.scene[asset_cfg.name]
    power = robot.data.applied_torque[:, asset_cfg.joint_ids] * robot.data.joint_vel[:, asset_cfg.joint_ids]
    return power.var(-1, unbiased=False).square()


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
    """지면 대비 발 높이 오차에 발 평면 속력을 곱한다."""
    robot = env.scene[asset_cfg.name]
    position = robot.data.body_pos_w[:, asset_cfg.body_ids]
    speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    height = position[:, :, 2] - terrain_height(env, position, sensor_cfg)
    return ((height - target_height).square() * speed).sum(-1)
