# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Two-Phase 논문의 위상 신호와 발 주변 국소 지형 관측을 제공한다.
"""

import torch
from isaaclab.utils.math import euler_xyz_from_quat


def gait_phase(env):
    """현재 보행 위상을 [sin(2*pi*phase), cos(2*pi*phase)]로 반환한다."""
    return env.gait.phase_signal


def foot_height_scan(env, sensor_cfg):
    """발 링크 기준의 주변 지면 높이차를 반환한다.

    발이 지면보다 높으면 양수이며, ray가 지면을 맞히지 못한 격자점은
    지면이 관측 범위보다 아래에 있다는 뜻이므로 양의 경계값으로 대체한다.
    """
    sensor = env.scene[sensor_cfg.name]
    rays = sensor.data.ray_hits_w[..., 2]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - rays
    return torch.where(torch.isfinite(rays), height, torch.ones_like(height))


def base_roll_pitch(env):
    """root quaternion의 roll·pitch를 [-pi, pi) 라디안으로 반환한다."""
    roll, pitch, _ = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    angles = torch.stack((roll, pitch), dim=-1)
    return torch.remainder(angles + torch.pi, 2.0 * torch.pi) - torch.pi
