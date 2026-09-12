# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""보행 위상과 사인파 기준 관절 동작을 한곳에서 계산한다.
"""

import math
from dataclasses import MISSING

import torch
from isaaclab.utils import configclass


@configclass
class GaitCfg:
    """보행 주기와 사인파 기준 동작의 대상 관절·진폭을 정의한다."""

    # 위상을 계산할 대상 자산
    asset_name: str = "robot"
    # 한 보행 주기의 길이(초). 좌우 다리가 한 번씩 스윙하는 시간이다.
    cycle_time_s: float = 0.8
    # |sin(2*pi*phase)|가 이 값보다 작으면 양발 지지로 본다.
    double_support_ratio: float = 0.1
    # 왼다리 기준 동작 대상 관절. joint_amplitudes와 순서가 일치해야 한다.
    left_joint_names: list[str] = MISSING
    # 오른다리 기준 동작 대상 관절. left_joint_names와 같은 순서의 대칭 관절이다.
    right_joint_names: list[str] = MISSING
    # 관절별 스윙 진폭(rad). 부호를 포함하며 좌우 다리에 동일하게 적용한다.
    joint_amplitudes: list[float] = MISSING


class GaitPhase:
    """episode 경과 시간에서 위상·지지 mask·기준 관절각을 계산한다.

    같은 제어 스텝에서 관측·보상이 여러 번 조회해도 계산은 한 번만 수행한다.
    위상은 episode_length_buf에서 유도하므로, reset된 환경은 invalidate() 이후
    다시 조회할 때 episode 시작 위상으로 돌아간다.
    """

    def __init__(self, cfg, env):
        """대상 관절 인덱스와 진폭 텐서를 준비한다."""
        if len(cfg.left_joint_names) != len(cfg.right_joint_names):
            raise ValueError("좌우 기준 동작 관절 수가 같아야 합니다.")
        if len(cfg.joint_amplitudes) != len(cfg.left_joint_names):
            raise ValueError("기준 동작 진폭 수가 대상 관절 수와 같아야 합니다.")
        if cfg.cycle_time_s <= 0.0 or not 0.0 <= cfg.double_support_ratio < 1.0:
            raise ValueError("보행 주기 또는 양발 지지 비율이 잘못되었습니다.")
        self._cfg = cfg
        self._env = env
        robot = env.scene[cfg.asset_name]

        # 좌우 관절을 같은 순서로 해석해 진폭과 1:1로 대응시킨다.
        left_ids, _ = robot.find_joints(list(cfg.left_joint_names), preserve_order=True)
        right_ids, _ = robot.find_joints(list(cfg.right_joint_names), preserve_order=True)
        self._joint_ids = list(left_ids) + list(right_ids)
        amplitudes = torch.tensor(list(cfg.joint_amplitudes), device=env.device)
        self._amplitudes = torch.cat((amplitudes, amplitudes)).unsqueeze(0)
        self._leg_count = len(left_ids)

        # 기준각의 기준점이 되는 기본 자세를 대상 관절만 잘라 보관한다.
        self._default_joint_pos = robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._phase = torch.zeros(env.num_envs, device=env.device)
        self._phase_signal = torch.zeros(env.num_envs, 2, device=env.device)
        self._stance_mask = torch.zeros(env.num_envs, 2, device=env.device)
        self._reference_joint_pos = self._default_joint_pos.clone()
        self._stamp = None

    @property
    def tracked_joint_ids(self):
        """기준 동작 추종 보상이 사용할 관절 인덱스를 [왼다리, 오른다리] 순서로 반환한다."""
        return list(self._joint_ids)

    @property
    def phase(self):
        """0~1로 정규화된 현재 보행 위상을 반환한다."""
        self._refresh()
        return self._phase

    @property
    def phase_signal(self):
        """관측에 사용할 [sin(2*pi*phase), cos(2*pi*phase)]를 반환한다."""
        self._refresh()
        return self._phase_signal

    @property
    def stance_mask(self):
        """[왼발, 오른발]의 지지 여부를 1.0/0.0으로 반환한다."""
        self._refresh()
        return self._stance_mask

    @property
    def swing_mask(self):
        """[왼발, 오른발]의 스윙 여부를 1.0/0.0으로 반환한다."""
        return 1.0 - self.stance_mask

    @property
    def reference_joint_pos(self):
        """대상 관절의 기준 관절각을 [왼다리, 오른다리] 순서로 반환한다."""
        self._refresh()
        return self._reference_joint_pos

    def invalidate(self):
        """episode reset처럼 같은 스텝 안에서 위상 입력이 바뀌면 캐시를 버린다."""
        self._stamp = None

    def _elapsed_time(self):
        """episode 경과 시간을 초 단위로 구한다.

        관측 차원을 재는 초기화 단계에서는 episode 길이 buffer가 아직 없으므로
        episode 시작 시점으로 본다.
        """
        steps = getattr(self._env, "episode_length_buf", None)
        if steps is None:
            return torch.zeros_like(self._phase)
        return steps.to(self._phase.dtype) * self._env.step_dt

    def _refresh(self):
        """같은 제어 스텝에서는 이전 계산 결과를 재사용한다."""
        step_counter = getattr(self._env, "common_step_counter", -1)
        if self._stamp == step_counter:
            return
        self._stamp = step_counter

        # episode 경과 시간을 주기로 나눈 나머지가 현재 위상이다.
        elapsed = self._elapsed_time()
        self._phase = torch.remainder(elapsed / self._cfg.cycle_time_s, 1.0)
        angle = 2.0 * math.pi * self._phase
        sin_phase, cos_phase = torch.sin(angle), torch.cos(angle)
        self._phase_signal = torch.stack((sin_phase, cos_phase), dim=-1)

        # 사인파의 음수 반주기는 왼다리 스윙, 양수 반주기는 오른다리 스윙에 대응한다.
        double_support = sin_phase.abs() < self._cfg.double_support_ratio
        left_swing = (sin_phase < 0.0) & ~double_support
        right_swing = (sin_phase > 0.0) & ~double_support
        self._stance_mask = torch.stack((~left_swing, ~right_swing), dim=-1).to(self._phase.dtype)

        # 스윙 중인 다리에만 반정류된 사인 진폭을 더하고 양발 지지에서는 기본 자세를 쓴다.
        left_signal = (-sin_phase).clamp_min(0.0) * left_swing
        right_signal = sin_phase.clamp_min(0.0) * right_swing
        signal = torch.cat((left_signal.unsqueeze(-1).expand(-1, self._leg_count),
                            right_signal.unsqueeze(-1).expand(-1, self._leg_count)), dim=-1)
        self._reference_joint_pos = self._default_joint_pos + self._amplitudes * signal
