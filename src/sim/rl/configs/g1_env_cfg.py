# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 보행 태스크 설정. src.sim.rl.env_cfg의 로봇 무관 공용 클래스를 상속해서
G1(다리 12 + 허리 3 = 15 DOF)에 맞는 값만 오버라이드한다. 팔은 USD에서 고정 관절이다.

공용 보상은 Two-Phase 논문의 수식·가중치를 기준으로 사용한다. G1은 안전 종료·관절
제한 및 하드웨어에 맞는 목표 자세·높이를 추가한다.

Two-Phase 논문 항목(사인파 기준 동작 대상 관절, 위상 기반 보상의 좌우 링크 순서,
좌우 발 지형 스캐너)도 여기에서 G1의 실제 관절·링크 이름으로 연결한다.
"""

import os
import sys

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# env_cfg.py와 동일한 mdp 모듈(isaaclab 기본 항목 + biped 전용 항목을 함께 재노출함)을 쓴다.
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from src.sim.rl.env_cfg import RewardsCfg, VelocityEnvCfg
from src.sim.rl.mdp import rewards
from src.sim.rl.paths import ROOT_DIR
from src.sim.rl.mdp.actuators import StrengthPDActuatorCfg

# data/robot/assets/humanoid/unitree_g1/unitree_g1_cfg.py를 찾기 위한 경로 추가.
# (paths.py에서 이미 검증된 ROOT_DIR을 재사용한다 — 이 파일에서 dirname을 직접 다시 세면
# 폴더 깊이가 바뀔 때마다 개수를 세는 실수가 반복되기 쉽다.)
_G1_ASSET_DIR = os.path.join(ROOT_DIR, "data", "robot", "assets", "humanoid", "unitree_g1")
sys.path.insert(0, _G1_ASSET_DIR)
from unitree_g1_cfg import UNITREE_G1_CFG

# G1의 RL 제어 대상은 다리 12 + 허리 3관절이다.
G1_CONTROLLED_ACTUATOR_NAMES = ("legs", "feet", "waist")

G1_ACTUATED_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]

# 허리 관절. 보행에 직접 쓰이지 않는 관절이라 다리와 분리해 편차를 억제한다.
G1_WAIST_JOINT_NAMES = ["waist_.*_joint"]

# 발의 좌우 위치를 결정하는 고관절 관절. 전후 구동을 담당하는 pitch는 제외한다.
G1_HIP_DEVIATION_JOINT_NAMES = [".*_hip_yaw_joint", ".*_hip_roll_joint"]

# 사인파 기준 동작 대상 관절. 좌우 목록은 같은 순서의 대칭 관절이어야 한다.
G1_REFERENCE_JOINT_NAMES = {
    "left": ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"],
    "right": ["right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"],
}

# 위 관절 순서에 대응하는 스윙 진폭(rad). 무릎은 고관절의 두 배로 굽힌다.
# 크기는 목표 발 높이를 내는 배율을 보행 후보 스윕에서 골라 정했다.
G1_REFERENCE_JOINT_AMPLITUDES = [-0.354, 0.707, -0.354]

# root 속도 변화량 보상의 지수 계수. 제어 주기 한 번 동안의 변화량에 적용한다.
G1_ROOT_ACCELERATION_COEFFICIENT = 3.0

# 스윙 최고점에서의 목표 발바닥 높이(m).
G1_FOOT_CLEARANCE_HEIGHT = 0.070

# 발 링크 원점에서 밑창 중심까지의 거리(m). 발 높이를 밑창 기준으로 환산한다.
G1_FOOT_SOLE_OFFSET = 0.035

# 좌우 발·무릎 간격 보상의 허용 범위(m). 범위 안에서는 최대 보상을 준다.
G1_LATERAL_DISTANCE_BOUNDS = (0.12, 0.35)

# 좌우 발 링크. 위상 기반 보상이 [왼발, 오른발] 순서를 유지해야 하므로 명시적으로 나열한다.
G1_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# 좌우 무릎 링크. 다리 간격 보상에서 발 간격과 함께 사용한다.
G1_KNEE_BODY_NAMES = ["left_knee_link", "right_knee_link"]


@configclass
class G1Rewards(RewardsCfg):
    """Two-Phase 보상에 G1의 링크 매핑과 안전 보상을 연결한다.

    기준 동작·접촉·공중 시간은 Two-Phase 수식과 가중치를 사용한다.
    생존 보너스와 허리 편차 벌점은 논문에 없는 G1용 항목이다.
    팔은 고정 관절이므로 별도의 팔 자세 보상을 두지 않는다.
    """

    # 종료되지 않고 살아있는 동안 매 스텝 보너스를 준다.
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # Two-Phase Table 4의 공중 시간 항목은 착지 시 음의 보상을 준다.
    feet_air_time_raw = RewTerm(
        func=rewards.feet_air_time, weight=-0.001,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES,
                                             preserve_order=True)},
    )
    contact_no_vel_xy_norm = RewTerm(
        func=rewards.feet_slip,
        weight=-0.005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    # 허리 관절이 기본 자세에서 벗어나는 정도를 다리와 분리해 억제한다.
    waist_pos_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_WAIST_JOINT_NAMES)},
    )
    # 고관절 yaw·roll이 기본 자세에서 벗어나는 정도를 억제해 발의 좌우 위치를 안정화한다.
    hip_pos_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_HIP_DEVIATION_JOINT_NAMES)},
    )
    # 사인파 기준 동작 추종. 학습 1단계에서만 활성화한다.
    ref_dof_pos = RewTerm(
        func=rewards.joint_position_tracking,
        weight=3.2,
        params={"coefficient": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 스윙 위상에서 발이 접촉하는 경우를 벌한다.
    contact_swing = RewTerm(
        func=rewards.gait_phase_contact,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES,
                                         preserve_order=True),
        },
    )
    # 발 간격이 허용 범위 안에 있을수록 보상한다.
    feet_distance = RewTerm(
        func=rewards.lateral_distance,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
            "minimum_distance": G1_LATERAL_DISTANCE_BOUNDS[0],
            "maximum_distance": G1_LATERAL_DISTANCE_BOUNDS[1],
        },
    )
    # 무릎 간격이 허용 범위 안에 있을수록 보상한다.
    knee_distance = RewTerm(
        func=rewards.lateral_distance,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_KNEE_BODY_NAMES, preserve_order=True),
            "minimum_distance": G1_LATERAL_DISTANCE_BOUNDS[0],
            "maximum_distance": G1_LATERAL_DISTANCE_BOUNDS[1],
        },
    )
    # 중력의 수평 성분 제곱합은 오차이므로 음의 가중치로 발의 기울기를 억제한다.
    # 경사면 법선 추종이 아니라 세계 수평 유지이며 Table 4의 모호한 축 표기를 구체화한다.
    feet_orientation = RewTerm(
        func=rewards.feet_orientation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES,
                                            preserve_order=True)},
    )
    # 몸통의 급격한 속도 변화를 억제한다. 계수는 제어 주기당 속도 변화량에 적용한다.
    base_acc = RewTerm(
        func=rewards.RootAcceleration,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "coefficient": G1_ROOT_ACCELERATION_COEFFICIENT},
    )
    # 관절 떨림을 억제한다.
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)},
    )


@configclass
class G1EnvCfg(VelocityEnvCfg):
    """G1 관절·링크 매핑과 이족보행에 맞춘 보상 가중치를 연결한다."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        """학습 관절을 지연 PD actuator에 연결하고 G1의 링크·관절 이름을 매핑한다."""
        super().__post_init__()
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 학습 관절을 출력 배율과 토크 지연을 적용하는 explicit PD actuator로 교체한다.
        for name in G1_CONTROLLED_ACTUATOR_NAMES:
            actuator = self.scene.robot.actuators[name]
            self.scene.robot.actuators[name] = StrengthPDActuatorCfg(**{
                key: value for key, value in vars(actuator).items()
                if not key.startswith("_") and key != "class_type"
            })
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        for event in (self.events.motor_strength, self.events.torque_delay):
            event.params["actuator_names"] = G1_CONTROLLED_ACTUATOR_NAMES
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.left_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/left_ankle_roll_link"
        self.scene.right_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/right_ankle_roll_link"
        # 사인파 기준 동작은 다리의 pitch 관절에만 적용한다. 진폭 부호는 G1의 기본 자세 방향을 따른다.
        self.gait.cycle_time_s = 0.66
        self.gait.double_support_ratio = 0.1
        self.gait.left_joint_names = G1_REFERENCE_JOINT_NAMES["left"]
        self.gait.right_joint_names = G1_REFERENCE_JOINT_NAMES["right"]
        self.gait.joint_amplitudes = G1_REFERENCE_JOINT_AMPLITUDES
        self.actions.joint_pos.joint_names = G1_ACTUATED_JOINT_NAMES
        # 관절 관측은 RL 제어 대상만 본다.
        for group in (self.observations.policy_current, self.observations.critic):
            for term in (group.joint_pos, group.joint_vel):
                term.params = {"asset_cfg": SceneEntityCfg("robot",
                                                           joint_names=G1_ACTUATED_JOINT_NAMES)}
        self.actions.joint_pos.delay_range_s = (0.0, 0.010)
        for event in (self.events.add_base_mass, self.events.base_com):
            event.params["asset_cfg"].body_names = ["torso_link"]
        self.rewards.base_height_terrain.params["target_height"] = UNITREE_G1_CFG.init_state.pos[2]
        self.rewards.feet_swing_height_terrain_phase.params["target_height"] = G1_FOOT_CLEARANCE_HEIGHT
        self.rewards.feet_swing_height_terrain_phase.params["sole_offset"] = G1_FOOT_SOLE_OFFSET
        self.rewards.feet_swing_height_terrain_phase.params["asset_cfg"].body_names = G1_FOOT_BODY_NAMES
        self.rewards.feet_swing_height_terrain_phase.params["asset_cfg"].preserve_order = True
        for reward in (self.rewards.dof_power, self.rewards.default_dof_pos):
            reward.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        self.rewards.dof_acc_physics_step.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
