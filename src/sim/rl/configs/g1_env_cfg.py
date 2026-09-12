# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 보행 태스크 설정. src.sim.rl.env_cfg의 로봇 무관 공용 클래스를 상속해서
G1(다리+허리 15 DOF만 RL 제어, 팔은 기본자세 PD 고정)에 맞는 값만 오버라이드한다.

공용 보상은 Two-Phase 논문의 명시 수식을 사용하며, G1용 안전 종료·관절 제한과
하드웨어에 맞춘 자세·가속도 가중치를 함께 사용한다.

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
from src.sim.rl.mdp import rewards as paper_rewards
from src.sim.rl.paths import ROOT_DIR
from src.sim.rl.mdp.actuators import StrengthPDActuatorCfg

# data/robot/assets/humanoid/unitree_g1/unitree_g1_cfg.py를 찾기 위한 경로 추가.
# (paths.py에서 이미 검증된 ROOT_DIR을 재사용한다 — 이 파일에서 dirname을 직접 다시 세면
# 폴더 깊이가 바뀔 때마다 개수를 세는 실수가 반복되기 쉽다.)
_G1_ASSET_DIR = os.path.join(ROOT_DIR, "data", "robot", "assets", "humanoid", "unitree_g1")
sys.path.insert(0, _G1_ASSET_DIR)
from unitree_g1_cfg import UNITREE_G1_CFG  # noqa: E402

# G1의 RL 제어 대상 관절(다리 12 + 허리 3 = 15). 나머지(팔)는 기본자세로 PD 고정된다.
G1_ACTUATED_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]

# 관절 가속도 페널티 대상. 공식 G1 예제와 동일하게 다리의 큰 관절만 대상으로 한다.
G1_ACCELERATION_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint"]

# 사인파 기준 동작 대상 관절. 좌우 목록은 같은 순서의 대칭 관절이어야 한다.
G1_REFERENCE_JOINT_NAMES = {
    "left": ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"],
    "right": ["right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"],
}

# 위 관절 순서에 대응하는 스윙 진폭(rad). 무릎은 고관절의 두 배로 굽힌다.
# 크기는 목표 발 높이를 내는 배율을 보행 후보 스윕에서 골라 정했다.
G1_REFERENCE_JOINT_AMPLITUDES = [-0.354, 0.707, -0.354]

# 스윙 최고점에서의 목표 발 높이(m). 위 진폭이 실제로 만드는 높이다.
G1_FOOT_CLEARANCE_HEIGHT = 0.070

# 좌우 발·무릎 간격 보상의 두 목표 거리(m). 논문 Table 4의 Feet&Knee distance 수식은
# 두 목표 각각에 대한 지수 보상의 평균이며, 논문은 N1 기준 0.3과 0.125를 쓴다.
G1_FOOT_DISTANCE_TARGETS = (0.12, 0.35)
G1_KNEE_DISTANCE_TARGETS = (0.10, 0.30)

# 좌우 발 링크. 위상 기반 보상이 [왼발, 오른발] 순서를 유지해야 하므로 명시적으로 나열한다.
G1_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# 좌우 무릎 링크. 다리 간격 보상에서 발 간격과 함께 사용한다.
G1_KNEE_BODY_NAMES = ["left_knee_link", "right_knee_link"]


@configclass
class G1Rewards(RewardsCfg):
    """Two-Phase 보상에 G1의 링크 매핑과 안전 보상을 연결한다.

    기준 동작·접촉·공중 시간은 Two-Phase 수식을 사용한다.
    관절 제한·종료 및 하드웨어 조정 가중치는 G1 설정을 사용한다.
    팔은 RL 제어 대상이 아니므로 별도의 팔 자세 보상을 두지 않는다.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # 위상 편차에 대한 응답이 반복 잡음보다 작아 민감도를 얻지 못했다.
    # 정상 동작 기여를 양의 보상 합의 2% 이내로 제한하는 크기다.
    feet_air_time = RewTerm(
        func=paper_rewards.feet_air_time, weight=-17.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES,
                                             preserve_order=True)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.86,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 정상 동작에서 원값이 0이라 수준으로도 크기를 정할 수 없다. 설정값을 유지한다.
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # 사인파 기준 동작 추종. 학습 1단계에서만 활성화한다.
    joint_position_tracking = RewTerm(
        func=paper_rewards.joint_position_tracking,
        weight=4.99,
        params={"coefficient": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 스윙 위상에서 발이 접촉하는 경우를 벌한다.
    gait_phase_contact = RewTerm(
        func=paper_rewards.gait_phase_contact,
        weight=-0.558,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "threshold": 1.0,
        },
    )
    # 두 목표 발 간격에 가까울수록 보상한다.
    feet_distance = RewTerm(
        func=paper_rewards.lateral_distance,
        weight=0.948,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True),
            "minimum_distance": G1_FOOT_DISTANCE_TARGETS[0],
            "maximum_distance": G1_FOOT_DISTANCE_TARGETS[1],
        },
    )
    # 두 목표 무릎 간격에 가까울수록 보상한다.
    knee_distance = RewTerm(
        func=paper_rewards.lateral_distance,
        weight=0.541,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_KNEE_BODY_NAMES, preserve_order=True),
            "minimum_distance": G1_KNEE_DISTANCE_TARGETS[0],
            "maximum_distance": G1_KNEE_DISTANCE_TARGETS[1],
        },
    )
    # 발바닥이 지면과 평행을 유지하도록 한다.
    feet_orientation = RewTerm(
        func=paper_rewards.feet_orientation,
        weight=-26.9,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES,
                                            preserve_order=True)},
    )
    # 몸통의 급격한 가속을 억제한다. 논문과 같은 유계 지수 보상이며 가중치도 논문값이다.
    # 보상이 [0, 1]이라 가중치가 곧 기여 상한이고, 보행 속도 편차에 대한 민감도는
    # 반복 잡음보다 작아 계측으로 정하지 않았다. 계수는 G1의 가속도 규모에 맞춘 값이다.
    root_acceleration = RewTerm(
        func=paper_rewards.root_acceleration,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "coefficient": 1.0e-5},
    )
    # 관절 떨림을 억제한다.
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.20e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)},
    )


@configclass
class G1EnvCfg(VelocityEnvCfg):
    """G1 관절·링크 매핑과 이족보행에 맞춘 보상 가중치를 연결한다."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        """자산 gain을 explicit PD에 연결하고 MDP 대상을 G1에 맞춘다."""
        super().__post_init__()
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 토크 배율을 직접 적용하는 explicit PD로 모든 관절을 제어한다.
        self.scene.robot.actuators = {
            name: StrengthPDActuatorCfg(**{
                key: value for key, value in vars(cfg).items()
                if not key.startswith("_") and key != "class_type"
            }) for name, cfg in self.scene.robot.actuators.items()
        }
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
        # 관절 관측은 RL 제어 대상만 본다. 기본값은 전체 관절이라, 
        # 팔처럼 기본자세로 PD 고정된 비제어 관절까지 들어가 정보가 없는 입력 차원이 생긴다.
        for group in (self.observations.policy_current, self.observations.critic):
            for term in (group.joint_pos, group.joint_vel):
                term.params = {"asset_cfg": SceneEntityCfg("robot",
                                                           joint_names=G1_ACTUATED_JOINT_NAMES)}
        self.actions.joint_pos.delay_range_s = (0.0, 0.010)
        # 질량·무게중심 랜덤화는 몸통 링크만 대상으로 한다. 외란은 전체 링크에 적용한다.
        for event in (self.events.add_base_mass, self.events.base_com):
            event.params["asset_cfg"].body_names = ["torso_link"]
        self.rewards.body_height.params["target_height"] = UNITREE_G1_CFG.init_state.pos[2]
        # 위상 mask와 순서를 맞추기 위해 발 링크를 [왼발, 오른발]로 고정한다.
        self.rewards.feet_clearance.params["asset_cfg"].body_names = G1_FOOT_BODY_NAMES
        self.rewards.feet_clearance.params["asset_cfg"].preserve_order = True
        # 토크·기본 자세는 RL 제어 관절 전체를, 가속도는 다리의 큰 관절을 대상으로 한다.
        for reward in (self.rewards.joint_power, self.rewards.default_joint_tracking):
            reward.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=G1_ACCELERATION_JOINT_NAMES)
        # 상하 진동은 이족보행의 정상 동작이므로 벌하지 않는다.
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # 휴머노이드는 사족보행용 리셋 랜덤화를 견디지 못하므로 공식 G1 예제 값을 쓴다.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
