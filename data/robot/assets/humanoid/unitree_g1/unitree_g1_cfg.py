# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1(29 DOF, 손가락 없는 unitree_g1.urdf 기준) Isaac Lab ArticulationCfg.

data/robot/assets/humanoid/unitree_g1/usd/unitree_g1.usd를 스폰한다.
변환 시 fixed 관절 8개 가 부모 바디로 병합되어, 실제 바디 수는 30개다.
fixed 관절: imu_in_pelvis, d435_link, head_link, imu_in_torso, logo_link, mid360_link, left/right_rubber_hand
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 이 파일과 usd/unitree_g1.usd가 같은 폴더(data/robot/assets/humanoid/unitree_g1/)에 있으므로,
# 프로젝트 루트를 거치지 않고 이 파일 위치 기준으로 바로 찾는다.
_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "unitree_g1.usd")

# 기본 자세의 무릎 굽힘각(rad). 자세 후보 스윕에서 토크 여유가 가장 큰 값
_KNEE_FLEXION = 0.30

# 고관절·발목 pitch의 배분 편차(rad). 
# 허벅지·정강이 길이가 같으므로 각각 무릎각의 절반이 기준이며, 이 편차만큼 무게중심을 앞뒤로 옮긴다.
_HIP_ANKLE_SPLIT = 0.043


UNITREE_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 아래 기본 자세에서 다리가 체중을 받은 채 실제로 안착하는 높이다.
        pos=(0.0, 0.0, 0.781),
        joint_pos={
            # 다리
            "left_hip_pitch_joint": -0.5 * _KNEE_FLEXION + _HIP_ANKLE_SPLIT,
            "right_hip_pitch_joint": -0.5 * _KNEE_FLEXION + _HIP_ANKLE_SPLIT,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_knee_joint": _KNEE_FLEXION,
            "right_knee_joint": _KNEE_FLEXION,
            "left_ankle_pitch_joint": -0.5 * _KNEE_FLEXION - _HIP_ANKLE_SPLIT,
            "right_ankle_pitch_joint": -0.5 * _KNEE_FLEXION - _HIP_ANKLE_SPLIT,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            # 허리
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # 팔 (RL 비제어, 기본자세로 PD 고정)
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 감쇠는 사인 스윕으로 식별한 등가 관성에서 목표 감쇠비 0.7을 내는 값이다.
        # 대역폭(고유진동수 4~10 Hz)은 이미 제어 주기 대비 충분하므로 stiffness는 유지한다.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=None,  # USD(URDF 유래)에 있는 관절별 실제 토크 한계를 그대로 사용
            velocity_limit_sim=None,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.6,
                ".*_hip_roll_joint": 5.5,
                ".*_hip_pitch_joint": 4.3,
                ".*_knee_joint": 4.5,
            },
            armature=0.03,
        ),
        # 발목은 감쇠비가 0.12/0.05로 사실상 무감쇠였다. 접촉 안정성을 위해 1.0으로 맞춘다.
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
            damping={".*_ankle_pitch_joint": 1.65, ".*_ankle_roll_joint": 1.6},
            armature=0.03,
        ),
        # 게인은 공식 Isaac Lab G1 예제(isaaclab_assets.G1_CFG)의 torso_joint 값을 유지한다.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=3000.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
