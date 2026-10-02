# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1(다리 12 + 허리 3 + 팔 14 = 29 DOF) Isaac Lab ArticulationCfg.

data/robot/assets/humanoid/unitree_g1/usd/unitree_g1.usd를 스폰한다.
USD는 URDF의 fixed 관절을 병합해 변환했다.
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
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": _KNEE_FLEXION,
            "left_ankle_pitch_joint": -0.5 * _KNEE_FLEXION - _HIP_ANKLE_SPLIT,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.5 * _KNEE_FLEXION + _HIP_ANKLE_SPLIT,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": _KNEE_FLEXION,
            "right_ankle_pitch_joint": -0.5 * _KNEE_FLEXION - _HIP_ANKLE_SPLIT,
            "right_ankle_roll_joint": 0.0,
            # 허리
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # 팔
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
        joint_vel={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
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
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 게인과 토크 한계는 Unitree 공식 G1 설정(unitree_rl_gym, g1_29dof.urdf)을 따른다.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            ],
            effort_limit={
                "left_hip_pitch_joint": 88.0, "left_hip_roll_joint": 88.0,
                "left_hip_yaw_joint": 88.0, "left_knee_joint": 139.0,
                "right_hip_pitch_joint": 88.0, "right_hip_roll_joint": 88.0,
                "right_hip_yaw_joint": 88.0, "right_knee_joint": 139.0,
            },
            velocity_limit_sim=None,
            stiffness={
                "left_hip_pitch_joint": 100.0, "left_hip_roll_joint": 100.0,
                "left_hip_yaw_joint": 100.0, "left_knee_joint": 150.0,
                "right_hip_pitch_joint": 100.0, "right_hip_roll_joint": 100.0,
                "right_hip_yaw_joint": 100.0, "right_knee_joint": 150.0,
            },
            damping={
                "left_hip_pitch_joint": 2.0, "left_hip_roll_joint": 2.0,
                "left_hip_yaw_joint": 2.0, "left_knee_joint": 4.0,
                "right_hip_pitch_joint": 2.0, "right_hip_roll_joint": 2.0,
                "right_hip_yaw_joint": 2.0, "right_knee_joint": 4.0,
            },
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_ankle_pitch_joint", "right_ankle_roll_joint",
            ],
            effort_limit=35.0,
            velocity_limit_sim=None,
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
        ),
        # 허리는 상체 자세를 유지하면서 실기 토크 한계 안에서 움직이는 게인을 쓴다.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit={"waist_yaw_joint": 88.0, "waist_roll_joint": 35.0, "waist_pitch_joint": 35.0},
            velocity_limit_sim=None,
            stiffness=300.0,
            damping=3.0,
            armature=0.001,
        ),
        # 팔은 기본 자세를 유지한다. 게인은 Isaac Lab G1 29-DOF 설정, 토크 한계는 URDF 값을 따른다.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ],
            effort_limit={
                "left_shoulder_pitch_joint": 25.0, "left_shoulder_roll_joint": 25.0,
                "left_shoulder_yaw_joint": 25.0, "left_elbow_joint": 25.0, "left_wrist_roll_joint": 25.0,
                "left_wrist_pitch_joint": 13.4, "left_wrist_yaw_joint": 13.4,
                "right_shoulder_pitch_joint": 25.0, "right_shoulder_roll_joint": 25.0,
                "right_shoulder_yaw_joint": 25.0, "right_elbow_joint": 25.0, "right_wrist_roll_joint": 25.0,
                "right_wrist_pitch_joint": 13.4, "right_wrist_yaw_joint": 13.4,
            },
            velocity_limit_sim=None,
            stiffness=3000.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
