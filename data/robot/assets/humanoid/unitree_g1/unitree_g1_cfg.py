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
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # 다리
            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.30,
            "right_knee_joint": 0.30,
            "left_ankle_pitch_joint": -0.20,
            "right_ankle_pitch_joint": -0.20,
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
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
            damping={".*_ankle_pitch_joint": 0.2, ".*_ankle_roll_joint": 0.1},
            armature=0.03,
        ),
        # 게인은 공식 Isaac Lab G1 예제(isaaclab_assets.G1_CFG)의 torso_joint 값을 따른다.
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
