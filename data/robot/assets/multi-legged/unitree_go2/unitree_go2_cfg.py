# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2(다리 4 × 3 = 12 DOF) Isaac Lab ArticulationCfg.

data/robot/assets/multi-legged/unitree_go2/usd/unitree_go2.usd를 스폰한다.
USD는 fixed 관절을 병합해 변환했으며, 발 링크(*_foot)는 URDF의 dont_collapse 속성으로 보존된다.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 이 파일과 usd/unitree_go2.usd가 같은 폴더(data/robot/assets/multi-legged/unitree_go2/)에 있으므로,
# 프로젝트 루트를 거치지 않고 이 파일 위치 기준으로 바로 찾는다.
_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "unitree_go2.usd")


UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 기본 자세와 높이는 Unitree 공식 Go2 설정(unitree_rl_gym go2_config.py)을 따른다.
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 게인은 unitree_rl_gym Go2 설정, 토크·속도 한계는 URDF 값을 따른다.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            velocity_limit={".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 15.7},
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
