# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Boston Dynamics Spot(다리 4 × 3 = 12 DOF) Isaac Lab ArticulationCfg.

data/robot/assets/multi-legged/boston_dynamics_spot/usd/boston_dynamics_spot.usd를 스폰한다.
USD는 Isaac Sim 5.1 Nucleus 에셋(Robots/BostonDynamics/spot/spot.usd)을 그대로 가져왔다.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 이 파일과 usd/boston_dynamics_spot.usd가 같은 폴더에 있으므로, 이 파일 위치 기준으로 바로 찾는다.
_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "boston_dynamics_spot.usd")


BOSTON_DYNAMICS_SPOT_CFG = ArticulationCfg(
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
        # 기본 자세와 높이는 Isaac Lab SPOT_CFG를 따른다.
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "[fh]l_hx": 0.1,
            "[fh]r_hx": -0.1,
            "f[rl]_hy": 0.9,
            "h[rl]_hy": 1.1,
            ".*_kn": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 게인과 고관절 토크 한계는 Isaac Lab SPOT_CFG를 따른다.
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=45.0,
            stiffness=60.0,
            damping=1.5,
        ),
        # 무릎 토크 한계는 Isaac Lab 무릎 remotized 룩업 테이블의 보행 구간(-2.0~-1.0 rad) 대표값으로 근사한다.
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit=100.0,
            stiffness=60.0,
            damping=1.5,
        ),
    },
)
