# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ 보행(velocity-tracking) 태스크의 Isaac Lab ManagerBasedRLEnv 설정.

- actor 관측 o_t = [ω, g, c, θ, θ̇, a_{t-1}] (proprioception만 사용)
- critic 특권 관측 s_t = [o_t, v_t, d_t, h_t] (몸통 선속도, 외란 힘, 높이맵)
- 보상은 DreamWaQ Table I, 도메인 랜덤화는 Table II를 따른다.
- 로봇별 asset·관절·링크는 build_env_config()가 `configs/rl/robot/legged/`의 YAML에서 연결한다.
"""

import importlib.util
import re
from dataclasses import MISSING
from pathlib import Path

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from src.sim.rl.envs.mdp import events, observations, rewards
from src.sim.rl.envs.mdp.actions import DelayedJointPositionActionCfg, StrengthPDActuatorCfg
from src.sim.rl.envs.terrains import DREAMWAQ_TERRAINS_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """지형, 로봇, 몸통 높이 스캐너, 접촉 센서, 조명."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DREAMWAQ_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING

    # 몸통 주변 높이맵 h_t. prim_path는 build_env_config()가 몸통 링크로 설정한다.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    """기본 자세 대비 관절 목표 오프셋 action. 시스템 지연 0~15 ms (Table II)."""

    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, delay_range_s=(0.0, 0.015)
    )


@configclass
class CommandsCfg:
    """10초마다 재샘플링하는 몸통 선속도·yaw 각속도 명령."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
                                                    ang_vel_z=(-1.0, 1.0)),
    )


@configclass
class ObservationsCfg:
    """actor 관측 o_t, critic 특권 관측 s_t, CENet 속도 정답 v_t. 관측 노이즈는 legged_gym 값을 쓴다."""

    @configclass
    class PolicyCfg(ObsGroup):
        """o_t = [몸통 각속도, 투영 중력, 속도 명령, 관절 위치, 관절 속도, 이전 action]."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(mean=0.0, std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """관측 노이즈를 켜고 항목을 하나의 벡터로 잇는다."""
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """s_t = [o_t, 몸통 선속도 v_t, 외란 힘 d_t, 높이맵 h_t]. 노이즈 없음."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        disturbance = ObsTerm(func=observations.disturbance_force,
                              params={"asset_cfg": SceneEntityCfg("robot", body_names="base")})
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))

        def __post_init__(self):
            """노이즈를 끄고 항목을 하나의 벡터로 잇는다."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """CENet 속도 추정 정답인 몸통 좌표계 선속도."""

        velocity = ObsTerm(func=mdp.base_lin_vel)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class RewardsCfg:
    """DreamWaQ Table I + 다리 접촉 페널티."""

    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0,
                               params={"command_name": "base_velocity", "std": 0.5})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5,
                               params={"command_name": "base_velocity", "std": 0.5})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_power = RewTerm(func=rewards.joint_power, weight=-2.0e-5, params={"asset_cfg": SceneEntityCfg("robot")})
    body_height = RewTerm(func=rewards.body_height, weight=-1.0,
                          params={"target_height": MISSING, "asset_cfg": SceneEntityCfg("robot"),
                                  "sensor_cfg": SceneEntityCfg("height_scanner")})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=rewards.ActionSmoothness, weight=-0.01)
    power_distribution = RewTerm(func=rewards.power_distribution, weight=-1.0e-5,
                                 params={"asset_cfg": SceneEntityCfg("robot")})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0,
                                 params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING),
                                         "threshold": 1.0})


@configclass
class TerminationsCfg:
    """시간 제한과 몸통 접촉 종료."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    """로봇 관절 구성 검사, DreamWaQ Table II 도메인 랜덤화, 몸통 외란 d_t, reset 상태."""

    # 로봇 YAML의 관절 목록이 USD 관절과 같은지 환경 생성 시 확인한다.
    check_joint_names = EventTerm(func=events.check_joint_names, mode="startup", params={"joint_names": MISSING})

    # 기체 특성: 마찰, payload, 무게중심, Kp·Kd, 모터 출력을 환경마다 한 번 샘플링한다.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.2, 1.25), "dynamic_friction_range": (0.2, 1.25),
                "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True},
    )
    add_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-1.0, 2.0), "operation": "add"},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1), "damping_distribution_params": (0.9, 1.1),
                "operation": "scale", "distribution": "uniform"},
    )
    motor_strength = EventTerm(func=events.randomize_motor_strength, mode="startup",
                               params={"factor_range": (0.9, 1.1), "actuator_names": MISSING})

    # 몸통 외란 d_t: 2~5초마다 ±10 N 힘을 새로 샘플링해 다음 샘플링까지 유지한다. (논문 미기재 값)
    disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (-10.0, 10.0), "torque_range": (0.0, 0.0)},
    )

    # reset 상태: 위치·yaw·속도와 관절 위치를 무작위로 둔다.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class CurriculumCfg:
    """게임식 지형 curriculum: 충분히 이동하면 어려운 지형으로, 못 가면 쉬운 지형으로 옮긴다."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class DreamWaQEnvCfg(ManagerBasedRLEnvCfg):
    """DreamWaQ 보행 환경. 정책 50 Hz, physics 200 Hz."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """제어·physics 주기, episode 길이, 센서 갱신 주기를 설정한다."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True


def _load_asset_config(project_root, robot_cfg):
    """data/robot/assets/{category}/{id}/{id}_cfg.py에서 {ID}_CFG 로봇 asset 설정을 읽는다."""
    robot_id = robot_cfg["id"]
    asset_path = Path(project_root) / "data" / "robot" / "assets" / robot_cfg["category"] / robot_id / f"{robot_id}_cfg.py"
    spec = importlib.util.spec_from_file_location(f"robot_asset_{robot_id}", asset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, f"{robot_id.upper()}_CFG")


def _controls_any_joint(actuator, joints):
    """actuator의 관절 목록에 제어 관절이 하나라도 있는지 확인한다."""
    return any(joint in joints for joint in actuator.joint_names_expr)


def build_env_config(config, project_root):
    """로봇 YAML의 asset·관절·링크·명령 범위를 DreamWaQ 환경 설정에 연결한다."""
    robot_cfg = config["robot"]
    command_cfg = config["command"]
    joints = config["controlled_joints"]
    base = config["links"]["base"]
    feet = config["links"]["feet"]
    cfg = DreamWaQEnvCfg()

    # 로봇 asset을 스폰하고, 제어 관절을 포함한 actuator를 출력 배율 PD로 바꾼다.
    asset = _load_asset_config(project_root, robot_cfg)
    cfg.scene.robot = asset.replace(prim_path="{ENV_REGEX_NS}/Robot")
    controlled_actuators = [name for name, actuator in cfg.scene.robot.actuators.items()
                            if _controls_any_joint(actuator, joints)]
    for name in controlled_actuators:
        actuator = cfg.scene.robot.actuators[name]
        cfg.scene.robot.actuators[name] = StrengthPDActuatorCfg(**{
            key: value for key, value in vars(actuator).items() if not key.startswith("_") and key != "class_type"
        })

    # action·관측 관절, 몸통 스캐너, 명령 범위를 연결한다.
    cfg.actions.joint_pos.joint_names = joints
    cfg.actions.joint_pos.scale = config["control"]["action_scale"]
    cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{base}"
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=joints)}
        group.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=joints)}
    cfg.observations.critic.disturbance.params["asset_cfg"] = SceneEntityCfg("robot", body_names=base)
    cfg.commands.base_velocity.ranges.lin_vel_x = tuple(command_cfg["linear_x"])
    cfg.commands.base_velocity.ranges.lin_vel_y = tuple(command_cfg["linear_y"])
    cfg.commands.base_velocity.ranges.ang_vel_z = tuple(command_cfg["angular_z"])

    # 보상의 관절·접촉 링크와 데이터시트 높이를 연결한다. SceneEntityCfg는 resolve 시 수정되므로 항목마다 새로 만든다.
    for term in (cfg.rewards.joint_acc, cfg.rewards.joint_power, cfg.rewards.power_distribution):
        term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joints)
    cfg.rewards.body_height.params["target_height"] = robot_cfg["standing_height"] - robot_cfg["root_to_top"]
    excluded_links = "|".join(re.escape(name) for name in (base, *feet))
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=f"(?!(?:{excluded_links})$).*")

    # 관절 구성 검사, 도메인 랜덤화·외란·종료 대상 링크와 actuator를 연결한다.
    cfg.events.check_joint_names.params["joint_names"] = config["joints"]
    for term in (cfg.events.add_payload, cfg.events.base_com, cfg.events.disturbance):
        term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=base)
    cfg.events.actuator_gains.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joints)
    cfg.events.motor_strength.params["actuator_names"] = controlled_actuators
    cfg.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=base)
    return cfg


# 평가에서 제거하는 도메인 랜덤화·외란 이벤트.
_RANDOMIZATION_EVENT_NAMES = ("add_payload", "base_com", "actuator_gains", "motor_strength", "disturbance")


def configure_evaluation(cfg, command, flat_terrain=False, observation_noise=False):
    """고정 명령·평가 지형·관측 노이즈·기본 동역학의 평가 조건을 적용한다. command는 [vx, vy, yaw rate]이다."""
    # 평가 지형과 고정 명령: 지형 curriculum을 끄고 명령을 지정 값으로 고정한다.
    if flat_terrain:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.curriculum.terrain_levels = None
    velocity_command = cfg.commands.base_velocity
    velocity_command.rel_standing_envs = 0.0
    velocity_command.ranges.lin_vel_x = (command[0], command[0])
    velocity_command.ranges.lin_vel_y = (command[1], command[1])
    velocity_command.ranges.ang_vel_z = (command[2], command[2])
    cfg.observations.policy.enable_corruption = observation_noise

    # 기본 동역학: 마찰을 1로 고정하고 랜덤화·외란 이벤트와 시스템 지연을 제거한다.
    cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
    for name in _RANDOMIZATION_EVENT_NAMES:
        setattr(cfg.events, name, None)
    cfg.actions.joint_pos.delay_range_s = (0.0, 0.0)
    return cfg
