"""reset 직전 물리 상태를 포함한 보행 진단 표본을 수집한다."""
import torch
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


class RolloutDiagnostics:
    """로봇 상태를 환경별로 보존하고 제어 스텝의 평균 지표를 제공한다."""

    def __init__(self, env):
        """제어 관절·비제어 관절과 접촉 링크 인덱스를 준비한다."""
        self._env = env
        self._robot = env.scene["robot"]
        self._action = env.action_manager.get_term("joint_pos")
        controlled, _ = self._robot.find_joints(self._action.joint_names, preserve_order=True)
        self._controlled = controlled
        self._uncontrolled = [i for i in range(self._robot.num_joints) if i not in controlled]
        self._values = {}
        self._command = torch.zeros(env.num_envs, 3, device=env.device)

    def begin_step(self):
        """물리 진행 동안 유효한 명령을 복사한다."""
        self._command.copy_(self._env.command_manager.get_command("base_velocity"))

    def capture(self, ids):
        """선택한 환경을 reset하기 전 또는 물리 진행 직후에 계측한다."""
        robot = self._robot.data
        velocity = quat_apply_inverse(yaw_quat(robot.root_quat_w), robot.root_lin_vel_w)
        values = {
            "Tracking/linear_error_mps": (velocity[:, :2] - self._command[:, :2]).norm(dim=-1),
            "Tracking/yaw_error_radps": (robot.root_ang_vel_w[:, 2] - self._command[:, 2]).abs(),
            "Control/requested_target_clipped_fraction": self._action.target_clipped.float().mean(-1),
        }
        limits = robot.joint_pos_limits[:, self._controlled]
        targets = robot.joint_pos_target[:, self._controlled]
        values["Control/applied_target_outside_limits_fraction"] = (
            (targets < limits[..., 0]) | (targets > limits[..., 1])).float().mean(-1)
        if self._uncontrolled:
            values["Control/arm_mean_abs_velocity_radps"] = robot.joint_vel[:, self._uncontrolled].abs().mean(-1)
            values["Control/arm_mean_abs_position_error_rad"] = (
                robot.joint_pos[:, self._uncontrolled] - robot.default_joint_pos[:, self._uncontrolled]).abs().mean(-1)
        for name, value in values.items():
            if name not in self._values:
                self._values[name] = torch.zeros_like(value)
            self._values[name][ids] = value[ids]

    def metrics(self):
        """상태 평균·실제 종료 사유·시간당 보상 기여를 GPU 텐서로 반환한다."""
        result = {name: value.mean() for name, value in self._values.items()}
        manager = self._env.termination_manager
        for name in manager.active_terms:
            result["TerminationCount/" + name] = manager.get_term(name).sum()
        # 설치된 Isaac Lab의 step reward는 weight 적용 후 dt를 나눈 시간당 기여다.
        manager = self._env.reward_manager
        for index, name in enumerate(manager.active_terms):
            result["RewardRate/" + name] = manager._step_reward[:, index].mean()
        return result
