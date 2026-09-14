"""reset 직전 물리 상태를 포함한 보행 진단 표본을 수집한다."""
import math

import torch
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

# 스텝 평균이 아닌 합계로 집계하는 종료 사유별 발생 수 진단값의 이름 접두사.
TERMINATION_COUNT_PREFIX = "TerminationCount/"

# 평가 진단값에서 집계 대상 환경 수를 전달하는 항목 이름.
MEASURED_ENVS_KEY = "Evaluation/measured_envs"


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
            result[TERMINATION_COUNT_PREFIX + name] = manager.get_term(name).sum()
        # 설치된 Isaac Lab의 step reward는 weight 적용 후 dt를 나눈 시간당 기여다.
        manager = self._env.reward_manager
        for index, name in enumerate(manager.active_terms):
            result["RewardRate/" + name] = manager._step_reward[:, index].mean()
        return result


class EvaluationDiagnostics(RolloutDiagnostics):
    """출발 구간이 지난 환경만 평균하고 명령 대비 부호 있는 yaw rate 오차를 함께 계측하는 평가용 수집기."""

    def __init__(self, env, warmup_steps):
        """부모 수집기를 준비하고 출발 구간 스텝 수를 보관한다."""
        super().__init__(env)
        self._warmup_steps = warmup_steps

    def capture(self, ids):
        """부모 진단값에 명령 대비 부호 있는 월드 yaw rate 오차와 그 제곱을 추가로 계측한다."""
        super().capture(ids)
        error = self._robot.data.root_ang_vel_w[:, 2] - self._command[:, 2]
        values = {"Yaw/rate_error_radps": error, "Yaw/rate_error_square_radps2": error.square()}
        for name, value in values.items():
            if name not in self._values:
                self._values[name] = torch.zeros_like(value)
            self._values[name][ids] = value[ids]

    def metrics(self):
        """종료 수는 전체 환경 합계로, 상태·보상 진단값은 출발 구간이 지난 환경의 평균으로 반환한다."""
        # 종료 사유별 발생 수는 부모 결과를 그대로 사용한다.
        result = {name: value for name, value in super().metrics().items()
                  if name.startswith(TERMINATION_COUNT_PREFIX)}

        # 출발 구간이 지난 환경의 mask와 수를 구한다.
        measured = (self._env.episode_length_buf >= self._warmup_steps).float()
        count = measured.sum()
        divisor = count.clamp_min(1.0)

        # 상태 진단값을 집계 대상 환경만 평균한다.
        for name, value in self._values.items():
            result[name] = (value * measured).sum() / divisor

        # 보상 항목별 시간당 기여를 집계 대상 환경만 평균한다.
        manager = self._env.reward_manager
        for index, name in enumerate(manager.active_terms):
            result["RewardRate/" + name] = (manager._step_reward[:, index] * measured).sum() / divisor

        result[MEASURED_ENVS_KEY] = count
        return result

    def summarize(self, diagnostics):
        """평가 기간의 진단값에서 yaw 오차 요약과 시간 제한을 제외한 종료 원인별 수를 정리한다.

        diagnostics는 종료 수 합계와 나머지 항목의 평가 기간 평균을 담은 CPU 값이다.
        """
        # 부호 있는 yaw rate 오차의 평균(편향)과 표준편차, 절대 오차 평균을 구한다.
        bias = diagnostics["Yaw/rate_error_radps"]
        std = math.sqrt(max(diagnostics["Yaw/rate_error_square_radps2"] - bias ** 2, 0.0))
        yaw = {"rate_bias_radps": bias, "rate_std_radps": std,
               "abs_error_radps": diagnostics["Tracking/yaw_error_radps"]}

        # 시간 제한 종료를 제외한 종료 원인별 발생 수를 넘어짐 원인으로 정리한다.
        manager = self._env.termination_manager
        fall_terms = {name: int(diagnostics[TERMINATION_COUNT_PREFIX + name])
                      for name in manager.active_terms if not manager.get_term_cfg(name).time_out}
        return {"yaw": yaw, "fall_terms": fall_terms}
