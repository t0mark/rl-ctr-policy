"""성공한 명령의 인접 grid를 여는 명령 curriculum."""
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


class GridVelocityCommand(UniformVelocityCommand):
    """저속 명령부터 시작해 추적 성공 시 인접 셀로 확장한다."""

    def __init__(self, cfg, env):
        """명령 공간을 고정 grid로 분할하고 원점 인접 셀을 연다."""
        super().__init__(cfg, env)
        bounds = [cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z]
        self._low = torch.tensor([value[0] for value in bounds], device=self.device)
        self._high = torch.tensor([value[1] for value in bounds], device=self.device)
        counts = [cfg.bins if high > low else 1 for low, high in bounds]
        axes = [torch.arange(n, device=self.device) for n in counts]
        self._cells = torch.cartesian_prod(*axes)
        self._width = (self._high-self._low) / torch.tensor(counts, device=self.device)
        centers = self._low + (self._cells + 0.5) * self._width
        self._active = (centers.abs() <= cfg.initial_limit).all(-1)
        if not self._active.any():
            self._active[centers.square().sum(-1).argmin()] = True
        self._chosen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._score = torch.zeros(self.num_envs, device=self.device)
        self._samples = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        """평면·yaw 속도 추적 중 더 낮은 점수를 누적한다."""
        super()._update_metrics()
        linear = (self.command[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).square().sum(-1)
        angular = (self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]).square()
        self._score += torch.minimum(torch.exp(-4*linear), torch.exp(-4*angular))
        self._samples += 1

    def _resample_command(self, env_ids):
        """충분히 오래 성공한 비정지 명령의 이웃을 열고 새 명령을 뽑는다."""
        if not self.cfg.curriculum_enabled:
            super()._resample_command(env_ids)
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        success = ((self._samples[ids] * self._env.step_dt >= self.cfg.minimum_duration_s)
                   & (self._score[ids] / self._samples[ids].clamp_min(1) >= self.cfg.success_threshold)
                   & ~self.is_standing_env[ids])
        terminated = getattr(self._env, "reset_terminated", None)
        if terminated is not None:
            success &= ~terminated[ids]
        passed = self._chosen[ids[success]].unique()
        # grid 크기를 제한해 병렬 환경 수에 비례한 거대한 임시 텐서를 피한다.
        if passed.numel():
            distance = (self._cells[:, None, :] - self._cells[passed][None, :, :]).abs().sum(-1)
            self._active |= (distance <= 1).any(-1)
        selection = torch.multinomial(self._active.float(), len(ids), replacement=True)
        self._chosen[ids] = selection
        self.vel_command_b[ids] = self._low + (
            self._cells[selection] + torch.rand(len(ids), 3, device=self.device)) * self._width
        self.is_standing_env[ids] = torch.rand(len(ids), device=self.device) < self.cfg.rel_standing_envs
        self._score[ids] = 0
        self._samples[ids] = 0

    def curriculum_state(self):
        """현재 활성 grid의 mask를 복사한다."""
        return {"active": self._active.clone()}

    def load_curriculum_state(self, state):
        """동일 grid 크기의 학습 진척도를 복원한다."""
        if state["active"].shape != self._active.shape:
            raise ValueError("명령 curriculum grid 크기가 다릅니다.")
        self._active.copy_(state["active"])

    def reset(self, env_ids=None):
        """완료 구간을 평가한 뒤 부모가 새 명령을 샘플링하도록 한다."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return super().reset(env_ids)


@configclass
class GridVelocityCommandCfg(UniformVelocityCommandCfg):
    """논문의 grid-adaptive 취지를 명시적인 성공 기준으로 구현한다."""

    class_type: type = GridVelocityCommand
    curriculum_enabled: bool = True
    bins: int = 11
    initial_limit: float = 0.25
    success_threshold: float = 0.8
    minimum_duration_s: float = 2.0
