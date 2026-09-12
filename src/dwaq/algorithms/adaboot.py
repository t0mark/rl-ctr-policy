"""완료된 episode return으로 추정 속도 사용 확률을 정한다."""
import torch


class AdaBoot:
    """환경별 최근 완료 return의 변동계수를 사용한다."""

    def __init__(self, num_envs, device, enabled=True, min_episodes=2, epsilon=1e-6):
        """통계가 부족할 때 실제 속도를 사용하도록 초기화한다."""
        self._returns = torch.zeros(num_envs, device=device)
        self._seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._enabled = enabled
        self._minimum = min(min_episodes, num_envs)
        self._epsilon = epsilon

    @torch.no_grad()
    def update(self, episode_returns, dones):
        """완료된 환경의 누적 return만 저장한다."""
        self._returns[dones] = episode_returns[dones]
        self._seen |= dones

    def probability(self):
        """음수 return에는 평균 절댓값, 영평균에는 보수적 확률을 적용한다."""
        if not self._enabled:
            return self._returns.new_tensor(1.0)
        values = self._returns[self._seen]
        if values.numel() < self._minimum:
            return self._returns.new_tensor(0.0)
        mean = values.mean().abs()
        cv = values.std(unbiased=False) / mean.clamp_min(self._epsilon)
        return torch.where(mean > self._epsilon, 1.0 - torch.tanh(cv), mean.new_zeros(()))

    def state_dict(self):
        """재개에 필요한 환경별 return 통계를 반환한다."""
        return {"returns": self._returns.clone(), "seen": self._seen.clone()}

    def load_state_dict(self, state):
        """같은 병렬 환경 수의 통계를 복원한다."""
        if state["returns"].shape != self._returns.shape:
            raise ValueError("AdaBoot 재개 시 환경 수가 일치해야 합니다.")
        self._returns.copy_(state["returns"])
        self._seen.copy_(state["seen"])
