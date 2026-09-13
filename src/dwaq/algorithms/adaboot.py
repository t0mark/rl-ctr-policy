"""완료된 episode의 보상 누적값으로 정책 입력에 추정 속도를 쓸 확률을 정한다."""
import torch


class AdaBoot:
    """환경별 최근 완료 episode return의 변동계수로 bootstrap 확률 1 - tanh(CV)를 계산한다."""

    def __init__(self, num_envs, device, enabled=True, min_episodes=2, epsilon=1e-6):
        """환경별 return 버퍼와 확률 계산 조건을 초기화한다."""
        # 진행 중인 episode와 최근 완료 episode의 return을 환경별로 보관한다.
        self._running_returns = torch.zeros(num_envs, device=device)
        self._returns = torch.zeros(num_envs, device=device)
        self._seen = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # 활성 여부, 확률 계산에 필요한 최소 환경 수, 평균 판정 기준을 저장한다.
        self._enabled = enabled
        self._minimum = max(2, min(min_episodes, num_envs))
        self._epsilon = epsilon

    @torch.no_grad()
    def update(self, rewards, dones):
        """스텝 보상을 누적하고 완료된 환경의 episode return을 저장한다."""
        # 음수 페널티를 포함한 스텝 보상을 부호 그대로 누적한다.
        self._running_returns += rewards

        # 완료된 환경의 누적값을 통계로 옮기고 다음 episode를 위해 초기화한다.
        self._returns[dones] = self._running_returns[dones]
        self._seen |= dones
        self._running_returns[dones] = 0.0

    def probability(self):
        """완료 episode return의 변동계수로 추정 속도 사용 확률을 반환한다."""
        # 비활성화하면 항상 추정 속도를 사용한다.
        if not self._enabled:
            return self._returns.new_tensor(1.0)

        # 완료 episode를 관측한 환경 수가 부족하면 실제 속도만 사용한다.
        values = self._returns[self._seen]
        if values.numel() < self._minimum:
            return self._returns.new_tensor(0.0)

        # 평균이 epsilon 이하이면 실제 속도만 사용하고, 양수 평균이면 1 - tanh(CV)를 적용한다.
        mean = values.mean()
        cv = values.std(unbiased=False) / mean.clamp_min(self._epsilon)
        return torch.where(mean > self._epsilon, 1.0 - torch.tanh(cv), mean.new_zeros(()))

    def state_dict(self):
        """재개에 필요한 환경별 완료 episode 통계를 반환한다."""
        return {"return_mode": "signed", "returns": self._returns.clone(),
                "seen": self._seen.clone()}

    def load_state_dict(self, state):
        """같은 병렬 환경 수의 완료 episode 통계를 복원하고 진행 중 누적값을 초기화한다."""
        if state["returns"].shape != self._returns.shape:
            raise ValueError("AdaBoot 재개 시 환경 수가 일치해야 합니다.")

        # return 방식이 기록되지 않은 통계는 버리고 새로 수집한다.
        if "return_mode" not in state:
            self._returns.zero_()
            self._seen.zero_()
            self._running_returns.zero_()
            return
        if state["return_mode"] != "signed":
            raise ValueError(f"지원하지 않는 AdaBoot return 방식: {state['return_mode']}")

        # 완료 episode 통계를 복원하고 진행 중 episode 누적값은 0부터 다시 시작한다.
        self._returns.copy_(state["returns"])
        self._seen.copy_(state["seen"])
        self._running_returns.zero_()
