"""학습 중 명시적으로 갱신하는 관측 통계."""
import torch
from torch import nn


class RunningNormalizer(nn.Module):
    """병합 가능한 평균·분산을 모델 버퍼에 보관한다."""

    def __init__(self, size: int, min_std: float = 0.1):
        """초기 통계와 최소 표준편차를 설정한다."""
        super().__init__()
        self.register_buffer("_mean", torch.zeros(size))
        self.register_buffer("_variance", torch.ones(size))
        self.register_buffer("_count", torch.zeros(()))
        self._min_std = min_std

    @torch.no_grad()
    def update(self, values):
        """새 표본을 병합하며 forward에서는 통계를 변경하지 않는다."""
        values = values.reshape(-1, self._mean.numel())
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise ValueError("정규화 입력에 비유한 값이 있습니다.")
        count = values.shape[0]
        mean = values.mean(0)
        variance = values.var(0, unbiased=False)
        total = self._count + count
        delta = mean - self._mean
        merged = (self._variance * self._count + variance * count
                  + delta.square() * self._count * count / total)
        self._mean.add_(delta * count / total)
        self._variance.copy_(merged / total)
        self._count.copy_(total)

    def forward(self, values):
        """최소 표준편차로 상수 관측의 과도한 증폭을 막는다."""
        return (values - self._mean) / self._variance.clamp_min(self._min_std ** 2).sqrt()

    def inverse(self, values):
        """정규화된 예측을 원래 물리 단위로 복원한다."""
        return values * self._variance.clamp_min(self._min_std ** 2).sqrt() + self._mean
