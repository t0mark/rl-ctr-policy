"""현재 속도와 다음 관측을 학습하는 DreamWaQ 네트워크."""
import torch
from torch import nn
from torch.distributions import Normal

from .normalizer import RunningNormalizer


def make_mlp(dimensions, activation):
    """차원 목록으로 독립 활성화 함수를 가진 MLP를 구성한다."""
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    if activation not in activations:
        raise ValueError(f"지원하지 않는 활성화: {activation}")
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(dimensions) - 2:
            layers.append(activations[activation]())
    return nn.Sequential(*layers)


class ActorCritic_DWAQ(nn.Module):
    """정책은 평균 context를, VAE는 재매개화한 latent를 사용한다."""

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, cenet_in_dim,
                 cenet_out_dim=19, activation="elu", init_noise_std=1.0,
                 normalization_min_std=0.1):
        """정책·가치·추정 네트워크와 정규화 통계를 생성한다."""
        super().__init__()
        self._obs_dim = num_actor_obs - cenet_out_dim
        if cenet_in_dim % self._obs_dim or init_noise_std <= 0:
            raise ValueError("관측 이력 차원 또는 초기 표준편차가 잘못되었습니다.")
        self._history_length = cenet_in_dim // self._obs_dim
        self._actor = make_mlp([num_actor_obs, 512, 256, 128, num_actions], activation)
        self._critic = make_mlp([num_critic_obs, 512, 256, 128, 1], activation)
        self._encoder = nn.Sequential(make_mlp([cenet_in_dim, 128, 64], activation),
                                      {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]())
        self._velocity_head = nn.Linear(64, 3)
        self._latent_mean = nn.Linear(64, cenet_out_dim - 3)
        self._latent_logvar = nn.Linear(64, cenet_out_dim - 3)
        self._decoder = make_mlp([cenet_out_dim, 64, 128, self._obs_dim], activation)
        self._log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())
        self._obs_normalizer = RunningNormalizer(self._obs_dim, normalization_min_std)
        self._critic_normalizer = RunningNormalizer(num_critic_obs, normalization_min_std)
        self._velocity_normalizer = RunningNormalizer(3, normalization_min_std)

    def encode(self, history):
        """프레임별 동일 정규화를 적용해 속도와 latent 분포를 반환한다."""
        frames = history.reshape(-1, self._history_length, self._obs_dim)
        encoded = self._encoder(self._obs_normalizer(frames).flatten(1))
        return (self._velocity_head(encoded), self._latent_mean(encoded),
                self._latent_logvar(encoded).clamp(-10.0, 10.0))

    def distribution(self, observations, history, velocity_target=None, use_estimate=None):
        """AdaBoot 선택을 유지한 조건부 action 분포를 반환한다."""
        velocity, latent, _ = self.encode(history)
        if velocity_target is not None:
            if use_estimate is None:
                raise ValueError("실제 속도 입력에는 AdaBoot 선택 mask가 필요합니다.")
            velocity = torch.where(use_estimate.bool().reshape(-1, 1), velocity,
                                   self._velocity_normalizer(velocity_target))
        features = torch.cat((velocity, latent, self._obs_normalizer(observations)), -1)
        mean = self._actor(features)
        std = self._log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def act_inference(self, observations, obs_history):
        """실제 속도를 입력받지 않는 결정적 배포 정책을 실행한다."""
        return self.distribution(observations, obs_history).mean

    def estimate_velocity(self, history):
        """추정 속도를 m/s 단위로 반환한다."""
        return self._velocity_normalizer.inverse(self.encode(history)[0])

    def evaluate(self, critic_observations):
        """특권 관측에서 상태 가치를 계산한다."""
        return self._critic(self._critic_normalizer(critic_observations)).squeeze(-1)

    def auxiliary_losses(self, history, velocity_target, next_obs, prediction_valid):
        """현재 속도·유효한 다음 관측·latent KL의 batch 평균을 계산한다."""
        velocity, mean, logvar = self.encode(history)
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        prediction = self._decoder(torch.cat((velocity, latent), -1))
        velocity_loss = (velocity - self._velocity_normalizer(velocity_target)).square().mean()
        per_sample = (prediction - self._obs_normalizer(next_obs)).square().mean(-1)
        valid = prediction_valid.to(per_sample.dtype)
        prediction_loss = (per_sample * valid).sum() / valid.sum().clamp_min(1)
        kl_loss = (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()
        return velocity_loss, prediction_loss, kl_loss

    @torch.no_grad()
    def update_normalizers(self, observations, critic_observations, velocity):
        """rollout와 PPO 갱신 사이 경계에서만 통계를 갱신한다."""
        self._obs_normalizer.update(observations)
        self._critic_normalizer.update(critic_observations)
        self._velocity_normalizer.update(velocity)
