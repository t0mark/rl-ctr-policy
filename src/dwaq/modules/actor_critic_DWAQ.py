"""관측 이력을 직접 입력받는 정책과 현재 속도·다음 관측 추정 네트워크."""
import torch
from torch import nn
from torch.distributions import Normal

from .normalizer import RunningNormalizer

# 논문 Table 6의 은닉층 구성. backbone은 actor/critic, encoder는 추정기에 사용한다.
BACKBONE_HIDDEN_DIMS = [512, 512, 128]
ENCODER_HIDDEN_DIMS = [768, 256, 64]
DECODER_HIDDEN_DIMS = [64, 128]


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
    """정책은 관측 이력과 평균 context를, VAE는 재매개화한 latent를 사용한다."""

    def __init__(self, obs_dim, num_critic_obs, num_actions, actor_history_length,
                 estimator_history_length, cenet_out_dim=19, activation="elu",
                 init_noise_std=1.0, normalization_min_std=0.1):
        """정책·가치·추정 네트워크와 정규화 통계를 생성한다."""
        super().__init__()
        if actor_history_length < 1 or estimator_history_length < 1:
            raise ValueError("관측 이력 길이는 1 이상이어야 합니다.")
        if estimator_history_length > actor_history_length:
            raise ValueError("추정기 이력은 정책 이력의 최근 구간이어야 합니다.")
        if obs_dim < 1 or cenet_out_dim <= 3 or init_noise_std <= 0:
            raise ValueError("관측 차원, context 차원 또는 초기 표준편차가 잘못되었습니다.")
        self._obs_dim = obs_dim
        self._actor_history_length = actor_history_length
        self._estimator_history_length = estimator_history_length
        actor_input_dim = obs_dim * actor_history_length + cenet_out_dim
        encoder_input_dim = obs_dim * estimator_history_length
        self._actor = make_mlp([actor_input_dim, *BACKBONE_HIDDEN_DIMS, num_actions], activation)
        self._critic = make_mlp([num_critic_obs, *BACKBONE_HIDDEN_DIMS, 1], activation)
        self._encoder = nn.Sequential(
            make_mlp([encoder_input_dim, *ENCODER_HIDDEN_DIMS], activation),
            {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]())
        self._velocity_head = nn.Linear(ENCODER_HIDDEN_DIMS[-1], 3)
        self._latent_mean = nn.Linear(ENCODER_HIDDEN_DIMS[-1], cenet_out_dim - 3)
        self._latent_logvar = nn.Linear(ENCODER_HIDDEN_DIMS[-1], cenet_out_dim - 3)
        self._decoder = make_mlp([cenet_out_dim, *DECODER_HIDDEN_DIMS, obs_dim], activation)
        self._log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())
        self._obs_normalizer = RunningNormalizer(obs_dim, normalization_min_std)
        self._critic_normalizer = RunningNormalizer(num_critic_obs, normalization_min_std)
        self._velocity_normalizer = RunningNormalizer(3, normalization_min_std)

    @property
    def actor_history_length(self):
        """정책이 입력받는 관측 프레임 수를 반환한다."""
        return self._actor_history_length

    @property
    def estimator_history_length(self):
        """추정기가 입력받는 관측 프레임 수를 반환한다."""
        return self._estimator_history_length

    def _normalized_frames(self, history):
        """정책 이력을 프레임 단위로 동일 정규화한다."""
        frames = history.reshape(-1, self._actor_history_length, self._obs_dim)
        return self._obs_normalizer(frames)

    def _encode_frames(self, frames):
        """이력의 최근 구간만 추정기에 넣어 속도와 latent 분포를 얻는다."""
        window = frames[:, -self._estimator_history_length:].flatten(1)
        encoded = self._encoder(window)
        return (self._velocity_head(encoded), self._latent_mean(encoded),
                self._latent_logvar(encoded).clamp(-10.0, 10.0))

    def _action_distribution(self, frames, velocity, latent, velocity_target, use_estimate):
        """AdaBoot 선택을 유지한 조건부 action 분포를 만든다."""
        if velocity_target is not None:
            if use_estimate is None:
                raise ValueError("실제 속도 입력에는 AdaBoot 선택 mask가 필요합니다.")
            velocity = torch.where(use_estimate.bool().reshape(-1, 1), velocity,
                                   self._velocity_normalizer(velocity_target))
        features = torch.cat((velocity, latent, frames.flatten(1)), -1)
        mean = self._actor(features)
        std = self._log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def encode(self, history):
        """프레임별 동일 정규화를 적용해 속도와 latent 분포를 반환한다."""
        return self._encode_frames(self._normalized_frames(history))

    def distribution(self, history, velocity_target=None, use_estimate=None):
        """관측 이력만으로 action 분포를 계산한다."""
        frames = self._normalized_frames(history)
        velocity, latent, _ = self._encode_frames(frames)
        return self._action_distribution(frames, velocity, latent, velocity_target, use_estimate)

    def act_inference(self, history):
        """실제 속도를 입력받지 않는 결정적 배포 정책을 실행한다."""
        return self.distribution(history).mean

    def estimate_velocity(self, history):
        """추정 속도를 m/s 단위로 반환한다."""
        return self._velocity_normalizer.inverse(self.encode(history)[0])

    def evaluate(self, critic_observations):
        """특권 관측에서 상태 가치를 계산한다."""
        return self._critic(self._critic_normalizer(critic_observations)).squeeze(-1)

    def training_terms(self, history, velocity_target, use_estimate, next_obs, prediction_valid):
        """한 번의 encoder 전파로 action 분포와 보조 손실을 함께 계산한다.

        정책은 latent 평균을, VAE는 재매개화한 표본을 쓰지만 encoder 몸통 출력은
        동일하므로 미니배치마다 encoder를 두 번 전파하지 않는다.
        """
        frames = self._normalized_frames(history)
        velocity, mean, logvar = self._encode_frames(frames)
        distribution = self._action_distribution(frames, velocity, mean, velocity_target, use_estimate)

        # 다음 관측 예측에는 평균 대신 재매개화한 latent 표본을 사용한다.
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        prediction = self._decoder(torch.cat((velocity, latent), -1))
        velocity_loss = (velocity - self._velocity_normalizer(velocity_target)).square().mean()
        per_sample = (prediction - self._obs_normalizer(next_obs)).square().mean(-1)
        valid = prediction_valid.to(per_sample.dtype)
        prediction_loss = (per_sample * valid).sum() / valid.sum().clamp_min(1)
        kl_loss = (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()
        return distribution, velocity_loss, prediction_loss, kl_loss

    @torch.no_grad()
    def update_normalizers(self, observations, critic_observations, velocity):
        """rollout와 PPO 갱신 사이 경계에서만 통계를 갱신한다."""
        self._obs_normalizer.update(observations)
        self._critic_normalizer.update(critic_observations)
        self._velocity_normalizer.update(velocity)
