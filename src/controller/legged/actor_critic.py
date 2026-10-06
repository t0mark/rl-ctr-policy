"""DreamWaQ 정책(actor)·가치(critic)·context-aided estimator(CENet) 네트워크."""
import torch
from torch import nn
from torch.distributions import Normal

from .normalizer import RunningNormalizer

# DreamWaQ Fig. 1·2의 은닉층 구성.
ACTOR_HIDDEN_DIMS = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [512, 256, 128]
ENCODER_HIDDEN_DIMS = [128, 64]
DECODER_HIDDEN_DIMS = [64, 128]

# CENet이 추정하는 몸통 선속도 v_t와 latent z_t의 차원.
VELOCITY_DIM = 3
LATENT_DIM = 16

# CENet encoder가 입력받는 관측 프레임 수(H).
HISTORY_LENGTH = 5

# 은닉층 활성화 함수, 행동 분포의 초기 표준편차, 관측 정규화의 최소 표준편차.
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0
NORMALIZATION_MIN_STD = 0.1

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


def make_mlp(dimensions, activation):
    """차원 목록으로 마지막 층을 제외한 층마다 활성화 함수를 둔 MLP를 구성한다."""
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(dimensions) - 2:
            layers.append(ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """DreamWaQ 비대칭 actor-critic.

    actor는 현재 관측 o_t와 CENet이 관측 이력 o_t^H로 추정한 속도 v_t·latent z_t를,
    critic은 특권 관측 s_t를 입력받는다. CENet은 β-VAE 구조로 decoder가 z_t에서 o_{t+1}을 복원한다.
    """

    def __init__(self, obs_dim, critic_dim, num_actions):
        """정책·가치·CENet 네트워크와 관측 정규화 통계를 생성한다."""
        super().__init__()
        self._obs_dim = obs_dim
        self._history_length = HISTORY_LENGTH

        # 정책: 현재 관측과 추정 context로 행동 분포를, 가치: 특권 관측으로 상태 가치를 출력한다.
        self._actor = make_mlp([obs_dim + VELOCITY_DIM + LATENT_DIM, *ACTOR_HIDDEN_DIMS, num_actions], ACTIVATION)
        self._critic = make_mlp([critic_dim, *CRITIC_HIDDEN_DIMS, 1], ACTIVATION)
        self._log_std = nn.Parameter(torch.full((num_actions,), float(INIT_NOISE_STD)).log())

        # CENet: encoder는 관측 이력으로 속도와 latent 분포를, decoder는 latent로 다음 관측을 출력한다.
        self._encoder = nn.Sequential(make_mlp([obs_dim * HISTORY_LENGTH, *ENCODER_HIDDEN_DIMS], ACTIVATION), ACTIVATIONS[ACTIVATION]())
        self._velocity_head = nn.Linear(ENCODER_HIDDEN_DIMS[-1], VELOCITY_DIM)
        self._latent_mean = nn.Linear(ENCODER_HIDDEN_DIMS[-1], LATENT_DIM)
        self._latent_logvar = nn.Linear(ENCODER_HIDDEN_DIMS[-1], LATENT_DIM)
        self._decoder = make_mlp([LATENT_DIM, *DECODER_HIDDEN_DIMS, obs_dim], ACTIVATION)

        # 관측·특권 관측 정규화 통계를 모델 버퍼로 보관한다.
        self._obs_normalizer = RunningNormalizer(obs_dim, NORMALIZATION_MIN_STD)
        self._critic_normalizer = RunningNormalizer(critic_dim, NORMALIZATION_MIN_STD)

    @property
    def history_length(self):
        """CENet이 입력받는 관측 프레임 수를 반환한다."""
        return self._history_length

    def encode(self, history):
        """관측 이력을 프레임별로 정규화해 추정 속도와 latent 평균·로그분산을 반환한다."""
        frames = self._obs_normalizer(history.reshape(-1, self._history_length, self._obs_dim))
        encoded = self._encoder(frames.flatten(1))
        return (self._velocity_head(encoded), self._latent_mean(encoded), self._latent_logvar(encoded).clamp(-10.0, 10.0))

    def _action_distribution(self, obs, velocity, latent):
        """현재 관측과 context로 행동 분포를 만든다. context는 PPO 그래디언트에서 분리한다."""
        features = torch.cat((velocity.detach(), latent.detach(), self._obs_normalizer(obs)), -1)
        mean = self._actor(features)
        return Normal(mean, self._log_std.clamp(-5.0, 2.0).exp().expand_as(mean))

    @staticmethod
    def _bootstrap_velocity(velocity, velocity_target, use_estimate):
        """AdaBoot가 실제 속도를 고른 환경은 추정 속도를 실제 속도로 바꾼다."""
        if velocity_target is None:
            return velocity
        return torch.where(use_estimate.bool().unsqueeze(-1), velocity, velocity_target)

    def distribution(self, obs, history, velocity_target=None, use_estimate=None):
        """행동 분포를 계산한다. 학습 rollout에서는 AdaBoot 선택에 따라 실제 속도를 쓴다."""
        velocity, latent, _ = self.encode(history)
        velocity = self._bootstrap_velocity(velocity, velocity_target, use_estimate)
        return self._action_distribution(obs, velocity, latent)

    def act_inference(self, obs, history):
        """특권 정보 없이 현재 관측과 관측 이력만으로 결정적 행동을 반환한다."""
        return self.distribution(obs, history).mean

    def estimate_velocity(self, history):
        """추정 몸통 선속도를 m/s 단위로 반환한다."""
        return self.encode(history)[0]

    def evaluate(self, critic_obs):
        """특권 관측으로 상태 가치를 계산한다."""
        return self._critic(self._critic_normalizer(critic_obs)).squeeze(-1)

    def training_terms(self, obs, history, velocity_target, use_estimate, next_obs, prediction_valid):
        """한 번의 encoder 전파로 행동 분포와 CENet 손실(속도 MSE·다음 관측 복원 MSE·latent KL)을 계산한다."""
        # 행동 분포: rollout과 같은 AdaBoot 선택으로 계산한다.
        velocity, mean, logvar = self.encode(history)
        policy_velocity = self._bootstrap_velocity(velocity, velocity_target, use_estimate)
        distribution = self._action_distribution(obs, policy_velocity, mean)

        # CENet 손실: 속도 MSE, 재매개화 latent로 복원한 다음 관측 MSE(비종료 전이), latent KL.
        velocity_loss = (velocity - velocity_target).square().mean()
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        per_sample = (self._decoder(latent) - self._obs_normalizer(next_obs)).square().mean(-1)
        valid = prediction_valid.to(per_sample.dtype)
        prediction_loss = (per_sample * valid).sum() / valid.sum().clamp_min(1)
        kl_loss = (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()
        return distribution, velocity_loss, prediction_loss, kl_loss

    @torch.no_grad()
    def update_normalizers(self, obs, critic_obs):
        """rollout 표본으로 관측·특권 관측 정규화 통계를 갱신한다."""
        self._obs_normalizer.update(obs)
        self._critic_normalizer.update(critic_obs)
