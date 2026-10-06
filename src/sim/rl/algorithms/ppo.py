# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import torch
from torch import nn

# 최적화 반복: rollout 하나를 나누는 미니배치 수와 반복 epoch 수.
NUM_LEARNING_EPOCHS = 5
NUM_MINI_BATCHES = 4

# 학습률: 초기값과 KL 기반 adaptive 조정의 목표 KL.
LEARNING_RATE = 1.0e-3
LEARNING_RATE_SCHEDULE = "adaptive"
DESIRED_KL = 0.01

# GAE 할인율과 λ 계수.
GAMMA = 0.99
GAE_LAMBDA = 0.95

# PPO 손실: clip 범위, 가치 손실 clip 사용 여부, 가치·엔트로피 가중치, gradient norm 한계.
CLIP_PARAM = 0.2
USE_CLIPPED_VALUE_LOSS = True
VALUE_LOSS_COEF = 1.0
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0

# CENet 손실 가중치: 속도 MSE, 다음 관측 복원 MSE, latent KL(β).
VELOCITY_LOSS_COEF = 1.0
PREDICTION_LOSS_COEF = 1.0
# 복원 손실은 관측 차원 평균이고 KL은 잠재 차원 합이므로, 두 항의 크기를 맞추는 배율이다.
BETA = 0.02


class PPO:
    """저장된 rollout으로 clipped surrogate·가치·엔트로피·추정기 손실을 최적화한다."""

    def __init__(self, actor_critic):
        """최적화 대상 모델과 옵티마이저를 저장한다."""
        # 정책·가치·추정기 파라미터 전체를 하나의 Adam으로 최적화한다.
        self._actor_critic = actor_critic
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    def update(self, storage):
        """rollout 전체를 epoch·미니배치 단위로 반복 학습하고 손실 평균을 반환한다."""
        totals = {}
        count = 0
        for batch in storage.batches(NUM_MINI_BATCHES, NUM_LEARNING_EPOCHS):
            # 모델 출력: rollout과 같은 입력 조건으로 현재 정책 분포·가치·추정기 손실을 계산한다.
            policy, velocity, prediction, latent_kl = self._actor_critic.training_terms(
                batch["obs"], batch["history"], batch["velocity"], batch["use_estimate"],
                batch["next_obs"], 1.0 - batch["dones"])
            log_prob = policy.log_prob(batch["actions"]).sum(-1)
            values = self._actor_critic.evaluate(batch["critic"])

            # 학습률 조정: rollout 정책과 현재 정책의 KL이 목표 범위를 벗어나면 학습률을 바꾼다.
            with torch.no_grad():
                kl = (torch.log(policy.scale / batch["std"])
                      + (batch["std"].square() + (batch["mean"] - policy.mean).square())
                      / (2 * policy.scale.square()) - 0.5).sum(-1).mean()
                if LEARNING_RATE_SCHEDULE == "adaptive":
                    rate = self._optimizer.param_groups[0]["lr"]
                    if kl > 2 * DESIRED_KL:
                        rate = max(1e-5, rate / 1.5)
                    elif 0 < kl < DESIRED_KL / 2:
                        rate = min(1e-2, rate * 1.5)
                    for group in self._optimizer.param_groups:
                        group["lr"] = rate

            # 전체 손실: clipped surrogate·clipped 가치·엔트로피 PPO 손실에 추정기 손실을 더한다.
            ratio = (log_prob - batch["log_prob"]).exp()
            surrogate = torch.maximum(
                -batch["advantages"] * ratio,
                -batch["advantages"] * ratio.clamp(1-CLIP_PARAM, 1+CLIP_PARAM)).mean()
            value_error = (values - batch["returns"]).square()
            if USE_CLIPPED_VALUE_LOSS:
                clipped = batch["values"] + (values - batch["values"]).clamp(-CLIP_PARAM, CLIP_PARAM)
                value_error = torch.maximum(value_error, (clipped - batch["returns"]).square())
            value_loss = value_error.mean()
            entropy = policy.entropy().sum(-1).mean()
            loss = (surrogate + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
                    + VELOCITY_LOSS_COEF * velocity + PREDICTION_LOSS_COEF * prediction
                    + BETA * latent_kl)

            # 파라미터 갱신: gradient norm을 제한해 한 번 갱신하고 지표를 GPU 텐서로 누적한다.
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._actor_critic.parameters(), MAX_GRAD_NORM)
            self._optimizer.step()
            metrics = {"surrogate": surrogate, "value": value_loss, "velocity": velocity,
                       "prediction": prediction, "latent_kl": latent_kl, "policy_kl": kl,
                       "entropy": entropy, "gradient_norm": grad_norm}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.detach()
            count += 1

        # 누적한 지표를 업데이트 횟수로 평균해 한 번에 CPU로 옮긴다.
        result = {key: (value / count).item() for key, value in totals.items()}
        result["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        return result

    def state_dict(self):
        """학습 재개에 필요한 옵티마이저 상태를 반환한다."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        """저장된 옵티마이저 상태를 복원한다."""
        self._optimizer.load_state_dict(state)
