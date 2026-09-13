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


"""명시적인 시점과 종료 구분을 가진 DreamWaQ rollout 저장소."""
import torch


class RolloutStorage:
    """전이를 복사해 보존하고 GAE 및 미니배치를 제공한다."""

    def __init__(self, num_envs, num_steps, obs_dim, critic_dim, history_dim, action_dim, device):
        """모든 학습 필드의 GPU 저장 공간을 할당한다."""
        shapes = {
            "obs": (obs_dim,), "critic": (critic_dim,), "history": (history_dim,),
            "velocity": (3,), "next_obs": (obs_dim,), "actions": (action_dim,),
            "mean": (action_dim,), "std": (action_dim,),
            "log_prob": (), "values": (), "rewards": (), "next_values": (),
            "terminated": (), "truncated": (), "prediction_valid": (), "use_estimate": (),
        }
        boolean = {"terminated", "truncated", "prediction_valid", "use_estimate"}
        self._data = {key: torch.zeros(num_steps, num_envs, *shape, device=device,
                                      dtype=torch.bool if key in boolean else torch.float32)
                      for key, shape in shapes.items()}
        self._returns = torch.zeros(num_steps, num_envs, device=device)
        self._advantages = torch.zeros_like(self._returns)
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._step = 0

    def add(self, transition):
        """한 시점의 모든 필드를 스냅샷으로 복사한다."""
        if self._step >= self._num_steps or transition.keys() != self._data.keys():
            raise ValueError("rollout 용량 또는 전이 필드가 맞지 않습니다.")
        for key, value in transition.items():
            self._data[key][self._step].copy_(value)
        self._step += 1

    def compute_returns(self, gamma, lam):
        """timeout은 terminal value로 bootstrap하고 episode 간 GAE는 끊는다."""
        if self._step != self._num_steps:
            raise ValueError("완료되지 않은 rollout입니다.")
        advantage = torch.zeros(self._num_envs, device=self._returns.device)
        for step in reversed(range(self._num_steps)):
            terminated = self._data["terminated"][step]
            done = terminated | self._data["truncated"][step]
            delta = (self._data["rewards"][step] + gamma * (~terminated)
                     * self._data["next_values"][step] - self._data["values"][step])
            advantage = delta + gamma * lam * (~done) * advantage
            self._advantages[step] = advantage
            self._returns[step] = advantage + self._data["values"][step]
        self._advantages.sub_(self._advantages.mean()).div_(
            self._advantages.std(unbiased=False).clamp_min(1e-8))

    def batches(self, num_mini_batches, epochs):
        """매 epoch 재섞으며 모든 표본을 정확히 한 번씩 제공한다."""
        total = self._num_steps * self._num_envs
        if total % num_mini_batches or num_mini_batches > total:
            raise ValueError("rollout 크기는 mini-batch 수로 나누어져야 합니다.")
        flat = {key: value.flatten(0, 1) for key, value in self._data.items()}
        flat["returns"] = self._returns.flatten()
        flat["advantages"] = self._advantages.flatten()
        for _ in range(epochs):
            indices = torch.randperm(total, device=self._returns.device)
            for ids in indices.chunk(num_mini_batches):
                yield {key: value[ids] for key, value in flat.items()}

    def normalization_data(self):
        """중복 history 대신 현재 시점 표본으로 통계를 갱신한다."""
        return tuple(self._data[key].flatten(0, 1) for key in ("obs", "critic"))

    def clear(self):
        """다음 rollout 수집을 시작한다."""
        self._step = 0
