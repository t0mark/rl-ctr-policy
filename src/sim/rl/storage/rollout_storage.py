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


"""DreamWaQ rollout 저장소."""
import torch


class RolloutStorage:
    """한 rollout의 전이를 GPU에 저장하고 GAE와 미니배치를 제공한다."""

    def __init__(self, num_envs, num_steps, obs_dim, critic_dim, history_dim, action_dim, device):
        """모든 학습 필드의 저장 공간을 할당한다."""
        shapes = {
            "obs": (obs_dim,), "history": (history_dim,), "critic": (critic_dim,), "velocity": (3,),
            "next_obs": (obs_dim,), "actions": (action_dim,), "mean": (action_dim,), "std": (action_dim,),
            "log_prob": (), "values": (), "rewards": (), "dones": (), "use_estimate": (),
        }
        self._data = {key: torch.zeros(num_steps, num_envs, *shape, device=device) for key, shape in shapes.items()}
        self._returns = torch.zeros(num_steps, num_envs, device=device)
        self._advantages = torch.zeros_like(self._returns)
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._step = 0

    def add(self, transition):
        """한 시점의 전이를 저장한다."""
        for key, value in transition.items():
            self._data[key][self._step].copy_(value)
        self._step += 1

    def compute_returns(self, last_values, gamma, lam):
        """rollout 끝 가치로 bootstrap해 GAE advantage와 return을 계산하고 advantage를 정규화한다."""
        advantage = torch.zeros_like(last_values)
        next_values = last_values
        for step in reversed(range(self._num_steps)):
            not_done = 1.0 - self._data["dones"][step]
            values = self._data["values"][step]
            delta = self._data["rewards"][step] + gamma * not_done * next_values - values
            advantage = delta + gamma * lam * not_done * advantage
            self._advantages[step] = advantage
            self._returns[step] = advantage + values
            next_values = values
        self._advantages.sub_(self._advantages.mean()).div_(self._advantages.std().clamp_min(1e-8))

    def batches(self, num_mini_batches, epochs):
        """epoch마다 표본을 섞어 미니배치를 제공한다."""
        total = self._num_steps * self._num_envs
        flat = {key: value.flatten(0, 1) for key, value in self._data.items()}
        flat["returns"] = self._returns.flatten()
        flat["advantages"] = self._advantages.flatten()
        for _ in range(epochs):
            for ids in torch.randperm(total, device=self._returns.device).chunk(num_mini_batches):
                yield {key: value[ids] for key, value in flat.items()}

    def normalization_data(self):
        """정규화 통계 갱신에 쓸 현재 관측과 특권 관측을 반환한다."""
        return self._data["obs"].flatten(0, 1), self._data["critic"].flatten(0, 1)

    def clear(self):
        """다음 rollout 수집을 시작한다."""
        self._step = 0
