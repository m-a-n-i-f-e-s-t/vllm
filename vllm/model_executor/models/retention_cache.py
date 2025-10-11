# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class RetentionCacheParams:
    state_tensor: torch.Tensor = torch.Tensor()
    sk_tensor: torch.Tensor = torch.Tensor()
    cache_tensor: torch.Tensor = torch.Tensor()
    block_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return RetentionCacheParams(self.state_tensor[layer_idx, ...],
                                  self.sk_tensor[layer_idx, ...],
                                  self.cache_tensor[layer_idx, ...],
                                  self.block_indices_tensor)


class RetentionCacheManager(ConstantSizeCache):

    def __init__(self, dtype, state_shape, sk_shape, cache_shape):
        super().__init__(cache_shape[1])  # max_batch_size is cache_shape[1]
        self._state_tensor = torch.zeros(size=state_shape,
                                         dtype=dtype,
                                         device="cuda")
        self._sk_tensor = torch.zeros(size=sk_shape,
                                      dtype=torch.float32,
                                      device="cuda")
        self._cache_tensor = torch.zeros(size=cache_shape,
                                         dtype=dtype,
                                         device="cuda")

    @property
    def cache(self):
        return self._state_tensor, self._sk_tensor, self._cache_tensor

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)
