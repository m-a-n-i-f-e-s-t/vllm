# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.lightning_attn import (
    lightning_attention,
    linear_decode_forward_triton,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


class Retention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "retention"

    @classmethod
    def sympow_dim(cls, d, power, d_tile=1):
        if d_tile == 1:
            return math.comb(d + power - 1, power)
        return cls.sympow_dim(d // d_tile, power) * d_tile**power

    def get_atnn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.retention_attn import RetentionAttentionBackend

        return RetentionAttentionBackend
    
    def get_state_type(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.retention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        assert self.model_config is not None
        hf_config = self.model_config.hf_config.to_dict()
        d_tile = hf_config.get("d_tile", 16)
        power = hf_config.get("power", 2)
        state_dim = self.sympow_dim(self.head_dim, power, d_tile)
        return MambaStateShapeCalculator.retention_state_shape(
            num_heads=self.num_heads,
            tp_size=self.tp_size,
            head_dim=self.head_dim,
            state_dim=state_dim,
        )
