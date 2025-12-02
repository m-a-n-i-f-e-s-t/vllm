# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.retention import RetentionMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from typing import TYPE_CHECKING

import torch
from transformers.configuration_utils import PretrainedConfig

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
from vllm.model_executor.layers.mamba.ops.retention import power_retention_varlen
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.retention import RetentionMetadata
from vllm.v1.kv_cache_interface import KVCacheSpec, RetentionSpec
from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


@CustomOp.register("retention")
class Retention(MambaBase, CustomOp):
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
        config: PretrainedConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.chunk_size = self.cache_config.mamba_block_size

        self.d_tile = config.to_dict().get("d_tile", 16)
        self.power = config.to_dict().get("p", 2)
        self.state_dim = self.sympow_dim(self.head_dim, self.power, self.d_tile)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),) * 5  # 5 caches: memory, ks, k_cache, v_cache, g_cache

    @property
    def mamba_type(self) -> str:
        return "retention"

    @classmethod
    def sympow_dim(cls, d, power, d_tile=1):
        if d_tile == 1:
            return math.comb(d + power - 1, power)
        return cls.sympow_dim(d // d_tile, power) * (d_tile**power)

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.retention import RetentionBackend

        return RetentionBackend
    
    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.retention_state_dtype(
            self.config.torch_dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.retention_state_shape(
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in ["qwen3_next"]
        ):
            raise NotImplementedError(
                "Mamba with speculative decoding is not supported yet."
            )
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return RetentionSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor
    ):
        output = torch.empty_like(query)
        torch.ops.vllm.retention(
            query,
            key,
            value,
            gate,
            output,
            self.prefix,
        )
        return output

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        prefix_caching_enabled = self.cache_config.enable_prefix_caching

        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, RetentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]

            memory = self_kv_cache[0]
            ks = self_kv_cache[1]
            k_cache = self_kv_cache[2]
            v_cache = self_kv_cache[3]
            g_cache = self_kv_cache[4]
            block_table = attn_metadata.block_table
            cu_seqlens_q = attn_metadata.cu_seqlens_q
            cu_seqlens_padded_q = attn_metadata.cu_seqlens_padded_q
            seq_lens = attn_metadata.seq_lens
            cache_lens = attn_metadata.cache_lens
            cu_cache_lens = attn_metadata.cu_cache_lens
            last_memorized_blk_idx = attn_metadata.last_memorized_blk_idx
        
        else: # profile run
            return output

        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        gate = gate.view(-1, self.num_kv_heads)
        output = output.view(-1, self.num_heads, self.head_dim)
        has_prefill = attn_metadata.num_prefills > 0
        
        # Call kernel
        power_retention_varlen(
            output,
            query,
            key,
            value,
            gate,
            memory,
            ks,
            k_cache,
            v_cache,
            g_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            cache_lens,
            cu_cache_lens,
            last_memorized_blk_idx,
            cu_seqlens_padded_q,
            self.scale,
            self.chunk_size,
            self.d_tile,
            self.power,
            has_prefill,
        )


def retention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(query=query, key=key, value=value, gate=gate, output=output)


def retention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="retention",
    op_func=retention,
    mutates_args=["output"],
    fake_impl=retention_fake,
)