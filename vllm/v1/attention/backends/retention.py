# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class RetentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_builder_cls() -> type["RetentionMetadataBuilder"]:
        return RetentionMetadataBuilder


@dataclass
class RetentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor

    # Retention keeps 3 types of tensor as cache:
    # 1. state_tensor: [num_blocks, num_kv_heads, state_dim, head_dim]
    # 2. sk_tensor: [num_blocks, num_kv_heads, state_dim], sum of keys, as normalization factor
    # 3. cache_tensor: [num_blocks, num_kv_heads, chunk_size, head_dim + value_dim + gating_dim]

    # When a request of size n comes in, it will have a block of state, sk, and cache tensors.
    # After prefill, the first [n // chunk_size * chunk_size] tokens will be "compressed"
    # into the state and sk; the last n % chunk_size tokens will be added to the cache.
    # As decoding happens, tokens are first added to the cache, and as the number of 
    # tokens grows beyond chunk_size, all tokens are flushed and compressed into the 
    # fixed-size state and sk.
    
    block_idx_tensor: torch.Tensor  # shape: [batch,], mapping from (req id) to the block index 
    


class RetentionMetadataBuilder(
        AttentionMetadataBuilder[RetentionMetadata]):

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> RetentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        if getattr(common_attn_metadata, "seq_lens_cpu", None) is not None:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            query_start_loc_cpu = query_start_loc.to("cpu", non_blocking=True)
            seq_lens_cpu = seq_lens.to("cpu", non_blocking=True)

        # Only one block for each request
        block_idx_tensor = common_attn_metadata.block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.reorder_batch_threshold))
        
        attn_metadata = RetentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_idx_tensor=block_idx_tensor,
        )
        return attn_metadata
            