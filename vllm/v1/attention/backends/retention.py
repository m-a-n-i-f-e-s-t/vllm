# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class RetentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["RetentionMetadataBuilder"]:
        return RetentionMetadataBuilder


@dataclass
class RetentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_computed_tokens: int
    num_actual_tokens: int
    max_query_len: int

    # metadata tensors
    block_table: torch.Tensor # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor # [num_reqs + 1]
    cu_seqlens_padded_q: torch.Tensor # [num_reqs + 1]
    seq_lens: torch.Tensor # [num_reqs]
    cache_lens: torch.Tensor # [num_reqs]
    cu_cache_lens: torch.Tensor # [num_reqs + 1]
    last_memorized_blk_idx: torch.Tensor # [num_reqs]


class RetentionMetadataBuilder(AttentionMetadataBuilder[RetentionMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def compute_prefix_caching_metadata(self, common_attn_metadata: CommonAttentionMetadata, mamba_block_size: int):
        device = common_attn_metadata.query_start_loc.device
        query_len = common_attn_metadata.query_start_loc[1:] - common_attn_metadata.query_start_loc[:-1]
        num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(device)
        cache_lens = num_computed_tokens % mamba_block_size
        cu_cache_lens = F.pad(torch.cumsum(cache_lens, dim=0), (1, 0), value=0)
        last_memorized_blks = num_computed_tokens // mamba_block_size
        non_memorized_lens = cache_lens + query_len
        # for query state, we artificially pad every request to block size to make kernel logic simpler
        padded_lens = ((non_memorized_lens + mamba_block_size - 1) // mamba_block_size) * mamba_block_size
        cu_seqlens_padded_q = F.pad(torch.cumsum(padded_lens, dim=0), (1, 0), value=0)
        return cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RetentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        
        if self.vllm_config.cache_config.enable_prefix_caching:
            block_table_tensor = common_attn_metadata.block_table_tensor # [#reqs, #max blocks]
            mamba_block_size = self.kv_cache_spec.block_size
            num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(
                self.device
            )
            cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q = self.compute_prefix_caching_metadata(common_attn_metadata, mamba_block_size)
        else:
            raise ValueError("Prefix caching is needs to be enabled for retention")
            # Always return just a single block per each request
            block_table_tensor = common_attn_metadata.block_table_tensor[:, 0] # [#reqs, 1]
            block_idx_last_scheduled_token = None
            block_idx_last_computed_token = None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        return RetentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_computed_tokens=num_computed_tokens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            block_table=block_table_tensor,
            cu_seqlens_q=common_attn_metadata.query_start_loc,
            cu_seqlens_padded_q=cu_seqlens_padded_q,
            seq_lens=seq_lens,
            cache_lens=cache_lens,
            cu_cache_lens=cu_cache_lens,
            last_memorized_blk_idx=last_memorized_blks,
        )
