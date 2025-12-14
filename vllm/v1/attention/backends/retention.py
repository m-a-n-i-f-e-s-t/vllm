# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
    AttentionCGSupport,
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

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)
        
        # Pre-allocate tensors for CUDA graph capture
        # These must be reused at the same memory addresses
        self.compilation_config = vllm_config.compilation_config
        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.max_cudagraph_capture_size = self.vllm_config.compilation_config.max_cudagraph_capture_size
        
        # Pre-allocate metadata tensors for CUDA graph compatibility
        # +1 for cumulative sum tensors
        self._cache_lens = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.int32,
            device=device,
        )
        self._cu_cache_lens = torch.empty(
            (self.max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self._last_memorized_blk_idx = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.int32,
            device=device,
        )
        self._cu_seqlens_padded_q = torch.empty(
            (self.max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self._num_computed_tokens = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.int32,
            device=device,
        )

    def compute_prefix_caching_metadata(
        self, 
        common_attn_metadata: CommonAttentionMetadata, 
        mamba_block_size: int,
        use_cuda_graph_tensors: bool = False,
    ):
        device = common_attn_metadata.query_start_loc.device
        num_reqs = common_attn_metadata.num_reqs
        query_len = common_attn_metadata.query_start_loc[1:] - common_attn_metadata.query_start_loc[:-1]
        self._num_computed_tokens[:num_reqs].copy_(common_attn_metadata.num_computed_tokens_cpu[:num_reqs])
        self._num_computed_tokens[num_reqs:].fill_(0)
        num_computed_tokens = self._num_computed_tokens[:num_reqs]
        
        if use_cuda_graph_tensors:
            # Use pre-allocated tensors for CUDA graph compatibility
            # Slice to padded size for consistent tensor shapes
            cache_lens = self._cache_lens[:num_reqs]
            cu_cache_lens = self._cu_cache_lens[:num_reqs + 1]
            last_memorized_blks = self._last_memorized_blk_idx[:num_reqs]
            cu_seqlens_padded_q = self._cu_seqlens_padded_q[:num_reqs + 1]
            num_computed_tokens = self._num_computed_tokens[:num_reqs]
            
            # Compute values on the fly and copy into pre-allocated tensors
            # Make sure to match dtype (int32)
            cache_lens.copy_(
                (num_computed_tokens % mamba_block_size).to(torch.int32), 
                non_blocking=True
            )
            cu_cache_lens[0] = 0
            cu_cache_lens[1:num_reqs + 1].copy_(
                torch.cumsum(cache_lens, dim=0), 
                non_blocking=True
            )
            last_memorized_blks.copy_(
                (num_computed_tokens // mamba_block_size - 1).to(torch.int32), 
                non_blocking=True
            )
            
            non_memorized_lens = cache_lens + query_len.to(torch.int32)
            padded_lens = ((non_memorized_lens + mamba_block_size - 1) // mamba_block_size) * mamba_block_size
            cu_seqlens_padded_q[0] = 0
            cu_seqlens_padded_q[1:].copy_(
                torch.cumsum(padded_lens, dim=0), 
                non_blocking=True
            )

            if self.max_num_seqs > num_reqs:
                self._cache_lens[num_reqs:].fill_(0)
                self._last_memorized_blk_idx[num_reqs:].fill_(-1)
                self._cu_cache_lens[num_reqs + 1:].fill_(cu_cache_lens[num_reqs])
                self._cu_seqlens_padded_q[num_reqs + 1:].fill_(cu_seqlens_padded_q[num_reqs])
            
            return cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q, num_computed_tokens
        else:
            # Create new tensors (for non-CUDA graph path)
            cache_lens = num_computed_tokens % mamba_block_size
            cu_cache_lens = F.pad(torch.cumsum(cache_lens, dim=0), (1, 0), value=0)
            last_memorized_blks = num_computed_tokens // mamba_block_size - 1
            non_memorized_lens = cache_lens + query_len
            padded_lens = ((non_memorized_lens + mamba_block_size - 1) // mamba_block_size) * mamba_block_size
            cu_seqlens_padded_q = F.pad(torch.cumsum(padded_lens, dim=0), (1, 0), value=0)
            return cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q, num_computed_tokens

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RetentionMetadata:
        """
        Build attention metadata for CUDA graph capture.
        Currently, only decode is supported for full cudagraphs with Retention.
        """
        m = common_attn_metadata
        assert m.num_reqs == m.num_actual_tokens, (
            "Retention only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )
        m.max_query_len = 1  # decode-only
        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RetentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        
        if not self.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Prefix caching needs to be enabled for retention")

        block_table_tensor = common_attn_metadata.block_table_tensor # [#reqs, #max blocks]
        mamba_block_size = self.kv_cache_spec.block_size

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        # Default: use tensors from common_attn_metadata
        cu_seqlens_q = common_attn_metadata.query_start_loc
        
        # Check if we have prefills - if so, use the non-CUDA graph path
        if num_prefills > 0:
            # Prefill path: create new tensors
            cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q, num_computed_tokens = (
                self.compute_prefix_caching_metadata(
                    common_attn_metadata, 
                    mamba_block_size,
                    use_cuda_graph_tensors=False,
                )
            )
        elif (
            num_decodes > 0
            and num_decodes <= self.max_cudagraph_capture_size
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q, num_computed_tokens = (
                self.compute_prefix_caching_metadata(
                    common_attn_metadata, 
                    mamba_block_size,
                    use_cuda_graph_tensors=True,
                )
            )
        else:
            # Decode path without CUDA graphs: create new tensors
            cache_lens, cu_cache_lens, last_memorized_blks, cu_seqlens_padded_q, num_computed_tokens = (
                self.compute_prefix_caching_metadata(
                    common_attn_metadata, 
                    mamba_block_size,
                    use_cuda_graph_tensors=False,
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_padded_q=cu_seqlens_padded_q,
            seq_lens=seq_lens,
            cache_lens=cache_lens,
            cu_cache_lens=cu_cache_lens,
            last_memorized_blk_idx=last_memorized_blks,
        )
