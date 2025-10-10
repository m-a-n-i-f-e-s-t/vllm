# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from typing import TYPE_CHECKING
from enum import Enum

import torch
import torch.distributed
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm import envs
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.v1.attention.backends.retention import RetentionMetadata

from vllm.model_executor.models.retention_cache import RetentionCacheParams

from retention import power_retention, power_retention_inference

logger = init_logger(__name__)

class RetentionBackend(Enum):

    triton = "triton"
    vidrial = "vidrial"


class Retention(nn.Module, MambaBase):

    @property
    def mamba_type(self) -> str:
        return "retention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.retention import (
            RetentionBackend)
        return RetentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.retention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateShapeCalculator.retention_state_shape(
            num_kv_heads=self.num_kv_heads,
            state_dim=self.state_dim,
            head_dim=self.head_dim,
            value_dim=self.head_dim,
            gating_dim=self.gating_dim,
            chunk_size=self.chunk_size,
        )

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        p: int,
        num_kv_heads: Optional[int] = None,
        chunk_size: Optional[int] = 1024,
        switch_over_seq_len: Optional[int] = 2048,
        bias: Optional[bool] = None,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        retention_backend: Optional[str] = None,
        layer_idx: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.bias = bias
        self.chunk_size = chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        # TODO(sean): enable vidrial backend
        self.retention_backend = retention_backend 
        self.p = p
        self.scale = scale
        self.layer_name = prefix

        assert self.p == 2, "only deg=2 is supported for now"
        assert self.chunk_size is not None, "chunk_size is required"
        assert self.switch_over_seq_len is not None, "switch_over_seq_len is required"

        # Check if chunked prefill is enabled (not supported by Retention)
        vllm_config = get_current_vllm_config()
        if vllm_config.scheduler_config.chunked_prefill_enabled:
            raise ValueError(
                "Chunked prefill is not supported by the Retention layer. "
                "Please disable chunked prefill by setting "
                "--disable-chunked-prefill when launching vLLM, or ensure "
                "--max-num-batched-tokens is large enough to avoid chunking."
            )


    def _prefill(self, query, key, gate, value, state_tensor, sk_tensor, block_idx_tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[int]]:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        assert num_prefills > 0
        assert attn_metadata is not None

        max_prefill_len = 0
        qs, ks, vs, gs = [], [], [], []
        seq_lens, block_ids = [], []
        for prefill_idx in range(num_prefills):
            if prefill_idx >= len(attn_metadata.query_start_loc) or prefill_idx >= len(block_idx_tensor):
                break
            offset = attn_metadata.num_decode_tokens if envs.VLLM_USE_V1 else 0
            q_start = attn_metadata.query_start_loc[offset + prefill_idx]
            q_end = attn_metadata.query_start_loc[offset + prefill_idx + 1]
            seq_len = attn_metadata.seq_lens[offset + prefill_idx]
            context_len = seq_len - (q_end - q_start)
            if context_len > 0:
                raise NotImplementedError("Chunked prefill is not supported by the Retention layer.")
            max_prefill_len = max(max_prefill_len, q_end - q_start)
            qs.append(query[q_start:q_end])
            ks.append(key[q_start:q_end])
            vs.append(value[q_start:q_end])
            gs.append(gate[q_start:q_end])
            block_ids.append(block_idx_tensor[offset + prefill_idx])
            seq_lens.append(q_end - q_start)

        for i in range(len(qs)):
            seq_len = qs[i].shape[1]
            qs[i] = F.pad(qs[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            ks[i] = F.pad(ks[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            vs[i] = F.pad(vs[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            gs[i] = F.pad(gs[i], (0, 0, 0, max_prefill_len - seq_len))

        q = torch.stack(qs, dim=0) # [num_prefills, max_prefill_len, num_heads, head_dim]
        k = torch.stack(ks, dim=0) # [num_prefills, max_prefill_len, num_kv_heads, head_dim]
        v = torch.stack(vs, dim=0) # [num_prefills, max_prefill_len, num_kv_heads, value_dim]
        g = torch.stack(gs, dim=0) # [num_prefills, max_prefill_len, num_kv_heads]
        out, final_state, final_sum_of_keys = power_retention(
            q, k, v, log_G=g,
            initial_state=None,
            chunk_size=self.chunk_size, 
            deg=self.p,
            scale=self.scale,
            switch_over_seq_len=self.switch_over_seq_len, return_final_state=True)

        output = []
        for i in range(len(out)):
            output.append(out.narrow(0, i, 1).squeeze(0).narrow(0, 0, seq_lens[i]))

        # update state and sk
        for prefill_idx in range(num_prefills):
            block_id = block_ids[prefill_idx]
            offset = attn_metadata.num_decode_tokens if envs.VLLM_USE_V1 else 0
            seq_len = attn_metadata.seq_lens[offset + prefill_idx]
            if seq_len >= self.chunk_size: # update state and sk
                state_tensor.narrow(0, block_id, 1).copy_(final_state.narrow(0, prefill_idx, 1))
                sk_tensor.narrow(0, block_id, 1).copy_(final_sum_of_keys.narrow(0, prefill_idx, 1))

        return output
        
    def _decode(self, query, key_cache, value_cache, gate_cache, state_tensor, sk_tensor, block_idx_tensor) -> list[torch.Tensor]:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert attn_metadata is not None
        num_decodes = getattr(attn_metadata, "num_decode_tokens", 0)
        assert num_decodes > 0

        if not envs.VLLM_USE_V1: # v1 and v0 packs decode prefill and decode in different order
            q = query[attn_metadata.num_prefill_tokens:].unsqueeze(1).contiguous() # [num_decodes, 1, num_heads, head_dim]
        else:
            q = query[:num_decodes].unsqueeze(1).contiguous()
        
        # Because each request potentially has a different number of tokens in kv cache, we need to run a for loop.
        output = []
        for i in range(len(num_decodes)):
            block_id = block_idx_tensor[i]
            key = key_cache[i].unsqueeze(0)
            value = value_cache[i].unsqueeze(0)
            gate = gate_cache[i].unsqueeze(0)
            state = state_tensor[block_id].unsqueeze(0)
            sum_of_keys = sk_tensor[block_id].unsqueeze(0)
            out, final_state, final_sum_of_keys = power_retention_inference(
                q, key, value, log_G=gate,
                initial_state=state,
                sum_of_keys=sum_of_keys,
                deg=self.p,
                scale=self.scale,
                # use chunk_size for switch_over_seq_len
                switch_over_seq_len=self.chunk_size)
            output.append(out.squeeze(0))

            # update state and sk
            key_len = key.shape[1]
            if key_len >= self.chunk_size:
                state_tensor[block_id].copy_(final_state.unsqueeze(0))
                sk_tensor[block_id].copy_(final_sum_of_keys.unsqueeze(0))
            
        return output


    def _get_state_and_cache(self):
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        block_idx_tensor = attn_metadata.block_idx_tensor
        state_tensor, sk_tensor, cache_tensor = self.kv_cache[forward_context.virtual_engine]

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        if num_prefills > 0:
            num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
            for prefill_idx in range(num_prefills):
                q_start = attn_metadata.query_start_loc[num_decode_tokens + prefill_idx]
                q_end = attn_metadata.query_start_loc[num_decode_tokens + prefill_idx + 1]
                query_len = q_end - q_start
                context_len = attn_metadata.seq_lens[num_decode_tokens + prefill_idx] - query_len
                if context_len == 0:
                    block_to_clear = block_idx_tensor[num_decode_tokens + prefill_idx]
                    state_tensor[block_to_clear, ...] = 0
                    sk_tensor[block_to_clear, ...] = 0
                    cache_tensor[block_to_clear, ...] = 0

        return state_tensor, sk_tensor, cache_tensor, block_idx_tensor

    
    def _update_cache(self, key, gate, value, cache_tensor, block_ids) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Retention keeps a short token cache in addition to the state and sum of keys. This function adds new tokens to the cache. Note that it may clear the cache for requests whose cache already grow past chunk_size.
        Args:
            key: [num_tokens, num_kv_heads, head_dim]
            gate: [num_tokens, num_kv_heads]
            value: [num_tokens, num_kv_heads, value_dim]
            cache_tensor: [num_blocks, chunk_size, num_kv_heads, head_dim + value_dim + gating_dim]
            block_ids: [num_reqs], mapping from (req id) to the block index for each request

        Returns:
            key_cache: list of tensors of shape [key_len, num_kv_heads, head_dim] for each request, note that key_len can vary across requests
            value_cache: list of tensors of shape [value_len, num_kv_heads, value_dim]
            gate_cache: list of tensors of shape [gate_len, num_kv_heads]
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata: RetentionMetadata = forward_context.attn_metadata
        assert attn_metadata is not None

        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        key_cache, value_cache, gate_cache = [], [], []

        for i in range(len(block_ids)):
            block_id = block_ids[i]
            start = query_start_loc[i]
            end = query_start_loc[i + 1]
            k = key[start:end] # [query_len, num_kv_heads, head_dim]
            v = value[start:end] # [query_len, num_kv_heads, value_dim]
            g = gate[start:end].unsqueeze(-1) # [query_len, num_kv_heads, 1]
            head_dim = k.shape[-1]
            value_dim = v.shape[-1]
            gating_dim = g.shape[-1]
            query_len = end - start
            seq_len = seq_lens[i] # computed tokens + newly scheduled tokens
            computed_len = seq_len - query_len
            already_retained = computed_len // self.chunk_size * self.chunk_size
            non_retained = seq_len - already_retained
            already_cached = computed_len % self.chunk_size
            if non_retained > self.chunk_size: # take out existing cache, clear and add trailing tokens
                existing_cache = cache_tensor[block_id].narrow(0, 0, already_cached)
                old_k, old_v, old_g = existing_cache.narrow(0, 0, head_dim), existing_cache.narrow(0, head_dim, value_dim), existing_cache.narrow(0, head_dim + value_dim, gating_dim)
                total_k = torch.cat([old_k, k], dim=0)
                total_v = torch.cat([old_v, v], dim=0)
                total_g = torch.cat([old_g, g], dim=0).squeeze(-1)
                key_cache.append(total_k)
                value_cache.append(total_v)
                gate_cache.append(total_g)
                
                to_cache = non_retained % self.chunk_size
                trailing_k = k.narrow(0, query_len - to_cache, to_cache)
                trailing_v = v.narrow(0, query_len - to_cache, to_cache)
                trailing_g = g.narrow(0, query_len - to_cache, to_cache).squeeze(-1)
                pack = torch.stack([trailing_k, trailing_v, trailing_g], dim=-1)

                # Maybe we can save this call?
                cache_tensor[block_id, ...] = 0
                cache_tensor[block_id].narrow(0, 0, to_cache).copy_(pack)

            else: # simply append
                pack = torch.stack([k, v, g], dim=-1) # [query_len, num_kv_heads, head_dim + value_dim + gating_dim]
                assert already_cached + query_len <= self.chunk_size
                assert already_cached + query_len == non_retained
                cache_tensor[block_id].narrow(0, already_cached, query_len).copy_(pack)

                key_cache.append(cache_tensor[block_id].narrow(0, 0, non_retained))
                value_cache.append(cache_tensor[block_id].narrow(0, 0, non_retained))
                gate_cache.append(cache_tensor[block_id].narrow(0, 0, non_retained))

        return key_cache, value_cache, gate_cache


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        cache_params: Optional[RetentionCacheParams] = None,
    ) -> None:
        """ 
        Relevant tensor objects:

        query: [num_tokens, num_heads, head_dim]
        key:   [num_tokens, num_kv_heads, head_dim]
        gate:  [num_tokens, num_kv_heads]
        value: [num_tokens, num_kv_heads, value_dim]
        output: [num_tokens, num_heads, value_dim]

        cache_params: RetentionCacheParams (only available for v0 path) containing:
            state_tensor: [num_blocks, *state_shape]
            sk_tensor: [num_blocks, *sk_shape]
            cache_tensor: [num_blocks, *cache_shape]

        self.kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] (only available for v1 path)
            state_tensor: [num_blocks, *state_shape]
            sk_tensor: [num_blocks, *sk_shape]
            cache_tensor: [num_blocks, *cache_shape]

        attn_metadata: RetentionMetadata (only available for v1 path) containing:
            query_start_loc: [num_reqs + 1]
                the start location of each request in query Tensor
            seq_lens: [num_reqs]
                the length of each request including both computed tokens
                and newly scheduled tokens
            state_indices: [num_reqs]
                mapping from (req id) to the state index for each request

            Note that in all the 3 above tensors, decoding requests come before prefill requests
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if envs.VLLM_USE_V1 and isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
            assert isinstance(attn_metadata, RetentionMetadata)
            num_actual_tokens = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            num_actual_tokens = query.shape[0]
            assert query.shape[0] == key.shape[0] == gate.shape[0] == value.shape[0], "batch size should match for profile run or first prefill"

        # check if there's new prefill, cleanup state and sk for them, add new tokens to cache
        if envs.VLLM_USE_V1:
            if attn_metadata is not None:
                state_tensor, sk_tensor, cache_tensor, block_idx_tensor = self._get_state_and_cache()
        else:
            assert cache_params is not None
            state_tensor, sk_tensor, cache_tensor = cache_params.state_tensor, cache_params.sk_tensor, cache_params.cache_tensor
            block_idx_tensor = cache_params.block_idx_tensor

        if attn_metadata is None: # profile run
            return

        # add new tokens to cache
        key_cache, value_cache, gate_cache = self._update_cache(key, gate, value, cache_tensor, block_idx_tensor)

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        num_decodes = getattr(attn_metadata, "num_decode_tokens", 0)

        # take care of prefill first, here we want to find the longest prefill sequence and pad the rest to the same length
        # TODO(sean): support chunked prefill
        if num_prefills > 0:
            outputs_prefill = self._prefill(query, key, gate, value, state_tensor, sk_tensor, block_idx_tensor)

        # take care of decode
        if num_decodes > 0:
            outputs_decode = self._decode(query, key_cache, value_cache, gate_cache, state_tensor, sk_tensor, block_idx_tensor)
            
        # Concat outputs
        if envs.VLLM_USE_V1:
            output.copy_(torch.cat(outputs_decode + outputs_prefill, dim=0))
        else:
            output.copy_(torch.cat(outputs_decode + outputs_prefill, dim=0))
