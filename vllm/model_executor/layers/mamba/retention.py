# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from multiprocessing import context
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

from vllm.utils.torch_utils import direct_register_custom_op
from vllm.platforms import current_platform

from retention import power_retention, power_retention_inference
from retention._utils import compute_expanded_dim

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
        prefill_chunk_size: Optional[int] = 64,
        switch_over_seq_len: Optional[int] = 2048,
        chunk_size: Optional[int] = 64,
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
        self.prefill_chunk_size = prefill_chunk_size
        self.chunk_size = chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        # TODO(sean): enable vidrial backend
        self.retention_backend = retention_backend 
        self.p = p
        self.scale = scale
        self.layer_name = prefix

        # Dimensions
        self.head_dim = head_size
        self.state_dim = compute_expanded_dim(head_size, deg=p)
        self.gating_dim = 1

        assert self.p == 2, "only deg=2 is supported for now"
        assert self.chunk_size is not None, "chunk_size is required"
        assert self.switch_over_seq_len is not None, "switch_over_seq_len is required"

        # Check if chunked prefill is enabled and validate chunk size compatibility
        vllm_config = get_current_vllm_config()
        if vllm_config.scheduler_config.chunked_prefill_enabled:
            vllm_chunk_size = vllm_config.scheduler_config.max_num_batched_tokens
            
            # Ensure retention chunk_size is a multiple of prefill chunk size
            if vllm_chunk_size % self.prefill_chunk_size != 0:
                raise ValueError(
                    f"When chunked prefill is enabled, the retention layer's "
                    f"prefill_chunk_size ({self.prefill_chunk_size}) must divide "
                    f"max_num_batched_tokens ({vllm_chunk_size}). "
                    f"Please adjust either --max-num-batched-tokens or the model's "
                    f"chunk_size configuration. Suggested retention chunk_size values: "
                    f"{[vllm_chunk_size * i for i in range(1, 5)]}"
                )

        if envs.VLLM_USE_V1:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self


    def _get_cpu_tensors(self, attn_metadata, block_idx_tensor=None):
        """Extract CPU versions of metadata tensors once to avoid repeated DtoH copies.
        
        Returns:
            tuple: (seq_lens_cpu, query_start_loc_cpu, block_ids_cpu)
        """
        # Handle seq_lens
        if hasattr(attn_metadata, 'seq_lens_cpu'):
            # V1 path: use pre-computed CPU tensor
            seq_lens_cpu = attn_metadata.seq_lens_cpu
            if not isinstance(seq_lens_cpu, (list, tuple)):
                seq_lens_cpu = seq_lens_cpu.cpu().numpy()
        elif hasattr(attn_metadata, 'seq_lens') and isinstance(attn_metadata.seq_lens, (list, tuple)):
            # V0 path: seq_lens is already a Python list (CPU)
            seq_lens_cpu = attn_metadata.seq_lens
        else:
            # Fallback: convert GPU tensor to CPU
            seq_lens_cpu = attn_metadata.seq_lens.cpu().numpy()
        
        # Handle query_start_loc
        if hasattr(attn_metadata, 'query_start_loc_cpu'):
            # V1 path: use pre-computed CPU tensor
            query_start_loc_cpu = attn_metadata.query_start_loc_cpu
            if not isinstance(query_start_loc_cpu, (list, tuple)):
                query_start_loc_cpu = query_start_loc_cpu.cpu().numpy()
        else:
            # V0 path: convert GPU tensor to CPU once
            query_start_loc_cpu = attn_metadata.query_start_loc.cpu().numpy()
        
        # Handle block_idx_tensor if provided
        block_ids_cpu = None
        if block_idx_tensor is not None:
            if block_idx_tensor.is_cuda:
                block_ids_cpu = block_idx_tensor.cpu().numpy()
            else:
                block_ids_cpu = block_idx_tensor.numpy()
        
        return seq_lens_cpu, query_start_loc_cpu, block_ids_cpu

    def _prefill(self, attn_metadata, query, key, gate, value, state_tensor, sk_tensor, block_idx_tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[int]]:

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        assert num_prefills > 0
        assert attn_metadata is not None

        # Extract CPU versions once to avoid repeated DtoH copies
        seq_lens_cpu, query_start_loc_cpu, block_ids_cpu = self._get_cpu_tensors(
            attn_metadata, block_idx_tensor)

        max_prefill_len = 0
        qs, ks, vs, gs = [], [], [], []
        states, sks = [], []
        context_lens, seq_lens, block_ids = [], [], []
        for prefill_idx in range(num_prefills):
            if prefill_idx >= len(query_start_loc_cpu) - 1 or prefill_idx >= len(block_ids_cpu):
                break
            offset = attn_metadata.num_decode_tokens if envs.VLLM_USE_V1 else 0
            q_start = int(query_start_loc_cpu[offset + prefill_idx])
            q_end = int(query_start_loc_cpu[offset + prefill_idx + 1])
            seq_len = int(seq_lens_cpu[offset + prefill_idx])
            context_len = seq_len - (q_end - q_start)
            context_lens.append(context_len)
            max_prefill_len = max(max_prefill_len, q_end - q_start)
            qs.append(query[q_start:q_end])
            ks.append(key[q_start:q_end])
            vs.append(value[q_start:q_end])
            gs.append(gate[q_start:q_end])
            block_id = int(block_ids_cpu[offset + prefill_idx])
            block_ids.append(block_id)
            seq_lens.append(q_end - q_start)
            states.append(state_tensor[block_id])
            sks.append(sk_tensor[block_id])

        for i in range(len(qs)):
            seq_len = qs[i].shape[0]
            qs[i] = F.pad(qs[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            ks[i] = F.pad(ks[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            vs[i] = F.pad(vs[i], (0, 0, 0, 0, 0, max_prefill_len - seq_len))
            gs[i] = F.pad(gs[i], (0, 0, 0, max_prefill_len - seq_len))

        # If all context_lens are 0, the initial states are all 0
        if all([context_len == 0 for context_len in context_lens]):
            initial_state = None
            initial_sk = None
        else:
            initial_state = torch.stack(states, dim=0) # [num_prefills, num_kv_heads, state_dim, head_dim]
            initial_sk = torch.stack(sks, dim=0) # [num_prefills, num_kv_heads, state_dim]

        q = torch.stack(qs, dim=0) # [num_prefills, max_prefill_len, num_heads, head_dim]
        k = torch.stack(ks, dim=0) # [num_prefills, max_prefill_len, num_kv_heads, head_dim]
        v = torch.stack(vs, dim=0) # [num_prefills, max_prefill_len, num_kv_heads, value_dim]
        g = torch.stack(gs, dim=0) # [num_prefills, max_prefill_len, num_kv_heads]
        out, final_state, final_sum_of_keys = power_retention(
            q, k, v, log_G=g,
            initial_state=initial_state,
            sum_of_keys=initial_sk,
            chunk_size=self.prefill_chunk_size, 
            deg=self.p,
            scale=self.scale,
            switch_over_seq_len=self.switch_over_seq_len, return_final_state=True)

        output = []
        for i in range(len(out)):
            output.append(out.narrow(0, i, 1).squeeze(0).narrow(0, 0, seq_lens[i]))

        # update state and sk
        if final_state is not None:
            for prefill_idx in range(num_prefills):
                block_id = block_ids[prefill_idx]
                offset = attn_metadata.num_decode_tokens if envs.VLLM_USE_V1 else 0
                # Use CPU version - fast access!
                seq_len = int(seq_lens_cpu[offset + prefill_idx])
                if seq_len >= self.chunk_size: # update state and sk
                    state_tensor.narrow(0, block_id, 1).copy_(final_state.narrow(0, prefill_idx, 1))
                    sk_tensor.narrow(0, block_id, 1).copy_(final_sum_of_keys.narrow(0, prefill_idx, 1))

        return output
        
    def _decode(self, attn_metadata, query, key_cache, value_cache, gate_cache, state_tensor, sk_tensor, block_idx_tensor) -> list[torch.Tensor]:
        assert attn_metadata is not None
        num_decodes = getattr(attn_metadata, "num_decode_tokens", 0)
        assert num_decodes > 0

        if not envs.VLLM_USE_V1: # v1 and v0 packs decode prefill and decode in different order
            q = query[attn_metadata.num_prefill_tokens:].unsqueeze(1).contiguous() # [num_decodes, 1, num_heads, head_dim]
        else:
            q = query[:num_decodes].unsqueeze(1).contiguous()

        # Get key lengths once to avoid repeated DtoH copies when accessing .shape[0]
        key_lens = [key.shape[0] for key in key_cache]
        max_key_len = max(key_lens) if key_lens else 0
        
        # Extract CPU version of block_idx_tensor once
        _, _, block_ids_cpu = self._get_cpu_tensors(attn_metadata, block_idx_tensor)
        
        coalesced_keys = torch.zeros(num_decodes, max_key_len, self.num_kv_heads, self.head_dim, device=key_cache[0].device, dtype=key_cache[0].dtype)
        coalesced_values = torch.zeros(num_decodes, max_key_len, self.num_kv_heads, self.head_dim, device=value_cache[0].device, dtype=value_cache[0].dtype)
        coalesced_gates = torch.zeros(num_decodes, max_key_len, self.num_kv_heads, device=gate_cache[0].device, dtype=gate_cache[0].dtype)
        coalesced_state = torch.zeros(num_decodes, self.num_kv_heads, self.state_dim, self.head_dim, device=state_tensor.device, dtype=state_tensor.dtype)
        coalesced_sk = torch.zeros(num_decodes, self.num_kv_heads, self.state_dim, device=sk_tensor.device, dtype=sk_tensor.dtype)
        for i in range(num_decodes):
            key_len = key_lens[i]
            coalesced_keys.narrow(0, i, 1).narrow(1, 0, key_len).copy_(key_cache[i].unsqueeze(0))
            coalesced_values.narrow(0, i, 1).narrow(1, 0, key_len).copy_(value_cache[i].unsqueeze(0))
            coalesced_gates.narrow(0, i, 1).narrow(1, 0, key_len).copy_(gate_cache[i].unsqueeze(0))
            # Use CPU version - fast access!
            block_id = int(block_ids_cpu[i])
            coalesced_state.narrow(0, i, 1).copy_(state_tensor[block_id])
            coalesced_sk.narrow(0, i, 1).copy_(sk_tensor[block_id])

        output, final_state, final_sum_of_keys = power_retention_inference(
            q, coalesced_keys, coalesced_values, log_G=coalesced_gates,
            initial_state=coalesced_state,
            sum_of_keys=coalesced_sk,
            deg=self.p,
            scale=self.scale,
            switch_over_seq_len=self.chunk_size)

        # update state and sk
        for i in range(num_decodes):
            # Use CPU version - fast access!
            block_id = int(block_ids_cpu[i])
            key_len = key_lens[i]
            if key_len >= self.chunk_size:
                state_tensor[block_id].copy_(final_state[i])
                sk_tensor[block_id].copy_(final_sum_of_keys[i])
    
        return list(output.unbind(0))


    def _get_state_and_cache(self, attn_metadata):
        forward_context: ForwardContext = get_forward_context()

        block_idx_tensor = attn_metadata.block_idx_tensor
        state_tensor, sk_tensor, cache_tensor = self.kv_cache[forward_context.virtual_engine]

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        if num_prefills > 0:
            # Extract CPU versions once
            seq_lens_cpu, query_start_loc_cpu, block_ids_cpu = self._get_cpu_tensors(
                attn_metadata, block_idx_tensor)
            
            num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
            for prefill_idx in range(num_prefills):
                # Use CPU versions - fast access!
                q_start = int(query_start_loc_cpu[num_decode_tokens + prefill_idx])
                q_end = int(query_start_loc_cpu[num_decode_tokens + prefill_idx + 1])
                query_len = q_end - q_start
                context_len = int(seq_lens_cpu[num_decode_tokens + prefill_idx]) - query_len
                if context_len == 0:
                    block_to_clear = int(block_ids_cpu[num_decode_tokens + prefill_idx])
                    state_tensor[block_to_clear, ...] = 0
                    sk_tensor[block_to_clear, ...] = 0
                    cache_tensor[block_to_clear, ...] = 0

        return state_tensor, sk_tensor, cache_tensor, block_idx_tensor

    
    def _update_cache(self, attn_metadata, key, gate, value, cache_tensor, block_ids) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Retention keeps a short token cache in addition to the state and sum of keys. This function adds new tokens to the cache. Note that it may clear the cache for requests whose cache already grow past chunk_size.
        Args:
            attn_metadata: RetentionMetadata
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
        assert attn_metadata is not None

        # Extract CPU versions once to avoid repeated DtoH copies
        seq_lens_cpu, query_start_loc_cpu, block_ids_cpu = self._get_cpu_tensors(
            attn_metadata, block_ids)

        key_cache, value_cache, gate_cache = [], [], []

        for i in range(len(block_ids_cpu)):
            # Use CPU versions - fast access!
            block_id = int(block_ids_cpu[i])
            start = int(query_start_loc_cpu[i])
            end = int(query_start_loc_cpu[i + 1])
            k = key[start:end] # [query_len, num_kv_heads, head_dim]
            v = value[start:end] # [query_len, num_kv_heads, value_dim]
            g = gate[start:end].unsqueeze(-1) # [query_len, num_kv_heads, 1]
            head_dim = k.shape[-1]
            value_dim = v.shape[-1]
            gating_dim = g.shape[-1]
            query_len = end - start
            seq_len = int(seq_lens_cpu[i]) # computed tokens + newly scheduled tokens
            computed_len = seq_len - query_len
            already_retained = computed_len // self.chunk_size * self.chunk_size
            non_retained = seq_len - already_retained
            already_cached = computed_len % self.chunk_size
            if non_retained > self.chunk_size: # take out existing cache, clear and add trailing tokens
                existing_cache = cache_tensor[block_id].narrow(0, 0, already_cached)
                # Split along the last dimension into k, v, g
                old_k = existing_cache[..., :head_dim]
                old_v = existing_cache[..., head_dim:head_dim + value_dim]
                old_g = existing_cache[..., head_dim + value_dim:]
                total_k = torch.cat([old_k, k], dim=0)
                total_v = torch.cat([old_v, v], dim=0)
                total_g = torch.cat([old_g, g], dim=0).squeeze(-1)
                key_cache.append(total_k)
                value_cache.append(total_v)
                gate_cache.append(total_g)
                
                to_cache = non_retained % self.chunk_size
                trailing_k = k.narrow(0, query_len - to_cache, to_cache)
                trailing_v = v.narrow(0, query_len - to_cache, to_cache)
                trailing_g = g.narrow(0, query_len - to_cache, to_cache)
                pack = torch.cat([trailing_k, trailing_v, trailing_g], dim=-1)

                # Maybe we can save this call?
                cache_tensor[block_id, ...] = 0
                cache_tensor[block_id].narrow(0, 0, to_cache).copy_(pack)

            else: # simply append
                pack = torch.cat([k, v, g], dim=-1) # [query_len, num_kv_heads, head_dim + value_dim + gating_dim]
                assert already_cached + query_len <= self.chunk_size
                assert already_cached + query_len == non_retained
                cache_tensor[block_id].narrow(0, already_cached, query_len).copy_(pack)

                cached = cache_tensor[block_id].narrow(0, 0, non_retained)
                key_cache.append(cached[..., :head_dim])
                value_cache.append(cached[..., head_dim:head_dim + value_dim])
                gate_cache.append(cached[..., head_dim + value_dim:].squeeze(-1))

        return key_cache, value_cache, gate_cache


    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor
    ) -> None:
        """ 
        Relevant tensor objects:

        query: [num_tokens, num_heads, head_dim]
        key:   [num_tokens, num_kv_heads, head_dim]
        gate:  [num_tokens, num_kv_heads]
        value: [num_tokens, num_kv_heads, value_dim]
        output: [num_tokens, num_heads, value_dim]

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

        if envs.VLLM_USE_V1 and attn_metadata:
            attn_metadata = attn_metadata[self.layer_name]
            assert isinstance(attn_metadata, RetentionMetadata)
            num_actual_tokens = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            num_actual_tokens = query.shape[0]
            assert query.shape[0] == key.shape[0] == gate.shape[0] == value.shape[0], "batch size should match for profile run or first prefill"

        # check if there's new prefill, cleanup state and sk for them, add new tokens to cache
        if attn_metadata is not None:
            state_tensor, sk_tensor, cache_tensor, block_idx_tensor = self._get_state_and_cache(attn_metadata)
        else:
            # Profile run: set dummy tensors (won't be used since we skip computation)
            state_tensor = sk_tensor = cache_tensor = block_idx_tensor = None

        # For profile run (attn_metadata is None), we create dummy caches
        # to let the full computation path execute (needed for CUDA graph capture)
        if attn_metadata is None:
            # Create empty lists for caches (will result in empty outputs)
            key_cache = []
            value_cache = []
            gate_cache = []
        else:
            # add new tokens to cache
            key_cache, value_cache, gate_cache = self._update_cache(attn_metadata, key, gate, value, cache_tensor, block_idx_tensor)

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        num_decodes = getattr(attn_metadata, "num_decode_tokens", 0)

        # take care of prefill first, here we want to find the longest prefill sequence and pad the rest to the same length
        # TODO(sean): support chunked prefill
        if num_prefills > 0:
            outputs_prefill = self._prefill(attn_metadata, query, key, gate, value, state_tensor, sk_tensor, block_idx_tensor)
        else:
            outputs_prefill = []

        # take care of decode
        if num_decodes > 0:
            outputs_decode = self._decode(attn_metadata, query, key_cache, value_cache, gate_cache, state_tensor, sk_tensor, block_idx_tensor)
        else:
            outputs_decode = []
            
        # Concat outputs
        all_outputs = outputs_decode + outputs_prefill
        if all_outputs:  # Only copy if we have actual outputs
            if envs.VLLM_USE_V1:
                output.narrow(0, 0, num_actual_tokens).copy_(torch.cat(all_outputs, dim=0))
            else:
                output.narrow(0, 0, num_actual_tokens).copy_(torch.cat(all_outputs, dim=0))
        else:
            # During profile run (attn_metadata=None), ensure output is deterministically written
            # This is required for CUDA graph capture to record a consistent operation
            output.zero_()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor
    ) -> None:
        torch.ops.vllm.retention(
            query,
            key,
            gate,
            value,
            output,
            self.layer_name,
        )


def retention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert isinstance(self, Retention)
    # For V1, attn_metadata is a dict keyed by layer_name
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None:
        attn_metadata = attn_metadata[layer_name]
    # Use v1 path (cache from forward_context)
    self._forward_impl(
        query=query,
        key=key,
        gate=gate,
        value=value,
        output=output
    )


def retention_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # Fake impl used during compile; does nothing but keeps shapes
    return


direct_register_custom_op(
    op_name="retention",
    op_func=retention_op,
    mutates_args=["output"],
    fake_impl=retention_op_fake,
    dispatch_key=current_platform.dispatch_key,
)
