# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import chunk
import pytest
import torch
import math
import torch.nn.functional as F
from einops import rearrange, repeat

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.layers.mamba.ops.retention import power_retention_varlen
from vllm.platforms import current_platform


def cumsum_intra_chunk_gate_ref(
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cache_lens: torch.Tensor, # [num_reqs]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    chunk_size: int, # int
    num_seqs: int, # int
) -> None:
    for seq_idx in range(num_seqs):
        query_len = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]
        cache_len = cache_lens[seq_idx]
        num_chunks = (query_len + cache_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_cache_len = cache_len if chunk_idx == 0 else 0
            chunk_query_len = min(chunk_size, query_len - chunk_idx * chunk_size) - chunk_cache_len
            if chunk_cache_len > 0:
                cache_blk = block_table[seq_idx, last_memorized_blk_idx[seq_idx] + 1]
                last_cached_gate = gate_cache[cache_blk, chunk_cache_len - 1, :]
            else:
                last_cached_gate = torch.zeros(gate.shape[1], dtype=gate.dtype)
            gates = gate[cu_seqlens_q[seq_idx]:cu_seqlens_q[seq_idx] + chunk_query_len]
            gates = gates.cumsum(dim=0)
            gates = gates + last_cached_gate
            gate[cu_seqlens_q[seq_idx]:cu_seqlens_q[seq_idx] + chunk_query_len] = gates

def sympow_dim(cls, d, power, d_tile=1):
    if d_tile == 1:
        return math.comb(d + power - 1, power)
    return sympow_dim(d // d_tile, power) * (d_tile**power)

def sympow_m_mma_ref(
    x: torch.Tensor, # [B, d, K]
    y: torch.Tensor, # [B, K, N]
    acc: torch.Tensor, # [B, D, N]
    l: torch.Tensor, # [B, D]
    d_tile: int, # int
    power: int, # int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert power == 2, "only power 2 is supported now"
    dim = x.shape[1]
    num_tiles = dim // d_tile
    B = x.shape[0]
    D = sympow_dim(dim, power, d_tile)
    acc = torch.zeros(B, D, y.shape[2], dtype=x.dtype, device=x.device)
    l = torch.zeros(B, D, dtype=torch.float32, device=x.device)
    for tile_0 in range(num_tiles):
        for tile_1 in range(num_tiles):
            x1_range = slice(tile_0 * d_tile, tile_0 * d_tile + d_tile)
            x2_range = slice(tile_1 * d_tile, tile_1 * d_tile + d_tile)
            x1 = x[:, x1_range, :]
            x2 = x[:, x2_range, :]
            x1_x2 = x1[:, None, :, :] * x2[:, :, None, :] # [d_tile, d_tile, K]
            phi_x = x1_x2.reshape(x.shape[0], d_tile**2, x.shape[2])
            if tile_0 != tile_1:
                phi_x *= 2
            start_offset = tile_0 * dim * d_tile + tile_1 * d_tile**2
            acc[:, start_offset:start_offset + d_tile**2, :] += phi_x @ y
            l[:, start_offset:start_offset + d_tile**2] += phi_x.sum(dim=1)
    return acc, l

def discount_keys_ref(
    keys: torch.Tensor, # [N, B, D]
    gates: torch.Tensor, # [N, B]
    last_gate: torch.Tensor, # [B]
) -> torch.Tensor:
    discounted_keys = (keys * (last_gate[None, :, None] - gates.unsqueeze(-1)).exp()).to(keys.dtype)
    return discounted_keys

def update_intra_chunk_memory_and_cache_3d_ref(
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    key_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cache_lens: torch.Tensor, # [num_reqs]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    chunk_size: int, # int
    num_seqs: int, # int
    head_dim: int, # int
    d_tile: int, # int
    power: int, # int
) -> None:
    for seq_idx in range(num_seqs):
        query_start = cu_seqlens_q[seq_idx]
        query_len = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]
        cache_len = cache_lens[seq_idx]
        num_chunks = (query_len + cache_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = query_start + chunk_idx * chunk_size
            chunk_cache_len = cache_len if chunk_idx == 0 else 0
            chunk_query_len = min(chunk_size, query_len - chunk_idx * chunk_size) - chunk_cache_len
            acc = torch.zeros(memory.shape[1], memory.shape[2], memory.shape[3], dtype=memory.dtype, device=memory.device)
            l = torch.zeros(memory.shape[1], memory.shape[2], dtype=torch.float32, device=memory.device)
            cache_blk = block_table[seq_idx, last_memorized_blk_idx[seq_idx] + 1]
            if chunk_cache_len > 0 and chunk_query_len + chunk_cache_len == chunk_size:
                cached_keys = key_cache[cache_blk, :chunk_cache_len]
                cached_values = value_cache[cache_blk, :chunk_cache_len]
                cached_gates = gate_cache[cache_blk, :chunk_cache_len]

                last_gate = gate[chunk_start + chunk_query_len - 1]
                discounted_keys = discount_keys_ref(cached_keys, cached_gates, last_gate).permute(1, 2, 0)
                cached_values = cached_values.permute(1, 0, 2)
                acc, l = sympow_m_mma_ref(discounted_keys, cached_values, acc, l, d_tile, power)
    
            if chunk_query_len + chunk_cache_len == chunk_size:
                keys = key[chunk_start : chunk_start + chunk_query_len]
                values = value[chunk_start : chunk_start + chunk_query_len]
                gates = gate[chunk_start : chunk_start + chunk_query_len]
                last_gate = gate[chunk_start + chunk_query_len - 1]
                discounted_keys = discount_keys_ref(keys, gates, last_gate).permute(1, 2, 0)
                values = values.permute(1, 0, 2)
                acc, l = sympow_m_mma_ref(discounted_keys, values, acc, l, d_tile, power)

                memory_blk = block_table[seq_idx, last_memorized_blk_idx[seq_idx] + chunk_idx + 1]
                memory[memory_blk] = acc
                ks[memory_blk] = l

            else:
                keys = key[chunk_start : chunk_start + chunk_query_len]
                values = value[chunk_start : chunk_start + chunk_query_len]
                gates = gate[chunk_start : chunk_start + chunk_query_len]
                key_cache[cache_blk, chunk_cache_len : chunk_cache_len + chunk_query_len, :] = keys
                value_cache[cache_blk, chunk_cache_len : chunk_cache_len + chunk_query_len, :] = values
                gate_cache[cache_blk, chunk_cache_len : chunk_cache_len + chunk_query_len, :] = gates


def cumsum_inter_chunk_memory_ref(
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cache_lens: torch.Tensor, # [num_reqs]
    chunk_size: int, # int
) -> None:
    num_seqs = cache_lens.shape[0]
    for seq_idx in range(num_seqs):
        query_start = cu_seqlens_q[seq_idx]
        query_len = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]
        cache_len = cache_lens[seq_idx]
        last_memorized_blk = last_memorized_blk_idx[seq_idx]
        if last_memorized_blk >= 0:
            prev_mem = memory[last_memorized_blk, :, :, :]
            prev_ks = ks[last_memorized_blk, :, :]
        else:
            prev_mem = torch.zeros(*memory.shape[1:], dtype=memory.dtype)
            prev_ks = torch.zeros(*ks.shape[1:], dtype=ks.dtype)
        num_full_chunks = (cache_len + query_len) // chunk_size
        cur_query_len = chunk_size - cache_len
        for chunk_idx in range(num_full_chunks):
            blk_idx = block_table[seq_idx, last_memorized_blk + chunk_idx + 1]
            mem = memory[blk_idx] #[num_kv_heads, state_dim, head_dim]
            ks = ks[blk_idx]
            gate = gate[query_start + chunk_idx * chunk_size + cur_query_len - 1, :]
            mem = prev_mem * gate[:, None, None] + mem
            ks = prev_ks * gate[:, None] + ks
            memory[blk_idx] = mem
            ks[blk_idx] = ks
            cur_query_len = chunk_size


def update_state_ref(
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    key_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    seq_lens: torch.Tensor, # [num_reqs]
    cache_lens: torch.Tensor, # [num_reqs]
    cu_cache_lens: torch.Tensor, # [num_reqs + 1]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    chunk_size: int, # int
    d_tile: int, # int
    power: int, # int
):
    num_seqs = seq_lens.shape[0]
    head_dim = key.shape[2]
    cumsum_intra_chunk_gate_ref(
        gate,
        gate_cache,
        block_table,
        cu_seqlens_q,
        cache_lens,
        last_memorized_blk_idx,
        chunk_size,
        num_seqs,
    )
    update_intra_chunk_memory_and_cache_3d_ref(
        key,
        value,
        gate,
        memory,
        ks,
        key_cache,
        value_cache,
        gate_cache,
        block_table,
        cu_seqlens_q,
        cache_lens,
        last_memorized_blk_idx,
        chunk_size,
        num_seqs,
        head_dim,
        d_tile,
        power,
    )
    cumsum_inter_chunk_memory_ref(
        memory,
        ks,
        gate,
        block_table,
        last_memorized_blk_idx,
        cu_seqlens_q,
        cache_lens,
        chunk_size,
    )


def attention_inner_ref(
    query: torch.Tensor, # [num_query_tokens, num_query_heads, head_dim]
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate_query: torch.Tensor, # [num_query_tokens, num_query_heads]
    gate_key: torch.Tensor, # [num_tokens, num_key_heads]
    scale: float, # float
    deg: int, # int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_queries_per_kv = query.shape[1] // key.shape[1]
    qk_offset = key.shape[0] - query.shape[0]
    keys = key.repeat_interleave(num_queries_per_kv, dim=1)
    values = value.repeat_interleave(num_queries_per_kv, dim=1)
    gates_key = gate_key.repeat_interleave(num_queries_per_kv, dim=1)
    query_by_head = query.transpose(0, 1)
    keys_by_head = keys.transpose(0, 1) # [num_query_heads, num_tokens, head_dim]
    values_by_head = values.transpose(0, 1)
    gq_by_head = gate_query.transpose(0, 1)
    gk_by_head = gates_key.transpose(0, 1)
    S = scale * (query_by_head @ keys_by_head.transpose(1, 2)) # [num_query_heads, num_query_tokens, num_tokens]
    S = (S.abs() + 1e-7) * deg
    S = S + gq_by_head[..., None] - gk_by_head[:, None, :]
    M = torch.tril(torch.ones(S.shape[1], S.shape[2], dtype=torch.bool, device=S.device), diagonal=qk_offset)
    S = S.where(M.unsqueeze(0), S, float("-inf"))
    m = torch.max(S, dim=1).values # [num_query_heads, num_query_tokens]
    P = torch.exp(S - m[None, :, None])
    l = P.sum(dim=-1).transpose(0, 1) # [num_query_tokens, num_query_heads]
    acc = (P @ values_by_head).transpose(0, 1) # [num_query_tokens, num_query_heads, head_dim]
    m = m.transpose(0, 1) # [num_query_tokens, num_query_heads]
    return acc, l, m


def discount_query_ref(
    query: torch.Tensor, # [num_query_tokens, num_query_heads, head_dim]
    gate_query: torch.Tensor, # [num_query_tokens, num_query_heads]
    deg: int, # int
) -> torch.Tensor:
    discounted_query = (query * (gate_query - deg).exp().unsqueeze(-1)).to(query.dtype)
    return discounted_query

def sympow_k_mma_ref(
    x: torch.Tensor, # [B, M, d]
    y: torch.Tensor, # [B, D, N]
    acc: torch.Tensor, # [B, M, N]
    l: torch.Tensor, # [M]
    scale: torch.Tensor, # [M]
    dim: int, # int
    power: int, # int
    d_tile: int, # int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert power == 2, "only power 2 is supported now"
    assert dim % d_tile == 0, "dim must be divisible by d_tile"
    for tile_0 in range(0, dim // d_tile):
        for tile_1 in range(0, dim // d_tile):
            x1_range = slice(tile_0 * d_tile, tile_0 * d_tile + d_tile)
            x2_range = slice(tile_1 * d_tile, tile_1 * d_tile + d_tile)
            x1 = x[:, x1_range]
            x2 = x[:, x2_range]
            x1_x2 = x1[:, :, None] * x2[:, None, :] # [M, d_tile, d_tile]
            phi_x = x1_x2.reshape(x.shape[0], d_tile**2)
            acc += (phi_x @ y) * scale[None, :, None]
            l += phi_x.sum(dim=1) * scale[None, :]
    return acc, l


def unified_query_state_ref(
    output: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    query: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_key_heads]
    key_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cu_seqlens_padded_q: torch.Tensor, # [num_reqs + 1]
    cache_lens: torch.Tensor, # [num_reqs]
    scale: float, # float
    num_query_heads: int, # int
    num_queries_per_kv: int, # int
    d_tile: int, # int
    deg: int, # int
    chunk_size: int, # int
    state_dim: int, # int
):
    num_seqs = cache_lens.shape[0]
    for seq_idx in range(num_seqs):
        query_start = cu_seqlens_q[seq_idx]
        query_len = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]
        cache_len = cache_lens[seq_idx]
        last_memorized_blk = last_memorized_blk_idx[seq_idx]
        cache_blk = block_table[seq_idx, last_memorized_blk + 1]
        combined_len = query_len + cache_len
        combined_keys = torch.cat([
            key_cache[cache_blk, :cache_len],
            key[query_start : query_start + query_len],
        ])
        combined_values = torch.cat([
            value_cache[cache_blk, :cache_len],
            value[query_start : query_start + query_len],
        ])
        combined_gates = torch.cat([
            gate_cache[cache_blk, :cache_len],
            gate[query_start : query_start + query_len],
        ])
        num_chunks = (combined_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            # first query cache and incoming contexts
            chunk_query_len = min(chunk_size, query_len - chunk_idx * chunk_size)
            chunk_ctx_len = min(chunk_size, (chunk_idx + 1) * chunk_size - combined_len)
            chunk_query = query[query_start + chunk_idx * chunk_size : query_start + chunk_idx * chunk_size + chunk_query_len]
            chunk_key = combined_keys[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_value = combined_values[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_gate = combined_gates[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            acc, l, m = attention_inner_ref(
                chunk_query,
                chunk_key,
                chunk_value,
                chunk_gate,
                scale,
                deg,
            )
            # then query memory
            if last_memorized_blk >= 0:
                alpha = torch.maximum(torch.sqrt(state_dim), torch.exp(m))
                adj_attn = torch.exp(m) / alpha
                acc = acc * adj_attn[:, None, None]
                l = l * adj_attn[:, None]
                mem_block_idx = block_table[seq_idx, last_memorized_blk + chunk_idx]
                scale_mem = (scale ** deg) / alpha
                discounted_query = discount_query_ref(chunk_query, chunk_gate, deg).permute(1, 0, 2)
                memory = memory[mem_block_idx].repeat_interleave(num_queries_per_kv, dim=0)
                ks = ks[mem_block_idx].repeat_interleave(num_queries_per_kv, dim=0)
                acc, l = sympow_k_mma_ref(discounted_query, memory, acc.permute(1, 0, 2), l.permute(1, 0), scale_mem, state_dim, deg, d_tile)
                acc = acc.permute(1, 0, 2)
                l = l.permute(1, 0)

            acc = acc / l[..., None] # [num_query_tokens, num_query_heads, head_dim]
            output[query_start + chunk_idx * chunk_size : query_start + chunk_idx * chunk_size + chunk_query_len] = acc