# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch._dynamo import cache_size
import pytest
import torch
import math
import torch.nn.functional as F
from einops import rearrange, repeat

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.layers.mamba.ops.retention import power_retention_varlen, update_state, query_state, cumsum_intra_chunk_gate, update_intra_chunk_memory_and_cache_3d, cumsum_inter_chunk_memory, unified_query_state_2d, find_block_sizes
from vllm.platforms import current_platform


def cumsum_intra_chunk_gate_ref(
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cache_lens: torch.Tensor, # [num_reqs]
    cu_cache_lens: torch.Tensor, # [num_reqs + 1]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    chunk_size: int, # int
    num_seqs: int, # int
) -> None:
    for seq_idx in range(num_seqs):
        query_len = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]).item()
        cache_len = cache_lens[seq_idx].item()
        num_chunks = (query_len + cache_len + chunk_size - 1) // chunk_size
        offset = cu_seqlens_q[seq_idx].item()
        for chunk_idx in range(num_chunks):
            chunk_cache_len = cache_len if chunk_idx == 0 else 0
            chunk_query_len = max(0, min(chunk_size, query_len + cache_len - chunk_idx * chunk_size) - chunk_cache_len)
            if chunk_cache_len > 0:
                cache_blk = block_table[seq_idx, last_memorized_blk_idx[seq_idx] + 1]
                last_cached_gate = gate_cache[cache_blk, chunk_cache_len - 1, :]
            else:
                last_cached_gate = torch.zeros(gate.shape[1], dtype=gate.dtype, device=gate.device)
            gates = gate[offset:offset + chunk_query_len]
            gates = gates.cumsum(dim=0)
            gates = gates + last_cached_gate
            gate[offset:offset + chunk_query_len] = gates
            offset += chunk_query_len

def sympow_dim(d, power, d_tile=1):
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
    D_offset = 0
    for tile_0 in range(num_tiles):
        for tile_1 in range(tile_0, num_tiles):
            x1_range = slice(tile_0 * d_tile, tile_0 * d_tile + d_tile)
            x2_range = slice(tile_1 * d_tile, tile_1 * d_tile + d_tile)
            x1 = x[:, x1_range, :]
            x2 = x[:, x2_range, :]
            x1_x2 = x1[:, :, None, :] * x2[:, None, :, :] # [B, d_tile, d_tile, K]
            phi_x = x1_x2.reshape(B, d_tile**2, x.shape[2]) # [B, d_tile**2, K]
            if tile_0 != tile_1:
                phi_x *= 2
            acc[:, D_offset:D_offset + d_tile**2, :] += phi_x @ y
            l[:, D_offset:D_offset + d_tile**2] += phi_x.sum(dim=-1)
            D_offset += d_tile**2
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
        query_start = cu_seqlens_q[seq_idx].item()
        query_len = (cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx]).item()
        cache_len = cache_lens[seq_idx].item()
        num_chunks = (query_len + cache_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = query_start + max(0, chunk_idx * chunk_size - cache_len)
            chunk_cache_len = cache_len if chunk_idx == 0 else 0
            chunk_query_len = max(0, min(chunk_size, query_len + cache_len - chunk_idx * chunk_size) - cache_len)
            acc = torch.zeros(memory.shape[1], memory.shape[2], memory.shape[3], dtype=memory.dtype, device=memory.device) # [num_kv_heads, state_dim, head_dim]
            l = torch.zeros(memory.shape[1], memory.shape[2], dtype=torch.float32, device=memory.device)
            cache_blk = (block_table[seq_idx, last_memorized_blk_idx[seq_idx].item() + 1 + chunk_idx]).item()
            if chunk_cache_len > 0 and chunk_query_len + chunk_cache_len == chunk_size:
                cached_keys = key_cache[cache_blk, :chunk_cache_len]
                cached_values = value_cache[cache_blk, :chunk_cache_len]
                cached_gates = gate_cache[cache_blk, :chunk_cache_len]

                last_gate = gate[chunk_start + chunk_query_len - 1]
                discounted_keys = discount_keys_ref(cached_keys, cached_gates, last_gate) # [tokens, num_kv_heads, head_dim]
                acc, l = sympow_m_mma_ref(discounted_keys.permute(1, 2, 0), cached_values.permute(1, 0, 2), acc, l, d_tile, power)
    
            if chunk_query_len + chunk_cache_len == chunk_size:
                keys = key[chunk_start : chunk_start + chunk_query_len]
                values = value[chunk_start : chunk_start + chunk_query_len]
                gates = gate[chunk_start : chunk_start + chunk_query_len]
                last_gate = gate[chunk_start + chunk_query_len - 1]
                discounted_keys = discount_keys_ref(keys, gates, last_gate) # [tokens, num_kv_heads, head_dim]
                acc, l = sympow_m_mma_ref(discounted_keys.permute(1, 2, 0), values.permute(1, 0, 2), acc, l, d_tile, power)

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
            blk_id = block_table[seq_idx, last_memorized_blk]
            prev_mem = memory[blk_id]
            prev_ks = ks[blk_id]
        else:
            prev_mem = torch.zeros(*memory.shape[1:], dtype=memory.dtype, device=memory.device)
            prev_ks = torch.zeros(*ks.shape[1:], dtype=ks.dtype, device=ks.device)
        num_full_chunks = (cache_len + query_len) // chunk_size
        gate_offset = query_start + chunk_size - cache_len - 1
        for chunk_idx in range(num_full_chunks):
            blk_idx = block_table[seq_idx, last_memorized_blk + chunk_idx + 1]
            mem_block = memory[blk_idx] # [num_kv_heads, state_dim, head_dim]
            ks_block = ks[blk_idx]
            gate_value = gate[gate_offset]
            mem_block = prev_mem * gate_value[:, None, None] + mem_block
            ks_block = prev_ks * gate_value[:, None] + ks_block
            memory[blk_idx] = mem_block
            ks[blk_idx] = ks_block
            prev_mem = mem_block
            prev_ks = ks_block
            gate_offset += chunk_size


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
        cu_cache_lens,
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
    gate_key: torch.Tensor, # [num_tokens, num_key_heads]
    gate_query: torch.Tensor, # [num_query_tokens, num_query_heads]
    padded_length: int, # int
    scale: float, # float
    deg: int, # int
    if_print: bool, # bool
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
    S = torch.log(S.abs() + 1e-7) * deg
    S = S + gq_by_head[..., None] - gk_by_head[:, None, :]
    M = torch.tril(torch.ones(S.shape[1], S.shape[2], dtype=torch.bool, device=S.device), diagonal=qk_offset)
    S = torch.where(M.unsqueeze(0), S, float("-inf"))
    m = torch.max(S, dim=-1).values # [num_query_heads, num_query_tokens]
    P = torch.exp(S - m[..., None])
    P[:, :padded_length] = 0 # avoid nan in the padded queries
    l = P.sum(dim=-1).transpose(0, 1) # [num_query_heads, num_query_tokens, num_query_heads]
    acc = (P.to(values.dtype) @ values_by_head).transpose(0, 1) # [num_query_tokens, num_query_heads, head_dim]
    m = m.transpose(0, 1) # [num_query_tokens, num_query_heads]
    return acc, l, m, P


def discount_query_ref(
    query: torch.Tensor, # [num_query_tokens, num_query_heads, head_dim]
    gate_query: torch.Tensor, # [num_query_tokens, num_query_heads]
    deg: int, # int
) -> torch.Tensor:
    discounted_query = (query * (gate_query / deg).exp().unsqueeze(-1)).to(query.dtype)
    return discounted_query

def sympow_k_mma_ref(
    x: torch.Tensor, # [B, M, d]
    y: torch.Tensor, # [B, D, N]
    s: torch.Tensor, # [B, D]
    acc: torch.Tensor, # [B, M, N]
    l: torch.Tensor, # [B, M]
    scale: torch.Tensor, # [B, M]
    dim: int, # int
    power: int, # int
    d_tile: int, # int
    if_print: bool, # bool
) -> tuple[torch.Tensor, torch.Tensor]:
    assert power == 2, "only power 2 is supported now"
    assert dim % d_tile == 0, "dim must be divisible by d_tile"
    phi_x = torch.empty(x.shape[0], x.shape[1], y.shape[1], dtype=x.dtype, device=x.device) # [B, M, D]
    D_range = torch.arange(0, d_tile**2, dtype=torch.int32, device=x.device)
    for tile_0 in range(0, dim // d_tile):
        for tile_1 in range(tile_0, dim // d_tile):
            x1_range = slice(tile_0 * d_tile, tile_0 * d_tile + d_tile)
            x2_range = slice(tile_1 * d_tile, tile_1 * d_tile + d_tile)
            x1 = x[..., x1_range]
            x2 = x[..., x2_range]
            x1_x2 = x1[..., None] * x2[:, :, None, :] # [B, M, d_tile, d_tile]
            phi_x[..., D_range] = x1_x2.reshape(x.shape[0], x.shape[1], d_tile**2)
            D_range += d_tile**2
    acc += (phi_x @ y).to(acc.dtype) * scale[..., None]
    l += (phi_x * s[:, None, :]).sum(dim=-1).to(l.dtype) * scale
    return acc, l


def unified_query_state_ref(
    output: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    query: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_kv_heads]
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
    head_dim = query.shape[2]
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
        combined_query = torch.cat([
            torch.zeros(cache_len, query.shape[1], query.shape[2], dtype=query.dtype, device=query.device),
            query[query_start : query_start + query_len],
        ])
        num_chunks = (combined_len + chunk_size - 1) // chunk_size
        chunk_query_start = query_start
        for chunk_idx in range(num_chunks):
            # first query cache and incoming contexts
            chunk_cache_len = cache_len if chunk_idx == 0 else 0
            chunk_query_len = min(query_len + cache_len - chunk_idx * chunk_size, chunk_size) - chunk_cache_len
            chunk_ctx_len = min(chunk_size, combined_len - chunk_idx * chunk_size)
            # we need to pad the query with "fake queries" to respect chunk attention semantics
            chunk_query = combined_query[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_key = combined_keys[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_value = combined_values[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_gate_key = combined_gates[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len]
            chunk_gate_query = combined_gates[chunk_idx * chunk_size : chunk_idx * chunk_size + chunk_ctx_len].repeat_interleave(num_queries_per_kv, dim=1)
            if_print = seq_idx == 0 and chunk_idx == 0
            acc, l, m, P = attention_inner_ref(
                chunk_query,
                chunk_key,
                chunk_value,
                chunk_gate_key,
                chunk_gate_query,
                chunk_cache_len,
                scale,
                deg,
                if_print
            )
            # then query memory
            if last_memorized_blk >= 0:
                alpha = torch.maximum(torch.tensor(math.sqrt(state_dim), dtype=torch.float32, device=m.device), torch.exp(m))
                scale_cache = torch.exp(m) / alpha
                acc = acc * scale_cache[..., None]
                l = l * scale_cache
                mem_block_idx = block_table[seq_idx, last_memorized_blk + chunk_idx]
                scale_mem = ((scale ** deg) / alpha) # [num_query_tokens, num_query_heads]
                discounted_query = discount_query_ref(chunk_query, chunk_gate_query, deg).permute(1, 0, 2)
                memory_block = memory[mem_block_idx].repeat_interleave(num_queries_per_kv, dim=0)
                ks_block = ks[mem_block_idx].repeat_interleave(num_queries_per_kv, dim=0)
                if_print = seq_idx == 1 and chunk_idx == 1
                acc, l = sympow_k_mma_ref(discounted_query, memory_block, ks_block, acc.permute(1, 0, 2), l.permute(1, 0), scale_mem.permute(1, 0), head_dim, deg, d_tile, if_print)
                acc = acc.permute(1, 0, 2)
                l = l.permute(1, 0)
            acc = acc / l[..., None] # [num_query_tokens, num_query_heads, head_dim]
            output[chunk_query_start: chunk_query_start + chunk_query_len] = acc[chunk_cache_len:]
            chunk_query_start += chunk_query_len


def query_state_ref(
    output: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    query: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_key_heads]
    key_cache: torch.Tensor, # [num_blocks, chunk_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blocks, chunk_size, num_kv_heads, head_dim]
    gate_cache: torch.Tensor, # [num_blocks, chunk_size, num_kv_heads]
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    cu_seqlens_padded_q: torch.Tensor, # [num_reqs + 1]
    seq_lens: torch.Tensor, # [num_reqs]
    cache_lens: torch.Tensor, # [num_reqs]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    scale: float, # float
    d_tile: int, # int
    deg: int, # int
):
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    chunk_size = key_cache.shape[1]
    state_dim = memory.shape[2]
    unified_query_state_ref(
    output,
    query,
    key,
    value,
    gate,
    key_cache,
    value_cache,
    gate_cache,
    memory,
    ks,
    block_table,
    last_memorized_blk_idx,
    cu_seqlens_q,
    cu_seqlens_padded_q,
    cache_lens,
    scale,
    num_query_heads,
    num_queries_per_kv,
    d_tile,
    deg,
    chunk_size,
    state_dim)



def create_state(num_seqs: int, max_num_blks: int, chunk_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, zero: bool = True, seed: int = 42, d_tile: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = torch.zeros if zero else torch.randn
    torch.manual_seed(seed)
    state_dim = sympow_dim(head_dim, 2, d_tile)
    total_num_blks = num_seqs * max_num_blks
    key_cache = fn(total_num_blks, chunk_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    value_cache = fn(total_num_blks, chunk_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    # gate_cache = F.logsigmoid(fn(total_num_blks, chunk_size, num_kv_heads, dtype=torch.float32, device="cuda")).cumsum(dim=1)
    gate_cache = torch.ones(total_num_blks, chunk_size, num_kv_heads, dtype=torch.float32, device="cuda") * 0.1
    all_blocks = torch.randperm(total_num_blks, device="cuda", dtype=torch.int32)
    block_table = all_blocks.reshape(num_seqs, max_num_blks)
    memory = fn(total_num_blks, num_kv_heads, state_dim, head_dim, dtype=dtype, device="cuda")
    ks = fn(total_num_blks, num_kv_heads, state_dim, dtype=torch.float32, device="cuda")
    return key_cache, value_cache, gate_cache, block_table, memory, ks

def create_input(num_tokens: int, num_query_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, seed: int = 12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    query = torch.randn(num_tokens, num_query_heads, head_dim, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    # gate = F.logsigmoid(torch.randn(num_tokens, num_kv_heads, dtype=torch.float32, device="cuda"))
    gate = torch.ones(num_tokens, num_kv_heads, dtype=torch.float32, device="cuda")
    return query, key, value, gate

def create_metadata(num_seqs: int, max_num_blks: int, block_size: int, query_lens: tuple[int,...], computed_lens: tuple[int,...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device="cuda")
    computed_lens_tensor = torch.tensor(computed_lens, dtype=torch.int32, device="cuda")
    cache_lens_tensor = computed_lens_tensor % block_size
    cu_seqlens_q = F.pad(torch.cumsum(query_lens_tensor, dim=0), (1, 0))
    cu_cache_lens = F.pad(torch.cumsum(cache_lens_tensor, dim=0), (1, 0))
    last_memorized_blk_idx = computed_lens_tensor // block_size - 1
    non_memorized_lens =cache_lens_tensor + query_lens_tensor
    padded_lens = ((non_memorized_lens + block_size - 1) // block_size) * block_size
    cu_seqlens_padded_q = F.pad(torch.cumsum(padded_lens, dim=0), (1, 0))
    seq_lens = query_lens_tensor + computed_lens_tensor
    return cu_seqlens_q, cu_seqlens_padded_q, cache_lens_tensor, cu_cache_lens, last_memorized_blk_idx, seq_lens


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [4, 16])
@pytest.mark.parametrize("query_per_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [32])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("query_lens", [(1, 23), (1, 41)])
@pytest.mark.parametrize("computed_lens", [(16, 0), (1, 64), (0, 0), (33, 129)])
def test_cumsum_intra_chunk_gate(dtype: torch.dtype, num_kv_heads: int, query_per_kv_heads: int, head_dim: int, chunk_size: int, query_lens: tuple[int, int], computed_lens: tuple[int, int]) -> None:
    num_tokens = sum(query_lens)
    max_num_blks = (max([q + c for q, c in zip(query_lens, computed_lens)]) + chunk_size - 1) // chunk_size
    num_query_heads = num_kv_heads * query_per_kv_heads
    num_seqs = len(query_lens)
    key_cache, value_cache, gate_cache, block_table, memory, ks = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False)
    query, key, value, gate = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q, cu_seqlens_padded_q, cache_lens, cu_cache_lens, last_memorized_blk_idx, seq_lens = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)
    total_chunks = num_tokens // chunk_size + num_seqs
    
    key_cache_ref, value_cache_ref, gate_cache_ref, block_table_ref, memory_ref, ks_ref = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False)
    query_ref, key_ref, value_ref, gate_ref = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q_ref, cu_seqlens_padded_q_ref, cache_lens_ref, cu_cache_lens_ref, last_memorized_blk_idx_ref, seq_lens_ref = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)

    torch.testing.assert_close(gate, gate_ref)
    torch.testing.assert_close(gate_cache, gate_cache_ref)

    cumsum_intra_chunk_gate[(total_chunks, num_kv_heads)](gate, gate_cache, block_table, cu_seqlens_q, cache_lens, cu_cache_lens, last_memorized_blk_idx, chunk_size, num_seqs, *gate.stride(), *gate_cache.stride(), *block_table.stride())
    cumsum_intra_chunk_gate_ref(gate_ref, gate_cache_ref, block_table_ref, cu_seqlens_q_ref, cache_lens_ref, cu_cache_lens_ref, last_memorized_blk_idx_ref, chunk_size, num_seqs)

    torch.testing.assert_close(gate, gate_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [4, 16])
@pytest.mark.parametrize("query_per_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("query_lens", [(23, 1), (1, 41), (1, 1)])
@pytest.mark.parametrize("computed_lens", [(16, 0), (1, 64), (0, 0), (33, 129)])
def test_update_intra_chunk_memory_and_cache_3d(dtype: torch.dtype, num_kv_heads: int, query_per_kv_heads: int, head_dim: int, chunk_size: int, query_lens: tuple[int, int], computed_lens: tuple[int, int]) -> None:
    if len(query_lens) != len(computed_lens):
        pytest.mark.skip("query_lens and computed_lens must have the same length")
    num_tokens = sum(query_lens)
    max_num_blks = (max([q + c for q, c in zip(query_lens, computed_lens)]) + chunk_size - 1) // chunk_size
    num_query_heads = num_kv_heads * query_per_kv_heads
    num_seqs = len(query_lens)
    d_tile = 16
    deg = 2
    state_dim = sympow_dim(head_dim, deg, d_tile)
    # KERNEL
    key_cache, value_cache, gate_cache, block_table, memory, ks = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query, key, value, gate = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q, cu_seqlens_padded_q, cache_lens, cu_cache_lens, last_memorized_blk_idx, seq_lens = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)
    total_chunks_including_cache = num_tokens // chunk_size + 2 * num_seqs
    BLOCK_S = d_tile ** deg
    num_state_blocks = state_dim // BLOCK_S
    BLOCK_T = 16

    # Ref
    key_cache_ref, value_cache_ref, gate_cache_ref, block_table_ref, memory_ref, ks_ref = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query_ref, key_ref, value_ref, gate_ref = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q_ref, cu_seqlens_padded_q_ref, cache_lens_ref, cu_cache_lens_ref, last_memorized_blk_idx_ref, seq_lens_ref = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)

    torch.testing.assert_close(memory, memory_ref)
    torch.testing.assert_close(ks, ks_ref)
    torch.testing.assert_close(key_cache, key_cache_ref)
    torch.testing.assert_close(value_cache, value_cache_ref)
    torch.testing.assert_close(gate_cache, gate_cache_ref)
    torch.testing.assert_close(key, key_ref)
    torch.testing.assert_close(value, value_ref)
    torch.testing.assert_close(gate, gate_ref)

    # Run kernel
    update_intra_chunk_memory_and_cache_3d[(total_chunks_including_cache, num_kv_heads, num_state_blocks)](
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
        cu_cache_lens,
        last_memorized_blk_idx,
        chunk_size,
        num_seqs,
        head_dim,
        d_tile,
        BLOCK_T,
        *key.stride(),
        *value.stride(),
        *gate.stride(),
        *memory.stride(),
        *ks.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        *gate_cache.stride(),
        *block_table.stride(),
    )

    # Run ref
    update_intra_chunk_memory_and_cache_3d_ref(
        key_ref,
        value_ref,
        gate_ref,
        memory_ref,
        ks_ref,
        key_cache_ref,
        value_cache_ref,
        gate_cache_ref,
        block_table_ref,
        cu_seqlens_q_ref,
        cache_lens_ref,
        last_memorized_blk_idx_ref,
        chunk_size,
        num_seqs,
        head_dim,
        d_tile,
        deg,
    )
    torch.testing.assert_close(memory, memory_ref, atol=5e-3, rtol=1e-2)
    # torch.testing.assert_close(ks / 2 * state_dim, ks_ref / 2 * state_dim, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(key_cache, key_cache_ref)
    torch.testing.assert_close(value_cache, value_cache_ref)
    torch.testing.assert_close(gate_cache, gate_cache_ref)
    torch.testing.assert_close(key, key_ref)
    torch.testing.assert_close(value, value_ref)
    torch.testing.assert_close(gate, gate_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [4, 16])
@pytest.mark.parametrize("query_per_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("query_lens", [(23, 1), (1, 41), (64, 127)])
@pytest.mark.parametrize("computed_lens", [(16, 0), (1, 64), (0, 0), (33, 129)])
def test_cumsum_inter_chunk_memory(dtype: torch.dtype, num_kv_heads: int, query_per_kv_heads: int, head_dim: int, chunk_size: int, query_lens: tuple[int, int], computed_lens: tuple[int, int]) -> None:
    if len(query_lens) != len(computed_lens):
        pytest.mark.skip("query_lens and computed_lens must have the same length")
    num_tokens = sum(query_lens)
    max_num_blks = (max([q + c for q, c in zip(query_lens, computed_lens)]) + chunk_size - 1) // chunk_size
    num_query_heads = num_kv_heads * query_per_kv_heads
    num_seqs = len(query_lens)
    d_tile = 16
    deg = 2
    state_dim = sympow_dim(head_dim, deg, d_tile)
    BLOCK_S = d_tile ** deg
    num_state_blocks = state_dim // BLOCK_S
    # kernel inputs
    key_cache, value_cache, gate_cache, block_table, memory, ks = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query, key, value, gate = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q, cu_seqlens_padded_q, cache_lens, cu_cache_lens, last_memorized_blk_idx, seq_lens = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens) 
    # ref inputs
    key_cache_ref, value_cache_ref, gate_cache_ref, block_table_ref, memory_ref, ks_ref = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query_ref, key_ref, value_ref, gate_ref = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q_ref, cu_seqlens_padded_q_ref, cache_lens_ref, cu_cache_lens_ref, last_memorized_blk_idx_ref, seq_lens_ref = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)

    # check inputs are created correctly
    torch.testing.assert_close(memory, memory_ref)
    torch.testing.assert_close(ks, ks_ref)
    torch.testing.assert_close(gate_cache, gate_cache_ref)
    torch.testing.assert_close(gate, gate_ref)

    # kernel call
    cumsum_inter_chunk_memory[(num_seqs, num_kv_heads, num_state_blocks)](
        memory,
        ks,
        gate,
        block_table,
        last_memorized_blk_idx,
        cu_seqlens_q,
        cache_lens,
        head_dim,
        BLOCK_S,
        chunk_size,
        *memory.stride(),
        *ks.stride(),
        *gate.stride(),
        *block_table.stride(),
    )

    # ref call
    cumsum_inter_chunk_memory_ref(
        memory_ref,
        ks_ref,
        gate_ref,
        block_table_ref,
        last_memorized_blk_idx_ref,
        cu_seqlens_q_ref,
        cache_lens_ref,
        chunk_size,
    )

    # Check
    torch.testing.assert_close(memory, memory_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ks, ks_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_kv_heads", [4, 16])
@pytest.mark.parametrize("query_per_kv_heads", [1, 2, 5])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("query_lens", [(1, 23,), (1, 41), (64, 127)])
@pytest.mark.parametrize("computed_lens", [(16, 0), (1, 64), (0, 0), (33, 129)])
@pytest.mark.parametrize("d_tile", [8])
def test_unified_query_state(dtype: torch.dtype, num_kv_heads: int, query_per_kv_heads: int, head_dim: int, chunk_size: int, query_lens: tuple[int, int], computed_lens: tuple[int, int], d_tile: int) -> None:
    if len(query_lens) != len(computed_lens):
        pytest.mark.skip("query_lens and computed_lens must have the same length")
    num_tokens = sum(query_lens)
    max_num_blks = (max([q + c for q, c in zip(query_lens, computed_lens)]) + chunk_size - 1) // chunk_size
    num_query_heads = num_kv_heads * query_per_kv_heads
    num_seqs = len(query_lens)
    deg = 2
    state_dim = sympow_dim(head_dim, deg, d_tile)
    # KERNEL
    key_cache, value_cache, gate_cache, block_table, memory, ks = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query, key, value, gate = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    output = torch.zeros_like(query)
    cu_seqlens_q, cu_seqlens_padded_q, cache_lens, cu_cache_lens, last_memorized_blk_idx, seq_lens = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)
    query_state(
        output,
        query,
        key,
        value,
        gate,
        key_cache,
        value_cache,
        gate_cache,
        memory,
        ks,
        block_table,
        cu_seqlens_q,
        cu_seqlens_padded_q,
        seq_lens,
        cache_lens,
        last_memorized_blk_idx,
        float(head_dim)**-0.5,
        d_tile,
        deg,
    )


    # Ref
    key_cache_ref, value_cache_ref, gate_cache_ref, block_table_ref, memory_ref, ks_ref = create_state(num_seqs, max_num_blks, chunk_size, num_kv_heads, head_dim, dtype, zero=False, d_tile=d_tile)
    query_ref, key_ref, value_ref, gate_ref = create_input(num_tokens, num_query_heads, num_kv_heads, head_dim, dtype)
    cu_seqlens_q_ref, cu_seqlens_padded_q_ref, cache_lens_ref, cu_cache_lens_ref, last_memorized_blk_idx_ref, seq_lens_ref = create_metadata(num_seqs, max_num_blks, chunk_size, query_lens, computed_lens)
    output_ref = torch.zeros_like(query_ref)
    query_state_ref(
        output_ref,
        query_ref,
        key_ref,
        value_ref,
        gate_ref,
        key_cache_ref,
        value_cache_ref,
        gate_cache_ref,
        memory_ref,
        ks_ref,
        block_table_ref,
        cu_seqlens_q_ref,
        cu_seqlens_padded_q_ref,
        seq_lens_ref,
        cache_lens_ref,
        last_memorized_blk_idx_ref,
        float(head_dim)**-0.5,
        d_tile,
        deg,
    )

    # Check
    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=5e-3)


def diff(a, b, if_abs: bool = True):
    if if_abs:
        return (a - b).abs()
    else:
        return (a - b).abs() / a.abs()
