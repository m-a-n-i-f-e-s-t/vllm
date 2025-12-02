import torch
import triton
import triton.language as tl
import math
from functools import lru_cache

@triton.jit
def binom(n: tl.constexpr, k: tl.constexpr):
    """
    """
    res = 1
    for i in tl.static_range(k):
        res = res * (n - i)
    
    div = 1
    for i in tl.static_range(1, k + 1):
        div = div * i
        
    return res // div

@triton.jit
def find_combinadic_root(val, k: tl.constexpr):
    """
    Finds the largest integer 'x' such that binom(x, k) <= val.
    Uses float approximation followed by a correction step.
    """
    if k == 1:
        return val
        
    # 1. Approximate root
    fact = 1
    for i in tl.static_range(1, k + 1):
        fact = fact * i
    
    approx = tl.exp(tl.log((val * fact).to(tl.float32)) / k)
    x = approx.to(tl.int32)
    
    # 2. Refine guess
    v = binom(x, k)
    if v > val:
        x = x - 1
    elif binom(x + 1, k) <= val:
        x = x + 1
        
    return x


@triton.jit
def get_sympow_coords_p2(linear_idx: tl.int32, N: tl.constexpr):
    """
    Maps a linear index to coordinates in a symmetric power tensor.
    Logic: Inverse Combinatorial Number System (Combinadics).
    
    Args:
        linear_idx: The linearized index.
        N: The size of the dimension (e.g., 8).
    
    Returns:
        tuple: (i_1, i_2, ... i_P) such that i_1 <= i_2 ...
    """
    P: tl.constexpr = 2
    # 1. Calculate total elements: (N + P - 1) choose P
    total = binom(N + P - 1, P)
    
    # 2. Invert index to switch to "growing" shape (Co-lexicographical)
    rem = total - 1 - linear_idx
    
    # 3. Greedy decomposition
    c1 = 0; c0 = 0
    
    for d in tl.static_range(P, 0, -1):
        # Find largest x such that binom(x, d) <= rem
        x = find_combinadic_root(rem, d)
        
        term = binom(x, d)
        rem = rem - term
        
        # Convert back to real coordinate
        real_coord = (N + d - 2) - x
        
        # Assign to specific variable based on d
        if d == 2: c0 = real_coord
        if d == 1: c1 = real_coord

    # Return ordered tuple (row, col, ...)
    return c0, c1


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_SIZE: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_SIZE + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1
    
    
@triton.jit
def find_seq_idx_with_offset(
    query_start_len_ptr,
    cu_cache_lens_ptr,
    target_idx,
    num_seqs,
    BLOCK_SIZE: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid) + tl.load(cu_cache_lens_ptr + mid)
        mid_val = val // BLOCK_SIZE + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def localize_this_pid(
    cu_seqlens_q_ptr, # [num_reqs + 1]
    pid: tl.int32, # int
    num_seqs: tl.int32, # int
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = find_seq_idx(cu_seqlens_q_ptr, pid, num_seqs, BLOCK_SIZE, True)
    q_block_start_idx = tl.load(cu_seqlens_q_ptr + seq_idx) // BLOCK_SIZE + seq_idx
    q_block_local_idx = pid - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(cu_seqlens_q_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(cu_seqlens_q_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    return seq_idx, q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index


@triton.jit
def localize_this_pid_in_chunk(
    cu_seqlens_q_ptr, # [num_seqs + 1]
    cu_seqlens_padded_q_ptr, # [num_reqs + 1]
    cache_lens_ptr, # [num_reqs]
    pid: tl.int32, # int
    num_seqs: tl.int32, # int
    BLOCK_SIZE: tl.constexpr,
    chunk_size: tl.constexpr,
):
    # For a sequence such as |cccs|ssss|ssss|sssx|, chunk size = 4, BLOCK_SIZE = 2
    # we launched 8 kernels: |(cc)(cs)|(ss)(ss)|(ss)(ss)|(ss)(sx)|
    # this function localizes the CTA 
    seq_idx = find_seq_idx(cu_seqlens_padded_q_ptr, pid, num_seqs, BLOCK_SIZE, True)
    q_block_start_idx = tl.load(cu_seqlens_padded_q_ptr + seq_idx) // BLOCK_SIZE + seq_idx
    query_block_idx = pid - q_block_start_idx
    chunk_idx = query_block_idx // (chunk_size // BLOCK_SIZE)
    local_block_idx = query_block_idx % (chunk_size // BLOCK_SIZE)

    # get cache length
    cache_len = (tl.load(cache_lens_ptr + seq_idx)).to(tl.int32)
    if chunk_idx == 0:
        chunk_cache_len = cache_len
    else:
        chunk_cache_len = 0
    
    # get scheduled length
    query_token_offset = tl.load(cu_seqlens_q_ptr + seq_idx)
    query_len = (tl.load(cu_seqlens_q_ptr + seq_idx + 1) - query_token_offset).to(tl.int32)

    block_scheduled_len = tl.maximum(0, tl.minimum(
        BLOCK_SIZE,
        tl.minimum(
            (query_block_idx + 1) * BLOCK_SIZE - cache_len,
            cache_len + query_len - query_block_idx * BLOCK_SIZE
            )
        )
    )

    return seq_idx, chunk_idx, local_block_idx, query_block_idx, query_len, query_token_offset, cache_len, chunk_cache_len, block_scheduled_len


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def localize_this_pid_with_offset(
    cu_seqlens_q_ptr, # [num_reqs + 1]
    cu_cache_lens_ptr, # [num_reqs + 1]
    pid: tl.int32, # int
    num_seqs: tl.int32, # int
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = find_seq_idx_with_offset(cu_seqlens_q_ptr, cu_cache_lens_ptr, pid, num_seqs, BLOCK_SIZE, True)
    q_block_start_idx = (tl.load(cu_seqlens_q_ptr + seq_idx) + tl.load(cu_cache_lens_ptr + seq_idx)) // BLOCK_SIZE + seq_idx

    q_block_local_idx = pid - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(cu_seqlens_q_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(cu_seqlens_q_ptr + seq_idx + 1)

    cur_cu_cache_len = tl.load(cu_cache_lens_ptr + seq_idx)
    next_cu_cache_len = tl.load(cu_cache_lens_ptr + seq_idx + 1)

    query_len = (cur_batch_in_all_stop_index - cur_batch_in_all_start_index).to(tl.int32)
    cache_len = (next_cu_cache_len - cur_cu_cache_len).to(tl.int32)

    return seq_idx, q_block_local_idx, query_len, cur_batch_in_all_start_index, cache_len


@triton.jit
def localize_d_tile_idxs(
    state_block_idx: tl.int32, # int
    dim: tl.constexpr, # int
    power: tl.constexpr, # int
):
    tl.static_assert(power == 2, "power must be 2")
    idx0, idx1 = get_sympow_coords_p2(state_block_idx, dim)
    return idx0, idx1


@triton.jit
def localize_cache_ptrs(
    key_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache_ptr, # [num_blks, chunk_size, num_kv_heads]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    last_memorized_blk_idx_ptr, # [num_reqs]
    seq_idx: tl.int32, # int
    head_idx: tl.int32, # int
    block_idx: tl.int32, # int
    block_table_stride_0: tl.int64, # int
    block_table_stride_1: tl.constexpr, # int
    key_cache_stride_0: tl.int64, # int
    key_cache_stride_2: tl.int64, # int
    value_cache_stride_0: tl.int64, # int
    value_cache_stride_2: tl.int64, # int
    gate_cache_stride_0: tl.int64, # int
    gate_cache_stride_2: tl.int64, # int
):
    cur_block_seq_idx = tl.load(last_memorized_blk_idx_ptr + seq_idx) + 1 + block_idx
    cur_block_idx = tl.load(block_table_ptr + seq_idx * block_table_stride_0 + cur_block_seq_idx * block_table_stride_1).to(tl.int64)
    key_cache_ptr += cur_block_idx * key_cache_stride_0 + head_idx * key_cache_stride_2
    value_cache_ptr += cur_block_idx * value_cache_stride_0 + head_idx * value_cache_stride_2
    gate_cache_ptr += cur_block_idx * gate_cache_stride_0 + head_idx * gate_cache_stride_2
    return key_cache_ptr, value_cache_ptr, gate_cache_ptr


@triton.jit
def localize_memory_and_ks_ptrs(
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    last_memorized_blk_idx_ptr, # [num_reqs]
    seq_idx: tl.int32, # int
    head_idx: tl.int32, # int
    cache_len: tl.int32, # int
    local_chunk_idx: tl.int32, # int
    block_table_stride_0: tl.int32, # int
    block_table_stride_1: tl.int32, # int
    memory_stride_0: tl.int64, # int
    memory_stride_1: tl.int64, # int
    ks_stride_0: tl.int64, # int
    ks_stride_1: tl.int64, # int
):
    start_blk_seq_idx = tl.load(last_memorized_blk_idx_ptr + seq_idx) + 1
    target_blk_seq_idx = start_blk_seq_idx + local_chunk_idx
    target_blk_idx = tl.load(block_table_ptr + seq_idx * block_table_stride_0 + target_blk_seq_idx * block_table_stride_1).to(tl.int64)
    memory_ptr = memory_ptr + target_blk_idx * memory_stride_0 + head_idx * memory_stride_1
    ks_ptr = ks_ptr + target_blk_idx * ks_stride_0 + head_idx * ks_stride_1
    return memory_ptr, ks_ptr


@triton.jit
def slice_read_2d(
    x, dim: tl.constexpr, pos,
    x_size0: tl.constexpr, x_size1: tl.constexpr,
    p_size: tl.constexpr):
    if dim == 0:
        idx = pos[:, None].broadcast_to((p_size, x_size1))
    else:
        idx = pos[None, :].broadcast_to((x_size0, p_size))
    return tl.gather(x, idx, dim)


@triton.jit
def cur_query_len(
    cu_seqlens_q_ptr, # [num_reqs + 1]
    seq_idx: tl.int32, # int
):
    cu_seqlens = tl.load(cu_seqlens_q_ptr + seq_idx)
    cu_seqlens_next = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    return cu_seqlens_next - cu_seqlens


@triton.jit
def block_sympow_m_mma(
    x, # [d, K]
    y, # [K, N]
    acc, # [d_tile**2, N]
    l, # [M]
    K: tl.constexpr, # int
    dim: tl.constexpr, # int
    power: tl.constexpr, # int
    d_tile: tl.constexpr, # int
    tile_idx_0: tl.int32, # int
    tile_idx_1: tl.int32, # int
):
    tl.static_assert(dim % d_tile == 0, "dim must be divisible by d_tile")
    tl.static_assert(power == 2, "power must be 2")
    d_tile_range = tl.arange(0, d_tile)

    x1_range = tile_idx_0 * d_tile + d_tile_range
    x2_range = tile_idx_1 * d_tile + d_tile_range
    x1 = slice_read_2d(x, 0, x1_range, dim, K, d_tile) # [d_tile, K]
    x2 = slice_read_2d(x, 0, x2_range, dim, K, d_tile) # [d_tile, K]
    x1_x2 = x1[:, None, :] * x2[None, :, :] # [d_tile, d_tile, K]
    phi_x = x1_x2.reshape(d_tile*d_tile, K) # [d_tile**2, K]
    if tile_idx_0 != tile_idx_1:
        phi_x *= 2
    acc = tl.dot(phi_x, y, acc, allow_tf32=False) # [d_tile**2, N]
    l = l + tl.sum(phi_x, axis=1) # [d_tile**2]
    return acc, l


@triton.jit
def block_sympow_k_mma(
    x, # [M, d]
    y, # [d_tile**2, N]
    s, # [d_tile**2]
    acc, # [M, N]
    l, # [M]
    M: tl.constexpr, # int
    dim: tl.constexpr, # int
    power: tl.constexpr, # int
    d_tile: tl.constexpr, # int
    tile_idx_0: tl.int32, # int
    tile_idx_1: tl.int32 # int
):
    tl.static_assert(power == 2, "power must be 2")
    tl.static_assert(dim % d_tile == 0, "dim must be divisible by d_tile")
    d_tile_range = tl.arange(0, d_tile)
    x1_range = tile_idx_0 * d_tile + d_tile_range
    x2_range = tile_idx_1 * d_tile + d_tile_range
    x1 = slice_read_2d(x, 1, x1_range, M, dim, d_tile) # [M, d_tile]
    x2 = slice_read_2d(x, 1, x2_range, M, dim, d_tile) # [M, d_tile]
    x1_x2 = x1[:, :, None] * x2[:, None, :] # [M, d_tile, d_tile]
    phi_x = x1_x2.reshape(M, d_tile*d_tile) # [M, d_tile**2]
    acc = tl.dot(phi_x.to(y.dtype), y, acc, allow_tf32=False) # [M, N]
    l = tl.sum(phi_x * s[None, :], axis=1) + l # [M]
    return acc, l


# === Update State Kernels ===

@triton.jit
def cumsum_intra_chunk_gate(
    gate_ptr, # [num_tokens, num_kv_heads]
    gate_cache_ptr, # [num_blks, chunk_size, num_kv_heads]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q_ptr, # [num_reqs + 1]
    cache_lens_ptr, # [num_reqs]
    cu_cache_lens_ptr, # [num_reqs + 1]
    last_memorized_blk_idx_ptr, # [num_reqs]
    chunk_size: tl.constexpr, # int
    num_seqs: tl.int32, # int
    gate_stride_0: tl.int64, # int
    gate_stride_1: tl.constexpr, # int
    gate_cache_stride_0: tl.int64, # int
    gate_cache_stride_1: tl.int64, # int
    gate_cache_stride_2: tl.constexpr, # int
    block_table_stride_0: tl.int64, # int
    block_table_stride_1: tl.constexpr, # int
):
    # For each request with some cached gates X and some scheduled gates Y,
    # split Y gates into many chunks, and for each chunk:
    #   read the scheduled gates, cumsum them
    #   if X > 0 and chunk_id == 0:
    #     read the last cached gate, add it to the cumsum of the scheduled gates
    #   store the result back to the gate pointer
    block_global_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx, local_chunk_idx, query_len, query_start_idx, cache_len = localize_this_pid_with_offset(cu_seqlens_q_ptr, cu_cache_lens_ptr, block_global_idx, num_seqs, chunk_size)


    if local_chunk_idx * chunk_size >= query_len + cache_len:
        return

    local_cache_len = tl.minimum(cache_len, tl.maximum(0, chunk_size - local_chunk_idx * chunk_size))
    local_query_len = tl.maximum(0,
        tl.minimum(chunk_size, query_len + cache_len - local_chunk_idx * chunk_size) - local_cache_len
    )
    local_query_start = tl.maximum(0, local_chunk_idx * chunk_size - cache_len)

    # cumsum on top of the last gate entry in the cache, if any
    cache_len = tl.load(cache_lens_ptr + seq_idx)
    if cache_len > 0 and local_chunk_idx == 0:
        cache_blk_seq_idx = tl.load(last_memorized_blk_idx_ptr + seq_idx) + 1
        cache_blk_idx = tl.load(block_table_ptr + seq_idx * block_table_stride_0 + cache_blk_seq_idx * block_table_stride_1)
        last_cached_gate = tl.load(gate_cache_ptr + cache_blk_idx * gate_cache_stride_0 + head_idx * gate_cache_stride_2 + (cache_len - 1) * gate_cache_stride_1).to(tl.float32)
    else:
        last_cached_gate = 0.0

    if query_len == 1: # just decoding
        gate_ptrs = gate_ptr + query_start_idx * gate_stride_0 + head_idx * gate_stride_1
        gate = tl.load(gate_ptrs)
        tl.store(gate_ptrs, gate + last_cached_gate)
        return

    gate_pos = tl.arange(0, chunk_size)
    offset_gate = query_start_idx + local_query_start + gate_pos
    mask_gate = gate_pos < local_query_len

    gate_ptrs = gate_ptr + offset_gate * gate_stride_0 + head_idx * gate_stride_1
    gates = tl.load(gate_ptrs, mask=mask_gate, other=0) # [chunk_size]
    gates = tl.cumsum(gates, axis=0) # [chunk_size]
    gates = gates + last_cached_gate # [chunk_size]
    tl.store(gate_ptrs, gates, mask=mask_gate)


@triton.jit
def discount_keys(
    x, # [D, N]
    cum_log_gates, # [N]
    last_gate, # [1]
    deg: tl.constexpr, # int
):
    reversed_cum_log_gates = last_gate - cum_log_gates
    discounted_x = x * (reversed_cum_log_gates / deg).exp()[None, :]
    return discounted_x.to(x.dtype)


@triton.jit
def update_intra_chunk_memory_and_cache_3d(
    key_ptr, # [num_tokens, num_kv_heads, head_dim]
    value_ptr, # [num_tokens, num_kv_heads, head_dim]
    gate_ptr, # [num_tokens, num_kv_heads]
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    key_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache_ptr, # [num_blks, chunk_size, num_kv_heads]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q_ptr, # [num_reqs + 1]
    cu_cache_lens_ptr, # [num_reqs + 1]
    last_memorized_blk_idx_ptr, # [num_reqs]
    chunk_size : tl.constexpr, # int
    num_seqs: tl.int32, # int
    head_dim: tl.constexpr, # int
    d_tile: tl.constexpr, # int
    deg: tl.constexpr, # int
    BLOCK_T: tl.constexpr, # int
    key_stride_0: tl.int64, # int
    key_stride_1: tl.int64, # int
    key_stride_2: tl.constexpr, # int
    value_stride_0: tl.int64, # int
    value_stride_1: tl.int64, # int
    value_stride_2: tl.constexpr, # int
    gate_stride_0: tl.int64, # int
    gate_stride_1: tl.constexpr, # int
    memory_stride_0: tl.int64, # int
    memory_stride_1: tl.int64, # int
    memory_stride_2: tl.int64, # int
    memory_stride_3: tl.constexpr, # int
    ks_stride_0: tl.int64, # int
    ks_stride_1: tl.int64, # int
    ks_stride_2: tl.constexpr, # int
    key_cache_stride_0: tl.int64, # int
    key_cache_stride_1: tl.int64, # int
    key_cache_stride_2: tl.int64, # int
    key_cache_stride_3: tl.constexpr, # int
    value_cache_stride_0: tl.int64, # int
    value_cache_stride_1: tl.int64, # int
    value_cache_stride_2: tl.int64, # int
    value_cache_stride_3: tl.constexpr, # int
    gate_cache_stride_0: tl.int64, # int
    gate_cache_stride_1: tl.int64, # int
    gate_cache_stride_2: tl.constexpr, # int
    block_table_stride_0: tl.int32, # int
    block_table_stride_1: tl.int32, # int
):
    # For each request, which has some cached tokens X and some scheduled tokens Y
    #   if X + Y < chunk size, append scheduled tokens to the cache
    #   else:
    #     split X + Y tokens into many chunks, and for each chunk:
    #       if assigned number of tokens < chunk size:
    #          append scheduled tokens to the cache of the corresponding block
    #          (this only happens for the last chunk of the request)
    #       elif assigned number of tokens contains cached tokens:
    #          memorize the cached tokens first
    #          discount it and add to the memorization of scheduled tokens
    #       else:
    #          memorize the scheduled tokens
    block_global_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    state_block_idx = tl.program_id(2)
    power: tl.constexpr = 2
    BLOCK_S: tl.constexpr = d_tile ** power
    seq_idx, local_chunk_idx, scheduled_len, query_start_idx, cache_len = localize_this_pid_with_offset(cu_seqlens_q_ptr, cu_cache_lens_ptr, block_global_idx, num_seqs, chunk_size)
    tile_idx_0, tile_idx_1 = localize_d_tile_idxs(state_block_idx, head_dim // d_tile, power)

    if local_chunk_idx * chunk_size >= cache_len + scheduled_len:
        return

    acc = tl.zeros((BLOCK_S, head_dim), dtype=tl.float32)
    l = tl.zeros((BLOCK_S,), dtype=tl.float32)

    key_cache_ptr, value_cache_ptr, gate_cache_ptr = localize_cache_ptrs(key_cache_ptr, value_cache_ptr, gate_cache_ptr, block_table_ptr, last_memorized_blk_idx_ptr, seq_idx, head_idx, local_chunk_idx, block_table_stride_0, block_table_stride_1, key_cache_stride_0, key_cache_stride_2, value_cache_stride_0, value_cache_stride_2, gate_cache_stride_0, gate_cache_stride_2)

    range_dim = tl.arange(0, head_dim)
    
    # assign all cached tokens to first chunk
    local_cache_len = tl.minimum(cache_len, tl.maximum(0, chunk_size - local_chunk_idx * chunk_size))
    local_query_start = (query_start_idx + tl.maximum(0, local_chunk_idx * chunk_size - cache_len)).to(tl.int64)

    # the number of scheduled tokens this CTA needs to handle
    local_schedule_len = tl.minimum(cache_len + scheduled_len - local_chunk_idx * chunk_size, chunk_size) - local_cache_len

    key_ptr = key_ptr + local_query_start * key_stride_0 + head_idx * key_stride_1
    value_ptr = value_ptr + local_query_start * value_stride_0 + head_idx * value_stride_1
    gate_ptr = gate_ptr + local_query_start * gate_stride_0 + head_idx * gate_stride_1

    # handle cached tokens
    # update memory if cached tokens + scheduled tokens fills a chunk
    if local_cache_len > 0 and local_cache_len + local_schedule_len == chunk_size:
        for tid in tl.range(0, tl.cdiv(cache_len, BLOCK_T)):
            range_t = tl.arange(0, BLOCK_T) + tid * BLOCK_T
            mask_t = range_t < cache_len
            cached_keys = tl.load(key_cache_ptr + range_t[None, :] * key_cache_stride_1 + range_dim[:, None] * key_cache_stride_3, mask=mask_t[None, :], other=0)
            cached_values = tl.load(value_cache_ptr + range_t[:, None] * value_cache_stride_1 + range_dim[None, :] * value_cache_stride_3, mask=mask_t[:, None], other=0)
            cached_gates = tl.load(gate_cache_ptr + range_t * gate_cache_stride_1, mask=mask_t, other=0)
            # discount keys, note that the last gate is actully in the scheduled tokens
            last_gate = tl.load(gate_ptr + (local_schedule_len - 1) * gate_stride_0).to(tl.float32)
            discounted_keys = discount_keys(cached_keys, cached_gates, last_gate, deg)
            acc, l = block_sympow_m_mma(discounted_keys, cached_values, acc, l, BLOCK_T, head_dim, power, d_tile, tile_idx_0, tile_idx_1)
        
    # handle scheduled tokens, update_state if enough tokens to fill a chunk
    if local_cache_len + local_schedule_len == chunk_size:
        for tid in tl.range(0, tl.cdiv(local_schedule_len, BLOCK_T)):
            range_t = tl.arange(0, BLOCK_T) + tid * BLOCK_T
            mask_t = (range_t < local_schedule_len)
            keys = tl.load(key_ptr + range_t[None, :] * key_stride_0 + range_dim[:, None] * key_stride_2, mask=mask_t[None, :], other=0)
            values = tl.load(value_ptr + range_t[:, None] * value_stride_0 + range_dim[None, :] * value_stride_2, mask=mask_t[:, None], other=0)
            gates = tl.load(gate_ptr + range_t * gate_stride_0, mask=mask_t, other=0)
            # discount keys
            last_gate = tl.load(gate_ptr + (local_schedule_len - 1) * gate_stride_0).to(tl.float32)
            discounted_keys = discount_keys(keys, gates, last_gate, deg)
            acc, l = block_sympow_m_mma(discounted_keys, values, acc, l, BLOCK_T, head_dim, power, d_tile, tile_idx_0, tile_idx_1)

        memory_ptr, ks_ptr = localize_memory_and_ks_ptrs(memory_ptr, ks_ptr, block_table_ptr, last_memorized_blk_idx_ptr, seq_idx, head_idx, cache_len, local_chunk_idx, block_table_stride_0, block_table_stride_1, memory_stride_0, memory_stride_1, ks_stride_0, ks_stride_1)

        # store memory and ks
        range_s = tl.arange(0, BLOCK_S)
        memory_ptrs = memory_ptr + (state_block_idx * BLOCK_S + range_s[:, None]) * memory_stride_2 + range_dim[None, :] * memory_stride_3
        ks_ptrs = ks_ptr + (state_block_idx * BLOCK_S + range_s) * ks_stride_2
        tl.store(memory_ptrs, acc)
        tl.store(ks_ptrs, l)

    # otherwise just append the scheduled tokens to cache
    else:
        if state_block_idx != 0: # only the first state block append tokens
            return
        if local_schedule_len > 1:
            for tid in tl.range(0, tl.cdiv(local_schedule_len, BLOCK_T)):
                range_t = tl.arange(0, BLOCK_T) + tid * BLOCK_T
                mask_t = range_t < local_schedule_len
                keys = tl.load(key_ptr + range_t[:, None] * key_stride_0 + range_dim[None, :] * key_stride_2, mask=mask_t[:, None], other=0)
                values = tl.load(value_ptr + range_t[:, None] * value_stride_0 + range_dim[None, :] * value_stride_2, mask=mask_t[:, None], other=0)
                gates = tl.load(gate_ptr + range_t * gate_stride_0, mask=mask_t, other=0)

                range_t_cache = range_t + local_cache_len
                key_cache_ptrs = key_cache_ptr + range_t_cache[:, None] * key_cache_stride_1 + range_dim[None, :] * key_cache_stride_3
                value_cache_ptrs = value_cache_ptr + range_t_cache[:, None] * value_cache_stride_1 + range_dim[None, :] * value_cache_stride_3
                gate_cache_ptrs = gate_cache_ptr + range_t_cache * gate_cache_stride_1

                tl.store(key_cache_ptrs, keys, mask=mask_t[:, None])
                tl.store(value_cache_ptrs, values, mask=mask_t[:, None])
                tl.store(gate_cache_ptrs, gates, mask=mask_t)

        else: # specialize for decoding
            key = tl.load(key_ptr + range_dim[None, :] * key_stride_2)
            value = tl.load(value_ptr + range_dim[None, :] * value_stride_2)
            gate = tl.load(gate_ptr)
            key_cache_ptrs = key_cache_ptr + local_cache_len * key_cache_stride_1 + range_dim[None, :] * key_cache_stride_3
            value_cache_ptrs = value_cache_ptr + local_cache_len * value_cache_stride_1 + range_dim[None, :] * value_cache_stride_3
            gate_cache_ptrs = gate_cache_ptr + local_cache_len * gate_cache_stride_1
            tl.store(key_cache_ptrs, key)
            tl.store(value_cache_ptrs, value)
            tl.store(gate_cache_ptrs, gate)



@triton.jit
def read_memory_and_ks(
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    block_idx: tl.int32, # int
    BLOCK_S: tl.constexpr, # int
    head_dim: tl.constexpr, # int
    memory_stride_0: tl.int64, # int
    memory_stride_2: tl.int64, # int
    memory_stride_3: tl.constexpr, # int
    ks_stride_0: tl.int64, # int
    ks_stride_2: tl.constexpr, # int
):
    # assumes memory_ptr and ks_ptr are already localized
    range_s = tl.arange(0, BLOCK_S)
    range_dim = tl.arange(0, head_dim)
    memory_ptrs = memory_ptr + block_idx * memory_stride_0 + range_s[:, None] * memory_stride_2 + range_dim[None, :] * memory_stride_3
    ks_ptrs = ks_ptr + block_idx * ks_stride_0 + range_s * ks_stride_2
    memory = tl.load(memory_ptrs).to(tl.float32)
    ks = tl.load(ks_ptrs).to(tl.float32)
    return memory, ks

@triton.jit
def write_memory_and_ks(
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    memory, # [BLOCK_S, head_dim]
    ks, # [BLOCK_S,]
    block_idx: tl.int32, # int
    BLOCK_S: tl.constexpr, # int
    head_dim: tl.constexpr, # int
    memory_stride_0: tl.int64, # int
    memory_stride_2: tl.int64, # int
    memory_stride_3: tl.constexpr, # int
    ks_stride_0: tl.int64, # int
    ks_stride_2: tl.constexpr, # int
):
    range_s = tl.arange(0, BLOCK_S)
    range_dim = tl.arange(0, head_dim)
    memory_ptrs = memory_ptr + block_idx * memory_stride_0 + range_s[:, None] * memory_stride_2 + range_dim[None, :] * memory_stride_3
    ks_ptrs = ks_ptr + block_idx * ks_stride_0 + range_s * ks_stride_2
    tl.store(memory_ptrs, memory)
    tl.store(ks_ptrs, ks)


@triton.jit
def cumsum_inter_chunk_memory(
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    gate_ptr, # [num_tokens, num_key_heads]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    last_memorized_blk_idx_ptr, # [num_reqs]
    cu_seqlens_q_ptr, # [num_reqs + 1]
    cache_lens_ptr, # [num_reqs + 1]
    head_dim: tl.constexpr, # int
    BLOCK_S: tl.constexpr, # int
    chunk_size: tl.constexpr, # int
    memory_stride_0: tl.int64, # int
    memory_stride_1: tl.int64, # int
    memory_stride_2: tl.int64, # int
    memory_stride_3: tl.constexpr, # int
    ks_stride_0: tl.int64, # int
    ks_stride_1: tl.int64, # int
    ks_stride_2: tl.constexpr, # int
    gate_stride_0: tl.int64, # int
    gate_stride_1: tl.constexpr, # int
    block_table_stride_0: tl.int64, # int
    block_table_stride_1: tl.constexpr, # int
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    state_block_idx = tl.program_id(2)
    cu_seqlens = tl.load(cu_seqlens_q_ptr + seq_idx)
    cu_seqlens_next = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    cur_query_len = cu_seqlens_next - cu_seqlens
    if cur_query_len == 0: # padded sequence
        return
    initial_memory_blk_seq_idx = tl.load(last_memorized_blk_idx_ptr + seq_idx)
    memory_ptr = memory_ptr + head_idx * memory_stride_1 + state_block_idx * BLOCK_S * memory_stride_2
    ks_ptr = ks_ptr + head_idx * ks_stride_1 + state_block_idx * BLOCK_S * ks_stride_2
    gate_ptr = gate_ptr + cu_seqlens * gate_stride_0 + head_idx * gate_stride_1
    blk_table_ptr = block_table_ptr + seq_idx * block_table_stride_0

    # only perform memory cumsum for full chunks
    cache_len = tl.load(cache_lens_ptr + seq_idx)
    num_full_chunks = (cache_len + cur_query_len) // chunk_size
    this_schedule_len = chunk_size - cache_len
    if num_full_chunks == 0: # no need to perform memory cumsum if there's no full chunks
        return

    # read initial memory and ks
    if initial_memory_blk_seq_idx >= 0:
        initial_memory_blk_idx = tl.load(blk_table_ptr + initial_memory_blk_seq_idx * block_table_stride_1).to(tl.int64)
        prev_memory, prev_ks = read_memory_and_ks(memory_ptr, ks_ptr, initial_memory_blk_idx, BLOCK_S, head_dim, memory_stride_0, memory_stride_2, memory_stride_3, ks_stride_0, ks_stride_2)
    else:
        prev_memory = tl.zeros((BLOCK_S, head_dim), dtype=tl.float32)
        prev_ks = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # cumsum through all full blocks for the request
    for chunk_idx in tl.range(num_full_chunks):
        blk_seq_idx = initial_memory_blk_seq_idx + chunk_idx + 1
        blk_idx = tl.load(blk_table_ptr + blk_seq_idx * block_table_stride_1).to(tl.int64)
        memory, ks = read_memory_and_ks(memory_ptr, ks_ptr, blk_idx, BLOCK_S, head_dim, memory_stride_0, memory_stride_2, memory_stride_3, ks_stride_0, ks_stride_2)
        gate = tl.load(gate_ptr + (this_schedule_len - 1) * gate_stride_0).to(tl.float32).exp() # [1]
        memory = prev_memory * gate + memory
        ks = prev_ks * gate + ks
        write_memory_and_ks(memory_ptr, ks_ptr, memory, ks, blk_idx, BLOCK_S, head_dim, memory_stride_0, memory_stride_2, memory_stride_3, ks_stride_0, ks_stride_2)
        prev_memory = memory
        prev_ks = ks
        gate_ptr = gate_ptr + this_schedule_len * gate_stride_0
        this_schedule_len = chunk_size

# === Query State Kernels ===
@triton.jit
def attention_inner(
    Q, # [BLOCK_M, HEAD_SIZE]
    K, # [HEAD_SIZE, TILE_SIZE]
    V, # [TILE_SIZE, HEAD_SIZE]
    GQ, # [BLOCK_M]
    GK, # [TILE_SIZE]
    mask, # [BLOCK_M, TILE_SIZE]
    acc, # [BLOCK_M, HEAD_SIZE]
    L, # [BLOCK_M]
    M, # [BLOCK_M]
    scale, # float32
    BLOCK_M: tl.constexpr, # int
    TILE_SIZE: tl.constexpr, # int
    deg: tl.constexpr, # int
):

    # S : (BLOCK_M, TILE_SIZE)
    S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

    S += scale * tl.dot(Q.to(K.dtype), K, allow_tf32=False)

    S = tl.log(S.abs() + 1e-7) * deg

    S = S + GQ[:, None] - GK[None, :]

    S = tl.where(mask, S, float("-inf"))

    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))

    # For blocks with cache tokens in front, there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - m_j[:, None])

    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)

    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - m_j)

    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = acc * alpha[:, None]

    # update constants
    L = L * alpha + l_j
    M = m_j

    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc += tl.dot(P.to(V.dtype), V, allow_tf32=False)
    return acc, L, M
    

@triton.jit
def query_cache(
    Q, # [BLOCK_M, HEAD_SIZE_PADDED]
    GQ, # [BLOCK_M]
    query_pos, # [BLOCK_M]
    acc, # [BLOCK_M, HEAD_SIZE_PADDED]
    L, # [BLOCK_M]
    M, # [BLOCK_M]
    key_ptr, # [num_tokens, num_kv_heads, head_dim]
    value_ptr, # [num_tokens, num_kv_heads, head_dim]
    gate_ptr, # [num_tokens, num_key_heads]
    key_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache_ptr, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache_ptr, # [num_blks, chunk_size, num_kv_heads]
    offs_t, # [TILE_SIZE]
    offs_d, # [HEAD_SIZE_PADDED]
    query_mask, # [BLOCK_M]
    dim_mask, # [HEAD_SIZE_PADDED]
    physical_block_idx, # [1]
    query_token_offset: tl.int32, # int
    kv_head_idx: tl.int32, # int
    chunk_idx: tl.int32, # int
    cache_len: tl.int32, # int
    query_len: tl.int32, # int
    chunk_cache_len: tl.int32, # int
    cached_tile_end: tl.int32, # int
    scheduled_tile_start: tl.int32, # int
    scheduled_tile_end: tl.int32, # int
    scale, # float32
    deg: tl.constexpr, # int
    chunk_size: tl.constexpr, # int
    BLOCK_M: tl.constexpr, # int
    TILE_SIZE: tl.constexpr, # int
    BLOCK_SIZE: tl.constexpr, # int
    key_cache_stride_0: tl.int64, # int
    key_cache_stride_1: tl.int64, # int
    key_cache_stride_2: tl.int64, # int
    key_cache_stride_3: tl.constexpr, # int
    value_cache_stride_0: tl.int64, # int
    value_cache_stride_1: tl.int64, # int
    value_cache_stride_2: tl.int64, # int
    value_cache_stride_3: tl.constexpr, # int
    gate_cache_stride_0: tl.int64, # int
    gate_cache_stride_1: tl.int64, # int
    gate_cache_stride_2: tl.constexpr, # int
    key_stride_0: tl.int64, # int
    key_stride_1: tl.int64, # int
    key_stride_2: tl.constexpr, # int
    value_stride_0: tl.int64, # int
    value_stride_1: tl.int64, # int
    value_stride_2: tl.constexpr, # int
    gate_stride_0: tl.int64, # int
    gate_stride_1: tl.constexpr, # int
):
    # # iterate through cached tiles
    for j in tl.range(0, cached_tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < chunk_cache_len

        v_offset = (
            physical_block_idx * value_cache_stride_0
            + kv_head_idx * value_cache_stride_2
            + offs_d[None, :] * value_cache_stride_3
            + (seq_offset % BLOCK_SIZE)[:, None] * value_cache_stride_1
        )

        k_offset = (
            physical_block_idx * key_cache_stride_0
            + kv_head_idx * key_cache_stride_2
            + offs_d[:, None] * key_cache_stride_3
            + (seq_offset % BLOCK_SIZE)[None, :] * key_cache_stride_1
        )

        gk_offset = (
            physical_block_idx * gate_cache_stride_0
            + kv_head_idx * gate_cache_stride_2
            + (seq_offset % BLOCK_SIZE) * gate_cache_stride_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        # V : (TILE_SIZE, HEAD_SIZE)
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # G : (TILE_SIZE, )
        GK = tl.load(
            gate_cache_ptr + gk_offset,
            mask=tile_mask,
            other=0.0,
        )

        causal_mask = seq_offset[None, :] < query_pos[:, None] + 1

        mask = query_mask[:, None] & causal_mask

        acc, L, M = attention_inner(Q, K, V, GQ, GK, mask, acc, L, M, scale, BLOCK_M, TILE_SIZE, deg)

    # iterate through scheduled tiles
    for j in tl.range(scheduled_tile_start, scheduled_tile_end):
        seq_pos = j * TILE_SIZE + chunk_size * chunk_idx + offs_t
        tile_mask = tl.where(seq_pos >= cache_len, 1, 0).to(tl.int1)
        tile_mask = tile_mask & tl.where(seq_pos < cache_len + query_len, 1, 0).to(tl.int1)
        kv_offset = query_token_offset + seq_pos - cache_len

        v_offset = (
            kv_offset[:, None] * value_stride_0
            + kv_head_idx * value_stride_1
            + offs_d[None, :] * value_stride_2
        )

        k_offset = (
            kv_offset[None, :] * key_stride_0
            + kv_head_idx * key_stride_1
            + offs_d[:, None] * key_stride_2
        )

        g_offset = (
            kv_offset * gate_stride_0
            + kv_head_idx * gate_stride_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K = tl.load(
            key_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        # V : (TILE_SIZE, HEAD_SIZE)
        V = tl.load(
            value_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # G : (TILE_SIZE, )
        GK = tl.load(
            gate_ptr + g_offset,
            mask=tile_mask,
            other=0.0,
        )

        causal_mask = seq_pos[None, :] < query_pos[:, None] + 1

        mask = query_mask[:, None] & causal_mask

        acc, L, M = attention_inner(Q, K, V, GQ, GK, mask, acc, L, M, scale, BLOCK_M, TILE_SIZE, deg)

    return acc, L, M


@triton.jit
def discount_query(
    query, # [M, D]
    cum_log_gates, # [M]
    deg: tl.constexpr, # int
):
    discounted_query = query * (cum_log_gates / deg).exp()[:, None]
    return discounted_query.to(query.dtype)


@triton.jit
def query_memory(
    Q, # [BLOCK_M, HEAD_SIZE]
    GQ, # [BLOCK_M]
    acc, # [BLOCK_M, HEAD_SIZE]
    L, # [BLOCK_M]
    M, # [BLOCK_M]
    memory_ptr, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks_ptr, # [num_blks, num_kv_heads, state_dim]
    block_table_ptr, # [num_reqs, max_num_blocks_per_req]
    offs_d, # [HEAD_SIZE]
    scale: tl.float32, # float32
    last_memorized_blk_idx: tl.int32, # int
    seq_idx: tl.int32, # int
    chunk_idx: tl.int32, # int
    kv_head_idx: tl.int32, # int
    d_tile: tl.constexpr, # int
    deg: tl.constexpr, # int
    state_dim: tl.constexpr, # int
    HEAD_SIZE: tl.constexpr, # int
    BLOCK_M: tl.constexpr, # int
    block_table_stride_0: tl.int64, # int
    memory_stride_0: tl.int64, # int
    memory_stride_1: tl.int64, # int
    memory_stride_2: tl.int64, # int
    memory_stride_3: tl.constexpr, # int
    ks_stride_0: tl.int64, # int
    ks_stride_1: tl.int64, # int
    ks_stride_2: tl.constexpr, # int
):
    # scale down attention output for numerical stability
    alpha = tl.maximum(tl.sqrt(state_dim), tl.exp(M)) # [BLOCK_M]
    scale_cache = tl.exp(M) / alpha # [BLOCK_M]
    scale_mem = _power(scale, deg) / alpha # [BLOCK_M]
    adj = scale_cache / scale_mem
    acc = acc * adj[:, None]
    L = L * adj
    # query memory
    memory_block_idx = tl.load(block_table_ptr + seq_idx * block_table_stride_0 + last_memorized_blk_idx + chunk_idx).to(tl.int64)
    BLOCK_S: tl.constexpr = d_tile ** deg
    offs_s = tl.arange(0, BLOCK_S)
    Q = discount_query(Q, GQ, deg)
    state_offset = offs_s
    for dtile_0 in tl.range(0, HEAD_SIZE // d_tile):
        for dtile_1 in tl.range(dtile_0, HEAD_SIZE // d_tile):
            memory = tl.load(memory_ptr + memory_block_idx * memory_stride_0 + kv_head_idx * memory_stride_1 + state_offset[:, None] * memory_stride_2 + offs_d[None, :] * memory_stride_3)
            ks = tl.load(ks_ptr + memory_block_idx * ks_stride_0 + kv_head_idx * ks_stride_1 + state_offset * ks_stride_2)
            acc, L = block_sympow_k_mma(Q, memory, ks, acc, L, BLOCK_M, HEAD_SIZE, deg, d_tile, dtile_0, dtile_1)
            state_offset = state_offset + d_tile * d_tile
    acc = acc * scale_mem[:, None]
    L = L * scale_mem
    return acc, L


@triton.jit
def _power(x, deg: tl.constexpr):
    if deg == 2:
        return x * x
    elif deg == 3:
        return x * x * x
    elif deg == 4:
        return x * x * x * x
    else:
        raise ValueError(f"Invalid degree: {deg}")


# Adapted from https://github.com/vllm-project/vllm/blob/c6fa3895e90f6daef4d223188f6b4156311f40c9/vllm/attention/ops/triton_unified_attention.py#L57
@triton.jit
def unified_query_state_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    gate_ptr,  # [num_tokens, num_key_heads]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    gate_cache_ptr,  # [num_blks, blk_size, num_kv_heads]
    memory_ptr,  # [num_blks, num_kv_heads, state_dim, head_size]
    ks_ptr,  # [num_blks, num_kv_heads, state_dim]
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    last_memorized_blk_idx_ptr, # [num_reqs]
    cu_seqlens_q_ptr,  # [num_seqs+1]
    cu_seqlens_padded_q_ptr, # [num_seqs + 1]
    cache_lens_ptr, # [num_reqs]
    scale: tl.float32,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    d_tile: tl.constexpr, # int
    BLOCK_M: tl.constexpr,  # int
    deg: tl.constexpr, # int
    chunk_size: tl.constexpr, # int
    has_prefill: tl.constexpr, # bool
    state_dim: tl.float32, # float (for stablization)
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    query_stride_2: tl.constexpr,  # int
    key_stride_0: tl.int64,  # int
    key_stride_1: tl.int64,  # int
    key_stride_2: tl.constexpr,  # int
    value_stride_0: tl.int64,  # int
    value_stride_1: tl.int64,  # int
    value_stride_2: tl.constexpr,  # int
    gate_stride_0: tl.int64,  # int
    gate_stride_1: tl.constexpr,  # int
    key_cache_stride_0: tl.int64,  # int
    key_cache_stride_1: tl.int64,  # int
    key_cache_stride_2: tl.int64,  # int
    key_cache_stride_3: tl.constexpr,  # int
    value_cache_stride_0: tl.int64,  # int
    value_cache_stride_1: tl.int64,  # int
    value_cache_stride_2: tl.int64,  # int
    value_cache_stride_3: tl.constexpr,  # int
    gate_cache_stride_0: tl.int64,  # int
    gate_cache_stride_1: tl.int64,  # int
    gate_cache_stride_2: tl.constexpr,  # int
    memory_stride_0: tl.int64,  # int
    memory_stride_1: tl.int64,  # int
    memory_stride_2: tl.int64,  # int
    memory_stride_3: tl.constexpr,  # int
    ks_stride_0: tl.int64,  # int
    ks_stride_1: tl.int64,  # int
    ks_stride_2: tl.constexpr,  # int
    block_table_stride_0: tl.int64,  # int
    block_table_stride_1: tl.constexpr,  # int
):
    # For each query block:
    # if there's cache, query against the cache using attention formula
    # if there's memory, query against the memory by expanding the query and do a matmul with memory
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    # A given block of tokens consists of:
    # [cached tokens] [scheduled tokens] [padded tokens]
    # Here we localize this CTA by computing:
    # seq_idx: index of the sequence in the batch
    # chunk_idx: index of the chunk in the (padded) sequence this CTA is responsible for
    # local_block_idx: index of the block in the chunk this CTA is responsible for
    # query_block_idx: index of the block in the (padded) sequence
    # query_len: the unpadded number of scheduled tokens for this sequence
    # query_token_offset: offset of the start of the query in num_tokens
    # cache_len: the number of cached tokens in this sequence
    # chunk_cache_len: the number of cached tokens in this chunk
    # block_scheduled_len: the number of scheduled tokens in this block
    if has_prefill:
        seq_idx, chunk_idx, local_block_idx, query_block_idx, query_len, query_token_offset, cache_len, chunk_cache_len, block_scheduled_len = localize_this_pid_in_chunk(cu_seqlens_q_ptr, cu_seqlens_padded_q_ptr, cache_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q, chunk_size)
    else:
        seq_idx = q_block_global_idx
        chunk_idx = 0
        local_block_idx = 0
        query_block_idx = 0
        query_len = 1
        query_token_offset = tl.load(cu_seqlens_q_ptr + seq_idx)
        cache_len = tl.load(cache_lens_ptr + seq_idx)
        chunk_cache_len = cache_len
        block_scheduled_len = 1


    # skip if there's no scheduled tokens in this block, could be due to:
    # 1. This is a padded sequence by vllm
    # 2. This block is at the start of the first chunk where it's all cached tokens
    # 3. This block is at the end of the last chunk where there's no scheduled tokens
    if block_scheduled_len <= 0 or seq_idx >= num_seqs:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_t = tl.arange(0, TILE_SIZE)
    # Account for the different grid semantics for mixed-prefill and pure-decode
    if has_prefill:
        query_pos = query_block_idx * BLOCK_Q + offs_m // num_queries_per_kv
    else:
        query_pos = query_block_idx * BLOCK_Q + offs_m // num_queries_per_kv + cache_len

    query_offset_0 = query_token_offset + query_pos - cache_len
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    
    gq_offset_0 = query_offset_0
    gq_offset_1 = kv_head_idx
    gq_offset = gq_offset_0 * gate_stride_0 + gq_offset_1 * gate_stride_1

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1) # not actually useful yet
    query_mask_0 = tl.where(query_pos < query_len + cache_len, 1, 0).to(tl.int1)
    query_mask_0 = query_mask_0 & tl.where(query_pos >= cache_len, 1, 0).to(tl.int1)
    query_mask_0 = query_mask_0 & tl.where(offs_m < num_queries_per_kv * BLOCK_Q, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # GQ : (BLOCK_M, )
    GQ = tl.load(
        gate_ptr + gq_offset,
        mask=query_mask_0 & query_mask_1,
        other=0.0,
    )

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    # cache tiles (only the first chunk deals with cached tokens)
    cached_tile_end = cdiv_fn(chunk_cache_len, TILE_SIZE)

    # scheduled tiles
    scheduled_tile_start = chunk_cache_len // TILE_SIZE
    if has_prefill:
        scheduled_tile_end = tl.cdiv((local_block_idx + 1) * BLOCK_Q, TILE_SIZE)
    else:
        scheduled_tile_end = tl.cdiv((local_block_idx + 1) * BLOCK_Q + cache_len, TILE_SIZE)

    # find cache block index
    last_memorized_blk_idx = tl.load(last_memorized_blk_idx_ptr + seq_idx).to(tl.int64)
    first_cached_block_idx = tl.load(
        block_table_ptr + seq_idx * block_table_stride_0 + last_memorized_blk_idx + 1
    ).to(tl.int64)

    # Query against cache and incoming scheduled tokens
    query_mask = query_mask_0 & query_mask_1
    acc, L, M = query_cache(Q, GQ, query_pos, acc, L, M, key_ptr, value_ptr, gate_ptr, key_cache_ptr, value_cache_ptr, gate_cache_ptr, offs_t, offs_d, query_mask, dim_mask, first_cached_block_idx, query_token_offset, kv_head_idx, chunk_idx, cache_len, query_len, chunk_cache_len, cached_tile_end, scheduled_tile_start, scheduled_tile_end, scale, deg, chunk_size, BLOCK_M, TILE_SIZE, BLOCK_SIZE, key_cache_stride_0, key_cache_stride_1, key_cache_stride_2, key_cache_stride_3, value_cache_stride_0, value_cache_stride_1, value_cache_stride_2, value_cache_stride_3, gate_cache_stride_0, gate_cache_stride_1, gate_cache_stride_2, key_stride_0, key_stride_1, key_stride_2, value_stride_0, value_stride_1, value_stride_2, gate_stride_0, gate_stride_1)

    # Query against memory
    if last_memorized_blk_idx + chunk_idx >= 0: # skip if no memory yet
        acc, L = query_memory(Q, GQ, acc, L, M, memory_ptr, ks_ptr, block_table_ptr, offs_d, scale, last_memorized_blk_idx, seq_idx, chunk_idx, kv_head_idx, d_tile, deg, state_dim, HEAD_SIZE, BLOCK_M, block_table_stride_0, memory_stride_0, memory_stride_1, memory_stride_2, memory_stride_3, ks_stride_0, ks_stride_1, ks_stride_2)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# === Interface ===

@lru_cache(maxsize=100)
def find_block_sizes(chunk_size: int, num_queries_per_kv: int) -> tuple[int, int]:
    """ Find BLOCK_Q and BLOCK_M such that
    1. chunk_size is divisible by BLOCK_Q
    2. BLOCK_M >= BLOCK_Q * num_queries_per_kv
    3. BLOCK_M is a power of 2 and >= 16 and <= 256
    4. BLOCK_Q is as small as possible to maximize parallelism
    
    Note: We use fixed block sizes independent of num_seqs to ensure
    consistent Triton kernel compilation between CUDA graph capture
    and eager execution. 
    """
    divisors = sorted({d for i in range(1, int(math.sqrt(chunk_size)) + 1) 
                       if chunk_size % i == 0 for d in (i, chunk_size // i)})
    
    # Find smallest valid BLOCK_Q that divides chunk_size
    for block_q in divisors:
        # Find smallest BLOCK_M that is a power of 2 and >= BLOCK_Q * num_queries_per_kv
        block_m = 16
        while block_m < block_q * num_queries_per_kv:
            block_m *= 2
        if block_m <= 256:
            return block_q, block_m
    
    # Fallback
    return chunk_size, max(16, chunk_size * num_queries_per_kv)

def query_state(
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
    has_prefill: bool, # bool
):
    # query_state kernel queries both cache and memory. 
    # The first is a chunked attention, the second is a symmetric-power-matmul with memory.
    num_tokens = query.shape[0]
    num_seqs = seq_lens.shape[0]
    num_query_heads = query.shape[1]
    num_kv_heads, head_dim = key.shape[1], key.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    chunk_size = key_cache.shape[1]
    state_dim = memory.shape[2]

    BLOCK_Q, BLOCK_M = find_block_sizes(chunk_size, num_queries_per_kv)
    
    # The attention pattern is a chunked attention pattern, but it's irregular at the
    # beginning and the end of the sequence. To illustrate, we use "c" for computed token,
    # "s" for scheduled token, "x" for placeholder, and "|" for chunk boundary, with a chunk size of 4.
    # |cccs|ssss|ssss|sssx|
    # Inside each chunk is causal self-attention.
    # For BLOCK_Q = 2, we only need to launch 7 kernels:
    # (cs)|(ss)(ss)|(ss)(ss)|(ss)(sx)
    # To launch a fixed size grid, however, we ideally want to launch 
    # \sum_i(\sum_j(cdiv(chunk_query_len[i][j], BLOCK_Q)) kernels along sequence dimension
    # where chunk_query_len[i][j] is the number of tokens in the j-th chunk of the i-th sequence.
    # but materializing chunk_query_len on cpu is slow, so we take its upper bound
    if has_prefill:
        # For mixed prefill and decode, we can't make assumptions about the number of tokens in the cache, so the
        # upper bound is loose:
        # \sum_i(\sum_j(cdiv(chunk_query_len[i][j], BLOCK_Q))
        #  <= \sum_i(\sum_j(cdiv(chunk_size, BLOCK_Q))
        #  <= \sum_i((chunk_size // BLOCK_Q) * (cdiv(query_len[i], chunk_size) + 1))
        #  = \sum_i(cdiv(query_len[i], chunk_size) * (chunk_size // BLOCK_Q) + num_seqs * (chunk_size // BLOCK_Q))
        #  <= (num_tokens // chunk_size + num_seqs) * (chunk_size // BLOCK_Q) + num_seqs * (chunk_size // BLOCK_Q)
        #  = num_tokens // BLOCK_Q + 2 * num_seqs * (chunk_size // BLOCK_Q)
        total_q_blocks = num_tokens // BLOCK_Q + 2 * num_seqs * (chunk_size // BLOCK_Q)
    else:
        # For decode, we only need to launch as many CTAs as there are queries, this reduces overhead massively
        total_q_blocks = num_seqs

    TILE_SIZE = 16
    BLOCK_SIZE = chunk_size

    unified_query_state_2d[(total_q_blocks, num_kv_heads)](
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
        BLOCK_SIZE,
        TILE_SIZE,
        head_dim,
        BLOCK_Q,
        num_seqs,
        d_tile,
        BLOCK_M,
        deg,
        chunk_size,
        has_prefill,
        float(state_dim),
        output.stride(0),
        output.stride(1),
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *gate.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        *gate_cache.stride(),
        *memory.stride(),
        *ks.stride(),
        *block_table.stride(),
    )



def update_state(
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
    chunk_size : int, # int
    d_tile: int, # int
    deg: int, # int
):
    # Most of the kernels in update_state are parallelized over chunks.
    # Here we want to launch \sum_i(cdiv(query_len[i], chunk_size)) kernels
    # but materializing cu_seqlens_q on cpu is slow, so we take its uppper bound
    num_seqs = seq_lens.shape[0]
    num_tokens = key.shape[0]
    total_chunks = num_tokens // chunk_size + num_seqs
    num_kv_heads, head_dim = key.shape[1], key.shape[2]
    state_dim = memory.shape[2]

    # First gating needs to be cumsummed intra-chunk in-place
    cumsum_intra_chunk_gate[(total_chunks, num_kv_heads)](
        gate,
        gate_cache,
        block_table,
        cu_seqlens_q,
        cache_lens,
        cu_cache_lens,
        last_memorized_blk_idx,
        chunk_size,
        num_seqs,
        *gate.stride(),
        *gate_cache.stride(),
        *block_table.stride(),
    )

    # Then perform intra-chunk memory updates and cache updates.
    # We want to launch \sum_i(cdiv(query_len[i] + cache_lens[i], chunk_size)) 
    # kernels along sequence dimension. To avoid materialization on cpu, we take
    # its upper bound: 
    #  \sum_i(cdiv(query_len[i] + cache_lens[i], chunk_size)) 
    #  <= \sum_i(cdiv(query_len[i] + chunk_size, chunk_size)) 
    #  = \sum_i(cdiv(query_len[i], chunk_size)) + \sum_i(1)
    #  <= num_tokens // chunk_size + num_seqs + num_seqs
    #  = num_tokens // chunk_size + 2 * num_seqs
    total_chunks_including_cache = num_tokens // chunk_size + 2 * num_seqs
    BLOCK_S = d_tile ** deg
    num_state_blocks = state_dim // BLOCK_S
    BLOCK_T = 16
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
        deg,
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

    # Then perform inter-chunk memory cumsum. This might be inefficient if one sequence
    # is much longer than the rest. Can be improved with associative scan.
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


# handles both prefill and decode
def power_retention_varlen(
    output: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    query: torch.Tensor, # [num_tokens, num_query_heads, head_dim]
    key: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
    gate: torch.Tensor, # [num_tokens, num_key_heads]
    memory: torch.Tensor, # [num_blks, num_kv_heads, state_dim, head_dim]
    ks: torch.Tensor, # [num_blks, num_kv_heads, state_dim]
    key_cache, # [num_blks, chunk_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads, head_dim]
    gate_cache: torch.Tensor, # [num_blks, chunk_size, num_kv_heads]
    block_table: torch.Tensor, # [num_reqs, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor, # [num_reqs + 1]
    seq_lens: torch.Tensor, # [num_reqs]
    cache_lens: torch.Tensor, # [num_reqs]
    cu_cache_lens: torch.Tensor, # [num_reqs + 1]
    last_memorized_blk_idx: torch.Tensor, # [num_reqs]
    cu_seqlens_padded_q: torch.Tensor, # [num_reqs + 1]
    scale: float, # float
    chunk_size : int, # int
    d_tile: int, # int
    deg: int, # int
    has_prefill: bool, # bool
):
    """
    last_hashed_memory_blk_idx[i] the last block index in request i's block table that has memory computed, which is also the last block hashed by vllm. If there is no memory computed, last_hashed_memory_blk_idx[i] = -1.
    """
    update_state(
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
        seq_lens,
        cache_lens,
        cu_cache_lens,
        last_memorized_blk_idx,
        chunk_size,
        d_tile,
        deg,
    )

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
        scale,
        d_tile,
        deg,
        has_prefill,
    )
