# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Powen3MoE model configuration"""

from transformers import Qwen3MoeConfig
from transformers import AutoConfig


class Powen3MoEConfig(Qwen3MoeConfig):
    """
    Configuration class for Powen3MoE models.
    
    Powen3MoE combines Qwen3MoE's Mixture-of-Experts architecture with 
    retention mechanism for efficient long-context modeling. It inherits 
    all Qwen3MoE parameters and adds retention-specific configuration.
    
    Args:
        chunk_size (int, optional, defaults to 64):
            Size of retention chunks for state management.
        switch_over_seq_len (int, optional, defaults to 2048):
            Sequence length threshold for switching retention computation modes.
        prefill_chunk_size (int, optional, defaults to 64):
            Chunk size for prefill operations in retention mechanism.
        **kwargs: Additional arguments passed to Qwen3MoeConfig.
    """
    
    model_type = "powen3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        chunk_size: int = 64,
        switch_over_seq_len: int = 2048,
        prefill_chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        self.prefill_chunk_size = prefill_chunk_size


# Register Powen3MoEConfig with transformers' AutoConfig
# This allows transformers to recognize 'powen3_moe' as a valid model_type
# when loading checkpoints
AutoConfig.register("powen3_moe", Powen3MoEConfig)

__all__ = ["Powen3MoEConfig"]

