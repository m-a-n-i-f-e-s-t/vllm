# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PowerCoder model configuration"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class PowerCoderConfig(PretrainedConfig):
    """
    Configuration class for PowerCoder models.
    
    This is a simplified version of the PowerCoder configuration that works
    with vLLM. It extends PretrainedConfig and defines the architecture
    parameters for the PowerCoder model.
    """

    model_type = "powercoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 3072,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        hidden_act: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.018042,
        norm_epsilon: float = 1e-5,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        sliding_window: Optional[int] = None,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        use_bias: bool = True,
        prefill_chunk_size: int = 1024,
        switch_over_seq_len: int = 2048,
        chunk_size: int = 64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.use_bias = use_bias
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_epsilon = norm_epsilon
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.prefill_chunk_size = prefill_chunk_size
        self.chunk_size = chunk_size
        self.switch_over_seq_len = switch_over_seq_len
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = ["PowerCoderConfig"]

