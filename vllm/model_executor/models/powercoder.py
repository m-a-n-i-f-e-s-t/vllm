# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PowerCoder model compatible with vLLM's retention layer.

PowerCoder uses power retention mechanism for efficient long-context modeling.
Unlike standard attention which has O(n) memory complexity for KV cache,
retention maintains constant-size state regardless of context length.

Architecture:
- Similar to decoder-only transformers (GPT/Llama-style)
- Replaces standard attention with power retention
- Uses LayerNorm (not RMSNorm), RoPE, and standard MLP
- Maintains state tensor, sum_of_keys, and short token cache

For more details on retention, see:
https://github.com/m-a-n-i-f-e-s-t/retention
"""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
# Use PyTorch's native LayerNorm since vLLM's custom LayerNorm doesn't support bias
# and PowerCoder uses bias=False in config but still expects LayerNorm behavior
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.retention import Retention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from retention._utils import compute_expanded_dim
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree, SupportsPP)
from vllm.model_executor.models.retention_cache import (RetentionCacheManager,
                                                         RetentionCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class PowerCoderMLP(nn.Module):
    """Standard MLP with vLLM's parallel linear layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.c_fc",
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.act = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class PowerCoderAttention(nn.Module):
    """
    Attention module using power retention mechanism.
    
    Projects inputs to Q, K, V, G (gate), applies RoPE to Q/K,
    then calls vLLM's Retention layer for the core computation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        p: int = 2,
        chunk_size: int = 1024,
        switch_over_seq_len: int = 2048,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = True,
        layer_idx: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.tp_heads = num_heads // tp_size
        self.tp_kv_heads = max(1, num_kv_heads // tp_size)

        self.scaling = self.head_dim ** -0.5

        # Q, K, V, G projections
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * head_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        # Gate projection (for retention mechanism)
        self.g_proj = ColumnParallelLinear(
            hidden_size,
            num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE embeddings
        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
        )

        # Retention layer
        layer_prefix = f"{prefix}.retention"
        self.retention = Retention(
            num_heads=self.tp_heads,
            head_size=head_dim,
            scale=self.scaling,
            p=p,
            num_kv_heads=self.tp_kv_heads,
            chunk_size=chunk_size,
            switch_over_seq_len=switch_over_seq_len,
            bias=bias,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
            prefix=layer_prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        retention_cache_params: Optional[RetentionCacheParams] = None,
    ) -> torch.Tensor:
        # Determine actual tokens for current step (V1) to avoid padding in capture
        # For profile run (attn_metadata=None), use full buffer size
        num_actual_tokens = hidden_states.shape[0]
        if envs.VLLM_USE_V1:
            from vllm.forward_context import get_forward_context
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                meta = attn_metadata.get(self.retention.layer_name, None)
                if meta is not None:
                    num_actual_tokens = meta.num_prefill_tokens + meta.num_decode_tokens

        # Always slice (even if no-op) for CUDA graph compatibility
        # During capture, num_actual_tokens == hidden_states.shape[0]
        hidden_states = hidden_states[:num_actual_tokens]
        positions = positions[:num_actual_tokens]

        # Project to Q, K, V, G on the sliced tokens
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        gate, _ = self.g_proj(hidden_states)

        # Reshape for multi-head attention
        # [batch * seq_len, tp_heads * head_dim] -> [batch * seq_len, tp_heads, head_dim]
        query = query.view(-1, self.tp_heads, self.head_dim)
        key = key.view(-1, self.tp_kv_heads, self.head_dim)
        value = value.view(-1, self.tp_kv_heads, self.head_dim)
        gate = gate.view(-1, self.tp_kv_heads)

        # Apply RoPE
        query, key = self.rotary_emb(positions, query, key)

        # Apply log-sigmoid to gate
        gate = torch.nn.functional.logsigmoid(gate.to(torch.float32))

        # Prepare output tensor sized to actual tokens
        output = torch.empty_like(query)

        # Call retention layer (V1 path will delegate to custom op internally)
        self.retention(
            query=query,
            key=key,
            gate=gate,
            value=value,
            output=output,
            cache_params=retention_cache_params,
        )

        # Reshape and project output (already trimmed)
        output2d = output.view(-1, self.tp_heads * self.head_dim)
        output2d, _ = self.o_proj(output2d)

        return output2d


class PowerCoderDecoderLayer(nn.Module):
    """Single decoder layer with retention and MLP."""

    def __init__(
        self,
        config,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Retention attention
        self.self_attn = PowerCoderAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            p=2,  # Power parameter (only p=2 supported currently)
            chunk_size=getattr(config, "chunk_size", 1024),
            switch_over_seq_len=getattr(config, "switch_over_seq_len", 2048),
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            bias=config.use_bias,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn",
        )

        # MLP
        self.mlp = PowerCoderMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=config.use_bias,
            prefix=f"{prefix}.mlp",
        )

        # Layer norms (PowerCoder uses LayerNorm, not RMSNorm)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        retention_cache_params: Optional[RetentionCacheParams] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Add residual and normalize
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            retention_cache_params=retention_cache_params,
        )

        # MLP with pre-norm
        # Add residual and normalize
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class PowerCoderModel(nn.Module):
    """PowerCoder model without the language modeling head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        # Decoder layers
        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return PowerCoderDecoderLayer(
                config,
                model_config,
                cache_config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

        # For pipeline parallelism
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        retention_cache_params: Optional[RetentionCacheParams] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Forward through decoder layers
        for global_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[global_idx]
            layer_retention_cache_params = None
            if retention_cache_params is not None:
                layer_retention_cache_params = retention_cache_params.at_layer_idx(
                    global_idx)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                retention_cache_params=layer_retention_cache_params,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        # Final norm with residual
        hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


class PowerCoderForCausalLM(nn.Module, HasInnerState, IsAttentionFree,
                              SupportsPP):
    """PowerCoder model with language modeling head."""

    # PowerCoder uses separate Q, K, V, G projections (not packed)
    # No packed_modules_mapping needed
    
    # Modules that should be fused
    supported_lora_modules = [
        "q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "c_fc", "c_proj"
    ]
    
    # Embedding modules
    embedding_modules = {
        "embed_tokens": "input_embeddings",
    }
    
    # Padding modules
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config

        self.model = PowerCoderModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()

        # Initialize retention cache manager for v0 path
        if not envs.VLLM_USE_V1:
            num_layers = config.num_hidden_layers
            # Derive dtypes and shapes from config
            state_dtypes = self.get_mamba_state_dtype_from_config(vllm_config)
            state_shapes = self.get_mamba_state_shape_from_config(
                vllm_config, use_v1=False)

            max_batch_size = vllm_config.scheduler_config.max_num_seqs
            if not vllm_config.model_config.enforce_eager:
                max_batch_size = vllm_config.pad_for_cudagraph(max_batch_size)

            state_shape = (num_layers, max_batch_size, *state_shapes[0])
            sk_shape = (num_layers, max_batch_size, *state_shapes[1])
            cache_shape = (num_layers, max_batch_size, *state_shapes[2])

            self.retention_cache_manager = RetentionCacheManager(
                dtype=state_dtypes[0],
                state_shape=state_shape,
                sk_shape=sk_shape,
                cache_shape=cache_shape,
            )
        else:
            self.retention_cache_manager = None

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        if self.retention_cache_manager:
            return self.retention_cache_manager.copy_inputs_before_cuda_graphs(
                input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        if self.retention_cache_manager:
            return self.retention_cache_manager.get_seqlen_agnostic_capture_inputs(
                batch_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Handle retention cache (V0 path)
        retention_cache_params = None
        if not envs.VLLM_USE_V1 and self.retention_cache_manager is not None:
            cache_tensors, state_indices_tensor = (
                self.retention_cache_manager.current_run_tensors(**kwargs))
            retention_cache_params = RetentionCacheParams(
                state_tensor=cache_tensors[0],
                sk_tensor=cache_tensors[1],
                cache_tensor=cache_tensors[2],
                state_indices_tensor=state_indices_tensor,
            )

        # V1 path: retention layer gets cache from forward_context
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            retention_cache_params=retention_cache_params,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.
        
        PowerCoder uses separate Q, K, V, G projections (not packed together).
        This method handles:
        - Separate projection weights (q_proj, k_proj, v_proj, g_proj)
        - Pipeline parallelism (skip missing layers)
        - Tensor parallelism (handled by ColumnParallelLinear/RowParallelLinear)
        - Bias parameters
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            # Skip rotary embedding inverse frequency buffer
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Skip bias if not in model (some configs may not use bias)
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            # Skip layers not on this pipeline parallel rank
            if is_pp_missing_parameter(name, self):
                continue
            
            # Handle LayerNorm parameters
            # PyTorch LayerNorm uses "weight" and "bias", but checkpoint might use
            # different names - the default loader should handle this
            if name not in params_dict:
                # Skip if parameter doesn't exist in model
                continue
            
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                   default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
                
        return loaded_params

    # HasInnerState implementation
    def get_mamba_type(self) -> str:
        """Return the type of mamba/inner state mechanism."""
        return "retention"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return dtypes for (state_tensor, sk_tensor, cache_tensor)."""
        first_layer = self.model.layers[0]
        return first_layer.self_attn.retention.get_state_dtype()

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return shapes for (state_tensor, sk_tensor, cache_tensor)."""
        first_layer = self.model.layers[0]
        return first_layer.self_attn.retention.get_state_shape()

    @classmethod
    def get_layers_block_type(cls, vllm_config: VllmConfig):
        """All layers use retention (no hybrid attention/retention)."""
        config = vllm_config.model_config.hf_config
        return ["retention"] * config.num_hidden_layers

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.retention_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, ...], ...]:
        config = vllm_config.model_config.hf_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        chunk_size = getattr(config, "chunk_size", 1024)
        state_dim = compute_expanded_dim(head_dim, deg=2)
        return MambaStateShapeCalculator.retention_state_shape(
            num_kv_heads=num_kv_heads,
            state_dim=state_dim,
            head_dim=head_dim,
            value_dim=head_dim,
            gating_dim=1,
            chunk_size=chunk_size,
        )


def get_sampler():
    """Get the sampler."""
    from vllm.model_executor.layers.sampler import Sampler
    return Sampler()

