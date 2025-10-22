# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Powen3 model compatible with vLLM's retention layer.

Powen3 combines Qwen3's architecture with retention mechanism for efficient
long-context modeling. Unlike standard attention which has O(n) memory complexity
for KV cache, retention maintains constant-size state regardless of context length.

Architecture:
- Based on Qwen3 (decoder-only transformer)
- Replaces standard attention with power retention
- Includes QK normalization like Qwen3
- Uses RMSNorm, RoPE, and standard MLP
- Maintains state tensor, sum_of_keys, and short token cache
"""

from collections.abc import Iterable
from typing import Any, Optional, Union

# Global flag to control whether to use Retention or standard Attention
# 
# When USE_RETENTION = True:
#   - Uses power retention mechanism for efficient long-context modeling
#   - Powen3ForCausalLM inherits from HasInnerState and IsAttentionFree
#   - Maintains retention state (state_tensor, sk_tensor, cache_tensor)
#   - Works with both V0 and V1 code paths
#
# When USE_RETENTION = False:
#   - Uses standard vLLM Attention layer with KV cache
#   - Powen3ForCausalLM only inherits from SupportsPP (not HasInnerState/IsAttentionFree)
#   - Compatible with all standard attention optimizations and profiling tools
#   - Works with both V0 and V1 code paths
#
# To switch modes, simply change this flag and restart vLLM
USE_RETENTION = True

import torch
from torch import nn
from transformers import Qwen3Config

# Import Powen3Config to ensure it's registered with transformers' AutoConfig
# This must happen before trying to load a Powen3 checkpoint
from vllm.transformers_utils.configs.powen3 import Powen3Config

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.retention import Retention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree, SupportsLoRA,
                                                   SupportsPP)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from retention._utils import compute_expanded_dim

from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    maybe_prefix)


class Powen3Retention(nn.Module):
    """
    Retention module using power retention mechanism with QK normalization.
    
    Similar to Qwen3Attention but replaces standard attention with retention.
    Projects inputs to Q, K, V using packed QKVParallelLinear, adds separate
    gate projection G, applies QK normalization and RoPE, then calls vLLM's
    Retention layer for the core computation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        p: int = 2,
        chunk_size: int = 64,
        switch_over_seq_len: int = 2048,
        model_config: Optional[Any] = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.layer_idx = layer_idx

        # Packed QKV projection (like Qwen3)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # Separate gate projection for retention
        self.g_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        # QK normalization (like Qwen3)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Conditionally initialize Retention or Attention layer
        if USE_RETENTION:
            self.retention = Retention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                p=p,
                num_kv_heads=self.num_kv_heads,
                chunk_size=chunk_size,
                switch_over_seq_len=switch_over_seq_len,
                bias=qkv_bias,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.retention",
            )
            self.attn = None
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=attn_type,
                prefix=f"{prefix}.attn",
                **{
                    "layer_idx": extract_layer_index(prefix),
                    "dual_chunk_attention_config": dual_chunk_attention_config,
                } if dual_chunk_attention_config else {},
            )
            self.retention = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if USE_RETENTION:
            # Retention path
            # Determine actual tokens for current step (V1) to avoid padding in capture
            num_actual_tokens = hidden_states.shape[0]
            if envs.VLLM_USE_V1:
                from vllm.forward_context import get_forward_context
                attn_metadata_v1 = get_forward_context().attn_metadata
                if isinstance(attn_metadata_v1, dict):
                    meta = attn_metadata_v1.get(self.retention.layer_name, None)
                    if meta is not None:
                        num_actual_tokens = meta.num_prefill_tokens + meta.num_decode_tokens

            # Always slice (even if no-op) for CUDA graph compatibility
            hidden_states = hidden_states[:num_actual_tokens]
            positions = positions[:num_actual_tokens]

            # Project to Q, K, V, G
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate, _ = self.g_proj(hidden_states)

            # Apply QK normalization
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                             self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                             self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            
            # Apply RoPE
            q, k = self.rotary_emb(positions, q, k)

            # Reshape for retention
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            gate = gate.view(-1, self.num_kv_heads)

            # Apply log-sigmoid to gate
            gate = torch.nn.functional.logsigmoid(gate.to(torch.float32))

            # Prepare output tensor sized to actual tokens
            output = torch.empty_like(q)

            # Call retention layer
            self.retention(
                query=q,
                key=k,
                gate=gate,
                value=v,
                output=output
            )

            # Reshape and project output
            output2d = output.view(-1, self.num_heads * self.head_dim)
            output2d, _ = self.o_proj(output2d)

            return output2d
        else:
            # Standard Attention path
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            # Apply QK normalization
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                             self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                             self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            
            # Apply RoPE
            q, k = self.rotary_emb(positions, q, k)
            
            # Call standard attention
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            
            return output


class Powen3DecoderLayer(nn.Module):
    """Single decoder layer with retention and MLP."""

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        model_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Powen3 uses causal attention/retention as it is a decoder-only model.
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Extract layer index for retention
        layer_idx = extract_layer_index(prefix)

        # get model config
        if model_config is None:
            vllm_config = get_current_vllm_config()
            model_config = vllm_config.model_config

        self.self_attn = Powen3Retention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            p=2,
            chunk_size=getattr(config, "chunk_size", 64),
            switch_over_seq_len=getattr(config, "switch_over_seq_len", 2048),
            model_config=model_config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "retention": Powen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Powen3Model(Qwen2Model):
    """Powen3 model without the language modeling head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Pass Powen3DecoderLayer as decoder_layer_type
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=Powen3DecoderLayer)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
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
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        # Final norm with residual
        hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states


class _Powen3ForCausalLMBase(nn.Module):
    """Base implementation for Powen3 model with language modeling head.
    
    This base class contains all the implementation details. The actual
    Powen3ForCausalLM class will inherit from this plus the appropriate
    interfaces based on whether USE_RETENTION is True or False.
    """

    # Powen3 uses packed QKV projection and separate gate projection
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    
    # Modules that should be fused
    supported_lora_modules = [
        "qkv_proj", "g_proj", "o_proj", "gate_up_proj", "down_proj"
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
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = Powen3Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list[torch.Tensor]] = None,
        attn_metadata: Optional[object] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Forward pass
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.
        
        Supports loading from Qwen3 checkpoints which don't have g_proj weights.
        When g_proj weights are missing, they are initialized to zeros with a warning.
        """
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        
        # Convert iterator to list to allow multiple passes
        weights_list = list(weights)
        weight_names = {name for name, _ in weights_list}
        
        # Check if any g_proj weights are missing (when loading from Qwen3 checkpoint)
        params_dict = dict(self.named_parameters())
        g_proj_params = {name for name in params_dict.keys() if '.g_proj.' in name}
        missing_g_proj = g_proj_params - weight_names
        
        if missing_g_proj and USE_RETENTION:
            logger.warning(
                "Loading Qwen3 checkpoint into Powen3 model. "
                "Gate projection (g_proj) weights not found in checkpoint. "
                "Initializing %d g_proj parameters to zeros. "
                "This is expected when loading a Qwen3 checkpoint.",
                len(missing_g_proj)
            )
            # Initialize missing g_proj weights to zeros
            for param_name in missing_g_proj:
                param = params_dict[param_name]
                param.data.zero_()
        
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(iter(weights_list))

    # HasInnerState implementation (only used when USE_RETENTION is True)
    def get_mamba_type(self) -> str:
        """Return the type of mamba/inner state mechanism."""
        if not USE_RETENTION:
            raise NotImplementedError("get_mamba_type not supported when USE_RETENTION is False")
        return "retention"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return dtypes for (state_tensor, sk_tensor, cache_tensor)."""
        if not USE_RETENTION:
            raise NotImplementedError("get_state_dtype not supported when USE_RETENTION is False")
        first_layer = self.model.layers[self.model.start_layer]
        return first_layer.self_attn.retention.get_state_dtype()

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return shapes for (state_tensor, sk_tensor, cache_tensor)."""
        if not USE_RETENTION:
            raise NotImplementedError("get_state_shape not supported when USE_RETENTION is False")
        first_layer = self.model.layers[self.model.start_layer]
        return first_layer.self_attn.retention.get_state_shape()

    @classmethod
    def get_layers_block_type(cls, vllm_config: VllmConfig):
        """All layers use retention (no hybrid attention/retention)."""
        if not USE_RETENTION:
            raise NotImplementedError("get_layers_block_type not supported when USE_RETENTION is False")
        config = vllm_config.model_config.hf_config
        return ["retention"] * config.num_hidden_layers

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, ...]:
        if not USE_RETENTION:
            raise NotImplementedError("get_mamba_state_dtype_from_config not supported when USE_RETENTION is False")
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
        if not USE_RETENTION:
            raise NotImplementedError("get_mamba_state_shape_from_config not supported when USE_RETENTION is False")
        config = vllm_config.model_config.hf_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        chunk_size = getattr(config, "chunk_size", 64)
        state_dim = compute_expanded_dim(head_dim, deg=2)
        return MambaStateShapeCalculator.retention_state_shape(
            num_kv_heads=num_kv_heads,
            state_dim=state_dim,
            head_dim=head_dim,
            value_dim=head_dim,
            gating_dim=1,
            chunk_size=chunk_size,
        )


# Conditionally create the final Powen3ForCausalLM class with appropriate base classes
# When USE_RETENTION is True: inherits from HasInnerState and IsAttentionFree
# When USE_RETENTION is False: only inherits from SupportsPP (standard attention model)
if USE_RETENTION:
    class Powen3ForCausalLM(_Powen3ForCausalLMBase, HasInnerState, 
                            IsAttentionFree, SupportsPP, SupportsLoRA):
        """Powen3 model with language modeling head using Retention mechanism."""
        pass
else:
    class Powen3ForCausalLM(_Powen3ForCausalLMBase, SupportsPP, SupportsLoRA):
        """Powen3 model with language modeling head using standard Attention."""
        pass

