# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class RetentionAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["RetentionAttentionMetadataBuilder"]:
        return RetentionAttentionMetadataBuilder


@dataclass
class RetentionAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    state_indices_tensor: torch.Tensor  # shape: [batch,]
    sk_indices_tensor: torch.Tensor  # shape: [batch,]
    k_cache_indices_tensor: torch.Tensor  # shape: [batch,]
    v_cache_indices_tensor: torch.Tensor  # shape: [batch,]
    g_cache_indices_tensor: torch.Tensor  # shape: [batch,]


class RetentionAttentionMetadataBuilder(AttentionMetadataBuilder[RetentionAttentionMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(
        self,
        common_prefix_len: int,
    )