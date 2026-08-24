# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.transformer import TransformerConfig


def _fake_build_module(_spec, **kwargs):
    module = nn.Module()
    module.config = kwargs["config"]
    module.layer_number = kwargs["layer_number"]
    return module


def _config(**kwargs):
    return TransformerConfig(
        num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True, **kwargs
    )


def test_hybrid_stack_builds_main_layers_with_resolved_configs():
    config = _config(
        per_layer_config_overrides=[
            {"mamba_num_heads": 8, "mamba_state_dim": 64},
            {"num_moe_experts": 4, "moe_ffn_hidden_size": 96},
            {"ffn_hidden_size": 384},
        ]
    )

    with patch(
        "megatron.core.models.hybrid.hybrid_block.build_module", side_effect=_fake_build_module
    ):
        stack = HybridStack(
            config=config,
            submodules=HybridStackSubmodules(),
            layer_type_list=["M", "E", "-"],
            post_layer_norm=False,
            pg_collection=SimpleNamespace(pp=object(), tp=object()),
        )

    assert stack.layers[0].config.mamba_num_heads == 8
    assert stack.layers[0].config.mamba_state_dim == 64
    assert stack.layers[1].config.num_moe_experts == 4
    assert stack.layers[1].config.moe_ffn_hidden_size == 96
    assert stack.layers[2].config.ffn_hidden_size == 384


def test_hybrid_stack_builds_mtp_positions_with_resolved_configs():
    config = _config(
        mtp_pattern_length=2,
        mtp_per_layer_config_overrides=[{"mamba_num_groups": 2}, {"moe_router_topk": 3}],
    )

    with patch(
        "megatron.core.models.hybrid.hybrid_block.build_module", side_effect=_fake_build_module
    ):
        stack = HybridStack(
            config=config,
            submodules=HybridStackSubmodules(),
            layer_type_list=["M", "E"],
            post_layer_norm=False,
            pg_collection=SimpleNamespace(pp=object(), tp=object()),
            is_mtp_layer=True,
        )

    assert stack.layers[0].config.mamba_num_groups == 2
    assert stack.layers[1].config.moe_router_topk == 3
