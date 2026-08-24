# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.transformer import TransformerConfig


def _config(**kwargs):
    return TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        **kwargs,
    )


def test_resolves_all_supported_main_layer_overrides():
    override = {
        "ffn_hidden_size": 384,
        "moe_ffn_hidden_size": 96,
        "num_moe_experts": 4,
        "moe_router_topk": 3,
        "moe_shared_expert_intermediate_size": 32,
        "mamba_state_dim": 64,
        "mamba_head_dim": 32,
        "mamba_num_groups": 4,
        "mamba_num_heads": 16,
    }
    config = _config(per_layer_config_overrides=[override, None, None])

    assert config.heterogeneous_block_specs
    first_layer = config.get_config_for_layer(1)
    for key, value in override.items():
        assert getattr(first_layer, key) == value
    assert not first_layer.heterogeneous_block_specs
    assert first_layer.per_layer_config_overrides is None
    assert first_layer.mtp_per_layer_config_overrides is None
    assert config.get_config_for_layer(2) is config
    assert config.num_moe_experts == 8


def test_none_override_values_preserve_architecture_semantics():
    config = _config(
        per_layer_config_overrides=[
            {"moe_shared_expert_intermediate_size": None, "mamba_num_heads": None},
            None,
            None,
        ]
    )

    layer = config.get_config_for_layer(1)
    assert layer.moe_shared_expert_intermediate_size is None
    assert layer.mamba_num_heads is None


def test_resolves_mtp_position_overrides():
    config = _config(
        mtp_pattern_length=2, mtp_per_layer_config_overrides=[{"ffn_hidden_size": 384}, None]
    )

    assert config.heterogeneous_block_specs
    assert config.get_config_for_mtp_layer(1).ffn_hidden_size == 384
    assert config.get_config_for_mtp_layer(2) is config


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"per_layer_config_overrides": [None]}, "must match num_layers"),
        (
            {"per_layer_config_overrides": [{"not_a_config_field": 4}, None, None]},
            "unknown override keys",
        ),
        (
            {"per_layer_config_overrides": [{"num_moe_experts": 0}, None, None]},
            "must be a positive int",
        ),
        (
            {"per_layer_config_overrides": [{"moe_router_topk": None}, None, None]},
            "does not accept None",
        ),
        (
            {"per_layer_config_overrides": [{"mamba_num_heads": True}, None, None]},
            "must be a positive int or None",
        ),
        ({"mtp_per_layer_config_overrides": [None]}, "mtp_pattern_length must be set"),
        (
            {"mtp_pattern_length": 2, "mtp_per_layer_config_overrides": [None]},
            "must match mtp_pattern_length",
        ),
    ],
)
def test_rejects_invalid_overrides(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _config(**kwargs)


def test_rejects_out_of_range_layer_numbers():
    config = _config(
        per_layer_config_overrides=[None, None, None],
        mtp_pattern_length=2,
        mtp_per_layer_config_overrides=[None, None],
    )

    with pytest.raises(ValueError, match="Invalid layer_number"):
        config.get_config_for_layer(0)
    with pytest.raises(ValueError, match="Invalid layer_number"):
        config.get_config_for_layer(4)
    with pytest.raises(ValueError, match="Invalid MTP layer_number"):
        config.get_config_for_mtp_layer(0)
    with pytest.raises(ValueError, match="Invalid MTP layer_number"):
        config.get_config_for_mtp_layer(3)
