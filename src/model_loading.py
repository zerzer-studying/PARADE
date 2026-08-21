"""Backbone loading helpers shared by WVS training entry points."""

import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def load_backbone_config(model_name: str):
    is_local = os.path.exists(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    return config, is_local


def is_mistral3_multimodal_config(config) -> bool:
    return str(getattr(config, "model_type", "") or "") == "mistral3"


def load_backbone_tokenizer(model_name: str, config=None, is_local=None):
    if config is None or is_local is None:
        config, is_local = load_backbone_config(model_name)
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": is_local,
    }
    if is_mistral3_multimodal_config(config):
        # Required by current Transformers for Mistral's Tekken conversion.
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_backbone_model(model_name: str, *, device_map, config=None,
                        is_local=None):
    if config is None or is_local is None:
        config, is_local = load_backbone_config(model_name)
    loader = (
        AutoModelForImageTextToText
        if is_mistral3_multimodal_config(config)
        else AutoModelForCausalLM
    )
    return loader.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=is_local,
    )
