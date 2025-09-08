"""
Inference modules for efficient text generation.

This module provides optimized inference capabilities including:
- KV-cache for autoregressive generation
- Multiple sampling strategies
- Batch generation support
- Performance optimization utilities
"""

from .cache import KVCache, KVCacheManager
from .generator import TextGenerator, GenerationConfig
from .samplers import (
    GreedySampler,
    TopKSampler, 
    TopPSampler,
    TemperatureSampler,
    MultinomialSampler,
    create_sampler,
)
from .utils import (
    prepare_input_ids,
    prepare_attention_mask,
    estimate_generation_memory,
)

__all__ = [
    "KVCache",
    "KVCacheManager", 
    "TextGenerator",
    "GenerationConfig",
    "GreedySampler",
    "TopKSampler",
    "TopPSampler", 
    "TemperatureSampler",
    "MultinomialSampler",
    "create_sampler",
    "prepare_input_ids",
    "prepare_attention_mask",
    "estimate_generation_memory",
]