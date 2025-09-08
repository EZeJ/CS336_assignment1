"""
KV-Cache Implementation for Efficient Autoregressive Generation

This module provides key-value caching to accelerate autoregressive text generation
by avoiding recomputation of attention keys and values for previous tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import logging
from einops import rearrange

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    max_batch_size: int = 1
    max_seq_len: int = 2048
    num_heads: int = 8
    head_dim: int = 64
    num_layers: int = 6
    dtype: torch.dtype = torch.float16
    device: torch.device = torch.device("cpu")


class KVCache:
    """
    Key-Value cache for a single transformer layer.
    
    Stores and manages cached key and value tensors to avoid recomputation
    during autoregressive generation.
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int, 
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Initialize cache tensors
        self.keys = torch.zeros(
            (max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.values = torch.zeros(
            (max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        
        # Track current sequence length for each batch item
        self.seq_lengths = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        
        # Track if cache is initialized
        self.initialized = False
    
    def get_shape(self) -> Tuple[int, ...]:
        """Get the shape of cached tensors."""
        return (self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim)
    
    def reset(self, batch_indices: Optional[List[int]] = None):
        """
        Reset cache for specified batch indices or all batches.
        
        Args:
            batch_indices: List of batch indices to reset, or None for all
        """
        if batch_indices is None:
            # Reset all
            self.keys.zero_()
            self.values.zero_()
            self.seq_lengths.zero_()
        else:
            # Reset specific batch indices
            for idx in batch_indices:
                if 0 <= idx < self.max_batch_size:
                    self.keys[idx].zero_()
                    self.values[idx].zero_()
                    self.seq_lengths[idx] = 0
        
        self.initialized = len(batch_indices) == 0 if batch_indices else False
    
    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        batch_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new keys and values.
        
        Args:
            keys: New key tensor [batch_size, num_heads, seq_len, head_dim]
            values: New value tensor [batch_size, num_heads, seq_len, head_dim]
            batch_indices: Batch indices being updated
        
        Returns:
            Tuple of (cached_keys, cached_values) including new additions
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Validate dimensions
        assert values.shape == keys.shape, "Keys and values must have same shape"
        assert num_heads == self.num_heads, f"Expected {self.num_heads} heads, got {num_heads}"
        assert head_dim == self.head_dim, f"Expected {self.head_dim} head dim, got {head_dim}"
        
        if batch_indices is None:
            batch_indices = list(range(batch_size))
        
        # Determine where to place new keys/values in cache
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx >= self.max_batch_size:
                continue
            
            current_len = self.seq_lengths[batch_idx].item()
            new_len = current_len + seq_len
            
            if new_len > self.max_seq_len:
                logger.warning(f"Sequence length {new_len} exceeds max_seq_len {self.max_seq_len}")
                # Truncate to fit
                available_space = self.max_seq_len - current_len
                if available_space <= 0:
                    continue
                seq_len_to_add = min(seq_len, available_space)
            else:
                seq_len_to_add = seq_len
            
            # Update cache
            self.keys[batch_idx, :, current_len:current_len + seq_len_to_add] = \
                keys[i, :, :seq_len_to_add]
            self.values[batch_idx, :, current_len:current_len + seq_len_to_add] = \
                values[i, :, :seq_len_to_add]
            
            # Update sequence length
            self.seq_lengths[batch_idx] = current_len + seq_len_to_add
        
        self.initialized = True
        
        # Return cached keys and values up to current lengths
        max_len = self.seq_lengths[batch_indices].max().item()
        cached_keys = self.keys[batch_indices, :, :max_len]
        cached_values = self.values[batch_indices, :, :max_len]
        
        return cached_keys, cached_values
    
    def get_cached(
        self,
        batch_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get currently cached keys and values.
        
        Args:
            batch_indices: Batch indices to retrieve, or None for all active
        
        Returns:
            Tuple of (cached_keys, cached_values)
        """
        if batch_indices is None:
            # Get all non-empty sequences
            batch_indices = [i for i in range(self.max_batch_size) if self.seq_lengths[i] > 0]
        
        if not batch_indices:
            # Return empty tensors
            return (
                torch.empty(0, self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(0, self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
            )
        
        max_len = self.seq_lengths[batch_indices].max().item()
        cached_keys = self.keys[batch_indices, :, :max_len]
        cached_values = self.values[batch_indices, :, :max_len]
        
        return cached_keys, cached_values
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_elements = self.keys.numel() + self.values.numel()
        memory_mb = (total_elements * element_size) / (1024 * 1024)
        
        return {
            "memory_mb": memory_mb,
            "keys_mb": (self.keys.numel() * element_size) / (1024 * 1024),
            "values_mb": (self.values.numel() * element_size) / (1024 * 1024),
            "total_elements": total_elements,
            "utilization": self.seq_lengths.float().mean().item() / self.max_seq_len,
        }


class KVCacheManager:
    """
    Manages KV caches for all layers of a transformer model.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.caches: List[KVCache] = []
        
        # Create cache for each layer
        for layer_idx in range(config.num_layers):
            cache = KVCache(
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                dtype=config.dtype,
                device=config.device,
            )
            self.caches.append(cache)
        
        self.current_length = 0
        logger.info(f"KVCacheManager initialized with {config.num_layers} layers")
    
    def reset(self, batch_indices: Optional[List[int]] = None):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset(batch_indices)
        self.current_length = 0
    
    def update_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        batch_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer."""
        if layer_idx >= len(self.caches):
            raise IndexError(f"Layer {layer_idx} not found in cache manager")
        
        return self.caches[layer_idx].update(keys, values, batch_indices)
    
    def get_layer_cache(
        self,
        layer_idx: int,
        batch_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a specific layer."""
        if layer_idx >= len(self.caches):
            raise IndexError(f"Layer {layer_idx} not found in cache manager")
        
        return self.caches[layer_idx].get_cached(batch_indices)
    
    def is_initialized(self) -> bool:
        """Check if cache is initialized."""
        return all(cache.initialized for cache in self.caches)
    
    def get_total_memory_usage(self) -> Dict[str, Any]:
        """Get total memory usage across all caches."""
        total_memory = 0
        layer_stats = []
        
        for i, cache in enumerate(self.caches):
            stats = cache.get_memory_usage()
            total_memory += stats["memory_mb"]
            layer_stats.append({"layer": i, **stats})
        
        return {
            "total_memory_mb": total_memory,
            "num_layers": len(self.caches),
            "avg_memory_per_layer_mb": total_memory / len(self.caches),
            "layer_stats": layer_stats,
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "max_seq_len": self.config.max_seq_len,
                "num_heads": self.config.num_heads,
                "head_dim": self.config.head_dim,
                "dtype": str(self.config.dtype),
            }
        }
    
    def optimize_memory(self):
        """Optimize memory usage by cleaning up unused cache entries."""
        for cache in self.caches:
            # Find batch indices with zero length (unused)
            unused_indices = [
                i for i in range(cache.max_batch_size) 
                if cache.seq_lengths[i] == 0
            ]
            
            if unused_indices:
                cache.reset(unused_indices)
        
        logger.debug("Cache memory optimization completed")


def create_kv_cache_manager(
    model: nn.Module,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> KVCacheManager:
    """
    Create a KV cache manager for a given model.
    
    Args:
        model: Transformer model to create cache for
        max_batch_size: Maximum batch size for generation
        max_seq_len: Maximum sequence length to cache
        dtype: Data type for cache tensors
        device: Device to place cache tensors on
    
    Returns:
        Configured KVCacheManager instance
    """
    
    # Auto-detect model configuration
    if device is None:
        device = next(model.parameters()).device
    
    if dtype is None:
        # Use same dtype as model parameters
        dtype = next(model.parameters()).dtype
    
    # Try to extract model dimensions
    num_layers = getattr(model, 'num_layers', 6)
    num_heads = getattr(model, 'num_heads', 8)
    d_model = getattr(model, 'd_model', 512)
    head_dim = d_model // num_heads
    
    # Create configuration
    config = CacheConfig(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype=dtype,
        device=device,
    )
    
    return KVCacheManager(config)