import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from cs336_basics.inference.cache import (
    CacheConfig,
    KVCache,
    KVCacheManager,
    create_kv_cache_manager
)
from ..fixtures.test_models import create_mock_model


class TestCacheConfig:
    """Test CacheConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default config initialization."""
        config = CacheConfig()
        
        assert config.max_batch_size == 1
        assert config.max_seq_len == 2048
        assert config.num_heads == 8
        assert config.head_dim == 64
        assert config.num_layers == 6
        assert config.dtype == torch.float16
        assert config.device == torch.device("cpu")
    
    def test_custom_initialization(self):
        """Test config with custom parameters."""
        config = CacheConfig(
            max_batch_size=4,
            max_seq_len=1024,
            num_heads=12,
            head_dim=32,
            num_layers=8,
            dtype=torch.float32,
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        assert config.max_batch_size == 4
        assert config.max_seq_len == 1024
        assert config.num_heads == 12
        assert config.head_dim == 32
        assert config.num_layers == 8
        assert config.dtype == torch.float32


class TestKVCache:
    """Test KVCache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.max_batch_size = 2
        self.max_seq_len = 16
        self.num_heads = 4
        self.head_dim = 8
        self.dtype = torch.float32
        
        self.cache = KVCache(
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device
        )
    
    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_batch_size == self.max_batch_size
        assert self.cache.max_seq_len == self.max_seq_len
        assert self.cache.num_heads == self.num_heads
        assert self.cache.head_dim == self.head_dim
        assert self.cache.dtype == self.dtype
        assert self.cache.device == self.device
        assert not self.cache.initialized
        
        # Check tensor shapes
        assert self.cache.keys.shape == (2, 4, 16, 8)
        assert self.cache.values.shape == (2, 4, 16, 8)
        assert self.cache.seq_lengths.shape == (2,)
        
        # Check tensors are zero-initialized
        assert torch.allclose(self.cache.keys, torch.zeros_like(self.cache.keys))
        assert torch.allclose(self.cache.values, torch.zeros_like(self.cache.values))
        assert torch.all(self.cache.seq_lengths == 0)
    
    def test_get_shape(self):
        """Test getting cache tensor shape."""
        shape = self.cache.get_shape()
        assert shape == (2, 4, 16, 8)
    
    def test_reset_all(self):
        """Test resetting all cache entries."""
        # Add some data first
        keys = torch.randn(2, 4, 3, 8, dtype=self.dtype, device=self.device)
        values = torch.randn(2, 4, 3, 8, dtype=self.dtype, device=self.device)
        self.cache.update(keys, values)
        
        assert self.cache.initialized
        assert torch.any(self.cache.seq_lengths > 0)
        
        # Reset all
        self.cache.reset()
        
        assert not self.cache.initialized
        assert torch.all(self.cache.seq_lengths == 0)
        assert torch.allclose(self.cache.keys, torch.zeros_like(self.cache.keys))
        assert torch.allclose(self.cache.values, torch.zeros_like(self.cache.values))
    
    def test_reset_specific_indices(self):
        """Test resetting specific batch indices."""
        # Add data for both batch indices
        keys = torch.randn(2, 4, 3, 8, dtype=self.dtype, device=self.device)
        values = torch.randn(2, 4, 3, 8, dtype=self.dtype, device=self.device)
        self.cache.update(keys, values)
        
        # Reset only batch index 0
        self.cache.reset(batch_indices=[0])
        
        assert self.cache.seq_lengths[0] == 0
        assert self.cache.seq_lengths[1] == 3
        assert torch.allclose(self.cache.keys[0], torch.zeros_like(self.cache.keys[0]))
        assert not torch.allclose(self.cache.keys[1], torch.zeros_like(self.cache.keys[1]))
    
    def test_update_basic(self):
        """Test basic cache update functionality."""
        batch_size = 2
        seq_len = 3
        keys = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim, dtype=self.dtype, device=self.device)
        
        cached_keys, cached_values = self.cache.update(keys, values)
        
        assert self.cache.initialized
        assert torch.all(self.cache.seq_lengths == seq_len)
        assert cached_keys.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        assert cached_values.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Check that returned values match input
        assert torch.allclose(cached_keys, keys)
        assert torch.allclose(cached_values, values)
    
    def test_update_incremental(self):
        """Test incremental cache updates."""
        # First update
        keys1 = torch.randn(1, self.num_heads, 2, self.head_dim, dtype=self.dtype, device=self.device)
        values1 = torch.randn(1, self.num_heads, 2, self.head_dim, dtype=self.dtype, device=self.device)
        
        cached_keys1, cached_values1 = self.cache.update(keys1, values1, batch_indices=[0])
        assert self.cache.seq_lengths[0] == 2
        assert cached_keys1.shape == (1, self.num_heads, 2, self.head_dim)
        
        # Second update (incremental)
        keys2 = torch.randn(1, self.num_heads, 3, self.head_dim, dtype=self.dtype, device=self.device)
        values2 = torch.randn(1, self.num_heads, 3, self.head_dim, dtype=self.dtype, device=self.device)
        
        cached_keys2, cached_values2 = self.cache.update(keys2, values2, batch_indices=[0])
        assert self.cache.seq_lengths[0] == 5  # 2 + 3
        assert cached_keys2.shape == (1, self.num_heads, 5, self.head_dim)
        
        # Check that first tokens are preserved
        assert torch.allclose(cached_keys2[0, :, :2], keys1[0])
        assert torch.allclose(cached_keys2[0, :, 2:], keys2[0])
    
    def test_update_sequence_length_overflow(self):
        """Test behavior when sequence length exceeds maximum."""
        # Try to add more tokens than max_seq_len allows
        keys = torch.randn(1, self.num_heads, self.max_seq_len + 5, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(1, self.num_heads, self.max_seq_len + 5, self.head_dim, dtype=self.dtype, device=self.device)
        
        cached_keys, cached_values = self.cache.update(keys, values, batch_indices=[0])
        
        # Should be truncated to max_seq_len
        assert self.cache.seq_lengths[0] == self.max_seq_len
        assert cached_keys.shape == (1, self.num_heads, self.max_seq_len, self.head_dim)
        
        # Should match the first max_seq_len tokens
        assert torch.allclose(cached_keys[0], keys[0, :, :self.max_seq_len])
    
    def test_update_dimension_validation(self):
        """Test validation of input dimensions."""
        # Wrong number of heads
        keys = torch.randn(1, self.num_heads + 1, 2, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(1, self.num_heads + 1, 2, self.head_dim, dtype=self.dtype, device=self.device)
        
        with pytest.raises(AssertionError):
            self.cache.update(keys, values)
        
        # Wrong head dimension
        keys = torch.randn(1, self.num_heads, 2, self.head_dim + 1, dtype=self.dtype, device=self.device)
        values = torch.randn(1, self.num_heads, 2, self.head_dim + 1, dtype=self.dtype, device=self.device)
        
        with pytest.raises(AssertionError):
            self.cache.update(keys, values)
        
        # Mismatched key/value shapes
        keys = torch.randn(1, self.num_heads, 2, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(1, self.num_heads, 3, self.head_dim, dtype=self.dtype, device=self.device)
        
        with pytest.raises(AssertionError):
            self.cache.update(keys, values)
    
    def test_get_cached_empty(self):
        """Test getting cached values when cache is empty."""
        cached_keys, cached_values = self.cache.get_cached()
        
        assert cached_keys.shape == (0, self.num_heads, 0, self.head_dim)
        assert cached_values.shape == (0, self.num_heads, 0, self.head_dim)
    
    def test_get_cached_with_data(self):
        """Test getting cached values with data."""
        # Add some data
        keys = torch.randn(2, self.num_heads, 4, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(2, self.num_heads, 4, self.head_dim, dtype=self.dtype, device=self.device)
        self.cache.update(keys, values)
        
        cached_keys, cached_values = self.cache.get_cached()
        
        assert cached_keys.shape == (2, self.num_heads, 4, self.head_dim)
        assert cached_values.shape == (2, self.num_heads, 4, self.head_dim)
        assert torch.allclose(cached_keys, keys)
        assert torch.allclose(cached_values, values)
    
    def test_get_cached_specific_indices(self):
        """Test getting cached values for specific batch indices."""
        # Add data for both batch indices but with different lengths
        keys1 = torch.randn(1, self.num_heads, 2, self.head_dim, dtype=self.dtype, device=self.device)
        values1 = torch.randn(1, self.num_heads, 2, self.head_dim, dtype=self.dtype, device=self.device)
        self.cache.update(keys1, values1, batch_indices=[0])
        
        keys2 = torch.randn(1, self.num_heads, 3, self.head_dim, dtype=self.dtype, device=self.device)
        values2 = torch.randn(1, self.num_heads, 3, self.head_dim, dtype=self.dtype, device=self.device)
        self.cache.update(keys2, values2, batch_indices=[1])
        
        # Get only batch index 0
        cached_keys, cached_values = self.cache.get_cached(batch_indices=[0])
        
        # Should be padded to max length (3) but only contain data for batch 0
        assert cached_keys.shape == (1, self.num_heads, 3, self.head_dim)
        assert torch.allclose(cached_keys[0, :, :2], keys1[0])
        # The remaining positions should be zeros (uninitialized)
        assert torch.allclose(cached_keys[0, :, 2:], torch.zeros(self.num_heads, 1, self.head_dim))
    
    def test_memory_usage_calculation(self):
        """Test memory usage calculation."""
        usage = self.cache.get_memory_usage()
        
        assert "memory_mb" in usage
        assert "keys_mb" in usage
        assert "values_mb" in usage
        assert "total_elements" in usage
        assert "utilization" in usage
        
        assert usage["memory_mb"] > 0
        assert usage["keys_mb"] == usage["values_mb"]  # Keys and values same size
        assert usage["total_elements"] == self.cache.keys.numel() + self.cache.values.numel()
        assert 0 <= usage["utilization"] <= 1
        
        # Add some data and check utilization changes
        keys = torch.randn(2, self.num_heads, 4, self.head_dim, dtype=self.dtype, device=self.device)
        values = torch.randn(2, self.num_heads, 4, self.head_dim, dtype=self.dtype, device=self.device)
        self.cache.update(keys, values)
        
        new_usage = self.cache.get_memory_usage()
        assert new_usage["utilization"] > usage["utilization"]
        assert new_usage["utilization"] == 4.0 / self.max_seq_len  # (4 + 4) / 2 / 16


class TestKVCacheManager:
    """Test KVCacheManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CacheConfig(
            max_batch_size=2,
            max_seq_len=16,
            num_heads=4,
            head_dim=8,
            num_layers=3,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        self.manager = KVCacheManager(self.config)
    
    def test_initialization(self):
        """Test cache manager initialization."""
        assert len(self.manager.caches) == 3
        assert self.manager.current_length == 0
        assert self.manager.config == self.config
        
        # Check that all caches have correct configuration
        for cache in self.manager.caches:
            assert cache.max_batch_size == self.config.max_batch_size
            assert cache.max_seq_len == self.config.max_seq_len
            assert cache.num_heads == self.config.num_heads
            assert cache.head_dim == self.config.head_dim
    
    def test_reset_all_layers(self):
        """Test resetting all layer caches."""
        # Add data to all layers
        for layer_idx in range(3):
            keys = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
            values = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
            self.manager.update_layer(layer_idx, keys, values)
        
        assert self.manager.is_initialized()
        
        # Reset all
        self.manager.reset()
        
        assert not self.manager.is_initialized()
        assert self.manager.current_length == 0
        
        for cache in self.manager.caches:
            assert not cache.initialized
            assert torch.all(cache.seq_lengths == 0)
    
    def test_reset_specific_batch_indices(self):
        """Test resetting specific batch indices across all layers."""
        # Add data
        for layer_idx in range(3):
            keys = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
            values = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
            self.manager.update_layer(layer_idx, keys, values)
        
        # Reset batch index 0
        self.manager.reset(batch_indices=[0])
        
        for cache in self.manager.caches:
            assert cache.seq_lengths[0] == 0
            assert cache.seq_lengths[1] == 3
    
    def test_update_layer_basic(self):
        """Test updating specific layer cache."""
        layer_idx = 1
        keys = torch.randn(2, 4, 5, 8, dtype=self.config.dtype, device=self.config.device)
        values = torch.randn(2, 4, 5, 8, dtype=self.config.dtype, device=self.config.device)
        
        cached_keys, cached_values = self.manager.update_layer(layer_idx, keys, values)
        
        assert cached_keys.shape == keys.shape
        assert cached_values.shape == values.shape
        assert torch.allclose(cached_keys, keys)
        assert torch.allclose(cached_values, values)
        
        # Other layers should remain uninitialized
        assert self.manager.caches[layer_idx].initialized
        assert not self.manager.caches[0].initialized
        assert not self.manager.caches[2].initialized
    
    def test_update_layer_invalid_index(self):
        """Test updating non-existent layer."""
        keys = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        values = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        
        with pytest.raises(IndexError):
            self.manager.update_layer(5, keys, values)  # Only 3 layers (0, 1, 2)
    
    def test_get_layer_cache_basic(self):
        """Test getting layer cache."""
        layer_idx = 0
        keys = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        values = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        
        # Update cache
        self.manager.update_layer(layer_idx, keys, values)
        
        # Retrieve cache
        cached_keys, cached_values = self.manager.get_layer_cache(layer_idx)
        
        assert torch.allclose(cached_keys, keys)
        assert torch.allclose(cached_values, values)
    
    def test_get_layer_cache_invalid_index(self):
        """Test getting cache for non-existent layer."""
        with pytest.raises(IndexError):
            self.manager.get_layer_cache(10)
    
    def test_is_initialized(self):
        """Test checking if cache is fully initialized."""
        assert not self.manager.is_initialized()
        
        # Initialize some layers
        keys = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        values = torch.randn(2, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        
        self.manager.update_layer(0, keys, values)
        assert not self.manager.is_initialized()  # Not all layers initialized
        
        self.manager.update_layer(1, keys, values)
        assert not self.manager.is_initialized()  # Still not all
        
        self.manager.update_layer(2, keys, values)
        assert self.manager.is_initialized()  # Now all layers initialized
    
    def test_get_total_memory_usage(self):
        """Test getting total memory usage statistics."""
        usage = self.manager.get_total_memory_usage()
        
        assert "total_memory_mb" in usage
        assert "num_layers" in usage
        assert "avg_memory_per_layer_mb" in usage
        assert "layer_stats" in usage
        assert "config" in usage
        
        assert usage["num_layers"] == 3
        assert len(usage["layer_stats"]) == 3
        assert usage["avg_memory_per_layer_mb"] == usage["total_memory_mb"] / 3
        
        # Check config section
        config = usage["config"]
        assert config["max_batch_size"] == self.config.max_batch_size
        assert config["max_seq_len"] == self.config.max_seq_len
        assert config["num_heads"] == self.config.num_heads
        assert config["head_dim"] == self.config.head_dim
        assert config["dtype"] == str(self.config.dtype)
    
    def test_optimize_memory(self):
        """Test memory optimization."""
        # Add data to some batch indices
        keys = torch.randn(1, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        values = torch.randn(1, 4, 3, 8, dtype=self.config.dtype, device=self.config.device)
        
        # Only update batch index 0, leaving batch index 1 unused
        for layer_idx in range(3):
            self.manager.update_layer(layer_idx, keys, values, batch_indices=[0])
        
        # Check that batch index 1 has zero length
        for cache in self.manager.caches:
            assert cache.seq_lengths[0] > 0
            assert cache.seq_lengths[1] == 0
        
        # Optimize memory (should clean up unused batch indices)
        self.manager.optimize_memory()
        
        # Batch index 1 should still be zero (already clean)
        for cache in self.manager.caches:
            assert cache.seq_lengths[1] == 0


class TestCreateKVCacheManager:
    """Test the create_kv_cache_manager utility function."""
    
    def test_create_with_mock_model(self):
        """Test creating cache manager from mock model."""
        model = create_mock_model(device="cpu")
        
        cache_manager = create_kv_cache_manager(
            model=model,
            max_batch_size=2,
            max_seq_len=64,
        )
        
        assert isinstance(cache_manager, KVCacheManager)
        assert cache_manager.config.max_batch_size == 2
        assert cache_manager.config.max_seq_len == 64
        assert cache_manager.config.device == torch.device("cpu")
        
        # Should auto-detect model parameters
        assert cache_manager.config.num_layers == model.num_layers
        assert cache_manager.config.num_heads == model.num_heads
        assert cache_manager.config.head_dim == model.d_model // model.num_heads
    
    def test_create_with_custom_params(self):
        """Test creating cache manager with custom parameters."""
        model = create_mock_model(device="cpu")
        
        cache_manager = create_kv_cache_manager(
            model=model,
            max_batch_size=4,
            max_seq_len=128,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        
        assert cache_manager.config.max_batch_size == 4
        assert cache_manager.config.max_seq_len == 128
        assert cache_manager.config.dtype == torch.float32
        assert cache_manager.config.device == torch.device("cpu")
    
    def test_create_auto_device_detection(self):
        """Test automatic device detection from model."""
        device = torch.device("cpu")
        model = create_mock_model(device=device)
        
        cache_manager = create_kv_cache_manager(model)
        
        assert cache_manager.config.device == device
    
    def test_create_auto_dtype_detection(self):
        """Test automatic dtype detection from model."""
        model = create_mock_model(device="cpu")
        
        # Get model's actual dtype
        model_dtype = next(model.parameters()).dtype
        
        cache_manager = create_kv_cache_manager(model)
        
        assert cache_manager.config.dtype == model_dtype


class TestKVCacheIntegration:
    """Integration tests for KV cache functionality."""
    
    def test_multi_layer_sequential_generation(self):
        """Test sequential generation with multi-layer caching."""
        config = CacheConfig(
            max_batch_size=1,
            max_seq_len=32,
            num_heads=4,
            head_dim=8,
            num_layers=2,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        manager = KVCacheManager(config)
        
        # Simulate sequential generation
        batch_size = 1
        current_seq_len = 0
        
        for step in range(5):  # Generate 5 tokens
            new_seq_len = 3  # Add 3 new tokens each step
            
            # Simulate keys and values for each layer
            for layer_idx in range(config.num_layers):
                keys = torch.randn(batch_size, config.num_heads, new_seq_len, config.head_dim,
                                 dtype=config.dtype, device=config.device)
                values = torch.randn(batch_size, config.num_heads, new_seq_len, config.head_dim,
                                   dtype=config.dtype, device=config.device)
                
                cached_keys, cached_values = manager.update_layer(layer_idx, keys, values, batch_indices=[0])
                
                # Check that cache grows correctly
                expected_total_len = current_seq_len + new_seq_len
                assert cached_keys.shape == (batch_size, config.num_heads, expected_total_len, config.head_dim)
                assert cached_values.shape == (batch_size, config.num_heads, expected_total_len, config.head_dim)
            
            current_seq_len += new_seq_len
            
            # Check that all layers have consistent sequence lengths
            seq_lengths = [cache.seq_lengths[0].item() for cache in manager.caches]
            assert all(length == current_seq_len for length in seq_lengths)
        
        assert manager.is_initialized()
        assert current_seq_len == 15  # 5 steps * 3 tokens per step
    
    def test_batch_generation_with_different_lengths(self):
        """Test batch generation where sequences have different lengths."""
        config = CacheConfig(
            max_batch_size=3,
            max_seq_len=20,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        manager = KVCacheManager(config)
        
        # Generate different length sequences for each batch item
        sequences_data = [
            (0, 3),  # batch 0: 3 tokens
            (1, 5),  # batch 1: 5 tokens
            (2, 2),  # batch 2: 2 tokens
        ]
        
        for batch_idx, seq_len in sequences_data:
            keys = torch.randn(1, config.num_heads, seq_len, config.head_dim,
                             dtype=config.dtype, device=config.device)
            values = torch.randn(1, config.num_heads, seq_len, config.head_dim,
                               dtype=config.dtype, device=config.device)
            
            manager.update_layer(0, keys, values, batch_indices=[batch_idx])
        
        # Check sequence lengths
        expected_lengths = [3, 5, 2]
        actual_lengths = manager.caches[0].seq_lengths.tolist()
        assert actual_lengths == expected_lengths
        
        # Test retrieving specific batch indices
        for batch_idx, expected_len in enumerate(expected_lengths):
            cached_keys, cached_values = manager.get_layer_cache(0, batch_indices=[batch_idx])
            max_len_in_batch = max(expected_lengths)  # Should be padded to max length
            assert cached_keys.shape == (1, config.num_heads, max_len_in_batch, config.head_dim)
    
    def test_cache_overflow_handling(self):
        """Test handling of cache overflow scenarios."""
        config = CacheConfig(
            max_batch_size=1,
            max_seq_len=8,  # Small max length to trigger overflow
            num_heads=2,
            head_dim=4,
            num_layers=1,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        manager = KVCacheManager(config)
        
        # First update: fill to near capacity
        keys1 = torch.randn(1, 2, 6, 4, dtype=config.dtype, device=config.device)
        values1 = torch.randn(1, 2, 6, 4, dtype=config.dtype, device=config.device)
        manager.update_layer(0, keys1, values1, batch_indices=[0])
        
        assert manager.caches[0].seq_lengths[0] == 6
        
        # Second update: exceed capacity
        keys2 = torch.randn(1, 2, 5, 4, dtype=config.dtype, device=config.device)  # Would make total 11
        values2 = torch.randn(1, 2, 5, 4, dtype=config.dtype, device=config.device)
        
        cached_keys, cached_values = manager.update_layer(0, keys2, values2, batch_indices=[0])
        
        # Should be capped at max_seq_len
        assert manager.caches[0].seq_lengths[0] == config.max_seq_len
        assert cached_keys.shape[2] == config.max_seq_len
        
        # Should contain original 6 tokens plus 2 new ones (6 + 2 = 8 = max_seq_len)
        expected_new_tokens = config.max_seq_len - 6
        assert torch.allclose(cached_keys[0, :, 6:], keys2[0, :, :expected_new_tokens])