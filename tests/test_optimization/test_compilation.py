import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import time

from cs336_basics.optimization.compilation import (
    CompilationOptimizer,
    compile_model,
    supports_compilation,
    get_compilation_mode
)
from ..fixtures.test_models import create_mock_model, create_real_model, create_sample_inputs


class TestCompilationSupport:
    """Test compilation support detection."""
    
    def test_supports_compilation_torch_version(self):
        """Test compilation support based on PyTorch version."""
        with patch('torch.__version__', '2.1.0'):
            assert supports_compilation() == True
        
        with patch('torch.__version__', '1.13.0'):
            assert supports_compilation() == False
    
    def test_get_compilation_mode_default(self):
        """Test default compilation mode selection."""
        mode = get_compilation_mode()
        assert mode in ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']
    
    def test_get_compilation_mode_custom(self):
        """Test custom compilation mode."""
        mode = get_compilation_mode('max-autotune')
        assert mode == 'max-autotune'


class TestCompileModel:
    """Test model compilation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compile_model_basic(self):
        """Test basic model compilation."""
        compiled_model = compile_model(self.model)
        
        # Check that model is compiled (has _dynamo attribute)
        assert hasattr(compiled_model, '_dynamo')
        
        # Test that compiled model produces same outputs
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        
        original_output = self.model(inputs)
        compiled_output = compiled_model(inputs)
        
        # Outputs should be close (compilation may introduce small numerical differences)
        assert torch.allclose(original_output, compiled_output, atol=1e-4, rtol=1e-4)
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compile_model_with_mode(self):
        """Test model compilation with specific mode."""
        compiled_model = compile_model(self.model, mode='reduce-overhead')
        
        assert hasattr(compiled_model, '_dynamo')
        
        # Should still produce correct outputs
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        original_output = self.model(inputs)
        compiled_output = compiled_model(inputs)
        
        assert torch.allclose(original_output, compiled_output, atol=1e-4, rtol=1e-4)
    
    def test_compile_model_unsupported(self):
        """Test compilation when not supported."""
        with patch('cs336_basics.optimization.compilation.supports_compilation', return_value=False):
            compiled_model = compile_model(self.model)
            
            # Should return original model unchanged
            assert compiled_model is self.model
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compile_model_error_handling(self):
        """Test error handling during compilation."""
        with patch('torch.compile', side_effect=RuntimeError("Compilation failed")):
            # Should fallback to original model on compilation error
            compiled_model = compile_model(self.model, fallback_on_error=True)
            assert compiled_model is self.model
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compile_model_no_fallback(self):
        """Test compilation without fallback on error."""
        with patch('torch.compile', side_effect=RuntimeError("Compilation failed")):
            with pytest.raises(RuntimeError, match="Compilation failed"):
                compile_model(self.model, fallback_on_error=False)


class TestCompilationOptimizer:
    """Test the CompilationOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
        self.optimizer = CompilationOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.enabled == supports_compilation()
        assert self.optimizer.mode == get_compilation_mode()
        assert self.optimizer.compiled_models == {}
    
    def test_initialization_disabled(self):
        """Test optimizer initialization when disabled."""
        optimizer = CompilationOptimizer(enabled=False)
        assert optimizer.enabled == False
    
    def test_optimize_model_enabled(self):
        """Test model optimization when compilation is enabled."""
        if not supports_compilation():
            pytest.skip("Compilation not supported")
        
        optimized_model = self.optimizer.optimize_model(self.model, model_id="test_model")
        
        # Model should be compiled
        assert hasattr(optimized_model, '_dynamo')
        
        # Should be cached
        assert "test_model" in self.optimizer.compiled_models
        assert self.optimizer.compiled_models["test_model"] is optimized_model
    
    def test_optimize_model_disabled(self):
        """Test model optimization when disabled."""
        optimizer = CompilationOptimizer(enabled=False)
        optimized_model = optimizer.optimize_model(self.model, model_id="test_model")
        
        # Should return original model
        assert optimized_model is self.model
        assert len(optimizer.compiled_models) == 0
    
    def test_optimize_model_caching(self):
        """Test model caching behavior."""
        model_id = "test_model"
        
        # First optimization
        optimized_1 = self.optimizer.optimize_model(self.model, model_id=model_id)
        
        # Second optimization should return cached model
        optimized_2 = self.optimizer.optimize_model(self.model, model_id=model_id)
        
        if supports_compilation():
            assert optimized_1 is optimized_2
            assert len(self.optimizer.compiled_models) == 1
        else:
            # When compilation not supported, should return original models
            assert optimized_1 is self.model
            assert optimized_2 is self.model
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        if not supports_compilation():
            pytest.skip("Compilation not supported")
        
        # Compile some models
        self.optimizer.optimize_model(self.model, model_id="model1")
        self.optimizer.optimize_model(create_mock_model(device=self.device), model_id="model2")
        
        assert len(self.optimizer.compiled_models) == 2
        
        # Clear cache
        self.optimizer.clear_cache()
        assert len(self.optimizer.compiled_models) == 0
    
    def test_get_stats(self):
        """Test statistics reporting."""
        stats = self.optimizer.get_stats()
        
        assert 'enabled' in stats
        assert 'mode' in stats
        assert 'compiled_models_count' in stats
        assert 'supports_compilation' in stats
        
        assert stats['enabled'] == self.optimizer.enabled
        assert stats['supports_compilation'] == supports_compilation()
        assert stats['compiled_models_count'] == 0
        
        # Add a model and check stats update
        if supports_compilation():
            self.optimizer.optimize_model(self.model, model_id="test")
            new_stats = self.optimizer.get_stats()
            assert new_stats['compiled_models_count'] == 1


class TestCompilationPerformance:
    """Test compilation performance benefits."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compilation_warmup(self):
        """Test that compilation has a warmup period."""
        compiled_model = compile_model(self.model)
        inputs = create_sample_inputs(batch_size=2, seq_len=16, device=self.device)
        
        # First few runs might be slow due to compilation
        warmup_times = []
        for _ in range(3):
            start_time = time.time()
            _ = compiled_model(inputs)
            warmup_times.append(time.time() - start_time)
        
        # Subsequent runs should be faster (in theory, hard to test reliably)
        runtime_times = []
        for _ in range(3):
            start_time = time.time()
            _ = compiled_model(inputs)
            runtime_times.append(time.time() - start_time)
        
        # All times should be reasonable (not hanging)
        assert all(t < 10.0 for t in warmup_times + runtime_times)
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compilation_consistency(self):
        """Test that compiled models produce consistent outputs."""
        compiled_model = compile_model(self.model)
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        
        # Run multiple times and check consistency
        outputs = []
        for _ in range(5):
            output = compiled_model(inputs)
            outputs.append(output)
        
        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.equal(outputs[0], outputs[i])
    
    @pytest.mark.skipif(not supports_compilation(), reason="Compilation not supported")
    def test_compilation_memory_usage(self):
        """Test memory usage with compilation."""
        # This is a smoke test - we can't easily measure memory precisely
        compiled_model = compile_model(self.model)
        
        # Should handle larger inputs without issues
        large_inputs = create_sample_inputs(batch_size=4, seq_len=32, device=self.device)
        
        # Should not raise memory errors
        for _ in range(10):
            output = compiled_model(large_inputs)
            assert output is not None
            assert output.shape[0] == 4  # Batch size preserved


class TestCompilationIntegration:
    """Integration tests for compilation optimization."""
    
    def test_training_loop_integration(self):
        """Test compilation in training loop."""
        device = 'cpu'
        model = create_mock_model(device=device)
        optimizer_torch = torch.optim.AdamW(model.parameters(), lr=1e-3)
        compilation_optimizer = CompilationOptimizer()
        
        # Optimize model
        compiled_model = compilation_optimizer.optimize_model(model, model_id="training_model")
        
        # Training loop
        for epoch in range(2):
            for batch in range(3):
                inputs = create_sample_inputs(batch_size=2, seq_len=8, device=device)
                targets = create_sample_inputs(batch_size=2, seq_len=8, device=device)
                
                optimizer_torch.zero_grad()
                outputs = compiled_model(inputs)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                loss.backward()
                optimizer_torch.step()
                
                assert loss.item() > 0
        
        # Should complete without errors
        stats = compilation_optimizer.get_stats()
        assert 'enabled' in stats
    
    def test_inference_optimization(self):
        """Test compilation for inference optimization."""
        device = 'cpu'
        model = create_mock_model(device=device)
        compilation_optimizer = CompilationOptimizer()
        
        # Set to evaluation mode
        model.eval()
        compiled_model = compilation_optimizer.optimize_model(model, model_id="inference_model")
        
        with torch.no_grad():
            inputs = create_sample_inputs(batch_size=1, seq_len=16, device=device)
            
            # Multiple inference runs
            outputs = []
            for _ in range(5):
                output = compiled_model(inputs)
                outputs.append(output)
            
            # All outputs should be consistent
            for i in range(1, len(outputs)):
                if supports_compilation():
                    assert torch.allclose(outputs[0], outputs[i], atol=1e-6, rtol=1e-6)
                else:
                    assert torch.equal(outputs[0], outputs[i])
    
    def test_multiple_models_compilation(self):
        """Test compiling multiple different models."""
        device = 'cpu'
        compilation_optimizer = CompilationOptimizer()
        
        # Create different models
        model1 = create_mock_model(device=device)
        model2 = create_real_model(device=device) if hasattr(create_real_model, '__call__') else create_mock_model(device=device)
        
        # Compile both
        compiled1 = compilation_optimizer.optimize_model(model1, model_id="model1")
        compiled2 = compilation_optimizer.optimize_model(model2, model_id="model2")
        
        # Test both work
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=device)
        
        output1 = compiled1(inputs)
        output2 = compiled2(inputs)
        
        assert output1 is not None
        assert output2 is not None
        assert output1.shape == output2.shape
        
        # Check caching
        if supports_compilation():
            assert len(compilation_optimizer.compiled_models) == 2
        else:
            assert len(compilation_optimizer.compiled_models) == 0