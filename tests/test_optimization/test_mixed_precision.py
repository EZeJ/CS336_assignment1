import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from cs336_basics.optimization.mixed_precision import (
    MixedPrecisionTrainer, 
    AutoCaster, 
    get_autocast_device_type,
    supports_mixed_precision
)
from ..fixtures.test_models import create_mock_model, create_sample_inputs


class TestAutoCaster:
    """Test the AutoCaster utility class."""
    
    def test_autocast_cuda_available(self):
        with patch('torch.cuda.is_available', return_value=True):
            caster = AutoCaster('cuda')
            assert caster.device_type == 'cuda'
            assert caster.dtype == torch.float16
    
    def test_autocast_mps_available(self):
        with patch('torch.backends.mps.is_available', return_value=True):
            caster = AutoCaster('mps')
            assert caster.device_type == 'cpu'  # MPS uses CPU autocast
            assert caster.dtype == torch.float16
    
    def test_autocast_cpu_fallback(self):
        caster = AutoCaster('cpu')
        assert caster.device_type == 'cpu'
        assert caster.dtype == torch.bfloat16
    
    def test_context_manager(self):
        caster = AutoCaster('cpu')
        with caster:
            # Should not raise any errors
            pass
    
    def test_get_autocast_device_type(self):
        with patch('torch.cuda.is_available', return_value=True):
            assert get_autocast_device_type('cuda') == 'cuda'
        
        with patch('torch.backends.mps.is_available', return_value=True):
            assert get_autocast_device_type('mps') == 'cpu'
        
        assert get_autocast_device_type('cpu') == 'cpu'
    
    def test_supports_mixed_precision(self):
        with patch('torch.cuda.is_available', return_value=True):
            assert supports_mixed_precision('cuda') == True
        
        with patch('torch.backends.mps.is_available', return_value=True):
            assert supports_mixed_precision('mps') == True
        
        assert supports_mixed_precision('cpu') == True  # CPU supports bfloat16


class TestMixedPrecisionTrainer:
    """Test the MixedPrecisionTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.trainer = MixedPrecisionTrainer(device=self.device)
    
    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.device == self.device
        assert isinstance(self.trainer.scaler, torch.cuda.amp.GradScaler)
        assert isinstance(self.trainer.autocaster, AutoCaster)
        assert self.trainer.enabled == True
    
    def test_initialization_disabled(self):
        """Test trainer initialization when disabled."""
        trainer = MixedPrecisionTrainer(device=self.device, enabled=False)
        assert trainer.enabled == False
        assert trainer.scaler is None
        assert trainer.autocaster is None
    
    def test_training_step_basic(self):
        """Test basic training step functionality."""
        inputs = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        targets = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        
        def loss_fn():
            with self.trainer.autocast():
                outputs = self.model(inputs)
                return nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
        
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Perform training step
        loss = self.trainer.training_step(self.model, self.optimizer, loss_fn)
        
        # Check that loss is computed and parameters updated
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
        # Check parameters were updated
        for initial, current in zip(initial_params, self.model.parameters()):
            assert not torch.equal(initial, current)
    
    def test_training_step_disabled(self):
        """Test training step when mixed precision is disabled."""
        trainer = MixedPrecisionTrainer(device=self.device, enabled=False)
        inputs = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        targets = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        
        def loss_fn():
            outputs = self.model(inputs)
            return nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
        
        loss = trainer.training_step(self.model, self.optimizer, loss_fn)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_gradient_scaling(self):
        """Test that gradient scaling works properly."""
        inputs = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        targets = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        
        # Create a loss function that should trigger scaling
        def loss_fn():
            with self.trainer.autocast():
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                return loss * 1000  # Artificially large loss to test scaling
        
        initial_scale = self.trainer.scaler.get_scale()
        
        # Multiple training steps to potentially trigger scaling adjustments
        for _ in range(5):
            self.trainer.training_step(self.model, self.optimizer, loss_fn)
        
        # Scale should be managed appropriately
        final_scale = self.trainer.scaler.get_scale()
        assert isinstance(final_scale, float)
        assert final_scale > 0
    
    def test_autocast_context(self):
        """Test autocast context manager."""
        with self.trainer.autocast():
            inputs = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
            outputs = self.model(inputs)
            
            # On CPU with bfloat16, output should be in bfloat16 or float32
            assert outputs.dtype in [torch.float32, torch.bfloat16]
    
    def test_state_dict_management(self):
        """Test saving and loading trainer state."""
        # Perform some training steps to change scaler state
        inputs = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        targets = create_sample_inputs(batch_size=2, seq_len=10, device=self.device)
        
        def loss_fn():
            with self.trainer.autocast():
                outputs = self.model(inputs)
                return nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
        
        for _ in range(3):
            self.trainer.training_step(self.model, self.optimizer, loss_fn)
        
        # Save state
        state_dict = self.trainer.state_dict()
        
        # Create new trainer and load state
        new_trainer = MixedPrecisionTrainer(device=self.device)
        new_trainer.load_state_dict(state_dict)
        
        # States should match
        assert new_trainer.scaler.get_scale() == self.trainer.scaler.get_scale()
    
    def test_get_stats(self):
        """Test statistics reporting."""
        stats = self.trainer.get_stats()
        
        assert 'enabled' in stats
        assert 'device_type' in stats
        assert 'dtype' in stats
        assert 'scale' in stats
        
        assert stats['enabled'] == True
        assert stats['device_type'] == 'cpu'
        assert stats['scale'] > 0
    
    def test_error_handling_invalid_device(self):
        """Test error handling for invalid device."""
        with pytest.raises(ValueError):
            MixedPrecisionTrainer(device='invalid_device')
    
    def test_memory_efficiency(self):
        """Test that mixed precision reduces memory usage (conceptually)."""
        # This is more of a smoke test since we can't easily measure memory on CPU
        inputs = create_sample_inputs(batch_size=4, seq_len=32, device=self.device)
        targets = create_sample_inputs(batch_size=4, seq_len=32, device=self.device)
        
        def loss_fn():
            with self.trainer.autocast():
                outputs = self.model(inputs)
                return nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
        
        # Should not raise memory errors
        for _ in range(10):
            loss = self.trainer.training_step(self.model, self.optimizer, loss_fn)
            assert loss.item() > 0


class TestMixedPrecisionIntegration:
    """Integration tests for mixed precision training."""
    
    def test_training_loop_integration(self):
        """Test integration with a realistic training loop."""
        device = 'cpu'
        model = create_mock_model(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(device=device)
        
        # Simulate training loop
        for epoch in range(2):
            for batch in range(3):
                inputs = create_sample_inputs(batch_size=2, seq_len=8, device=device)
                targets = create_sample_inputs(batch_size=2, seq_len=8, device=device)
                
                def loss_fn():
                    with trainer.autocast():
                        outputs = model(inputs)
                        return nn.functional.cross_entropy(
                            outputs.view(-1, outputs.size(-1)), 
                            targets.view(-1)
                        )
                
                loss = trainer.training_step(model, optimizer, loss_fn)
                assert loss.item() > 0
        
        # Training should complete without errors
        stats = trainer.get_stats()
        assert stats['enabled'] == True
    
    def test_checkpoint_compatibility(self):
        """Test that mixed precision state can be checkpointed."""
        device = 'cpu'
        model = create_mock_model(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(device=device)
        
        # Do some training
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=device)
        targets = create_sample_inputs(batch_size=2, seq_len=8, device=device)
        
        def loss_fn():
            with trainer.autocast():
                outputs = model(inputs)
                return nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
        
        for _ in range(5):
            trainer.training_step(model, optimizer, loss_fn)
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'trainer_state_dict': trainer.state_dict(),
            'epoch': 1,
            'loss': 2.5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(f.name, map_location=device)
            
            # Create new trainer and load state
            new_trainer = MixedPrecisionTrainer(device=device)
            new_trainer.load_state_dict(loaded_checkpoint['trainer_state_dict'])
            
            # States should match
            assert new_trainer.get_stats()['scale'] == trainer.get_stats()['scale']
            
            # Clean up
            Path(f.name).unlink()