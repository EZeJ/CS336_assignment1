import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from cs336_basics.optimization.checkpointing import (
    CheckpointMetadata,
    EnhancedCheckpoint,
    GradientCheckpointing
)
from ..fixtures.test_models import create_mock_model, create_sample_inputs


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""
    
    def test_initialization(self):
        """Test metadata initialization."""
        metadata = CheckpointMetadata(
            iteration=100,
            epoch=1,
            timestamp=time.time(),
            metrics={"loss": 2.5, "accuracy": 0.8},
            model_config={"d_model": 128},
            optimizer_config={"lr": 1e-3},
            file_path="/path/to/checkpoint.pt",
            file_size=1024
        )
        
        assert metadata.iteration == 100
        assert metadata.epoch == 1
        assert metadata.metrics["loss"] == 2.5
        assert metadata.file_path == "/path/to/checkpoint.pt"
        assert metadata.file_size == 1024


class TestEnhancedCheckpoint:
    """Test the EnhancedCheckpoint class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=3,
            save_interval=1,  # Save every iteration for testing
        )
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test checkpoint manager initialization."""
        assert self.checkpoint_manager.checkpoint_dir.exists()
        assert self.checkpoint_manager.max_checkpoints == 3
        assert self.checkpoint_manager.save_interval == 1
        assert self.checkpoint_manager.best_metric == "val_loss"
        assert self.checkpoint_manager.best_mode == "min"
        assert len(self.checkpoint_manager.checkpoints) == 0
    
    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        metrics = {"train_loss": 2.5, "val_loss": 2.8}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=100,
            epoch=1,
            metrics=metrics,
            force_save=True
        )
        
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        assert len(self.checkpoint_manager.checkpoints) == 1
        
        # Check metadata
        metadata = self.checkpoint_manager.checkpoints[0]
        assert metadata.iteration == 100
        assert metadata.epoch == 1
        assert metadata.metrics == metrics
        assert metadata.file_size > 0
    
    def test_save_checkpoint_interval(self):
        """Test checkpoint saving respects interval."""
        # Create manager with larger interval
        manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_interval=10
        )
        
        # Should not save (iteration 5, interval 10)
        result = manager.save_checkpoint(
            self.model, self.optimizer, iteration=5, epoch=0
        )
        assert result is None
        assert len(manager.checkpoints) == 0
        
        # Should save (iteration 10, interval 10)
        result = manager.save_checkpoint(
            self.model, self.optimizer, iteration=10, epoch=0
        )
        assert result is not None
        assert len(manager.checkpoints) == 1
    
    def test_load_checkpoint_basic(self):
        """Test basic checkpoint loading."""
        # Save a checkpoint
        metrics = {"train_loss": 2.5}
        extra_state = {"learning_rate": 1e-3}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=100,
            epoch=1,
            metrics=metrics,
            extra_state=extra_state,
            force_save=True
        )
        
        # Create new model and optimizer
        new_model = create_mock_model(device=self.device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=2e-3)
        
        # Load checkpoint
        loaded_data = self.checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            checkpoint_path=checkpoint_path
        )
        
        assert loaded_data is not None
        assert loaded_data["iteration"] == 100
        assert loaded_data["epoch"] == 1
        assert loaded_data["metrics"] == metrics
        assert loaded_data["extra_state"] == extra_state
        
        # Check that model states match
        for (name1, param1), (name2, param2) in zip(
            self.model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)
    
    def test_load_latest_checkpoint(self):
        """Test loading latest checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=100 + i,
                epoch=i,
                force_save=True
            )
        
        # Load latest (should be iteration 102)
        new_model = create_mock_model(device=self.device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer
        )
        
        assert loaded_data["iteration"] == 102
        assert loaded_data["epoch"] == 2
    
    def test_best_checkpoint_tracking_min(self):
        """Test best checkpoint tracking with min mode."""
        manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_best=True,
            best_metric="val_loss",
            best_mode="min"
        )
        
        # Save checkpoints with decreasing loss
        losses = [3.0, 2.5, 2.8, 2.0, 2.1]
        for i, loss in enumerate(losses):
            manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=i,
                epoch=0,
                metrics={"val_loss": loss},
                force_save=True
            )
        
        # Best should be 2.0 (minimum)
        assert manager.best_value == 2.0
        assert len(manager.best_checkpoints) > 0
        
        # Best checkpoint file should exist
        best_path = Path(self.temp_dir) / "best_val_loss.pt"
        assert best_path.exists()
    
    def test_best_checkpoint_tracking_max(self):
        """Test best checkpoint tracking with max mode."""
        manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_best=True,
            best_metric="accuracy",
            best_mode="max"
        )
        
        # Save checkpoints with varying accuracy
        accuracies = [0.6, 0.7, 0.65, 0.8, 0.75]
        for i, acc in enumerate(accuracies):
            manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=i,
                epoch=0,
                metrics={"accuracy": acc},
                force_save=True
            )
        
        # Best should be 0.8 (maximum)
        assert manager.best_value == 0.8
        
        # Best checkpoint file should exist
        best_path = Path(self.temp_dir) / "best_accuracy.pt"
        assert best_path.exists()
    
    def test_load_best_checkpoint(self):
        """Test loading best checkpoint."""
        manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_best=True,
            best_metric="val_loss",
            best_mode="min"
        )
        
        # Save checkpoints with varying loss
        for i, loss in enumerate([3.0, 2.5, 2.8]):
            manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=100 + i,
                epoch=0,
                metrics={"val_loss": loss},
                force_save=True
            )
        
        # Load best checkpoint
        new_model = create_mock_model(device=self.device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        
        loaded_data = manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            load_best=True
        )
        
        # Should load iteration 101 (loss 2.5, the best)
        assert loaded_data["iteration"] == 101
        assert loaded_data["metrics"]["val_loss"] == 2.5
    
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup."""
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=i,
                epoch=0,
                force_save=True
            )
        
        # Should only keep max_checkpoints (3)
        assert len(self.checkpoint_manager.checkpoints) == 3
        
        # Should be the latest 3
        iterations = [cp.iteration for cp in self.checkpoint_manager.checkpoints]
        assert iterations == [2, 3, 4]
    
    def test_metadata_persistence(self):
        """Test metadata saving and loading."""
        # Save some checkpoints
        for i in range(2):
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=i,
                epoch=0,
                metrics={"loss": 2.5 - i * 0.1},
                force_save=True
            )
        
        # Create new manager in same directory
        new_manager = EnhancedCheckpoint(checkpoint_dir=self.temp_dir)
        
        # Should load existing metadata
        assert len(new_manager.checkpoints) == 2
        assert new_manager.checkpoints[0].iteration == 0
        assert new_manager.checkpoints[1].iteration == 1
    
    def test_get_checkpoint_info(self):
        """Test checkpoint information retrieval."""
        # Initially no checkpoints
        info = self.checkpoint_manager.get_checkpoint_info()
        assert info["num_checkpoints"] == 0
        assert info["latest_checkpoint"] is None
        
        # Save a checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=100,
            epoch=1,
            metrics={"val_loss": 2.5},
            force_save=True
        )
        
        info = self.checkpoint_manager.get_checkpoint_info()
        assert info["num_checkpoints"] == 1
        assert info["latest_checkpoint"] is not None
        assert info["total_size_mb"] > 0
    
    def test_model_config_extraction(self):
        """Test model configuration extraction."""
        # Add config to model
        self.model.config = {"d_model": 128, "num_layers": 2}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=100,
            epoch=0,
            force_save=True
        )
        
        # Load and check config
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint_data["model_config"]
        
        assert "d_model" in model_config
        assert "num_layers" in model_config
        assert "model_class" in model_config
        assert "num_parameters" in model_config
        assert model_config["model_class"] == "MockTransformer"
    
    def test_checkpoint_save_failure_handling(self):
        """Test handling of checkpoint save failures."""
        # Mock torch.save to raise an exception
        with patch('torch.save', side_effect=RuntimeError("Save failed")):
            result = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                iteration=100,
                epoch=0,
                force_save=True
            )
        
        assert result is None
        assert len(self.checkpoint_manager.checkpoints) == 0
    
    def test_checkpoint_load_failure_handling(self):
        """Test handling of checkpoint load failures."""
        # Try to load non-existent checkpoint
        result = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path="/nonexistent/path.pt"
        )
        
        assert result is None


class TestGradientCheckpointing:
    """Test the GradientCheckpointing utility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_mock_model(device=self.device)
    
    def test_enable_gradient_checkpointing_basic(self):
        """Test enabling gradient checkpointing."""
        original_forward = self.model.layers[0].forward
        
        # Enable checkpointing
        GradientCheckpointing.enable_gradient_checkpointing(
            self.model,
            checkpoint_modules=["TransformerDecoderLayer"]
        )
        
        # Forward method should be wrapped
        assert self.model.layers[0].forward != original_forward
    
    def test_gradient_checkpointing_functionality(self):
        """Test that gradient checkpointing actually works."""
        # Enable checkpointing
        GradientCheckpointing.enable_gradient_checkpointing(
            self.model,
            checkpoint_modules=["TransformerDecoderLayer"]
        )
        
        # Test forward pass
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        output = self.model(inputs)
        
        assert output is not None
        assert output.shape == (2, 8, 1000)  # batch, seq, vocab
    
    def test_gradient_checkpointing_with_training(self):
        """Test gradient checkpointing during training."""
        # Enable checkpointing
        GradientCheckpointing.enable_gradient_checkpointing(
            self.model,
            checkpoint_modules=["TransformerDecoderLayer"]
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Training step
        self.model.train()
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        targets = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_estimate_memory_savings(self):
        """Test memory savings estimation."""
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        
        savings = GradientCheckpointing.estimate_memory_savings(
            model=self.model,
            sample_input=inputs,
            checkpoint_modules=["TransformerDecoderLayer"]
        )
        
        assert "total_parameter_memory_mb" in savings
        assert "estimated_activation_memory_savings_ratio" in savings
        assert "estimated_total_memory_savings_mb" in savings
        
        assert savings["total_parameter_memory_mb"] > 0
        assert 0 < savings["estimated_activation_memory_savings_ratio"] < 1
        assert savings["estimated_total_memory_savings_mb"] > 0
    
    def test_checkpoint_modules_filtering(self):
        """Test that only specified modules get checkpointed."""
        # Count original forward methods
        original_forwards = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'forward'):
                original_forwards[name] = module.forward
        
        # Enable checkpointing for specific modules only
        GradientCheckpointing.enable_gradient_checkpointing(
            self.model,
            checkpoint_modules=["NonExistentModule"]  # Should not match anything
        )
        
        # No modules should be modified
        for name, module in self.model.named_modules():
            if name in original_forwards:
                assert module.forward == original_forwards[name]
    
    def test_checkpointing_with_custom_modules(self):
        """Test checkpointing with custom module list."""
        # Enable checkpointing for embedding layer
        GradientCheckpointing.enable_gradient_checkpointing(
            self.model,
            checkpoint_modules=["Embedding"]
        )
        
        # Test that model still works
        inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
        output = self.model(inputs)
        
        assert output is not None
        assert output.shape == (2, 8, 1000)


class TestCheckpointingIntegration:
    """Integration tests for checkpointing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_training_with_checkpointing(self):
        """Test complete training loop with checkpointing."""
        model = create_mock_model(device=self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Enable gradient checkpointing
        GradientCheckpointing.enable_gradient_checkpointing(model)
        
        # Create checkpoint manager
        checkpoint_manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_interval=2,
            save_best=True,
            best_metric="val_loss"
        )
        
        # Simulate training loop
        for iteration in range(5):
            model.train()
            
            # Training step
            optimizer.zero_grad()
            inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            targets = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            
            outputs = model(inputs)
            train_loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(inputs)
                val_loss = nn.functional.cross_entropy(
                    val_outputs.view(-1, val_outputs.size(-1)),
                    targets.view(-1)
                )
            
            # Save checkpoint
            metrics = {
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
            }
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                epoch=0,
                metrics=metrics
            )
        
        # Should have saved checkpoints based on interval
        info = checkpoint_manager.get_checkpoint_info()
        assert info["num_checkpoints"] > 0
        assert info["latest_checkpoint"] is not None
        
        if info["num_best_checkpoints"] > 0:
            assert info["best_checkpoint"] is not None
    
    def test_checkpoint_resume_training(self):
        """Test resuming training from checkpoint."""
        # Initial training
        model1 = create_mock_model(device=self.device)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        
        checkpoint_manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_interval=1
        )
        
        # Train for a few steps
        for i in range(3):
            inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            targets = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            loss.backward()
            optimizer1.step()
            
            checkpoint_manager.save_checkpoint(
                model=model1,
                optimizer=optimizer1,
                iteration=i,
                epoch=0,
                metrics={"loss": loss.item()}
            )
        
        # Resume training with new model
        model2 = create_mock_model(device=self.device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        # Load latest checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(
            model=model2,
            optimizer=optimizer2
        )
        
        assert checkpoint_data is not None
        assert checkpoint_data["iteration"] == 2  # Latest iteration
        
        # Models should have same parameters after loading
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_checkpoint_with_mixed_precision(self):
        """Test checkpointing compatibility with mixed precision."""
        from cs336_basics.optimization.mixed_precision import MixedPrecisionTrainer
        
        model = create_mock_model(device=self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mp_trainer = MixedPrecisionTrainer(device=self.device)
        
        checkpoint_manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_interval=1
        )
        
        # Training with mixed precision
        for i in range(2):
            inputs = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            targets = create_sample_inputs(batch_size=2, seq_len=8, device=self.device)
            
            def loss_fn():
                with mp_trainer.autocast():
                    outputs = model(inputs)
                    return nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
            
            loss = mp_trainer.training_step(model, optimizer, loss_fn)
            
            # Save checkpoint with mixed precision state
            extra_state = {"mp_trainer_state": mp_trainer.state_dict()}
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=i,
                epoch=0,
                metrics={"loss": loss.item()},
                extra_state=extra_state
            )
        
        # Load checkpoint
        new_model = create_mock_model(device=self.device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        
        checkpoint_data = checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer
        )
        
        assert checkpoint_data is not None
        assert "mp_trainer_state" in checkpoint_data["extra_state"]
        
        # Create new MP trainer and load state
        new_mp_trainer = MixedPrecisionTrainer(device=self.device)
        new_mp_trainer.load_state_dict(checkpoint_data["extra_state"]["mp_trainer_state"])
        
        # Should work without issues
        assert new_mp_trainer.get_stats()["enabled"] == mp_trainer.get_stats()["enabled"]