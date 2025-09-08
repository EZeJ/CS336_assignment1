"""
Enhanced Checkpointing System

Provides advanced checkpointing capabilities including:
- Automatic checkpoint management with rotation
- Best model tracking based on metrics
- Gradient checkpointing for memory efficiency
- Comprehensive state management
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import logging
import heapq
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    iteration: int
    epoch: int
    timestamp: float
    metrics: Dict[str, float]
    model_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    file_path: str
    file_size: int


class EnhancedCheckpoint:
    """
    Enhanced checkpointing system with automatic management and best model tracking.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min",
        save_interval: int = 1000,
        async_save: bool = False,
    ):
        """
        Initialize enhanced checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            save_best: Whether to save best model checkpoints
            best_metric: Metric to use for best model selection
            best_mode: "min" or "max" for best metric
            save_interval: Iteration interval for automatic saves
            async_save: Whether to save checkpoints asynchronously
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.save_interval = save_interval
        self.async_save = async_save
        
        # Track checkpoints and best models
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoints: List[tuple] = []  # Min-heap for best models
        self.best_value = float('inf') if best_mode == "min" else float('-inf')
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.load_metadata()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        force_save: bool = False,
    ) -> Optional[str]:
        """
        Save a checkpoint with comprehensive state.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            iteration: Current iteration
            epoch: Current epoch
            metrics: Training/validation metrics
            extra_state: Additional state to save
            force_save: Force save regardless of interval
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        
        # Check if we should save
        if not force_save and iteration % self.save_interval != 0:
            return None
        
        metrics = metrics or {}
        extra_state = extra_state or {}
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "epoch": epoch,
            "metrics": metrics,
            "extra_state": extra_state,
            "timestamp": time.time(),
            "model_config": self._extract_model_config(model),
            "optimizer_config": self._extract_optimizer_config(optimizer),
        }
        
        # Create checkpoint filename
        timestamp = int(time.time())
        checkpoint_name = f"checkpoint_iter_{iteration}_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        try:
            if self.async_save:
                # TODO: Implement async saving with threading
                self._save_checkpoint_sync(checkpoint_data, checkpoint_path)
            else:
                self._save_checkpoint_sync(checkpoint_data, checkpoint_path)
            
            # Create metadata
            file_size = checkpoint_path.stat().st_size
            metadata = CheckpointMetadata(
                iteration=iteration,
                epoch=epoch,
                timestamp=time.time(),
                metrics=metrics,
                model_config=checkpoint_data["model_config"],
                optimizer_config=checkpoint_data["optimizer_config"],
                file_path=str(checkpoint_path),
                file_size=file_size,
            )
            
            # Add to tracking
            self.checkpoints.append(metadata)
            
            # Check if this is a best model
            if self.save_best and self.best_metric in metrics:
                self._update_best_checkpoint(metadata, metrics[self.best_metric])
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            # Save metadata
            self.save_metadata()
            
            logger.info(f"Checkpoint saved: {checkpoint_path} (size: {file_size/1024/1024:.1f}MB)")
            return str(checkpoint_path)
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _save_checkpoint_sync(self, data: Dict[str, Any], path: Path):
        """Synchronously save checkpoint data."""
        torch.save(data, path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        strict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint and restore model/optimizer state.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            checkpoint_path: Specific checkpoint to load (if None, loads latest)
            load_best: Whether to load the best checkpoint instead of latest
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            Loaded checkpoint data or None if failed
        """
        
        if checkpoint_path is None:
            if load_best and self.best_checkpoints:
                # Load best checkpoint
                _, metadata = max(self.best_checkpoints)
                checkpoint_path = metadata.file_path
            elif self.checkpoints:
                # Load latest checkpoint
                checkpoint_path = self.checkpoints[-1].file_path
            else:
                logger.warning("No checkpoints found to load")
                return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            
            # Load model state
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)
            
            # Load optimizer state
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            return checkpoint_data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _update_best_checkpoint(self, metadata: CheckpointMetadata, metric_value: float):
        """Update best checkpoint tracking."""
        is_best = False
        
        if self.best_mode == "min":
            if metric_value < self.best_value:
                self.best_value = metric_value
                is_best = True
        else:  # max mode
            if metric_value > self.best_value:
                self.best_value = metric_value
                is_best = True
        
        if is_best:
            # Add to best checkpoints (use negative value for max heap behavior)
            heap_value = metric_value if self.best_mode == "min" else -metric_value
            heapq.heappush(self.best_checkpoints, (heap_value, metadata))
            
            # Save best checkpoint copy
            best_path = self.checkpoint_dir / f"best_{self.best_metric}.pt"
            shutil.copy2(metadata.file_path, best_path)
            
            logger.info(f"New best checkpoint: {self.best_metric}={metric_value:.4f}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to stay within limits."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = self.checkpoints[:-self.max_checkpoints]
            
            for metadata in to_remove:
                try:
                    os.remove(metadata.file_path)
                    logger.debug(f"Removed old checkpoint: {metadata.file_path}")
                except FileNotFoundError:
                    pass
            
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def _extract_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration for metadata."""
        config = {}
        
        # Try to get model configuration
        if hasattr(model, 'config'):
            config = model.config if isinstance(model.config, dict) else {}
        
        # Add basic model info
        config.update({
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        
        return config
    
    def _extract_optimizer_config(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Extract optimizer configuration for metadata."""
        return {
            "optimizer_class": optimizer.__class__.__name__,
            "param_groups": len(optimizer.param_groups),
            "state_dict_keys": list(optimizer.state_dict().keys()),
        }
    
    def save_metadata(self):
        """Save checkpoint metadata to file."""
        metadata = {
            "checkpoints": [asdict(cp) for cp in self.checkpoints],
            "best_checkpoints": [(val, asdict(meta)) for val, meta in self.best_checkpoints],
            "best_value": self.best_value,
            "config": {
                "max_checkpoints": self.max_checkpoints,
                "save_best": self.save_best,
                "best_metric": self.best_metric,
                "best_mode": self.best_mode,
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self):
        """Load checkpoint metadata from file."""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Restore checkpoints
            self.checkpoints = [
                CheckpointMetadata(**cp) for cp in metadata.get("checkpoints", [])
            ]
            
            # Restore best checkpoints
            best_data = metadata.get("best_checkpoints", [])
            self.best_checkpoints = [
                (val, CheckpointMetadata(**meta)) for val, meta in best_data
            ]
            
            self.best_value = metadata.get("best_value", 
                float('inf') if self.best_mode == "min" else float('-inf'))
            
            logger.info(f"Loaded metadata for {len(self.checkpoints)} checkpoints")
        
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        return {
            "num_checkpoints": len(self.checkpoints),
            "num_best_checkpoints": len(self.best_checkpoints),
            "latest_checkpoint": self.checkpoints[-1].file_path if self.checkpoints else None,
            "best_checkpoint": self.best_checkpoints[0][1].file_path if self.best_checkpoints else None,
            "best_value": self.best_value,
            "total_size_mb": sum(cp.file_size for cp in self.checkpoints) / 1024 / 1024,
        }


class GradientCheckpointing:
    """
    Utility for applying gradient checkpointing to reduce memory usage during training.
    """
    
    @staticmethod
    def enable_gradient_checkpointing(
        model: nn.Module,
        checkpoint_modules: List[str] = None,
        preserve_rng_state: bool = True,
    ):
        """
        Enable gradient checkpointing for specified modules.
        
        Args:
            model: Model to apply checkpointing to
            checkpoint_modules: List of module names to checkpoint
            preserve_rng_state: Whether to preserve RNG state during checkpointing
        """
        
        if checkpoint_modules is None:
            # Default modules to checkpoint
            checkpoint_modules = ["TransformerBlock", "transformer_layers"]
        
        checkpointed_count = 0
        
        for name, module in model.named_modules():
            # Check if module should be checkpointed
            should_checkpoint = any(
                target in name or target in module.__class__.__name__ 
                for target in checkpoint_modules
            )
            
            if should_checkpoint and hasattr(module, 'forward'):
                # Wrap forward method with checkpointing
                original_forward = module.forward
                
                def checkpointed_forward(*args, use_reentrant=False, **kwargs):
                    def forward_func(*args):
                        return original_forward(*args, **kwargs)
                    
                    return checkpoint(
                        forward_func,
                        *args,
                        use_reentrant=use_reentrant,
                        preserve_rng_state=preserve_rng_state,
                    )
                
                module.forward = checkpointed_forward
                checkpointed_count += 1
                logger.debug(f"Applied gradient checkpointing to: {name}")
        
        logger.info(f"Applied gradient checkpointing to {checkpointed_count} modules")
    
    @staticmethod
    def estimate_memory_savings(
        model: nn.Module,
        sample_input: torch.Tensor,
        checkpoint_modules: List[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate memory savings from gradient checkpointing.
        
        Returns:
            Dictionary with memory usage estimates
        """
        
        # This is a simplified estimation
        total_params = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Rough estimation: checkpointing can save 50-80% of activation memory
        # This depends heavily on model architecture and sequence length
        estimated_savings_ratio = 0.6  # 60% savings estimate
        
        return {
            "total_parameter_memory_mb": total_params / 1024 / 1024,
            "estimated_activation_memory_savings_ratio": estimated_savings_ratio,
            "estimated_total_memory_savings_mb": (total_params * estimated_savings_ratio) / 1024 / 1024,
        }