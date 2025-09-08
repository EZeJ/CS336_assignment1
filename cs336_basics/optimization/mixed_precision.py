"""
Mixed Precision Training Implementation

This module provides FP16/BF16 training support with automatic loss scaling
for improved training speed and memory efficiency while maintaining numerical stability.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class AutoCaster:
    """
    Automatic mixed precision context manager that works across different devices.
    
    Supports both CUDA (FP16) and CPU/MPS (BF16) with device-appropriate precision.
    """
    
    def __init__(self, device: torch.device, enabled: bool = True, dtype: Optional[torch.dtype] = None):
        self.device = device
        self.enabled = enabled
        
        # Auto-select appropriate dtype based on device
        if dtype is None:
            if device.type == "cuda":
                self.dtype = torch.float16  # FP16 for CUDA
            else:
                self.dtype = torch.bfloat16  # BF16 for CPU/MPS
        else:
            self.dtype = dtype
        
        # Only use CUDA autocast on CUDA devices
        self.use_cuda_autocast = device.type == "cuda"
    
    def __enter__(self):
        if not self.enabled:
            return self
        
        if self.use_cuda_autocast:
            self.autocast_context = autocast(enabled=True, dtype=self.dtype)
        else:
            # For non-CUDA devices, use torch.autocast with device_type
            self.autocast_context = torch.autocast(device_type=self.device.type, 
                                                 enabled=True, 
                                                 dtype=self.dtype)
        
        return self.autocast_context.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'autocast_context'):
            return self.autocast_context.__exit__(exc_type, exc_val, exc_tb)


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper with automatic loss scaling and gradient management.
    
    Features:
    - Automatic device-appropriate precision selection
    - Dynamic loss scaling for numerical stability
    - Gradient accumulation support
    - Comprehensive logging and metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        enabled: bool = True,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.enabled = enabled
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize gradient scaler for CUDA
        if device.type == "cuda" and enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None
        
        # Initialize autocaster
        self.autocaster = AutoCaster(device=device, enabled=enabled)
        
        # Training state
        self.step_count = 0
        self.accumulation_count = 0
        
        # Metrics tracking
        self.metrics = {
            "scale_updates": 0,
            "overflow_skips": 0,
            "successful_steps": 0,
            "current_scale": init_scale if self.scaler else 1.0,
        }
        
        logger.info(f"Mixed precision training initialized: enabled={enabled}, device={device.type}")
    
    def forward_step(self, inputs: Dict[str, torch.Tensor], compute_loss_fn) -> torch.Tensor:
        """
        Perform a forward step with mixed precision.
        
        Args:
            inputs: Dictionary of input tensors
            compute_loss_fn: Function that computes loss given model inputs
        
        Returns:
            Computed loss tensor
        """
        with self.autocaster:
            loss = compute_loss_fn(inputs)
        
        # Scale loss for gradient accumulation
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        
        return loss
    
    def backward_step(self, loss: torch.Tensor) -> bool:
        """
        Perform backward pass with mixed precision scaling.
        
        Args:
            loss: Loss tensor to backpropagate
        
        Returns:
            True if gradients were computed successfully, False if overflow occurred
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accumulation_count += 1
        return True
    
    def optimizer_step(self) -> bool:
        """
        Perform optimizer step with gradient unscaling and clipping.
        
        Returns:
            True if optimizer step was successful, False if skipped due to overflow
        """
        if self.accumulation_count < self.accumulation_steps:
            return True  # Wait for more accumulation
        
        success = True
        
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        if self.max_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                logger.warning(f"Gradient norm is {total_norm}, skipping step")
                success = False
        
        if success:
            if self.scaler is not None:
                # Check for overflow and update
                self.scaler.step(self.optimizer)
                old_scale = self.scaler.get_scale()
                self.scaler.update()
                new_scale = self.scaler.get_scale()
                
                # Track scaling updates
                if new_scale != old_scale:
                    self.metrics["scale_updates"] += 1
                    logger.debug(f"Loss scale updated: {old_scale} -> {new_scale}")
                
                self.metrics["current_scale"] = new_scale
                
                # Check if step was skipped due to overflow
                if new_scale < old_scale:
                    self.metrics["overflow_skips"] += 1
                    success = False
                else:
                    self.metrics["successful_steps"] += 1
            else:
                self.optimizer.step()
                self.metrics["successful_steps"] += 1
        
        # Reset gradients and accumulation counter
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_count = 0
        self.step_count += 1
        
        return success
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "mixed_precision_enabled": self.enabled,
            "current_scale": self.metrics["current_scale"],
            "scale_updates": self.metrics["scale_updates"],
            "overflow_skips": self.metrics["overflow_skips"],
            "successful_steps": self.metrics["successful_steps"],
            "step_count": self.step_count,
            "overflow_rate": self.metrics["overflow_skips"] / max(self.step_count, 1),
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        state = {
            "step_count": self.step_count,
            "accumulation_count": self.accumulation_count,
            "metrics": self.metrics.copy(),
        }
        
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.step_count = state_dict.get("step_count", 0)
        self.accumulation_count = state_dict.get("accumulation_count", 0)
        self.metrics.update(state_dict.get("metrics", {}))
        
        if self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])


def enable_mixed_precision_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **kwargs
) -> MixedPrecisionTrainer:
    """
    Convenience function to enable mixed precision training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        device: Training device
        **kwargs: Additional arguments for MixedPrecisionTrainer
    
    Returns:
        Configured MixedPrecisionTrainer instance
    """
    return MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        **kwargs
    )