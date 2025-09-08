"""
PyTorch 2.0+ Compilation Utilities

This module provides utilities for optimizing transformer models using torch.compile
for automatic optimization and acceleration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging
import functools

logger = logging.getLogger(__name__)


def compile_transformer(
    model: nn.Module,
    mode: str = "default",
    dynamic: bool = False,
    backend: str = "inductor",
    fullgraph: bool = False,
    disable_on_unsupported: bool = True,
) -> nn.Module:
    """
    Compile a transformer model using torch.compile for optimization.
    
    Args:
        model: The transformer model to compile
        mode: Compilation mode - "default", "reduce-overhead", "max-autotune"
        dynamic: Whether to enable dynamic shapes
        backend: Compilation backend to use
        fullgraph: Whether to require full graph compilation
        disable_on_unsupported: Disable compilation if not supported
    
    Returns:
        Compiled model or original model if compilation is not supported
    """
    
    # Check if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        if not disable_on_unsupported:
            raise RuntimeError("torch.compile requires PyTorch 2.0 or later")
        logger.warning("torch.compile not available, returning uncompiled model")
        return model
    
    # Check device compatibility and adjust backend
    device_type = None
    for param in model.parameters():
        device_type = param.device.type
        break
    
    # Adjust backend based on device compatibility
    if device_type == "mps":
        # MPS doesn't fully support inductor backend yet
        if backend == "inductor":
            if disable_on_unsupported:
                logger.warning("Inductor backend not fully supported on MPS, skipping compilation")
                return model
            else:
                # Try aot_eager as fallback for MPS
                backend = "aot_eager"
                logger.info("Switching to aot_eager backend for MPS compatibility")
    elif device_type == "cpu":
        # CPU works well with inductor
        pass
    elif device_type == "cuda":
        # CUDA works well with inductor
        pass
    
    try:
        compiled_model = torch.compile(
            model,
            mode=mode,
            dynamic=dynamic,
            backend=backend,
            fullgraph=fullgraph,
        )
        
        logger.info(f"Model compiled successfully with mode='{mode}', backend='{backend}'")
        
        # Add compilation info to model
        compiled_model._compilation_config = {
            "mode": mode,
            "dynamic": dynamic,
            "backend": backend,
            "fullgraph": fullgraph,
        }
        
        return compiled_model
    
    except Exception as e:
        if not disable_on_unsupported:
            raise RuntimeError(f"Model compilation failed: {e}")
        
        logger.warning(f"Model compilation failed: {e}. Returning uncompiled model.")
        return model


class CompilationManager:
    """
    Manager for handling model compilation with different strategies.
    """
    
    COMPILATION_MODES = {
        "fast": {"mode": "default", "dynamic": False},
        "memory": {"mode": "reduce-overhead", "dynamic": False},  
        "performance": {"mode": "max-autotune", "dynamic": False},
        "dynamic": {"mode": "default", "dynamic": True},
    }
    
    def __init__(self, strategy: str = "fast"):
        if strategy not in self.COMPILATION_MODES:
            raise ValueError(f"Unknown compilation strategy: {strategy}")
        
        self.strategy = strategy
        self.config = self.COMPILATION_MODES[strategy].copy()
        self.compiled_models: Dict[str, nn.Module] = {}
    
    def compile_model(self, model: nn.Module, name: Optional[str] = None) -> nn.Module:
        """Compile a model with the current strategy."""
        if name is None:
            name = model.__class__.__name__
        
        compiled_model = compile_transformer(model, **self.config)
        self.compiled_models[name] = compiled_model
        
        return compiled_model
    
    def get_compilation_info(self) -> Dict[str, Any]:
        """Get information about compiled models."""
        return {
            "strategy": self.strategy,
            "config": self.config,
            "compiled_models": list(self.compiled_models.keys()),
            "torch_compile_available": hasattr(torch, 'compile'),
        }


def benchmark_compilation(
    model: nn.Module,
    sample_input: torch.Tensor,
    strategies: list = None,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different compilation strategies.
    
    Args:
        model: Model to benchmark
        sample_input: Sample input tensor for benchmarking
        strategies: List of strategies to test (default: all)
        warmup_steps: Number of warmup iterations
        benchmark_steps: Number of benchmark iterations
    
    Returns:
        Dictionary with benchmark results for each strategy
    """
    
    if strategies is None:
        strategies = list(CompilationManager.COMPILATION_MODES.keys()) + ["none"]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Benchmarking strategy: {strategy}")
        
        if strategy == "none":
            test_model = model
        else:
            manager = CompilationManager(strategy)
            test_model = manager.compile_model(model.cpu())  # Compile on CPU first
        
        # Move to appropriate device
        test_model = test_model.to(sample_input.device)
        test_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = test_model(sample_input)
        
        # Benchmark
        torch.cuda.synchronize() if sample_input.device.type == "cuda" else None
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(benchmark_steps):
                _ = test_model(sample_input)
        
        torch.cuda.synchronize() if sample_input.device.type == "cuda" else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / benchmark_steps
        throughput = benchmark_steps / total_time
        
        results[strategy] = {
            "avg_time_ms": avg_time * 1000,
            "throughput_iter_per_sec": throughput,
            "total_time_s": total_time,
        }
        
        logger.info(f"{strategy}: {avg_time*1000:.2f}ms/iter, {throughput:.1f} iter/s")
    
    return results


def selective_compile(
    compile_attention: bool = True,
    compile_ffn: bool = True,
    compile_embeddings: bool = False,
    compile_full_model: bool = False,
    **compile_kwargs
) -> Callable:
    """
    Decorator for selective compilation of transformer components.
    
    Args:
        compile_attention: Whether to compile attention layers
        compile_ffn: Whether to compile feed-forward layers
        compile_embeddings: Whether to compile embedding layers
        compile_full_model: Whether to compile the entire model
        **compile_kwargs: Additional arguments for torch.compile
    
    Returns:
        Decorator function
    """
    
    def decorator(model_class):
        
        @functools.wraps(model_class)
        def wrapper(*args, **kwargs):
            model = model_class(*args, **kwargs)
            
            if not hasattr(torch, 'compile'):
                logger.warning("torch.compile not available, skipping selective compilation")
                return model
            
            if compile_full_model:
                return compile_transformer(model, **compile_kwargs)
            
            # Selective compilation of components
            for name, module in model.named_modules():
                compile_module = False
                
                if compile_attention and "attention" in name.lower():
                    compile_module = True
                elif compile_ffn and any(x in name.lower() for x in ["ffn", "feedforward", "swiglu"]):
                    compile_module = True
                elif compile_embeddings and "embed" in name.lower():
                    compile_module = True
                
                if compile_module:
                    try:
                        compiled_module = torch.compile(module, **compile_kwargs)
                        # Replace module in model
                        parent = model
                        parts = name.split('.')
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], compiled_module)
                        logger.debug(f"Compiled module: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to compile module {name}: {e}")
            
            return model
        
        return wrapper
    
    return decorator