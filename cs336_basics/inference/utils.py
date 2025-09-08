"""
Inference Utilities

Helper functions for text generation and inference optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_input_ids(
    text: Union[str, List[str]],
    tokenizer: Any,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Prepare input text for generation by tokenizing and formatting.
    
    Args:
        text: Input text or list of texts
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate long sequences
        device: Device to place tensors on
    
    Returns:
        Input IDs tensor [batch_size, seq_len]
    """
    
    if isinstance(text, str):
        text = [text]
    
    # Encode texts
    input_ids = []
    for t in text:
        ids = tokenizer.encode(t)
        if not isinstance(ids, list):
            ids = ids.tolist()
        
        # Apply truncation
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        input_ids.append(ids)
    
    # Apply padding
    if padding:
        max_len = max(len(ids) for ids in input_ids)
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        
        padded_ids = []
        for ids in input_ids:
            padded = ids + [pad_token_id] * (max_len - len(ids))
            padded_ids.append(padded)
        input_ids = padded_ids
    
    # Convert to tensor
    tensor = torch.tensor(input_ids, dtype=torch.long)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def prepare_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create attention mask from input IDs.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        pad_token_id: Token ID used for padding
    
    Returns:
        Attention mask tensor [batch_size, seq_len]
    """
    return (input_ids != pad_token_id).long()


def estimate_generation_memory(
    model: nn.Module,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    use_cache: bool = True,
    precision: str = "float16",
) -> Dict[str, float]:
    """
    Estimate memory requirements for text generation.
    
    Args:
        model: Transformer model
        batch_size: Generation batch size
        max_seq_len: Maximum sequence length
        use_cache: Whether KV-cache is used
        precision: Model precision ("float32", "float16", "bfloat16")
    
    Returns:
        Dictionary with memory estimates in MB
    """
    
    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    num_layers = getattr(model, 'num_layers', 6)
    num_heads = getattr(model, 'num_heads', 8)
    d_model = getattr(model, 'd_model', 512)
    head_dim = d_model // num_heads
    vocab_size = getattr(model, 'vocab_size', 50257)
    
    # Precision mapping
    precision_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }
    bytes_per_element = precision_bytes.get(precision, 4)
    
    # Model parameters memory
    model_memory = num_params * bytes_per_element / (1024 * 1024)
    
    # Activation memory (rough estimate)
    activation_memory = (
        batch_size * max_seq_len * d_model * num_layers * bytes_per_element
    ) / (1024 * 1024)
    
    # KV-cache memory
    if use_cache:
        kv_cache_memory = (
            2 * batch_size * num_heads * max_seq_len * head_dim * 
            num_layers * bytes_per_element
        ) / (1024 * 1024)
    else:
        kv_cache_memory = 0
    
    # Output logits memory
    output_memory = (
        batch_size * max_seq_len * vocab_size * bytes_per_element
    ) / (1024 * 1024)
    
    # Total memory (with some overhead)
    total_memory = (model_memory + activation_memory + kv_cache_memory + output_memory) * 1.2
    
    return {
        "model_memory_mb": model_memory,
        "activation_memory_mb": activation_memory,
        "kv_cache_memory_mb": kv_cache_memory,
        "output_memory_mb": output_memory,
        "total_estimated_mb": total_memory,
        "precision": precision,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
    }


def load_model_for_inference(
    model_path: Union[str, Path],
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
) -> nn.Module:
    """
    Load a model checkpoint optimized for inference.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        torch_dtype: Data type for model parameters
        low_cpu_mem_usage: Whether to minimize CPU memory usage during loading
    
    Returns:
        Loaded model ready for inference
    """
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        extra_info = {
            "iteration": checkpoint.get("iteration", 0),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        model_state = checkpoint
        extra_info = {}
    
    # TODO: In practice, you'd need to instantiate the model architecture here
    # This is a placeholder - you'd need the actual model class and configuration
    logger.warning("Model loading is incomplete - need to implement model architecture instantiation")
    
    return None, extra_info


def optimize_model_for_inference(
    model: nn.Module,
    enable_torch_compile: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Optimize model for inference performance.
    
    Args:
        model: Model to optimize
        enable_torch_compile: Whether to use torch.compile
        enable_mixed_precision: Whether to use mixed precision
        enable_gradient_checkpointing: Whether to use gradient checkpointing (not recommended for inference)
    
    Returns:
        Optimized model
    """
    
    # Set to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Apply torch.compile if available
    if enable_torch_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    # Apply mixed precision
    if enable_mixed_precision:
        try:
            model = model.half()  # Convert to FP16
            logger.info("Applied FP16 mixed precision")
        except Exception as e:
            logger.warning(f"Mixed precision conversion failed: {e}")
    
    return model


def batch_encode_texts(
    texts: List[str],
    tokenizer: Any,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Efficiently batch encode a list of texts.
    
    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer instance
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to place tensors on
    
    Returns:
        List of encoded tensor batches
    """
    
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Encode batch
        input_ids = prepare_input_ids(
            batch_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=True,
            truncation=True,
            device=device,
        )
        
        batches.append(input_ids)
    
    return batches


def create_generation_pipeline(
    model: nn.Module,
    tokenizer: Any,
    device: Optional[torch.device] = None,
    **kwargs
) -> 'TextGenerator':
    """
    Create a text generation pipeline with optimizations.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        device: Device for generation
        **kwargs: Additional arguments for TextGenerator
    
    Returns:
        Configured TextGenerator instance
    """
    
    from .generator import TextGenerator
    
    if device is None:
        device = next(model.parameters()).device
    
    # Optimize model for inference
    optimized_model = optimize_model_for_inference(model)
    
    # Create generator
    generator = TextGenerator(
        model=optimized_model,
        tokenizer=tokenizer,
        device=device,
        **kwargs
    )
    
    return generator


def benchmark_generation_speed(
    generator: 'TextGenerator',
    prompts: List[str],
    max_new_tokens: int = 50,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> Dict[str, float]:
    """
    Benchmark text generation speed.
    
    Args:
        generator: TextGenerator instance
        prompts: List of prompts to test with
        max_new_tokens: Maximum tokens to generate
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with benchmark results
    """
    
    import time
    
    logger.info(f"Benchmarking generation speed with {len(prompts)} prompts")
    
    # Warmup runs
    for _ in range(warmup_runs):
        for prompt in prompts[:1]:  # Just use first prompt for warmup
            _ = generator.generate_text(prompt, max_new_tokens=max_new_tokens)
    
    # Benchmark runs
    times = []
    total_tokens = 0
    
    for run in range(num_runs):
        start_time = time.time()
        
        for prompt in prompts:
            output = generator.generate_text(prompt, max_new_tokens=max_new_tokens)
            # Count generated tokens (approximate)
            total_tokens += len(output.split()) * 0.75  # Rough token count
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    tokens_per_second = (total_tokens / num_runs) / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time_seconds": avg_time,
        "tokens_per_second": tokens_per_second,
        "total_tokens_generated": total_tokens / num_runs,
        "num_prompts": len(prompts),
        "num_runs": num_runs,
        "all_run_times": times,
    }