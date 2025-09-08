"""
Text Generation Engine

High-level interface for generating text with transformer models using
various sampling strategies, KV-caching, and generation controls.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, asdict
import time
import logging
from tqdm import tqdm

from .cache import KVCacheManager, create_kv_cache_manager, CacheConfig
from .samplers import BaseSampler, create_sampler, analyze_distribution

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Generation parameters
    max_new_tokens: int = 50
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    
    # Sampling parameters
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    sampling_strategy: str = "top_p"
    
    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # Generation control
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Performance
    use_cache: bool = True
    batch_size: int = 1
    
    # Output control
    return_dict_in_generate: bool = True
    output_scores: bool = False
    output_attentions: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenerationOutput:
    """Output from text generation."""
    
    sequences: torch.Tensor
    scores: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    
    # Generation statistics
    generation_time: float = 0.0
    num_generated_tokens: int = 0
    tokens_per_second: float = 0.0
    
    # Additional info
    finished_sequences: Optional[List[bool]] = None
    generation_config: Optional[GenerationConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sequences": self.sequences.tolist() if self.sequences is not None else None,
            "generation_time": self.generation_time,
            "num_generated_tokens": self.num_generated_tokens,
            "tokens_per_second": self.tokens_per_second,
            "finished_sequences": self.finished_sequences,
            "config": self.generation_config.to_dict() if self.generation_config else None,
        }


class TextGenerator:
    """
    High-level text generation interface with KV-caching and advanced sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        Initialize text generator.
        
        Args:
            model: Transformer model for generation
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run generation on
            use_cache: Whether to use KV-caching
            cache_config: Configuration for KV-cache
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.use_cache = use_cache
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize KV-cache manager
        if use_cache:
            if cache_config is None:
                self.cache_manager = create_kv_cache_manager(
                    model=model,
                    max_batch_size=8,  # Default batch size
                    max_seq_len=2048,  # Default max length
                    device=self.device,
                )
            else:
                self.cache_manager = KVCacheManager(cache_config)
        else:
            self.cache_manager = None
        
        # Generation statistics
        self.generation_stats = {
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "num_generations": 0,
        }
        
        logger.info(f"TextGenerator initialized with caching={'enabled' if use_cache else 'disabled'}")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[Callable] = None,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate text given input token IDs.
        
        Args:
            input_ids: Input token tensor [batch_size, seq_len]
            generation_config: Generation configuration
            stopping_criteria: Custom stopping criteria function
            **kwargs: Additional generation parameters
        
        Returns:
            GenerationOutput with generated sequences and metadata
        """
        
        # Merge configs
        config = generation_config or GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Prepare inputs
        input_ids = input_ids.to(self.device)
        batch_size, input_length = input_ids.shape
        
        # Initialize cache
        if self.cache_manager is not None:
            self.cache_manager.reset()
        
        # Create sampler
        sampler = create_sampler(
            strategy=config.sampling_strategy,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        ) if config.do_sample else create_sampler("greedy")
        
        # Generation loop
        start_time = time.time()
        generated_tokens = []
        scores = [] if config.output_scores else None
        attentions = [] if config.output_attentions else None
        finished = [False] * batch_size
        
        current_ids = input_ids
        
        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Forward pass
                if step == 0:
                    # First step: process all input tokens
                    outputs = self._forward_with_cache(current_ids, use_cache=False)
                    logits = outputs[:, -1, :]  # Last token logits
                else:
                    # Subsequent steps: only process new token
                    outputs = self._forward_with_cache(
                        next_token_ids.unsqueeze(-1), 
                        use_cache=self.use_cache
                    )
                    logits = outputs[:, -1, :]
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, current_ids, config.repetition_penalty
                    )
                
                # Sample next tokens
                next_token_ids = sampler.sample(logits)
                generated_tokens.append(next_token_ids)
                
                # Store scores if requested
                if config.output_scores:
                    scores.append(logits)
                
                # Update current sequence
                current_ids = torch.cat([current_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                
                # Check stopping criteria
                if config.eos_token_id is not None:
                    finished = [
                        finished[i] or (next_token_ids[i].item() == config.eos_token_id)
                        for i in range(batch_size)
                    ]
                
                # Check if all sequences finished
                if all(finished):
                    break
                
                # Check max length
                if config.max_length and current_ids.size(1) >= config.max_length:
                    break
                
                # Custom stopping criteria
                if stopping_criteria and stopping_criteria(current_ids, step):
                    break
        
        # Calculate generation statistics
        end_time = time.time()
        generation_time = end_time - start_time
        num_generated = len(generated_tokens)
        tokens_per_second = (num_generated * batch_size) / generation_time if generation_time > 0 else 0
        
        # Update global stats
        self.generation_stats["total_tokens_generated"] += num_generated * batch_size
        self.generation_stats["total_generation_time"] += generation_time
        self.generation_stats["num_generations"] += 1
        
        # Create output
        output = GenerationOutput(
            sequences=current_ids,
            scores=scores,
            attentions=attentions,
            generation_time=generation_time,
            num_generated_tokens=num_generated,
            tokens_per_second=tokens_per_second,
            finished_sequences=finished,
            generation_config=config,
        )
        
        logger.info(
            f"Generated {num_generated} tokens in {generation_time:.2f}s "
            f"({tokens_per_second:.1f} tokens/sec)"
        )
        
        return output
    
    def _forward_with_cache(
        self, 
        input_ids: torch.Tensor, 
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional KV-caching.
        
        This is a simplified version - in practice, you'd need to modify
        the model's forward method to support KV-cache.
        """
        # TODO: Integrate with actual model that supports KV-cache
        # For now, just do regular forward pass
        logits = self.model(input_ids)
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        
        batch_size, seq_len = input_ids.shape
        vocab_size = logits.size(-1)
        
        # Create penalty mask
        for batch_idx in range(batch_size):
            for token_id in input_ids[batch_idx]:
                token_id = token_id.item()
                if 0 <= token_id < vocab_size:
                    # Penalize tokens that have appeared before
                    if logits[batch_idx, token_id] > 0:
                        logits[batch_idx, token_id] /= penalty
                    else:
                        logits[batch_idx, token_id] *= penalty
        
        return logits
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Text prompt to continue
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Generated text as string
        """
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        
        # Generate
        config = GenerationConfig(max_new_tokens=max_new_tokens, **generation_kwargs)
        output = self.generate(input_ids, config)
        
        # Decode output
        generated_ids = output.sequences[0]  # Take first batch item
        generated_text = self.tokenizer.decode(generated_ids.tolist())
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        **generation_kwargs
    ) -> str:
        """
        Generate response for a chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        """
        
        # Simple chat template - could be made more sophisticated
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                prompt += f"User: {content}\\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\\n"
        
        prompt += "Assistant: "
        
        # Generate response
        full_response = self.generate_text(
            prompt, 
            max_new_tokens=max_new_tokens, 
            **generation_kwargs
        )
        
        # Extract just the assistant's response
        if "Assistant: " in full_response:
            response = full_response.split("Assistant: ")[-1]
        else:
            response = full_response[len(prompt):]
        
        return response.strip()
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        avg_time = (
            self.generation_stats["total_generation_time"] / 
            max(self.generation_stats["num_generations"], 1)
        )
        avg_tokens_per_sec = (
            self.generation_stats["total_tokens_generated"] /
            max(self.generation_stats["total_generation_time"], 0.001)
        )
        
        stats = {
            **self.generation_stats,
            "avg_generation_time": avg_time,
            "avg_tokens_per_second": avg_tokens_per_sec,
        }
        
        # Add cache stats if available
        if self.cache_manager:
            stats["cache_memory"] = self.cache_manager.get_total_memory_usage()
        
        return stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "num_generations": 0,
        }
    
    def optimize_cache(self):
        """Optimize KV-cache memory usage."""
        if self.cache_manager:
            self.cache_manager.optimize_memory()