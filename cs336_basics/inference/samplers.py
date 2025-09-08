"""
Sampling Strategies for Text Generation

This module provides various sampling methods for text generation including:
- Greedy sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature sampling
- Multinomial sampling
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseSampler(ABC):
    """Base class for all sampling strategies."""
    
    def __init__(self, temperature: float = 1.0, **kwargs):
        self.temperature = temperature
        
    @abstractmethod
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample next tokens from logits.
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            **kwargs: Additional sampling parameters
        
        Returns:
            Sampled token indices [batch_size]
        """
        pass
    
    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if self.temperature != 1.0 and self.temperature > 0:
            return logits / self.temperature
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """Get sampler configuration."""
        return {"temperature": self.temperature}


class GreedySampler(BaseSampler):
    """Greedy sampling - always select the most likely token."""
    
    def __init__(self):
        super().__init__(temperature=1.0)  # Temperature doesn't affect greedy
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample greedily by selecting argmax."""
        return torch.argmax(logits, dim=-1)
    
    def get_config(self) -> Dict[str, Any]:
        return {"sampler": "greedy"}


class MultinomialSampler(BaseSampler):
    """Basic multinomial sampling with temperature."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature=temperature)
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample using multinomial distribution."""
        scaled_logits = self.apply_temperature(logits)
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampler(BaseSampler):
    """Top-k sampling - sample from top k most likely tokens."""
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.k = k
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from top-k most likely tokens."""
        k = kwargs.get('k', self.k)
        
        if k <= 0 or k >= logits.size(-1):
            # Fall back to multinomial sampling
            return MultinomialSampler(self.temperature).sample(logits)
        
        scaled_logits = self.apply_temperature(logits)
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
        
        # Set all other values to -inf
        top_k_logits = torch.full_like(scaled_logits, float('-inf'))
        top_k_logits.scatter_(-1, top_k_indices, top_k_values)
        
        # Sample from the filtered distribution
        probs = F.softmax(top_k_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def get_config(self) -> Dict[str, Any]:
        return {"sampler": "top_k", "k": self.k, "temperature": self.temperature}


class TopPSampler(BaseSampler):
    """Top-p (nucleus) sampling - sample from smallest set of tokens with cumulative probability >= p."""
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.p = p
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample using nucleus (top-p) sampling."""
        p = kwargs.get('p', self.p)
        
        if p <= 0 or p >= 1:
            # Fall back to multinomial sampling
            return MultinomialSampler(self.temperature).sample(logits)
        
        scaled_logits = self.apply_temperature(logits)
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to keep (cumulative prob <= p)
        # We keep at least one token
        sorted_indices_to_remove = cumulative_probs > p
        # Shift right to keep the first token that exceeds p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample from the filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def get_config(self) -> Dict[str, Any]:
        return {"sampler": "top_p", "p": self.p, "temperature": self.temperature}


class TemperatureSampler(BaseSampler):
    """Temperature-only sampling (equivalent to MultinomialSampler but with explicit naming)."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature=temperature)
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample with temperature scaling."""
        temperature = kwargs.get('temperature', self.temperature)
        temp_sampler = MultinomialSampler(temperature)
        return temp_sampler.sample(logits)
    
    def get_config(self) -> Dict[str, Any]:
        return {"sampler": "temperature", "temperature": self.temperature}


class MixedSampler(BaseSampler):
    """
    Mixed sampling strategy that combines multiple approaches.
    
    Can use both top-k and top-p filtering together.
    """
    
    def __init__(
        self, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_tokens_to_keep: int = 1,
    ):
        super().__init__(temperature=temperature)
        self.top_k = top_k
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def sample(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample using mixed strategy."""
        top_k = kwargs.get('top_k', self.top_k)
        top_p = kwargs.get('top_p', self.top_p)
        temperature = kwargs.get('temperature', self.temperature)
        
        scaled_logits = logits / temperature if temperature > 0 else logits
        
        # Apply top-k filtering if specified
        if top_k is not None and top_k > 0:
            k = min(top_k, scaled_logits.size(-1))
            top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
            scaled_logits = torch.full_like(scaled_logits, float('-inf'))
            scaled_logits.scatter_(-1, top_k_indices, top_k_values)
        
        # Apply top-p filtering if specified
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create mask for tokens to keep
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least min_tokens_to_keep
            if self.min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
            else:
                # Shift to keep first token that exceeds p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample from filtered distribution
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "sampler": "mixed",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_tokens_to_keep": self.min_tokens_to_keep,
        }


# Sampler factory function
def create_sampler(
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs
) -> BaseSampler:
    """
    Create a sampler based on strategy name.
    
    Args:
        strategy: Sampling strategy ("greedy", "multinomial", "top_k", "top_p", "mixed")
        temperature: Temperature for scaling
        top_k: Top-k parameter
        top_p: Top-p parameter
        **kwargs: Additional parameters
    
    Returns:
        Configured sampler instance
    """
    
    strategy = strategy.lower()
    
    if strategy == "greedy":
        return GreedySampler()
    
    elif strategy == "multinomial" or strategy == "temperature":
        return MultinomialSampler(temperature=temperature)
    
    elif strategy == "top_k":
        k = top_k if top_k is not None else kwargs.get('k', 50)
        return TopKSampler(k=k, temperature=temperature)
    
    elif strategy == "top_p" or strategy == "nucleus":
        p = top_p if top_p is not None else kwargs.get('p', 0.9)
        return TopPSampler(p=p, temperature=temperature)
    
    elif strategy == "mixed":
        return MixedSampler(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


# Utility functions for sampling analysis
def analyze_distribution(logits: torch.Tensor, top_k: int = 10) -> Dict[str, Any]:
    """
    Analyze the probability distribution of logits.
    
    Args:
        logits: Logits tensor [batch_size, vocab_size] or [vocab_size]
        top_k: Number of top tokens to analyze
    
    Returns:
        Dictionary with distribution statistics
    """
    
    # Handle single batch item
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        single_batch = True
    else:
        single_batch = False
    
    batch_size, vocab_size = logits.shape
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get top-k tokens and probabilities
    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, vocab_size), dim=-1)
    
    # Compute entropy
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Compute perplexity
    perplexity = torch.exp(entropy)
    
    results = {
        "entropy": entropy.tolist() if not single_batch else entropy.item(),
        "perplexity": perplexity.tolist() if not single_batch else perplexity.item(),
        "top_k_indices": top_k_indices.tolist(),
        "top_k_probs": top_k_probs.tolist(),
        "vocab_size": vocab_size,
        "max_prob": probs.max(dim=-1)[0].tolist() if not single_batch else probs.max().item(),
    }
    
    return results