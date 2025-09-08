"""
Test fixtures for creating mock models and data for testing.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from pathlib import Path
import tempfile
import json

import cs336_basics.Transformers_cs336 as my_tf


class MockTransformer(nn.Module):
    """Simplified transformer for testing purposes."""
    
    def __init__(self, d_model=128, num_heads=4, d_ff=256, vocab_size=1000, 
                 context_length=64, num_layers=2, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.to(device)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # Create causal mask
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, x, tgt_mask=mask)
        
        x = self.ln_final(x)
        return self.lm_head(x)


class MockTokenizer:
    """Simple mock tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text: str) -> list:
        # Simple hash-based encoding for consistent results
        tokens = [self.bos_token_id]
        for char in text[:50]:  # Limit length
            tokens.append((hash(char) % (self.vocab_size - 10)) + 10)
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: list) -> str:
        # Simple decode - just return a placeholder
        return f"decoded_text_{len(token_ids)}_tokens"


def create_mock_model(device='cpu') -> MockTransformer:
    """Create a mock transformer model for testing."""
    return MockTransformer(device=device)


def create_real_model(device='cpu') -> my_tf.transformer.Transformer:
    """Create a real CS336 transformer model for testing."""
    return my_tf.transformer.Transformer(
        d_model=128,
        num_heads=4,
        d_ff=256,
        vocab_size=1000,
        context_length=64,
        num_layers=2,
        max_seq_len=64,
        theta=10000.0,
        device=device
    )


def create_mock_tokenizer() -> MockTokenizer:
    """Create a mock tokenizer for testing."""
    return MockTokenizer()


def create_sample_inputs(batch_size=2, seq_len=10, vocab_size=1000, device='cpu') -> torch.Tensor:
    """Create sample input token IDs for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def create_mock_checkpoint(model: nn.Module, iteration=100, epoch=1) -> Dict[str, Any]:
    """Create a mock checkpoint dictionary."""
    return {
        "model_state_dict": model.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "metrics": {
            "train_loss": 2.5,
            "val_loss": 2.8,
            "perplexity": 15.2
        },
        "model_config": {
            "d_model": getattr(model, 'd_model', 128),
            "num_heads": getattr(model, 'num_heads', 4),
            "d_ff": getattr(model, 'd_ff', 256),
            "vocab_size": getattr(model, 'vocab_size', 1000),
            "context_length": getattr(model, 'context_length', 64),
            "num_layers": getattr(model, 'num_layers', 2),
        }
    }


def create_temporary_checkpoint(model: nn.Module) -> str:
    """Create a temporary checkpoint file and return its path."""
    checkpoint = create_mock_checkpoint(model)
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        torch.save(checkpoint, f)
        return f.name


def create_mock_vocab_and_merges() -> Tuple[str, str]:
    """Create temporary vocab and merges files for testing."""
    # Create mock vocabulary
    vocab = {str(i): i for i in range(1000)}
    vocab_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(vocab, vocab_file)
    vocab_file.close()
    
    # Create mock merges
    merges = ["a b", "c d", "e f", "g h"]
    merges_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    merges_file.write("\n".join(merges))
    merges_file.close()
    
    return vocab_file.name, merges_file.name


def create_mock_config() -> Dict[str, Any]:
    """Create a mock configuration dictionary."""
    return {
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 256,
            "num_layers": 2,
            "vocab_size": 1000,
            "context_length": 64,
            "rope_theta": 10000.0
        },
        "optimizer": {
            "learning_rate_max": 0.001,
            "learning_rate_min": 0.0001,
            "warmup_iters": 100,
            "cosine_iters": 1000,
            "weight_decay": 0.01,
            "max_l2_norm": 1.0
        },
        "training": {
            "batch_size": 8,
            "max_iters": 1000,
            "log_every": 100,
            "val_every": 200,
            "device": "cpu"
        }
    }


def cleanup_temp_files(*file_paths: str):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            Path(file_path).unlink()
        except FileNotFoundError:
            pass


# Test data generators
def generate_sample_text() -> str:
    """Generate sample text for testing."""
    return "This is a sample text for testing the transformer model. It should be long enough to create meaningful tokens."


def generate_conversation_history() -> list:
    """Generate sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you explain quantum computing?"},
        {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena to process information differently than classical computers."}
    ]


def generate_large_text(size_kb: int = 10) -> str:
    """Generate large text for performance testing."""
    base_text = "The quick brown fox jumps over the lazy dog. "
    target_size = size_kb * 1024
    repeat_count = target_size // len(base_text) + 1
    return (base_text * repeat_count)[:target_size]