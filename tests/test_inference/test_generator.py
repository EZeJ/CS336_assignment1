import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import time

from cs336_basics.inference.generator import (
    GenerationConfig,
    GenerationOutput,
    TextGenerator
)
from cs336_basics.inference.cache import CacheConfig
from ..fixtures.test_models import create_mock_model, MockTokenizer


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default config initialization."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 50
        assert config.max_length is None
        assert config.min_length is None
        assert config.do_sample == True
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.sampling_strategy == "top_p"
        assert config.repetition_penalty == 1.0
        assert config.use_cache == True
        assert config.batch_size == 1
    
    def test_custom_initialization(self):
        """Test config with custom parameters."""
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=1.5,
            top_k=40,
            top_p=0.9,
            sampling_strategy="top_k",
            repetition_penalty=1.1,
            eos_token_id=2,
            use_cache=False
        )
        
        assert config.max_new_tokens == 100
        assert config.temperature == 1.5
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.sampling_strategy == "top_k"
        assert config.repetition_penalty == 1.1
        assert config.eos_token_id == 2
        assert config.use_cache == False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = GenerationConfig(max_new_tokens=30, temperature=0.8)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["max_new_tokens"] == 30
        assert config_dict["temperature"] == 0.8
        assert "do_sample" in config_dict
        assert "sampling_strategy" in config_dict


class TestGenerationOutput:
    """Test GenerationOutput dataclass."""
    
    def test_initialization(self):
        """Test output initialization."""
        sequences = torch.tensor([[1, 2, 3, 4]])
        output = GenerationOutput(
            sequences=sequences,
            generation_time=1.5,
            num_generated_tokens=3,
            tokens_per_second=2.0
        )
        
        assert torch.equal(output.sequences, sequences)
        assert output.generation_time == 1.5
        assert output.num_generated_tokens == 3
        assert output.tokens_per_second == 2.0
        assert output.scores is None
        assert output.attentions is None
    
    def test_with_optional_fields(self):
        """Test output with optional fields."""
        sequences = torch.tensor([[1, 2, 3]])
        scores = [torch.randn(1, 1000), torch.randn(1, 1000)]
        config = GenerationConfig()
        
        output = GenerationOutput(
            sequences=sequences,
            scores=scores,
            generation_config=config,
            finished_sequences=[True]
        )
        
        assert output.scores == scores
        assert output.generation_config == config
        assert output.finished_sequences == [True]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        sequences = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=10)
        
        output = GenerationOutput(
            sequences=sequences,
            generation_time=0.5,
            num_generated_tokens=2,
            tokens_per_second=4.0,
            generation_config=config,
            finished_sequences=[True]
        )
        
        output_dict = output.to_dict()
        
        assert isinstance(output_dict, dict)
        assert output_dict["sequences"] == [[1, 2, 3]]
        assert output_dict["generation_time"] == 0.5
        assert output_dict["num_generated_tokens"] == 2
        assert output_dict["tokens_per_second"] == 4.0
        assert output_dict["finished_sequences"] == [True]
        assert isinstance(output_dict["config"], dict)


class TestTextGenerator:
    """Test TextGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        self.generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False  # Disable cache for simpler testing
        )
    
    def test_initialization_basic(self):
        """Test generator initialization."""
        assert self.generator.model is self.model
        assert self.generator.tokenizer is self.tokenizer
        assert self.generator.device == self.device
        assert self.generator.use_cache == False
        assert self.generator.cache_manager is None
        assert "total_tokens_generated" in self.generator.generation_stats
    
    def test_initialization_with_cache(self):
        """Test generator initialization with caching enabled."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=True
        )
        
        assert generator.use_cache == True
        assert generator.cache_manager is not None
    
    def test_initialization_with_custom_cache_config(self):
        """Test initialization with custom cache configuration."""
        cache_config = CacheConfig(
            max_batch_size=2,
            max_seq_len=1024,
            num_heads=8,
            head_dim=64,
            num_layers=4
        )
        
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=True,
            cache_config=cache_config
        )
        
        assert generator.cache_manager is not None
        assert generator.cache_manager.config.max_batch_size == 2
        assert generator.cache_manager.config.max_seq_len == 1024
    
    def test_generate_basic(self):
        """Test basic text generation."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=5, do_sample=False)  # Greedy
        
        output = self.generator.generate(input_ids, config)
        
        assert isinstance(output, GenerationOutput)
        assert output.sequences.shape[0] == 1  # batch size
        assert output.sequences.shape[1] > 3  # should be longer than input
        assert output.num_generated_tokens <= 5
        assert output.generation_time > 0
        assert output.tokens_per_second > 0
    
    def test_generate_with_sampling(self):
        """Test generation with sampling enabled."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_new_tokens=3,
            do_sample=True,
            sampling_strategy="multinomial",
            temperature=0.8
        )
        
        output = self.generator.generate(input_ids, config)
        
        assert output.sequences.shape[1] == 6  # 3 input + 3 generated
        assert output.num_generated_tokens == 3
    
    def test_generate_with_eos_token(self):
        """Test generation with EOS token stopping."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_new_tokens=10,
            eos_token_id=1,  # Token 1 will stop generation
            do_sample=False
        )
        
        # Mock model to return EOS token
        with patch.object(self.model, 'forward') as mock_forward:
            # Create logits that heavily favor EOS token
            mock_logits = torch.zeros(1, 4, 1000)
            mock_logits[:, -1, 1] = 10.0  # High logit for EOS token
            mock_forward.return_value = mock_logits
            
            output = self.generator.generate(input_ids, config)
            
            # Should stop early due to EOS token
            assert output.num_generated_tokens < 10
            assert output.finished_sequences == [True]
    
    def test_generate_batch(self):
        """Test batch generation."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Batch size 2
        config = GenerationConfig(max_new_tokens=3, do_sample=False)
        
        output = self.generator.generate(input_ids, config)
        
        assert output.sequences.shape[0] == 2  # Batch size preserved
        assert output.sequences.shape[1] == 6  # 3 input + 3 generated
        assert len(output.finished_sequences) == 2
    
    def test_generate_with_max_length(self):
        """Test generation with max_length constraint."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_new_tokens=20,  # High number
            max_length=5,  # Should stop at total length 5
            do_sample=False
        )
        
        output = self.generator.generate(input_ids, config)
        
        assert output.sequences.shape[1] <= 5
        assert output.num_generated_tokens <= 2  # Only 2 tokens to reach max_length
    
    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        input_ids = torch.tensor([[1, 2, 1, 2]])  # Repeated pattern
        config = GenerationConfig(
            max_new_tokens=3,
            repetition_penalty=1.5,
            do_sample=False
        )
        
        output = self.generator.generate(input_ids, config)
        
        # Should successfully generate without error
        assert output.sequences.shape[1] == 7  # 4 input + 3 generated
        assert output.num_generated_tokens == 3
    
    def test_generate_with_scores_output(self):
        """Test generation with score output enabled."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(
            max_new_tokens=2,
            output_scores=True,
            do_sample=False
        )
        
        output = self.generator.generate(input_ids, config)
        
        assert output.scores is not None
        assert len(output.scores) == 2  # One score per generated token
        assert all(isinstance(score, torch.Tensor) for score in output.scores)
    
    def test_generate_with_kwargs_override(self):
        """Test generation with kwargs overriding config."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=5)
        
        # Override max_new_tokens via kwargs
        output = self.generator.generate(input_ids, config, max_new_tokens=2)
        
        assert output.num_generated_tokens == 2  # Should use kwargs value
    
    def test_generate_with_custom_stopping_criteria(self):
        """Test generation with custom stopping criteria."""
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=10)
        
        # Stop after 2 tokens
        def stop_after_2_tokens(current_ids, step):
            return step >= 2
        
        output = self.generator.generate(
            input_ids, 
            config, 
            stopping_criteria=stop_after_2_tokens
        )
        
        assert output.num_generated_tokens == 2
    
    def test_apply_repetition_penalty(self):
        """Test repetition penalty application."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 1.5]])
        input_ids = torch.tensor([[0, 2, 0]])  # Tokens 0 and 2 have appeared
        
        penalized_logits = self.generator._apply_repetition_penalty(
            logits, input_ids, penalty=2.0
        )
        
        # Tokens 0 and 2 should be penalized
        assert penalized_logits[0, 0] < logits[0, 0]  # Token 0 penalized
        assert penalized_logits[0, 2] < logits[0, 2]  # Token 2 penalized
        assert penalized_logits[0, 1] == logits[0, 1]  # Token 1 unchanged
        assert penalized_logits[0, 3] == logits[0, 3]  # Token 3 unchanged
    
    def test_apply_repetition_penalty_no_change(self):
        """Test repetition penalty with penalty=1.0 (no change)."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 1, 2]])
        
        unchanged_logits = self.generator._apply_repetition_penalty(
            logits, input_ids, penalty=1.0
        )
        
        assert torch.equal(unchanged_logits, logits)
    
    def test_generate_text_basic(self):
        """Test text generation from string prompt."""
        prompt = "Hello world"
        
        generated_text = self.generator.generate_text(
            prompt, 
            max_new_tokens=3, 
            do_sample=False
        )
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)  # Should be longer than input
    
    def test_generate_text_no_tokenizer_error(self):
        """Test error when tokenizer is not provided."""
        generator = TextGenerator(model=self.model, device=self.device)
        
        with pytest.raises(ValueError, match="Tokenizer is required"):
            generator.generate_text("Hello")
    
    def test_chat_basic(self):
        """Test basic chat functionality."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = self.generator.chat(messages, max_new_tokens=5)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_chat_prompt_formatting(self):
        """Test chat prompt formatting."""
        messages = [
            {"role": "user", "content": "Test message"},
        ]
        
        # Mock generate_text to capture the formatted prompt
        with patch.object(self.generator, 'generate_text') as mock_generate:
            mock_generate.return_value = "Assistant: Test response"
            
            self.generator.chat(messages)
            
            # Check that generate_text was called with formatted prompt
            called_prompt = mock_generate.call_args[0][0]
            assert "User: Test message\\n" in called_prompt
            assert called_prompt.endswith("Assistant: ")
    
    def test_get_generation_stats(self):
        """Test generation statistics tracking."""
        # Perform some generations
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=2, do_sample=False)
        
        self.generator.generate(input_ids, config)
        self.generator.generate(input_ids, config)
        
        stats = self.generator.get_generation_stats()
        
        assert stats["total_tokens_generated"] > 0
        assert stats["total_generation_time"] > 0
        assert stats["num_generations"] == 2
        assert "avg_generation_time" in stats
        assert "avg_tokens_per_second" in stats
    
    def test_get_generation_stats_with_cache(self):
        """Test generation statistics with cache information."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=True
        )
        
        stats = generator.get_generation_stats()
        
        assert "cache_memory" in stats
        assert isinstance(stats["cache_memory"], dict)
    
    def test_reset_stats(self):
        """Test statistics reset."""
        # Generate something to create non-zero stats
        input_ids = torch.tensor([[1, 2, 3]])
        self.generator.generate(input_ids)
        
        # Verify stats are non-zero
        stats_before = self.generator.get_generation_stats()
        assert stats_before["total_tokens_generated"] > 0
        
        # Reset and verify
        self.generator.reset_stats()
        stats_after = self.generator.get_generation_stats()
        
        assert stats_after["total_tokens_generated"] == 0
        assert stats_after["total_generation_time"] == 0.0
        assert stats_after["num_generations"] == 0
    
    def test_optimize_cache(self):
        """Test cache optimization."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=True
        )
        
        # Should not raise error
        generator.optimize_cache()
    
    def test_optimize_cache_no_cache(self):
        """Test cache optimization when cache is disabled."""
        # Should not raise error even without cache
        self.generator.optimize_cache()
    
    def test_forward_with_cache(self):
        """Test forward pass with cache (simplified version)."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        output = self.generator._forward_with_cache(input_ids, use_cache=False)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 3  # Sequence length
        assert output.shape[2] == 1000  # Vocab size


class TestTextGeneratorIntegration:
    """Integration tests for TextGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end generation workflow."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test text generation
        prompt = "The quick brown fox"
        generated_text = generator.generate_text(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
            sampling_strategy="top_p"
        )
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
        
        # Test chat generation
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = generator.chat(messages, max_new_tokens=15)
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check statistics
        stats = generator.get_generation_stats()
        assert stats["num_generations"] >= 2
        assert stats["total_tokens_generated"] > 0
    
    def test_different_sampling_strategies(self):
        """Test generation with different sampling strategies."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        input_ids = torch.tensor([[1, 2, 3]])
        strategies = ["greedy", "multinomial", "top_k", "top_p", "mixed"]
        
        for strategy in strategies:
            config = GenerationConfig(
                max_new_tokens=3,
                sampling_strategy=strategy,
                temperature=0.8,
                top_k=10,
                top_p=0.9
            )
            
            output = generator.generate(input_ids, config)
            
            assert output.sequences.shape[1] == 6  # 3 input + 3 generated
            assert output.num_generated_tokens == 3
    
    def test_batch_processing_consistency(self):
        """Test that batch processing produces consistent results."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=False
        )
        
        # Single item processing
        single_input = torch.tensor([[1, 2, 3]])
        single_config = GenerationConfig(max_new_tokens=3, do_sample=False)
        single_output = generator.generate(single_input, single_config)
        
        # Batch processing with same input
        batch_input = torch.tensor([[1, 2, 3], [1, 2, 3]])
        batch_config = GenerationConfig(max_new_tokens=3, do_sample=False)
        batch_output = generator.generate(batch_input, batch_config)
        
        # Results should be consistent (for greedy sampling)
        assert batch_output.sequences.shape[0] == 2
        assert torch.equal(single_output.sequences[0], batch_output.sequences[0])
        assert torch.equal(single_output.sequences[0], batch_output.sequences[1])
    
    def test_memory_efficiency_large_sequence(self):
        """Test memory efficiency with longer sequences."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=True
        )
        
        # Generate longer sequence
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        config = GenerationConfig(max_new_tokens=20, do_sample=False)
        
        output = generator.generate(input_ids, config)
        
        assert output.sequences.shape[1] == 25  # 5 input + 20 generated
        assert output.num_generated_tokens == 20
        
        # Should complete without memory errors
        assert output.generation_time > 0
    
    def test_error_handling_invalid_input(self):
        """Test error handling with invalid inputs."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=False
        )
        
        # Empty input
        empty_input = torch.empty(1, 0, dtype=torch.long)
        config = GenerationConfig(max_new_tokens=5)
        
        # Should handle gracefully (may generate from empty context)
        output = generator.generate(empty_input, config)
        assert output.sequences.shape[0] == 1
    
    def test_performance_measurement_accuracy(self):
        """Test accuracy of performance measurements."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            use_cache=False
        )
        
        input_ids = torch.tensor([[1, 2, 3]])
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        start_time = time.time()
        output = generator.generate(input_ids, config)
        actual_time = time.time() - start_time
        
        # Measured time should be close to actual time (within reasonable tolerance)
        assert abs(output.generation_time - actual_time) < 0.1  # 100ms tolerance
        
        # Tokens per second should be reasonable
        expected_tps = output.num_generated_tokens / output.generation_time
        assert abs(output.tokens_per_second - expected_tps) < 0.1