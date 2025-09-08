import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch

from cs336_basics.inference.samplers import (
    BaseSampler,
    GreedySampler,
    MultinomialSampler,
    TopKSampler,
    TopPSampler,
    TemperatureSampler,
    MixedSampler,
    create_sampler,
    analyze_distribution
)


class TestBaseSampler:
    """Test BaseSampler abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseSampler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSampler()
    
    def test_apply_temperature(self):
        """Test temperature application."""
        # Create a concrete implementation for testing
        class ConcreteSampler(BaseSampler):
            def sample(self, logits):
                return torch.argmax(logits, dim=-1)
        
        sampler = ConcreteSampler(temperature=2.0)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        scaled_logits = sampler.apply_temperature(logits)
        expected = logits / 2.0
        assert torch.allclose(scaled_logits, expected)
    
    def test_apply_temperature_edge_cases(self):
        """Test temperature edge cases."""
        class ConcreteSampler(BaseSampler):
            def sample(self, logits):
                return torch.argmax(logits, dim=-1)
        
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Temperature = 1.0 (no change)
        sampler = ConcreteSampler(temperature=1.0)
        scaled_logits = sampler.apply_temperature(logits)
        assert torch.allclose(scaled_logits, logits)
        
        # Temperature = 0 (should not scale)
        sampler = ConcreteSampler(temperature=0.0)
        scaled_logits = sampler.apply_temperature(logits)
        assert torch.allclose(scaled_logits, logits)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        class ConcreteSampler(BaseSampler):
            def sample(self, logits):
                return torch.argmax(logits, dim=-1)
        
        sampler = ConcreteSampler(temperature=1.5)
        config = sampler.get_config()
        assert config == {"temperature": 1.5}


class TestGreedySampler:
    """Test GreedySampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = GreedySampler()
    
    def test_initialization(self):
        """Test greedy sampler initialization."""
        assert self.sampler.temperature == 1.0
        config = self.sampler.get_config()
        assert config == {"sampler": "greedy"}
    
    def test_greedy_sampling_single_batch(self):
        """Test greedy sampling with single batch."""
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])  # Max is at index 1
        
        sample = self.sampler.sample(logits)
        
        assert sample.shape == torch.Size([1])
        assert sample[0] == 1  # Index of maximum value
    
    def test_greedy_sampling_multiple_batch(self):
        """Test greedy sampling with multiple batches."""
        logits = torch.tensor([
            [1.0, 3.0, 2.0],  # Max at index 1
            [5.0, 2.0, 1.0],  # Max at index 0
            [0.1, 0.2, 0.9],  # Max at index 2
        ])
        
        samples = self.sampler.sample(logits)
        
        assert samples.shape == torch.Size([3])
        assert samples[0] == 1
        assert samples[1] == 0
        assert samples[2] == 2
    
    def test_greedy_sampling_deterministic(self):
        """Test that greedy sampling is deterministic."""
        logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
        
        sample1 = self.sampler.sample(logits)
        sample2 = self.sampler.sample(logits)
        
        assert torch.equal(sample1, sample2)
        assert sample1[0] == 2  # Index 2 has highest value (3.0)
    
    def test_greedy_sampling_ties(self):
        """Test greedy sampling behavior with tied values."""
        # When values are equal, argmax returns the first index
        logits = torch.tensor([[2.0, 2.0, 1.0]])  # Two maximum values
        
        sample = self.sampler.sample(logits)
        assert sample[0] == 0  # Should return first occurrence


class TestMultinomialSampler:
    """Test MultinomialSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = MultinomialSampler(temperature=1.0)
    
    def test_initialization(self):
        """Test multinomial sampler initialization."""
        assert self.sampler.temperature == 1.0
        config = self.sampler.get_config()
        assert config == {"temperature": 1.0}
    
    def test_multinomial_sampling_shape(self):
        """Test that multinomial sampling returns correct shape."""
        batch_size, vocab_size = 3, 100
        logits = torch.randn(batch_size, vocab_size)
        
        samples = self.sampler.sample(logits)
        
        assert samples.shape == torch.Size([batch_size])
        assert all(0 <= s < vocab_size for s in samples)
    
    def test_multinomial_sampling_with_temperature(self):
        """Test multinomial sampling with different temperatures."""
        torch.manual_seed(42)  # For reproducibility
        
        # Skewed logits to test temperature effect
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])  # Heavily favors first token
        
        # High temperature (more random)
        high_temp_sampler = MultinomialSampler(temperature=2.0)
        
        # Low temperature (more deterministic)
        low_temp_sampler = MultinomialSampler(temperature=0.1)
        
        # Sample multiple times to observe distribution
        high_temp_samples = [high_temp_sampler.sample(logits).item() for _ in range(100)]
        low_temp_samples = [low_temp_sampler.sample(logits).item() for _ in range(100)]
        
        # Low temperature should favor the first token more
        high_temp_first_count = sum(1 for s in high_temp_samples if s == 0)
        low_temp_first_count = sum(1 for s in low_temp_samples if s == 0)
        
        assert low_temp_first_count > high_temp_first_count
    
    def test_multinomial_sampling_probability_distribution(self):
        """Test that sampling follows the probability distribution."""
        torch.manual_seed(42)
        
        # Create logits where token 0 is much more likely
        logits = torch.tensor([[5.0, 1.0, 1.0, 1.0]])  # Token 0 is heavily favored
        
        # Sample many times
        samples = [self.sampler.sample(logits).item() for _ in range(1000)]
        
        # Count occurrences
        token_0_count = sum(1 for s in samples if s == 0)
        
        # Token 0 should be sampled much more frequently
        assert token_0_count > 700  # Should be > 70% due to high logit


class TestTopKSampler:
    """Test TopKSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = TopKSampler(k=3, temperature=1.0)
    
    def test_initialization(self):
        """Test top-k sampler initialization."""
        assert self.sampler.k == 3
        assert self.sampler.temperature == 1.0
        
        config = self.sampler.get_config()
        assert config == {"sampler": "top_k", "k": 3, "temperature": 1.0}
    
    def test_top_k_sampling_constraints(self):
        """Test that top-k sampling only samples from top-k tokens."""
        torch.manual_seed(42)
        
        # Create logits where we know the top-3 tokens
        logits = torch.tensor([[5.0, 4.0, 3.0, 1.0, 0.5, 0.1]])  # Top-3: indices 0, 1, 2
        
        # Sample many times
        samples = [self.sampler.sample(logits).item() for _ in range(100)]
        
        # All samples should be from top-3 tokens (indices 0, 1, 2)
        assert all(s in [0, 1, 2] for s in samples)
    
    def test_top_k_sampling_fallback(self):
        """Test fallback behavior when k is invalid."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])  # vocab_size = 3
        
        # k = 0 should fallback to multinomial
        sampler = TopKSampler(k=0)
        sample = sampler.sample(logits)
        assert 0 <= sample[0] < 3
        
        # k >= vocab_size should fallback to multinomial
        sampler = TopKSampler(k=5)  # k > vocab_size
        sample = sampler.sample(logits)
        assert 0 <= sample[0] < 3
    
    def test_top_k_dynamic_parameter(self):
        """Test passing k as a dynamic parameter."""
        logits = torch.tensor([[5.0, 4.0, 3.0, 1.0, 0.5]])
        
        # Use different k value than initialization
        samples = [self.sampler.sample(logits, k=2).item() for _ in range(50)]
        
        # Should only sample from top-2 (indices 0, 1)
        assert all(s in [0, 1] for s in samples)
    
    def test_top_k_with_temperature(self):
        """Test top-k sampling with temperature scaling."""
        torch.manual_seed(42)
        
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.5]])
        
        # High temperature should make distribution more uniform within top-k
        high_temp_sampler = TopKSampler(k=3, temperature=2.0)
        low_temp_sampler = TopKSampler(k=3, temperature=0.1)
        
        high_temp_samples = [high_temp_sampler.sample(logits).item() for _ in range(100)]
        low_temp_samples = [low_temp_sampler.sample(logits).item() for _ in range(100)]
        
        # Low temperature should favor the highest logit more
        high_temp_top_count = sum(1 for s in high_temp_samples if s == 0)
        low_temp_top_count = sum(1 for s in low_temp_samples if s == 0)
        
        assert low_temp_top_count > high_temp_top_count


class TestTopPSampler:
    """Test TopPSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = TopPSampler(p=0.8, temperature=1.0)
    
    def test_initialization(self):
        """Test top-p sampler initialization."""
        assert self.sampler.p == 0.8
        assert self.sampler.temperature == 1.0
        
        config = self.sampler.get_config()
        assert config == {"sampler": "top_p", "p": 0.8, "temperature": 1.0}
    
    def test_top_p_sampling_nucleus(self):
        """Test that top-p sampling respects the nucleus constraint."""
        torch.manual_seed(42)
        
        # Create a highly skewed distribution
        # After softmax, first few tokens will dominate
        logits = torch.tensor([[10.0, 5.0, 4.0, 1.0, 1.0, 1.0, 1.0]])
        probs = F.softmax(logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        
        # Find which tokens are in top-p=0.8
        nucleus_mask = cumsum[0] <= 0.8
        nucleus_tokens = torch.where(nucleus_mask)[0].tolist()
        
        # Sample many times
        samples = [self.sampler.sample(logits, p=0.8).item() for _ in range(100)]
        
        # All samples should be from nucleus tokens (with some tolerance for the boundary)
        # The algorithm keeps at least one token, so we expect mostly nucleus tokens
        nucleus_samples = sum(1 for s in samples if s in nucleus_tokens)
        assert nucleus_samples > 80  # Should be majority
    
    def test_top_p_sampling_fallback(self):
        """Test fallback behavior when p is invalid."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # p <= 0 should fallback to multinomial
        sampler = TopPSampler(p=0.0)
        sample = sampler.sample(logits)
        assert 0 <= sample[0] < 3
        
        # p >= 1 should fallback to multinomial
        sampler = TopPSampler(p=1.0)
        sample = sampler.sample(logits)
        assert 0 <= sample[0] < 3
    
    def test_top_p_dynamic_parameter(self):
        """Test passing p as a dynamic parameter."""
        logits = torch.tensor([[8.0, 2.0, 1.0, 1.0]])
        
        # Very small p should mostly sample the top token
        samples = [self.sampler.sample(logits, p=0.1).item() for _ in range(50)]
        
        # Should heavily favor token 0
        top_token_count = sum(1 for s in samples if s == 0)
        assert top_token_count > 40  # > 80%
    
    def test_top_p_minimum_tokens(self):
        """Test that top-p keeps at least one token."""
        # Even with very small p, should keep at least one token
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # Uniform distribution
        
        sampler = TopPSampler(p=0.01)  # Very small p
        
        # Should still be able to sample (not fail)
        sample = sampler.sample(logits)
        assert 0 <= sample[0] < 4


class TestTemperatureSampler:
    """Test TemperatureSampler functionality."""
    
    def test_initialization(self):
        """Test temperature sampler initialization."""
        sampler = TemperatureSampler(temperature=1.5)
        assert sampler.temperature == 1.5
        
        config = sampler.get_config()
        assert config == {"sampler": "temperature", "temperature": 1.5}
    
    def test_temperature_sampling_equivalence(self):
        """Test that TemperatureSampler is equivalent to MultinomialSampler."""
        torch.manual_seed(42)
        
        logits = torch.tensor([[2.0, 1.0, 3.0]])
        temperature = 1.5
        
        temp_sampler = TemperatureSampler(temperature=temperature)
        multi_sampler = MultinomialSampler(temperature=temperature)
        
        # They should produce the same distribution behavior
        # (exact equality is hard due to randomness, so we test multiple samples)
        temp_samples = [temp_sampler.sample(logits).item() for _ in range(100)]
        
        torch.manual_seed(42)  # Reset seed
        multi_samples = [multi_sampler.sample(logits).item() for _ in range(100)]
        
        # Should have similar distributions (not exact due to potential implementation differences)
        assert set(temp_samples) == set(multi_samples)  # Same possible outcomes
    
    def test_dynamic_temperature(self):
        """Test dynamic temperature parameter."""
        sampler = TemperatureSampler(temperature=1.0)
        logits = torch.tensor([[5.0, 1.0, 1.0]])
        
        # Use different temperature than initialization
        sample = sampler.sample(logits, temperature=0.1)  # Very low temperature
        
        # Should work without error
        assert 0 <= sample[0] < 3


class TestMixedSampler:
    """Test MixedSampler functionality."""
    
    def test_initialization(self):
        """Test mixed sampler initialization."""
        sampler = MixedSampler(
            temperature=1.2,
            top_k=50,
            top_p=0.9,
            min_tokens_to_keep=2
        )
        
        assert sampler.temperature == 1.2
        assert sampler.top_k == 50
        assert sampler.top_p == 0.9
        assert sampler.min_tokens_to_keep == 2
        
        config = sampler.get_config()
        expected_config = {
            "sampler": "mixed",
            "temperature": 1.2,
            "top_k": 50,
            "top_p": 0.9,
            "min_tokens_to_keep": 2,
        }
        assert config == expected_config
    
    def test_mixed_sampling_top_k_only(self):
        """Test mixed sampling with only top-k filtering."""
        torch.manual_seed(42)
        
        sampler = MixedSampler(top_k=3, top_p=None)
        logits = torch.tensor([[5.0, 4.0, 3.0, 1.0, 0.5]])
        
        # Sample many times
        samples = [sampler.sample(logits).item() for _ in range(100)]
        
        # Should only sample from top-3 tokens
        assert all(s in [0, 1, 2] for s in samples)
    
    def test_mixed_sampling_top_p_only(self):
        """Test mixed sampling with only top-p filtering."""
        torch.manual_seed(42)
        
        sampler = MixedSampler(top_k=None, top_p=0.1)  # Very restrictive
        logits = torch.tensor([[8.0, 2.0, 1.0, 1.0]])
        
        samples = [sampler.sample(logits).item() for _ in range(100)]
        
        # Should heavily favor the top token
        top_count = sum(1 for s in samples if s == 0)
        assert top_count > 80
    
    def test_mixed_sampling_both_filters(self):
        """Test mixed sampling with both top-k and top-p filtering."""
        torch.manual_seed(42)
        
        sampler = MixedSampler(top_k=5, top_p=0.8)
        logits = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]])
        
        # Should apply both filters
        samples = [sampler.sample(logits).item() for _ in range(100)]
        
        # All samples should be valid
        assert all(0 <= s < 8 for s in samples)
        
        # Should be constrained by both top-k and top-p
        unique_samples = set(samples)
        assert len(unique_samples) <= 5  # Top-k constraint
    
    def test_mixed_sampling_min_tokens_to_keep(self):
        """Test minimum tokens to keep parameter."""
        sampler = MixedSampler(top_p=0.01, min_tokens_to_keep=3)  # Very restrictive p
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])  # Uniform
        
        # Should still be able to sample (keeps at least 3 tokens)
        samples = [sampler.sample(logits).item() for _ in range(50)]
        
        # Should have diversity due to min_tokens_to_keep
        unique_samples = set(samples)
        assert len(unique_samples) >= 3
    
    def test_mixed_sampling_dynamic_parameters(self):
        """Test dynamic parameters in mixed sampling."""
        sampler = MixedSampler()  # No preset parameters
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
        
        # Pass parameters dynamically
        sample = sampler.sample(logits, top_k=2, top_p=0.9, temperature=0.5)
        
        assert 0 <= sample[0] < 4


class TestCreateSampler:
    """Test the create_sampler factory function."""
    
    def test_create_greedy_sampler(self):
        """Test creating greedy sampler."""
        sampler = create_sampler("greedy")
        assert isinstance(sampler, GreedySampler)
    
    def test_create_multinomial_sampler(self):
        """Test creating multinomial sampler."""
        sampler = create_sampler("multinomial", temperature=1.5)
        assert isinstance(sampler, MultinomialSampler)
        assert sampler.temperature == 1.5
        
        # Test temperature alias
        sampler = create_sampler("temperature", temperature=2.0)
        assert isinstance(sampler, MultinomialSampler)
        assert sampler.temperature == 2.0
    
    def test_create_top_k_sampler(self):
        """Test creating top-k sampler."""
        sampler = create_sampler("top_k", top_k=25, temperature=1.2)
        assert isinstance(sampler, TopKSampler)
        assert sampler.k == 25
        assert sampler.temperature == 1.2
        
        # Test with k parameter
        sampler = create_sampler("top_k", k=30)
        assert isinstance(sampler, TopKSampler)
        assert sampler.k == 30
    
    def test_create_top_p_sampler(self):
        """Test creating top-p sampler."""
        sampler = create_sampler("top_p", top_p=0.85, temperature=0.9)
        assert isinstance(sampler, TopPSampler)
        assert sampler.p == 0.85
        assert sampler.temperature == 0.9
        
        # Test nucleus alias
        sampler = create_sampler("nucleus", p=0.7)
        assert isinstance(sampler, TopPSampler)
        assert sampler.p == 0.7
    
    def test_create_mixed_sampler(self):
        """Test creating mixed sampler."""
        sampler = create_sampler(
            "mixed",
            temperature=1.1,
            top_k=40,
            top_p=0.95,
            min_tokens_to_keep=2
        )
        assert isinstance(sampler, MixedSampler)
        assert sampler.temperature == 1.1
        assert sampler.top_k == 40
        assert sampler.top_p == 0.95
        assert sampler.min_tokens_to_keep == 2
    
    def test_create_sampler_case_insensitive(self):
        """Test that strategy names are case insensitive."""
        samplers = [
            create_sampler("GREEDY"),
            create_sampler("Top_K", top_k=10),
            create_sampler("TOP_P", top_p=0.9),
            create_sampler("Mixed")
        ]
        
        assert isinstance(samplers[0], GreedySampler)
        assert isinstance(samplers[1], TopKSampler)
        assert isinstance(samplers[2], TopPSampler)
        assert isinstance(samplers[3], MixedSampler)
    
    def test_create_sampler_unknown_strategy(self):
        """Test error handling for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            create_sampler("unknown_strategy")


class TestAnalyzeDistribution:
    """Test the analyze_distribution utility function."""
    
    def test_analyze_single_batch(self):
        """Test analyzing single batch distribution."""
        logits = torch.tensor([3.0, 1.0, 2.0, 0.5, 0.1])
        
        analysis = analyze_distribution(logits, top_k=3)
        
        assert "entropy" in analysis
        assert "perplexity" in analysis
        assert "top_k_indices" in analysis
        assert "top_k_probs" in analysis
        assert "vocab_size" in analysis
        assert "max_prob" in analysis
        
        assert analysis["vocab_size"] == 5
        assert isinstance(analysis["entropy"], float)
        assert isinstance(analysis["perplexity"], float)
        assert isinstance(analysis["max_prob"], float)
        
        # Top-3 should be indices [0, 2, 1] (sorted by logit values 3.0, 2.0, 1.0)
        assert len(analysis["top_k_indices"]) == 3
        assert analysis["top_k_indices"][0] == 0  # Highest logit
    
    def test_analyze_multiple_batch(self):
        """Test analyzing multiple batch distribution."""
        logits = torch.tensor([
            [3.0, 1.0, 2.0],
            [0.5, 2.5, 1.0]
        ])
        
        analysis = analyze_distribution(logits, top_k=2)
        
        # Should return lists for batch dimensions
        assert isinstance(analysis["entropy"], list)
        assert isinstance(analysis["perplexity"], list)
        assert isinstance(analysis["max_prob"], list)
        assert len(analysis["entropy"]) == 2
        assert len(analysis["top_k_indices"]) == 2
        assert len(analysis["top_k_indices"][0]) == 2  # top_k=2
    
    def test_analyze_entropy_calculation(self):
        """Test entropy calculation correctness."""
        # Uniform distribution should have high entropy
        uniform_logits = torch.zeros(100)  # All equal logits
        uniform_analysis = analyze_distribution(uniform_logits)
        
        # Peaked distribution should have low entropy
        peaked_logits = torch.tensor([10.0] + [0.0] * 99)  # One very high logit
        peaked_analysis = analyze_distribution(peaked_logits)
        
        # Uniform should have higher entropy than peaked
        assert uniform_analysis["entropy"] > peaked_analysis["entropy"]
        
        # Perplexity should follow same pattern
        assert uniform_analysis["perplexity"] > peaked_analysis["perplexity"]
    
    def test_analyze_top_k_correctness(self):
        """Test that top-k results are correctly sorted."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])  # Sorted indices should be [1, 4, 2, 3, 0]
        
        analysis = analyze_distribution(logits, top_k=3)
        
        top_k_indices = analysis["top_k_indices"]
        top_k_probs = analysis["top_k_probs"]
        
        # Should be sorted by probability (descending)
        assert top_k_indices == [1, 4, 2]  # Indices of highest logits
        
        # Probabilities should be in descending order
        assert top_k_probs[0] > top_k_probs[1] > top_k_probs[2]
    
    def test_analyze_edge_cases(self):
        """Test edge cases in distribution analysis."""
        # Single token vocabulary
        single_logit = torch.tensor([5.0])
        analysis = analyze_distribution(single_logit, top_k=10)
        
        assert analysis["vocab_size"] == 1
        assert len(analysis["top_k_indices"]) == 1
        assert analysis["top_k_indices"] == [0]
        assert abs(analysis["top_k_probs"][0] - 1.0) < 1e-6  # Should be probability 1
        
        # Large vocabulary
        large_logits = torch.randn(10000)
        analysis = analyze_distribution(large_logits, top_k=5)
        
        assert analysis["vocab_size"] == 10000
        assert len(analysis["top_k_indices"]) == 5
        assert len(analysis["top_k_probs"]) == 5


class TestSamplersIntegration:
    """Integration tests for samplers."""
    
    def test_sampling_consistency_across_strategies(self):
        """Test that all samplers handle the same input correctly."""
        torch.manual_seed(42)
        logits = torch.tensor([[2.0, 1.5, 3.0, 0.5, 1.0]])
        
        samplers = [
            GreedySampler(),
            MultinomialSampler(temperature=1.0),
            TopKSampler(k=3, temperature=1.0),
            TopPSampler(p=0.8, temperature=1.0),
            MixedSampler(temperature=1.0),
        ]
        
        for sampler in samplers:
            sample = sampler.sample(logits)
            
            # All should return valid token indices
            assert sample.shape == torch.Size([1])
            assert 0 <= sample[0] < 5
            
            # Should be tensor of long type
            assert sample.dtype == torch.long
    
    def test_sampler_reproducibility_with_seed(self):
        """Test that samplers are reproducible with fixed seed."""
        logits = torch.tensor([[1.0, 2.0, 1.5, 0.8]])
        
        # Test stochastic samplers
        stochastic_samplers = [
            MultinomialSampler(temperature=1.0),
            TopKSampler(k=3, temperature=1.0),
            TopPSampler(p=0.9, temperature=1.0),
        ]
        
        for sampler in stochastic_samplers:
            # First run
            torch.manual_seed(123)
            sample1 = sampler.sample(logits)
            
            # Second run with same seed
            torch.manual_seed(123)
            sample2 = sampler.sample(logits)
            
            # Should be identical
            assert torch.equal(sample1, sample2)
    
    def test_large_vocabulary_handling(self):
        """Test samplers with large vocabulary sizes."""
        vocab_size = 50000
        batch_size = 2
        logits = torch.randn(batch_size, vocab_size)
        
        samplers = [
            GreedySampler(),
            MultinomialSampler(),
            TopKSampler(k=1000),
            TopPSampler(p=0.95),
            MixedSampler(top_k=500, top_p=0.9),
        ]
        
        for sampler in samplers:
            samples = sampler.sample(logits)
            
            assert samples.shape == torch.Size([batch_size])
            assert all(0 <= s < vocab_size for s in samples)
    
    def test_extreme_temperature_values(self):
        """Test samplers with extreme temperature values."""
        logits = torch.tensor([[5.0, 1.0, 2.0, 0.1]])
        
        # Very high temperature (almost uniform)
        high_temp_sampler = MultinomialSampler(temperature=10.0)
        high_temp_samples = [high_temp_sampler.sample(logits).item() for _ in range(100)]
        
        # Very low temperature (almost greedy)
        low_temp_sampler = MultinomialSampler(temperature=0.01)
        low_temp_samples = [low_temp_sampler.sample(logits).item() for _ in range(100)]
        
        # High temperature should have more diversity
        high_temp_unique = len(set(high_temp_samples))
        low_temp_unique = len(set(low_temp_samples))
        
        # Low temperature should favor token 0 more heavily
        low_temp_top_count = sum(1 for s in low_temp_samples if s == 0)
        high_temp_top_count = sum(1 for s in high_temp_samples if s == 0)
        
        assert low_temp_top_count > high_temp_top_count
        assert high_temp_unique >= low_temp_unique
    
    def test_batch_processing_efficiency(self):
        """Test that samplers handle batches efficiently."""
        batch_sizes = [1, 8, 32, 128]
        vocab_size = 1000
        
        sampler = TopKSampler(k=50)
        
        for batch_size in batch_sizes:
            logits = torch.randn(batch_size, vocab_size)
            
            # Should handle any batch size without error
            samples = sampler.sample(logits)
            assert samples.shape == torch.Size([batch_size])
            assert all(0 <= s < vocab_size for s in samples)