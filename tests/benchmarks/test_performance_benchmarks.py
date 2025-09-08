"""
Performance benchmarks and stress tests for CS336 Transformer system.

These tests measure performance characteristics and validate system behavior
under various load conditions and stress scenarios.
"""

import pytest
import torch
import time
import statistics
import gc
import psutil
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

from cs336_basics.inference.generator import TextGenerator, GenerationConfig
from cs336_basics.inference.cache import KVCacheManager, CacheConfig
from cs336_basics.inference.samplers import create_sampler
from cs336_basics.inference.utils import estimate_generation_memory, benchmark_generation_speed
from cs336_basics.optimization.mixed_precision import MixedPrecisionTrainer
from cs336_basics.optimization.compilation import compile_model
from cs336_basics.optimization.checkpointing import EnhancedCheckpoint
from ..fixtures.test_models import create_mock_model, MockTokenizer


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution and return result and time taken."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


class TestGenerationPerformanceBenchmarks:
    """Benchmark text generation performance under various conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        
        # Clear memory before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.mark.benchmark
    def test_basic_generation_speed_benchmark(self):
        """Benchmark basic text generation speed."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Benchmark parameters
        num_runs = 10
        max_new_tokens = 20
        prompt = "Performance test prompt for benchmarking"
        
        # Warmup runs
        for _ in range(3):
            generator.generate_text(prompt, max_new_tokens=5)
        
        # Benchmark runs
        times = []
        tokens_generated = []
        
        for run in range(num_runs):
            start_time = time.time()
            result = generator.generate_text(prompt, max_new_tokens=max_new_tokens)
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            
            # Estimate token count (rough approximation)
            estimated_tokens = len(result.split()) * 0.75
            tokens_generated.append(estimated_tokens)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        avg_tokens = statistics.mean(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        # Performance assertions
        assert avg_time < 10.0, f"Generation too slow: {avg_time:.2f}s average"
        assert tokens_per_second > 0.1, f"Token generation rate too low: {tokens_per_second:.2f} tokens/s"
        
        # Log performance metrics
        print(f"\n=== Basic Generation Performance ===")
        print(f"Runs: {num_runs}")
        print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Min time: {min_time:.3f}s")
        print(f"Max time: {max_time:.3f}s") 
        print(f"Average tokens: {avg_tokens:.1f}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
        # Store results for comparison
        generator._benchmark_results = {
            "avg_time": avg_time,
            "tokens_per_second": tokens_per_second,
            "std_time": std_time
        }
    
    @pytest.mark.benchmark
    def test_cache_vs_no_cache_performance_benchmark(self):
        """Benchmark performance difference with and without KV-cache."""
        # Generator without cache
        generator_no_cache = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Generator with cache
        generator_with_cache = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test parameters
        num_runs = 8
        max_new_tokens = 15
        prompts = [f"Cache test prompt {i}" for i in range(4)]
        
        # Benchmark without cache
        no_cache_times = []
        for run in range(num_runs):
            prompt = prompts[run % len(prompts)]
            start_time = time.time()
            generator_no_cache.generate_text(prompt, max_new_tokens=max_new_tokens)
            end_time = time.time()
            no_cache_times.append(end_time - start_time)
        
        # Benchmark with cache
        with_cache_times = []
        for run in range(num_runs):
            prompt = prompts[run % len(prompts)]
            start_time = time.time()
            generator_with_cache.generate_text(prompt, max_new_tokens=max_new_tokens)
            end_time = time.time()
            with_cache_times.append(end_time - start_time)
        
        # Calculate averages
        avg_no_cache = statistics.mean(no_cache_times)
        avg_with_cache = statistics.mean(with_cache_times)
        
        # Performance comparison
        speedup = avg_no_cache / avg_with_cache if avg_with_cache > 0 else 1.0
        
        print(f"\n=== Cache Performance Comparison ===")
        print(f"No cache average: {avg_no_cache:.3f}s")
        print(f"With cache average: {avg_with_cache:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Cache should not significantly slow down (might not speed up on CPU)
        assert avg_with_cache < avg_no_cache * 2.0, "Cache causing significant slowdown"
        
        # Check cache memory usage
        if generator_with_cache.cache_manager:
            cache_stats = generator_with_cache.cache_manager.get_total_memory_usage()
            print(f"Cache memory usage: {cache_stats['total_memory_mb']:.2f} MB")
            assert cache_stats["total_memory_mb"] > 0
    
    @pytest.mark.benchmark
    def test_batch_generation_performance_benchmark(self):
        """Benchmark batch generation performance."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        max_new_tokens = 10
        seq_len = 8
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create batch input
            input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=self.device)
            config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
            
            # Warmup
            generator.generate(input_ids, config)
            
            # Benchmark
            times = []
            for _ in range(5):
                start_time = time.time()
                output = generator.generate(input_ids, config)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Verify output shape
                assert output.sequences.shape[0] == batch_size
                assert output.sequences.shape[1] > seq_len
            
            avg_time = statistics.mean(times)
            tokens_per_second_per_sequence = max_new_tokens / avg_time if avg_time > 0 else 0
            total_tokens_per_second = tokens_per_second_per_sequence * batch_size
            
            results[batch_size] = {
                "avg_time": avg_time,
                "tokens_per_second_per_sequence": tokens_per_second_per_sequence,
                "total_tokens_per_second": total_tokens_per_second
            }
        
        print(f"\n=== Batch Generation Performance ===")
        for batch_size, stats in results.items():
            print(f"Batch size {batch_size}: {stats['avg_time']:.3f}s, "
                  f"{stats['total_tokens_per_second']:.2f} tokens/s total")
        
        # Larger batches should generally be more efficient per sequence
        # (though this might not hold on CPU)
        for batch_size in batch_sizes:
            assert results[batch_size]["avg_time"] < 30.0, f"Batch size {batch_size} too slow"
    
    @pytest.mark.benchmark
    def test_sampling_strategy_performance_benchmark(self):
        """Benchmark performance of different sampling strategies."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test different sampling strategies
        strategies = [
            {"sampling_strategy": "greedy", "do_sample": False},
            {"sampling_strategy": "multinomial", "temperature": 1.0, "do_sample": True},
            {"sampling_strategy": "top_k", "top_k": 10, "temperature": 0.8, "do_sample": True},
            {"sampling_strategy": "top_p", "top_p": 0.9, "temperature": 0.8, "do_sample": True},
            {"sampling_strategy": "mixed", "top_k": 20, "top_p": 0.95, "temperature": 0.7, "do_sample": True}
        ]
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        max_new_tokens = 12
        num_runs = 6
        
        results = {}
        
        for strategy_config in strategies:
            strategy_name = strategy_config["sampling_strategy"]
            config = GenerationConfig(max_new_tokens=max_new_tokens, **strategy_config)
            
            # Warmup
            generator.generate(input_ids, config)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                output = generator.generate(input_ids, config)
                end_time = time.time()
                times.append(end_time - start_time)
                
                assert output.sequences.shape[1] > 5  # Generated tokens
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            results[strategy_name] = {
                "avg_time": avg_time,
                "std_time": std_time
            }
        
        print(f"\n=== Sampling Strategy Performance ===")
        for strategy, stats in results.items():
            print(f"{strategy}: {stats['avg_time']:.3f}s ± {stats['std_time']:.3f}s")
        
        # All strategies should complete in reasonable time
        for strategy, stats in results.items():
            assert stats["avg_time"] < 15.0, f"{strategy} sampling too slow: {stats['avg_time']:.2f}s"


class TestMemoryBenchmarks:
    """Benchmark memory usage and efficiency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        
        # Clear memory before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.mark.benchmark
    def test_memory_usage_scaling_benchmark(self):
        """Benchmark memory usage scaling with different parameters."""
        base_memory = get_memory_usage()
        
        # Test different configurations
        configs = [
            {"batch_size": 1, "max_seq_len": 128},
            {"batch_size": 2, "max_seq_len": 128},
            {"batch_size": 1, "max_seq_len": 256},
            {"batch_size": 4, "max_seq_len": 64},
        ]
        
        memory_results = []
        
        for config in configs:
            gc.collect()
            start_memory = get_memory_usage()
            
            # Create cache manager with config
            cache_config = CacheConfig(
                max_batch_size=config["batch_size"],
                max_seq_len=config["max_seq_len"],
                num_heads=4,
                head_dim=32,
                num_layers=2,
                device=self.device
            )
            
            cache_manager = KVCacheManager(cache_config)
            
            # Use the cache
            for layer_idx in range(2):
                keys = torch.randn(config["batch_size"], 4, 10, 32, device=self.device)
                values = torch.randn(config["batch_size"], 4, 10, 32, device=self.device)
                cache_manager.update_layer(layer_idx, keys, values)
            
            end_memory = get_memory_usage()
            memory_used = end_memory - start_memory
            
            # Get cache statistics
            cache_stats = cache_manager.get_total_memory_usage()
            
            memory_results.append({
                "config": config,
                "memory_used_mb": memory_used,
                "cache_estimated_mb": cache_stats["total_memory_mb"],
                "cache_stats": cache_stats
            })
            
            # Clean up
            del cache_manager
            gc.collect()
        
        print(f"\n=== Memory Usage Scaling ===")
        print(f"Base memory: {base_memory:.2f} MB")
        
        for result in memory_results:
            config = result["config"]
            print(f"Config {config}: {result['memory_used_mb']:.2f} MB used, "
                  f"{result['cache_estimated_mb']:.2f} MB estimated")
        
        # Memory usage should be reasonable
        for result in memory_results:
            assert result["memory_used_mb"] < 500, f"Memory usage too high: {result['memory_used_mb']:.2f} MB"
            assert result["cache_estimated_mb"] > 0, "Cache estimation should be positive"
    
    @pytest.mark.benchmark
    def test_memory_efficiency_benchmark(self):
        """Benchmark memory efficiency of different optimization techniques."""
        # Test memory with different optimization settings
        optimizations = [
            {"use_cache": False, "mixed_precision": False},
            {"use_cache": True, "mixed_precision": False},
            {"use_cache": False, "mixed_precision": True},
            {"use_cache": True, "mixed_precision": True},
        ]
        
        memory_results = []
        
        for opt_config in optimizations:
            gc.collect()
            start_memory = get_memory_usage()
            
            # Create generator with optimization settings
            if opt_config["mixed_precision"]:
                # Note: Mixed precision affects training more than inference
                # but we test the setup
                mp_trainer = MixedPrecisionTrainer(device=self.device, enabled=True)
                mp_stats = mp_trainer.get_stats()
                assert mp_stats["enabled"] is True
            
            generator = TextGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                use_cache=opt_config["use_cache"]
            )
            
            # Perform some generations
            for i in range(3):
                generator.generate_text(
                    f"Memory test prompt {i}",
                    max_new_tokens=8,
                    do_sample=False
                )
            
            end_memory = get_memory_usage()
            memory_used = end_memory - start_memory
            
            # Get generation stats
            gen_stats = generator.get_generation_stats()
            
            memory_results.append({
                "optimization": opt_config,
                "memory_used_mb": memory_used,
                "generations": gen_stats["num_generations"],
                "avg_time": gen_stats.get("avg_generation_time", 0)
            })
            
            # Clean up
            del generator
            if opt_config["mixed_precision"]:
                del mp_trainer
            gc.collect()
        
        print(f"\n=== Memory Efficiency Comparison ===")
        for result in memory_results:
            opt = result["optimization"]
            print(f"Cache: {opt['use_cache']}, MP: {opt['mixed_precision']} - "
                  f"Memory: {result['memory_used_mb']:.2f} MB, "
                  f"Avg time: {result['avg_time']:.3f}s")
        
        # All configurations should use reasonable memory
        for result in memory_results:
            assert result["memory_used_mb"] < 1000, f"Memory usage too high: {result['memory_used_mb']:.2f} MB"


class TestStressTests:
    """Stress tests to evaluate system behavior under extreme conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    @pytest.mark.stress
    def test_high_volume_generation_stress(self):
        """Stress test with high volume of generations."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Parameters for stress test
        num_generations = 100
        max_new_tokens = 10
        
        # Track performance over time
        times = []
        memory_samples = []
        
        print(f"\n=== High Volume Generation Stress Test ===")
        print(f"Running {num_generations} generations...")
        
        start_memory = get_memory_usage()
        overall_start = time.time()
        
        for i in range(num_generations):
            # Sample memory every 20 generations
            if i % 20 == 0:
                memory_samples.append(get_memory_usage())
            
            # Vary prompt to avoid caching benefits
            prompt = f"Stress test generation number {i} with varied content"
            
            start_time = time.time()
            result = generator.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic for consistency
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify generation succeeded
            assert isinstance(result, str)
            assert len(result) > len(prompt)
            
            # Progress indicator
            if (i + 1) % 25 == 0:
                avg_time_so_far = statistics.mean(times[-25:])
                print(f"Progress: {i + 1}/{num_generations}, "
                      f"Recent avg time: {avg_time_so_far:.3f}s")
        
        overall_end = time.time()
        final_memory = get_memory_usage()
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        min_time = min(times)
        max_time = max(times)
        total_time = overall_end - overall_start
        memory_growth = final_memory - start_memory
        
        # Get final stats
        final_stats = generator.get_generation_stats()
        
        print(f"Completed {num_generations} generations in {total_time:.2f}s")
        print(f"Average time per generation: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"Min/Max time: {min_time:.3f}s / {max_time:.3f}s")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Tokens per second: {final_stats['avg_tokens_per_second']:.2f}")
        
        # Stress test assertions
        assert avg_time < 5.0, f"Average generation time too slow under stress: {avg_time:.3f}s"
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f} MB"
        assert max_time < avg_time * 10, f"Max time too much higher than average: {max_time:.3f}s vs {avg_time:.3f}s"
        assert final_stats["num_generations"] == num_generations
        
        # Check memory stability (no major leaks)
        if len(memory_samples) > 2:
            memory_trend = memory_samples[-1] - memory_samples[0]
            print(f"Memory trend over test: {memory_trend:.2f} MB")
            assert memory_trend < 200, f"Potential memory leak detected: {memory_trend:.2f} MB growth"
    
    @pytest.mark.stress
    def test_long_sequence_generation_stress(self):
        """Stress test with very long sequence generation."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test parameters
        long_sequence_lengths = [50, 100, 200]
        prompt = "Generate a very long sequence"
        
        results = {}
        
        for max_tokens in long_sequence_lengths:
            print(f"\nTesting {max_tokens} token generation...")
            
            start_memory = get_memory_usage()
            start_time = time.time()
            
            try:
                result = generator.generate_text(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
                
                end_time = time.time()
                end_memory = get_memory_usage()
                
                generation_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                # Verify result
                assert isinstance(result, str)
                assert len(result) > len(prompt)
                
                results[max_tokens] = {
                    "success": True,
                    "time": generation_time,
                    "memory": memory_used,
                    "result_length": len(result)
                }
                
                print(f"{max_tokens} tokens: {generation_time:.2f}s, "
                      f"{memory_used:.2f}MB memory")
                
            except Exception as e:
                results[max_tokens] = {
                    "success": False,
                    "error": str(e),
                    "time": None,
                    "memory": None
                }
                print(f"{max_tokens} tokens: FAILED - {e}")
        
        # Analyze results
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        print(f"\n=== Long Sequence Stress Results ===")
        print(f"Successful generations: {len(successful_results)}/{len(long_sequence_lengths)}")
        
        # At least shorter sequences should succeed
        assert results[50]["success"], "Even short sequences failed"
        
        # Check performance scaling
        if len(successful_results) > 1:
            for tokens, result in successful_results.items():
                tokens_per_second = tokens / result["time"] if result["time"] > 0 else 0
                print(f"{tokens} tokens: {tokens_per_second:.2f} tokens/s")
                
                # Performance should be reasonable
                assert result["time"] < tokens * 0.5, f"Generation too slow for {tokens} tokens"
                assert result["memory"] < 500, f"Memory usage too high for {tokens} tokens"
    
    @pytest.mark.stress
    def test_concurrent_model_usage_stress(self):
        """Stress test simulating concurrent model usage."""
        # Create multiple generators (simulating concurrent usage)
        generators = []
        for i in range(3):
            generator = TextGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                use_cache=True
            )
            generators.append(generator)
        
        # Simulate concurrent-style usage (actually sequential)
        num_rounds = 15
        prompts_per_generator = 3
        
        start_memory = get_memory_usage()
        start_time = time.time()
        
        total_generations = 0
        all_times = []
        
        for round_num in range(num_rounds):
            for gen_idx, generator in enumerate(generators):
                for prompt_idx in range(prompts_per_generator):
                    prompt = f"Concurrent test R{round_num} G{gen_idx} P{prompt_idx}"
                    
                    gen_start = time.time()
                    result = generator.generate_text(
                        prompt,
                        max_new_tokens=8,
                        do_sample=False
                    )
                    gen_end = time.time()
                    
                    all_times.append(gen_end - gen_start)
                    total_generations += 1
                    
                    assert isinstance(result, str)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        total_time = end_time - start_time
        memory_growth = end_memory - start_memory
        avg_gen_time = statistics.mean(all_times)
        
        print(f"\n=== Concurrent Usage Stress Test ===")
        print(f"Total generations: {total_generations}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average generation time: {avg_gen_time:.3f}s")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Generations per second: {total_generations / total_time:.2f}")
        
        # Assertions for concurrent usage
        expected_generations = num_rounds * len(generators) * prompts_per_generator
        assert total_generations == expected_generations
        assert avg_gen_time < 2.0, f"Average generation time too slow: {avg_gen_time:.3f}s"
        assert memory_growth < 150, f"Excessive memory growth: {memory_growth:.2f} MB"
        
        # Cleanup
        for generator in generators:
            del generator
        gc.collect()


class TestOptimizationPerformanceBenchmarks:
    """Benchmark performance of optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    @pytest.mark.benchmark
    def test_mixed_precision_performance_benchmark(self):
        """Benchmark mixed precision training performance."""
        # Create trainers
        mp_trainer = MixedPrecisionTrainer(device=self.device, enabled=True)
        regular_trainer = MixedPrecisionTrainer(device=self.device, enabled=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Benchmark parameters
        num_steps = 10
        
        # Benchmark regular training
        regular_times = []
        for step in range(num_steps):
            inputs = torch.randint(0, 1000, (2, 8), device=self.device)
            targets = torch.randint(0, 1000, (2, 8), device=self.device)
            
            def loss_fn():
                outputs = self.model(inputs)
                return torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1)
                )
            
            start_time = time.time()
            loss = regular_trainer.training_step(self.model, optimizer, loss_fn)
            end_time = time.time()
            
            regular_times.append(end_time - start_time)
            assert loss.item() > 0
        
        # Benchmark mixed precision training
        mp_times = []
        for step in range(num_steps):
            inputs = torch.randint(0, 1000, (2, 8), device=self.device)
            targets = torch.randint(0, 1000, (2, 8), device=self.device)
            
            def loss_fn():
                with mp_trainer.autocast():
                    outputs = self.model(inputs)
                    return torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
            
            start_time = time.time()
            loss = mp_trainer.training_step(self.model, optimizer, loss_fn)
            end_time = time.time()
            
            mp_times.append(end_time - start_time)
            assert loss.item() > 0
        
        # Calculate statistics
        regular_avg = statistics.mean(regular_times)
        mp_avg = statistics.mean(mp_times)
        speedup = regular_avg / mp_avg if mp_avg > 0 else 1.0
        
        print(f"\n=== Mixed Precision Training Performance ===")
        print(f"Regular training: {regular_avg:.4f}s average")
        print(f"Mixed precision: {mp_avg:.4f}s average")
        print(f"Speedup: {speedup:.2f}x")
        
        # Mixed precision should not significantly slow down training
        assert mp_avg < regular_avg * 2.0, "Mixed precision causing significant slowdown"
        
        # Check trainer statistics
        regular_stats = regular_trainer.get_stats()
        mp_stats = mp_trainer.get_stats()
        
        assert regular_stats["enabled"] is False
        assert mp_stats["enabled"] is True
        print(f"MP scale: {mp_stats['scale']}")
    
    @pytest.mark.benchmark
    def test_checkpointing_performance_benchmark(self):
        """Benchmark checkpointing performance."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create checkpoint manager
            checkpoint_manager = EnhancedCheckpoint(
                checkpoint_dir=temp_dir,
                max_checkpoints=5,
                save_best=True
            )
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            
            # Benchmark checkpoint saving
            save_times = []
            num_checkpoints = 10
            
            for iteration in range(num_checkpoints):
                metrics = {
                    "train_loss": 2.5 - iteration * 0.1,
                    "val_loss": 2.8 - iteration * 0.1,
                }
                
                start_time = time.time()
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    iteration=iteration,
                    epoch=0,
                    metrics=metrics,
                    force_save=True
                )
                end_time = time.time()
                
                save_times.append(end_time - start_time)
                assert checkpoint_path is not None
                assert Path(checkpoint_path).exists()
            
            # Benchmark checkpoint loading
            load_times = []
            for _ in range(5):
                new_model = create_mock_model(device=self.device)
                new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
                
                start_time = time.time()
                checkpoint_data = checkpoint_manager.load_checkpoint(
                    model=new_model,
                    optimizer=new_optimizer,
                    load_best=True
                )
                end_time = time.time()
                
                load_times.append(end_time - start_time)
                assert checkpoint_data is not None
            
            # Calculate statistics
            avg_save_time = statistics.mean(save_times)
            avg_load_time = statistics.mean(load_times)
            
            print(f"\n=== Checkpointing Performance ===")
            print(f"Average save time: {avg_save_time:.3f}s")
            print(f"Average load time: {avg_load_time:.3f}s")
            
            # Checkpointing should be reasonably fast
            assert avg_save_time < 5.0, f"Checkpoint saving too slow: {avg_save_time:.3f}s"
            assert avg_load_time < 3.0, f"Checkpoint loading too slow: {avg_load_time:.3f}s"
            
            # Check checkpoint info
            info = checkpoint_manager.get_checkpoint_info()
            assert info["num_checkpoints"] <= 5  # Should respect max_checkpoints
            assert info["best_checkpoint"] is not None
            print(f"Final checkpoint count: {info['num_checkpoints']}")
            print(f"Total size: {info['total_size_mb']:.2f} MB")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestRealWorldBenchmarks:
    """Benchmarks simulating real-world usage patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    @pytest.mark.benchmark
    def test_interactive_chat_simulation_benchmark(self):
        """Benchmark simulating interactive chat usage."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Simulate chat conversation
        conversation_history = []
        response_times = []
        
        # Simulate user messages of varying lengths
        user_messages = [
            "Hello",
            "How are you doing today?",
            "Can you tell me about artificial intelligence and machine learning?",
            "That's interesting. What about deep learning?",
            "Thanks for the explanation. What should I learn next?",
            "Great advice!",
        ]
        
        print(f"\n=== Interactive Chat Simulation ===")
        
        total_start = time.time()
        
        for i, user_message in enumerate(user_messages):
            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_message})
            
            # Generate response
            start_time = time.time()
            response = generator.chat(
                conversation_history[-6:],  # Keep last 6 messages for context
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Add assistant response to conversation
            conversation_history.append({"role": "assistant", "content": response})
            
            print(f"Turn {i+1}: User({len(user_message)} chars) -> "
                  f"Assistant({len(response)} chars) in {response_time:.2f}s")
            
            assert isinstance(response, str)
            assert len(response) > 0
        
        total_time = time.time() - total_start
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        total_turns = len(user_messages)
        
        print(f"\nChat completed: {total_turns} turns in {total_time:.2f}s")
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Max response time: {max_response_time:.2f}s")
        
        # Interactive chat should be responsive
        assert avg_response_time < 5.0, f"Average response too slow for chat: {avg_response_time:.2f}s"
        assert max_response_time < 10.0, f"Max response too slow for chat: {max_response_time:.2f}s"
        
        # Check generation statistics
        final_stats = generator.get_generation_stats()
        assert final_stats["num_generations"] == total_turns
        print(f"Total tokens generated: {final_stats['total_tokens_generated']}")
        print(f"Average tokens/sec: {final_stats['avg_tokens_per_second']:.2f}")
    
    @pytest.mark.benchmark
    def test_batch_processing_workflow_benchmark(self):
        """Benchmark simulating batch processing workflow."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False  # Less beneficial for batch processing
        )
        
        # Simulate batch processing of documents
        documents = [
            f"Document {i}: This is a sample document for batch processing. "
            f"It contains content that needs to be processed by the model."
            for i in range(20)
        ]
        
        # Process in batches
        batch_sizes = [1, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            
            batch_times = []
            total_docs_processed = 0
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Create batch input
                batch_input_ids = []
                for doc in batch_docs:
                    # Simulate tokenization
                    tokens = self.tokenizer.encode(doc)
                    if len(tokens) > 20:
                        tokens = tokens[:20]  # Truncate for consistency
                    batch_input_ids.append(tokens)
                
                # Pad batch to same length
                max_len = max(len(tokens) for tokens in batch_input_ids)
                padded_batch = []
                for tokens in batch_input_ids:
                    padded = tokens + [0] * (max_len - len(tokens))
                    padded_batch.append(padded)
                
                batch_tensor = torch.tensor(padded_batch, device=self.device)
                
                # Generate batch
                config = GenerationConfig(max_new_tokens=15, do_sample=False)
                
                start_time = time.time()
                output = generator.generate(batch_tensor, config)
                end_time = time.time()
                
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                total_docs_processed += len(batch_docs)
                
                # Verify output
                assert output.sequences.shape[0] == len(batch_docs)
                assert output.sequences.shape[1] > max_len
            
            # Calculate batch processing statistics
            total_batch_time = sum(batch_times)
            avg_batch_time = statistics.mean(batch_times)
            docs_per_second = total_docs_processed / total_batch_time if total_batch_time > 0 else 0
            
            results[batch_size] = {
                "total_time": total_batch_time,
                "avg_batch_time": avg_batch_time,
                "docs_per_second": docs_per_second,
                "total_docs": total_docs_processed
            }
            
            print(f"Batch size {batch_size}: {docs_per_second:.2f} docs/sec, "
                  f"{avg_batch_time:.3f}s per batch")
        
        print(f"\n=== Batch Processing Results ===")
        for batch_size, stats in results.items():
            efficiency = stats["docs_per_second"] / batch_size if batch_size > 0 else 0
            print(f"Batch {batch_size}: {stats['docs_per_second']:.2f} docs/sec, "
                  f"efficiency: {efficiency:.2f}")
        
        # Larger batches should generally be more efficient
        # (though this might not hold strongly on CPU)
        for batch_size, stats in results.items():
            assert stats["docs_per_second"] > 0, f"No throughput for batch size {batch_size}"
            assert stats["avg_batch_time"] < 30.0, f"Batch processing too slow: {stats['avg_batch_time']:.2f}s"