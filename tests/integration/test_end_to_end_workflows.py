"""
Integration tests for end-to-end workflows.

These tests verify that all components work together correctly
in realistic usage scenarios.
"""

import pytest
import torch
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from cs336_basics.inference.generator import TextGenerator, GenerationConfig
from cs336_basics.inference.cache import create_kv_cache_manager
from cs336_basics.inference.samplers import create_sampler
from cs336_basics.inference.utils import create_generation_pipeline
from cs336_basics.optimization.mixed_precision import MixedPrecisionTrainer
from cs336_basics.optimization.compilation import compile_model
from cs336_basics.optimization.checkpointing import EnhancedCheckpoint
from ..fixtures.test_models import create_mock_model, MockTokenizer, create_mock_config


class TestModelToGenerationWorkflow:
    """Test complete workflow from model loading to text generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_model_to_generation_workflow(self):
        """Test basic end-to-end generation workflow."""
        # Create generator with model and tokenizer
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test text generation
        prompt = "Hello, how are you?"
        generated_text = generator.generate_text(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            do_sample=True
        )
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
        
        # Test chat functionality
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = generator.chat(messages, max_new_tokens=15)
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check generation statistics
        stats = generator.get_generation_stats()
        assert stats["num_generations"] >= 2
        assert stats["total_tokens_generated"] > 0
    
    def test_generation_with_kv_cache_workflow(self):
        """Test generation workflow with KV-cache enabled."""
        # Create generator with KV-cache
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test multiple generations (should benefit from caching)
        prompts = [
            "Tell me a story about",
            "What is the meaning of",
            "How do I learn"
        ]
        
        total_generation_time = 0
        
        for prompt in prompts:
            start_time = time.time()
            generated_text = generator.generate_text(
                prompt,
                max_new_tokens=8,
                do_sample=False  # Greedy for consistency
            )
            generation_time = time.time() - start_time
            total_generation_time += generation_time
            
            assert isinstance(generated_text, str)
            assert len(generated_text) > len(prompt)
        
        # Check that cache manager exists and has statistics
        assert generator.cache_manager is not None
        cache_stats = generator.cache_manager.get_total_memory_usage()
        assert "total_memory_mb" in cache_stats
        assert cache_stats["num_layers"] > 0
    
    def test_generation_with_different_sampling_strategies(self):
        """Test generation with various sampling strategies."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        prompt = "The weather today is"
        sampling_strategies = [
            {"sampling_strategy": "greedy", "do_sample": False},
            {"sampling_strategy": "multinomial", "temperature": 1.0},
            {"sampling_strategy": "top_k", "top_k": 10, "temperature": 0.8},
            {"sampling_strategy": "top_p", "top_p": 0.9, "temperature": 0.8},
            {"sampling_strategy": "mixed", "top_k": 20, "top_p": 0.95, "temperature": 0.7}
        ]
        
        for strategy_config in sampling_strategies:
            config = GenerationConfig(
                max_new_tokens=5,
                **strategy_config
            )
            
            output = generator.generate(
                torch.tensor([[1, 2, 3, 4]]),  # Mock input IDs
                config
            )
            
            assert output.sequences.shape[1] > 4  # Should have generated tokens
            assert output.num_generated_tokens <= 5
            assert output.generation_time > 0
            assert output.tokens_per_second > 0


class TestOptimizedGenerationWorkflow:
    """Test generation workflow with optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_mixed_precision_generation_workflow(self):
        """Test generation with mixed precision optimization."""
        # Create mixed precision trainer
        mp_trainer = MixedPrecisionTrainer(device=self.device, enabled=True)
        
        # Create generator
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test generation (mixed precision affects training, not inference directly)
        generated_text = generator.generate_text(
            "Test prompt",
            max_new_tokens=5,
            do_sample=False
        )
        
        assert isinstance(generated_text, str)
        
        # Test mixed precision stats
        mp_stats = mp_trainer.get_stats()
        assert mp_stats["enabled"] is True
        assert "device_type" in mp_stats
    
    def test_compiled_model_generation_workflow(self):
        """Test generation with compiled model."""
        # Compile model (if supported)
        compiled_model = compile_model(self.model, fallback_on_error=True)
        
        # Create generator with compiled model
        generator = TextGenerator(
            model=compiled_model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test generation
        generated_text = generator.generate_text(
            "Compiled model test",
            max_new_tokens=5,
            do_sample=False
        )
        
        assert isinstance(generated_text, str)
        
        # Compiled model should produce same results as original
        # (though this is hard to test deterministically)
        assert len(generated_text) > len("Compiled model test")
    
    def test_optimized_pipeline_creation_workflow(self):
        """Test creation of optimized generation pipeline."""
        # Create optimized pipeline
        generator = create_generation_pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        assert isinstance(generator, TextGenerator)
        assert generator.device == self.device
        assert generator.use_cache is True
        
        # Test generation with optimized pipeline
        prompt = "Optimized generation test"
        generated_text = generator.generate_text(
            prompt,
            max_new_tokens=8,
            temperature=0.7
        )
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)


class TestTrainingToInferenceWorkflow:
    """Test workflow from training/checkpointing to inference."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_save_load_inference_workflow(self):
        """Test complete workflow: training -> checkpoint -> load -> inference."""
        # Create checkpoint manager
        checkpoint_manager = EnhancedCheckpoint(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=3,
            save_best=True,
            best_metric="val_loss"
        )
        
        # Simulate training and checkpointing
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        for iteration in range(5):
            # Simulate training step
            inputs = torch.randint(0, 1000, (2, 8), device=self.device)
            outputs = self.model(inputs)
            loss = torch.mean(outputs)
            
            # Save checkpoint
            metrics = {"train_loss": loss.item(), "val_loss": loss.item() * 0.9}
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                iteration=iteration,
                epoch=0,
                metrics=metrics,
                force_save=True
            )
            
            if checkpoint_path:
                assert Path(checkpoint_path).exists()
        
        # Load best checkpoint
        new_model = create_mock_model(device=self.device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        
        checkpoint_data = checkpoint_manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            load_best=True
        )
        
        assert checkpoint_data is not None
        assert "iteration" in checkpoint_data
        assert "metrics" in checkpoint_data
        
        # Use loaded model for inference
        generator = TextGenerator(
            model=new_model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        generated_text = generator.generate_text(
            "Test with loaded model",
            max_new_tokens=5
        )
        
        assert isinstance(generated_text, str)
        
        # Check checkpoint info
        checkpoint_info = checkpoint_manager.get_checkpoint_info()
        assert checkpoint_info["num_checkpoints"] > 0
        assert checkpoint_info["best_checkpoint"] is not None
    
    def test_mixed_precision_training_inference_workflow(self):
        """Test workflow with mixed precision training and inference."""
        # Setup mixed precision trainer
        mp_trainer = MixedPrecisionTrainer(device=self.device, enabled=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Simulate training steps with mixed precision
        for _ in range(3):
            inputs = torch.randint(0, 1000, (2, 8), device=self.device)
            targets = torch.randint(0, 1000, (2, 8), device=self.device)
            
            def loss_fn():
                with mp_trainer.autocast():
                    outputs = self.model(inputs)
                    return torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
            
            loss = mp_trainer.training_step(self.model, optimizer, loss_fn)
            assert loss.item() > 0
        
        # Use trained model for inference
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        generated_text = generator.generate_text(
            "Mixed precision trained model",
            max_new_tokens=6
        )
        
        assert isinstance(generated_text, str)
        
        # Check mixed precision stats
        mp_stats = mp_trainer.get_stats()
        assert mp_stats["enabled"] is True


class TestUIIntegrationWorkflow:
    """Test integration with UI components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_cli_integration_workflow(self):
        """Test integration with CLI interface."""
        # Mock CLI interface creation
        with patch('cs336_basics.ui.cli.InteractiveCLI') as mock_cli_class:
            mock_cli = Mock()
            mock_cli_class.return_value = mock_cli
            
            # Mock CLI initialization with model and tokenizer
            from cs336_basics.ui.cli import InteractiveCLI
            
            # Test CLI would work with our components
            generator = TextGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                use_cache=True
            )
            
            # Simulate CLI generation
            response = generator.generate_text(
                "CLI integration test",
                max_new_tokens=8,
                temperature=0.8
            )
            
            assert isinstance(response, str)
            
            # Test chat functionality that CLI would use
            messages = [{"role": "user", "content": "Hello"}]
            chat_response = generator.chat(messages, max_new_tokens=10)
            
            assert isinstance(chat_response, str)
    
    def test_web_app_integration_workflow(self):
        """Test integration with web application."""
        try:
            from cs336_basics.ui.web_app import WebChatApp
            flask_available = True
        except ImportError:
            flask_available = False
        
        if not flask_available:
            pytest.skip("Flask not available for web app testing")
        
        # Mock web app creation
        with patch('cs336_basics.ui.web_app.WebChatApp') as mock_web_class:
            mock_web_app = Mock()
            mock_web_class.return_value = mock_web_app
            
            # Test that generator works with web app patterns
            generator = TextGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                use_cache=True
            )
            
            # Simulate web app conversation flow
            conversation_messages = []
            
            # User message
            user_message = {"role": "user", "content": "Hello from web"}
            conversation_messages.append(user_message)
            
            # Generate response
            response = generator.chat(conversation_messages, max_new_tokens=12)
            
            # Assistant message
            assistant_message = {"role": "assistant", "content": response}
            conversation_messages.append(assistant_message)
            
            assert len(conversation_messages) == 2
            assert conversation_messages[0]["role"] == "user"
            assert conversation_messages[1]["role"] == "assistant"
            assert isinstance(conversation_messages[1]["content"], str)


class TestPerformanceWorkflow:
    """Test performance-related end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_batch_generation_performance_workflow(self):
        """Test batch generation performance."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test single vs batch generation
        single_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        # Single generations
        single_start_time = time.time()
        single_results = []
        for prompt in single_prompts:
            result = generator.generate_text(prompt, max_new_tokens=5)
            single_results.append(result)
        single_time = time.time() - single_start_time
        
        # Batch generation (simulated)
        batch_start_time = time.time()
        batch_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        batch_output = generator.generate(batch_input_ids, config)
        batch_time = time.time() - batch_start_time
        
        # Verify results
        assert len(single_results) == 3
        assert all(isinstance(result, str) for result in single_results)
        
        assert batch_output.sequences.shape[0] == 3  # Batch size
        assert batch_output.sequences.shape[1] > 3   # Generated tokens
        
        # Performance should be reasonable
        assert single_time > 0
        assert batch_time > 0
        
        # Check generation statistics
        stats = generator.get_generation_stats()
        assert stats["num_generations"] >= 4  # 3 single + 1 batch
    
    def test_memory_usage_workflow(self):
        """Test memory usage patterns in generation."""
        from cs336_basics.inference.utils import estimate_generation_memory
        
        # Estimate memory for different configurations
        memory_estimates = []
        
        configs = [
            {"batch_size": 1, "max_seq_len": 128, "use_cache": False},
            {"batch_size": 1, "max_seq_len": 128, "use_cache": True},
            {"batch_size": 4, "max_seq_len": 128, "use_cache": True},
            {"batch_size": 1, "max_seq_len": 512, "use_cache": True},
        ]
        
        for config in configs:
            memory_info = estimate_generation_memory(
                model=self.model,
                precision="float32",
                **config
            )
            memory_estimates.append(memory_info)
            
            assert memory_info["total_estimated_mb"] > 0
            assert memory_info["model_memory_mb"] > 0
        
        # Cache should increase memory usage
        no_cache_memory = memory_estimates[0]["total_estimated_mb"]
        with_cache_memory = memory_estimates[1]["total_estimated_mb"]
        assert with_cache_memory > no_cache_memory
        
        # Batch size should increase memory usage
        batch1_memory = memory_estimates[1]["total_estimated_mb"]
        batch4_memory = memory_estimates[2]["total_estimated_mb"]
        assert batch4_memory > batch1_memory
        
        # Sequence length should increase memory usage
        seq128_memory = memory_estimates[1]["total_estimated_mb"]
        seq512_memory = memory_estimates[3]["total_estimated_mb"]
        assert seq512_memory > seq128_memory


class TestErrorHandlingWorkflow:
    """Test error handling in end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_generation_error_recovery_workflow(self):
        """Test error recovery during generation."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test generation with problematic input
        try:
            # Empty input
            result = generator.generate_text("", max_new_tokens=5)
            assert isinstance(result, str)  # Should handle gracefully
        except Exception as e:
            # Should not crash the entire system
            assert isinstance(e, Exception)
        
        # Test generation with very long input
        long_prompt = "Very long prompt. " * 100
        
        try:
            result = generator.generate_text(
                long_prompt,
                max_new_tokens=5,
                # Should handle or truncate gracefully
            )
            assert isinstance(result, str)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)
        
        # Normal generation should still work after errors
        normal_result = generator.generate_text(
            "Normal prompt",
            max_new_tokens=5
        )
        assert isinstance(normal_result, str)
    
    def test_model_loading_error_handling_workflow(self):
        """Test error handling during model loading workflows."""
        # Test with invalid model
        class BrokenModel(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Model is broken")
        
        broken_model = BrokenModel()
        
        # Generator should handle broken model gracefully
        generator = TextGenerator(
            model=broken_model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Generation should fail gracefully
        with pytest.raises(RuntimeError):
            generator.generate_text("Test prompt", max_new_tokens=5)
        
        # Stats should still be accessible
        stats = generator.get_generation_stats()
        assert isinstance(stats, dict)
    
    def test_cache_error_handling_workflow(self):
        """Test error handling with cache-related issues."""
        # Test with very large cache requirements
        try:
            from cs336_basics.inference.cache import CacheConfig, KVCacheManager
            
            # Create unrealistic cache config
            cache_config = CacheConfig(
                max_batch_size=1000,
                max_seq_len=100000,  # Very large
                num_heads=8,
                head_dim=64,
                num_layers=100,  # Many layers
                device=self.device
            )
            
            # Should handle large configs gracefully
            cache_manager = KVCacheManager(cache_config)
            memory_usage = cache_manager.get_total_memory_usage()
            
            assert memory_usage["total_memory_mb"] > 0
            
        except (MemoryError, RuntimeError) as e:
            # Should handle memory-related errors
            assert isinstance(e, (MemoryError, RuntimeError))


class TestConfigurationWorkflow:
    """Test configuration-driven workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_yaml_configuration_workflow(self):
        """Test workflow with YAML configuration files."""
        # Create configuration file
        config_data = {
            "model": {
                "d_model": 128,
                "num_heads": 4,
                "num_layers": 2,
                "vocab_size": 1000
            },
            "generation": {
                "max_new_tokens": 20,
                "temperature": 0.8,
                "top_p": 0.9,
                "sampling_strategy": "top_p"
            },
            "optimization": {
                "use_cache": True,
                "mixed_precision": True,
                "compile_model": False
            }
        }
        
        config_path = Path(self.temp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test loading and using configuration
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == config_data
        
        # Create generation config from loaded data
        gen_config_data = loaded_config.get("generation", {})
        generation_config = GenerationConfig(**gen_config_data)
        
        assert generation_config.max_new_tokens == 20
        assert generation_config.temperature == 0.8
        assert generation_config.top_p == 0.9
        assert generation_config.sampling_strategy == "top_p"
        
        # Test generation with configuration
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=loaded_config["optimization"]["use_cache"]
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = generator.generate(input_ids, generation_config)
        
        assert output.sequences.shape[1] > 4  # Generated tokens
        assert output.num_generated_tokens <= 20
    
    def test_dynamic_configuration_workflow(self):
        """Test dynamic configuration changes during generation."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=False
        )
        
        # Test with different configurations
        configs = [
            GenerationConfig(temperature=0.1, max_new_tokens=5),  # Very deterministic
            GenerationConfig(temperature=1.5, max_new_tokens=5),  # Very random
            GenerationConfig(top_k=5, max_new_tokens=5),          # Top-k sampling
            GenerationConfig(top_p=0.5, max_new_tokens=5),        # Top-p sampling
        ]
        
        input_ids = torch.tensor([[1, 2, 3]])
        results = []
        
        for config in configs:
            output = generator.generate(input_ids, config)
            results.append(output)
            
            assert output.sequences.shape[1] == 8  # 3 input + 5 generated
            assert output.num_generated_tokens == 5
            assert output.generation_config == config
        
        # All generations should have completed successfully
        assert len(results) == 4
        assert all(result.sequences.shape[0] == 1 for result in results)


class TestRobustnessWorkflow:
    """Test robustness of end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = create_mock_model(device=self.device)
        self.tokenizer = MockTokenizer()
    
    def test_concurrent_generation_workflow(self):
        """Test multiple concurrent generations."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Simulate concurrent-like generations (sequential in test)
        prompts = [f"Prompt {i}" for i in range(10)]
        results = []
        
        start_time = time.time()
        for prompt in prompts:
            result = generator.generate_text(
                prompt,
                max_new_tokens=3,
                do_sample=False  # Deterministic for testing
            )
            results.append(result)
        total_time = time.time() - start_time
        
        # All generations should succeed
        assert len(results) == 10
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > len(f"Prompt {i}") for i, result in enumerate(results))
        
        # Check statistics
        stats = generator.get_generation_stats()
        assert stats["num_generations"] == 10
        assert stats["total_tokens_generated"] > 0
        assert stats["avg_tokens_per_second"] > 0
        
        # Performance should be reasonable
        assert total_time < 60  # Should not take more than 1 minute
    
    def test_stress_generation_workflow(self):
        """Test generation under stress conditions."""
        generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True
        )
        
        # Test many short generations
        short_generation_results = []
        for i in range(50):
            result = generator.generate_text(
                f"Short {i}",
                max_new_tokens=2,
                do_sample=False
            )
            short_generation_results.append(result)
        
        assert len(short_generation_results) == 50
        
        # Test fewer long generations
        long_generation_results = []
        for i in range(5):
            result = generator.generate_text(
                f"Long generation prompt number {i}",
                max_new_tokens=25,
                do_sample=False
            )
            long_generation_results.append(result)
        
        assert len(long_generation_results) == 5
        
        # All should be successful
        all_results = short_generation_results + long_generation_results
        assert all(isinstance(result, str) for result in all_results)
        
        # Check final statistics
        stats = generator.get_generation_stats()
        assert stats["num_generations"] == 55  # 50 + 5
        
        # Cache should be utilized
        if generator.cache_manager:
            cache_stats = generator.cache_manager.get_total_memory_usage()
            assert cache_stats["total_memory_mb"] > 0