# CS336 Assignment 1 - Testing Guide

## Overview

This document provides comprehensive guidance for testing the enhanced CS336 Transformer implementation. The codebase includes core CS336 functionality plus advanced features like interactive interfaces, mixed precision training, and optimized inference.

## Current Test Status

### ✅ Core Functionality (All Working)
The original CS336 assignment tests are fully functional:
- **46 tests passing** out of 48 total tests (2 skipped due to memory constraints)
- All core transformer components tested and verified
- Model architecture, tokenization, training utilities all validated

```bash
# Run core tests
uv run pytest tests/test_model.py tests/test_tokenizer.py tests/test_data.py tests/test_train_bpe.py tests/test_optimizer.py tests/test_nn_utils.py tests/test_serialization.py
```

### 🚧 Enhanced Features (Needs Test Development)
The following modules exist but need comprehensive testing:

#### Optimization Module (`cs336_basics/optimization/`)
- **Mixed Precision Training**: `mixed_precision.py`
  - Classes: `AutoCaster`, `MixedPrecisionTrainer`
  - Features: FP16/BF16 support, automatic loss scaling, gradient management
  
- **Compilation**: `compilation.py` 
  - Functions: `compile_transformer()`, `benchmark_compilation()`
  - Classes: `CompilationManager`
  - Features: PyTorch 2.0+ torch.compile integration

- **Checkpointing**: `checkpointing.py`
  - Advanced checkpoint management with best model tracking

#### Inference Module (`cs336_basics/inference/`)
- **KV Cache**: `cache.py` - Efficient autoregressive generation caching
- **Text Generator**: `generator.py` - Main generation engine with streaming
- **Sampling**: `samplers.py` - Multiple sampling strategies (top-k, top-p, temperature)
- **Utilities**: `utils.py` - Inference helper functions

#### UI Module (`cs336_basics/ui/`)
- **CLI Interface**: `cli.py` - Rich terminal interface for model interaction
- **Web App**: `web_app.py` - Flask-based web interface
- **Interactive Launcher**: `interactive.py` - Main interface coordinator

## Test Structure

### Core Tests (Working)
```
tests/
├── test_model.py           # Transformer architecture tests
├── test_tokenizer.py       # BPE tokenization tests  
├── test_data.py            # Data loading tests
├── test_train_bpe.py       # BPE training tests
├── test_optimizer.py       # Optimization tests
├── test_nn_utils.py        # Neural network utilities
└── test_serialization.py  # Model checkpointing tests
```

### Test Development Framework
A comprehensive test framework was designed but needs alignment with actual implementation:

```
tests/
├── fixtures/               # Shared test fixtures and mock objects
├── test_optimization/      # Mixed precision, compilation, checkpointing tests  
├── test_inference/         # Cache, generation, sampling tests
├── test_ui/               # CLI and web interface tests
├── integration/           # End-to-end workflow tests
└── benchmarks/           # Performance and stress tests
```

## Running Tests

### Quick Validation
```bash
# Verify core functionality still works
uv run pytest tests/test_model.py tests/test_tokenizer.py -v

# Run all working tests
uv run pytest --tb=short -q
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test complete workflows and component interactions
3. **Performance Tests**: Benchmark generation speed and memory usage
4. **UI Tests**: Validate interactive interfaces (CLI and web)

## Test Development Guidelines

### For New Module Tests

When developing tests for the enhanced modules:

1. **Study the Implementation First**
   ```bash
   # Examine actual module structure
   ls -la cs336_basics/optimization/
   ls -la cs336_basics/inference/  
   ls -la cs336_basics/ui/
   ```

2. **Create Focused Test Files**
   - One test file per module
   - Test actual classes and functions that exist
   - Use realistic test scenarios

3. **Mock External Dependencies**
   - Mock Flask for web app tests
   - Mock Rich for CLI interface tests
   - Mock CUDA operations for device-agnostic tests

4. **Test Error Conditions**
   - Invalid inputs
   - Device compatibility issues
   - Missing optional dependencies

### Example Test Patterns

#### Testing Mixed Precision Training
```python
def test_mixed_precision_trainer():
    model = create_simple_transformer()
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cpu")
    
    trainer = MixedPrecisionTrainer(model, optimizer, device)
    
    # Test forward pass
    inputs = {"input_ids": torch.randint(0, 1000, (2, 10))}
    loss = trainer.forward_step(inputs, lambda x: F.cross_entropy(...))
    
    # Test backward pass
    success = trainer.backward_step(loss)
    assert success
```

#### Testing Text Generation
```python
def test_text_generator():
    model = load_mock_model()
    tokenizer = load_mock_tokenizer() 
    
    generator = TextGenerator(model, tokenizer)
    
    result = generator.generate_text("Hello", max_new_tokens=10)
    
    assert len(result.generated_text) > len("Hello")
    assert result.num_tokens >= 10
```

## Development Priority

### High Priority (Core Enhancement Tests)
1. **Mixed Precision Training** - Critical for training performance
2. **KV Cache** - Essential for efficient inference
3. **Text Generator** - Core functionality for interactive features
4. **Basic Sampling** - Required for text generation

### Medium Priority (Advanced Features)
1. **Compilation Optimization** - Performance enhancement
2. **Advanced Checkpointing** - Training workflow improvement
3. **Web Interface** - User experience feature

### Low Priority (Nice to Have)
1. **Performance Benchmarks** - Optimization validation
2. **Stress Tests** - Robustness validation
3. **CLI Rich Interface** - Enhanced user experience

## Contributing Tests

When adding new tests:

1. **Follow pytest conventions**
2. **Use descriptive test names**
3. **Include docstrings for complex tests**
4. **Add appropriate fixtures in `conftest.py`**
5. **Ensure tests are deterministic and fast**

## Dependencies for Testing

```bash
# Core testing
uv add --dev pytest pytest-cov

# Optional for enhanced features (may need mocking)
uv add --dev flask  # For web interface tests
uv add --dev rich   # For CLI interface tests
```

## Future Work

The comprehensive testing framework outlined above provides a roadmap for:

- Ensuring all enhanced features are properly validated
- Maintaining code quality as new features are added  
- Performance regression detection
- Cross-platform compatibility validation

When implementing these tests, focus on testing the actual implementations rather than assumed interfaces, and ensure all tests align with the specific classes and functions that exist in the codebase.

---

**Note**: This testing guide reflects the current state where core CS336 functionality is fully tested and working, while enhanced features exist but need targeted test development aligned with their actual implementations.