import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import torch

from cs336_basics.ui.cli import InteractiveCLI
from ..fixtures.test_models import create_mock_model, MockTokenizer


class TestInteractiveCLIInitialization:
    """Test InteractiveCLI initialization and setup."""
    
    def test_init_minimal(self):
        """Test CLI initialization with minimal parameters."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI()
            
            assert cli.console is None
            assert cli.use_rich is False
            assert cli.model is None
            assert cli.tokenizer is None
            assert cli.generator is None
            assert cli.conversation_history == []
            assert "messages_generated" in cli.session_stats
    
    @patch('cs336_basics.ui.cli.RICH_AVAILABLE', True)
    @patch('cs336_basics.ui.cli.Console')
    def test_init_with_rich(self, mock_console_class):
        """Test CLI initialization with Rich interface."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        cli = InteractiveCLI(use_rich=True)
        
        assert cli.console is mock_console
        assert cli.use_rich is True
        mock_console_class.assert_called_once()
    
    def test_init_with_config(self):
        """Test CLI initialization with configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'model': {'d_model': 512, 'num_heads': 8},
                'generation': {'temperature': 0.7, 'top_p': 0.9}
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                cli = InteractiveCLI(config_path=config_path)
                
                assert cli.config == config
        finally:
            Path(config_path).unlink()
    
    def test_init_invalid_config(self):
        """Test CLI initialization with invalid config file."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI(config_path="/nonexistent/config.yaml")
            
            assert cli.config == {}  # Should fallback to empty config
    
    def test_device_setup_cuda_available(self):
        """Test device setup when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                cli = InteractiveCLI()
                assert cli.device.type == "cuda"
    
    def test_device_setup_mps_available(self):
        """Test device setup when MPS is available but CUDA is not."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                    cli = InteractiveCLI()
                    assert cli.device.type == "mps"
    
    def test_device_setup_cpu_fallback(self):
        """Test device setup falls back to CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                    cli = InteractiveCLI()
                    assert cli.device.type == "cpu"
    
    def test_device_setup_explicit_device(self):
        """Test device setup with explicitly specified device."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI(device="cpu")
            assert cli.device == torch.device("cpu")
    
    def test_generation_config_initialization(self):
        """Test that generation config is properly initialized."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI()
            
            assert cli.generation_config.max_new_tokens == 100
            assert cli.generation_config.temperature == 0.8
            assert cli.generation_config.top_p == 0.9
            assert cli.generation_config.do_sample is True
            assert cli.generation_config.sampling_strategy == "top_p"
    
    def test_session_stats_initialization(self):
        """Test that session statistics are properly initialized."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI()
            
            expected_keys = [
                "messages_generated", "total_tokens_generated", 
                "total_generation_time", "session_start_time"
            ]
            
            for key in expected_keys:
                assert key in cli.session_stats
            
            assert cli.session_stats["messages_generated"] == 0
            assert cli.session_stats["total_tokens_generated"] == 0
            assert cli.session_stats["total_generation_time"] == 0.0
            assert cli.session_stats["session_start_time"] > 0


class TestInteractiveCLIConfiguration:
    """Test CLI configuration and model loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            self.cli = InteractiveCLI()
    
    def test_load_config_valid_yaml(self):
        """Test loading valid YAML configuration."""
        config_data = {"model": {"d_model": 256}, "device": "cpu"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = self.cli._load_config(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_file(self):
        """Test loading invalid configuration file."""
        config = self.cli._load_config("/nonexistent/file.yaml")
        assert config == {}
    
    def test_load_config_invalid_yaml(self):
        """Test loading malformed YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            config = self.cli._load_config(config_path)
            assert config == {}
        finally:
            Path(config_path).unlink()


class TestInteractiveCLIMessaging:
    """Test CLI messaging and display functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Test both with and without Rich
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            self.cli_no_rich = InteractiveCLI()
        
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', True):
            with patch('cs336_basics.ui.cli.Console') as mock_console_class:
                self.mock_console = Mock()
                mock_console_class.return_value = self.mock_console
                self.cli_rich = InteractiveCLI(use_rich=True)
    
    def test_print_info_no_rich(self):
        """Test info message printing without Rich."""
        with patch('builtins.print') as mock_print:
            self.cli_no_rich._print_info("Test message")
            mock_print.assert_called_once_with("INFO: Test message")
    
    def test_print_info_with_rich(self):
        """Test info message printing with Rich."""
        self.cli_rich._print_info("Test message")
        self.mock_console.print.assert_called_once_with("[blue]ℹ[/blue] Test message")
    
    def test_print_success_no_rich(self):
        """Test success message printing without Rich."""
        with patch('builtins.print') as mock_print:
            self.cli_no_rich._print_success("Success message")
            mock_print.assert_called_once_with("SUCCESS: Success message")
    
    def test_print_success_with_rich(self):
        """Test success message printing with Rich."""
        self.cli_rich._print_success("Success message")
        self.mock_console.print.assert_called_once_with("[green]✓[/green] Success message")
    
    def test_print_warning_no_rich(self):
        """Test warning message printing without Rich."""
        with patch('builtins.print') as mock_print:
            self.cli_no_rich._print_warning("Warning message")
            mock_print.assert_called_once_with("WARNING: Warning message")
    
    def test_print_warning_with_rich(self):
        """Test warning message printing with Rich."""
        self.cli_rich._print_warning("Warning message")
        self.mock_console.print.assert_called_once_with("[yellow]⚠[/yellow] Warning message")
    
    def test_print_error_no_rich(self):
        """Test error message printing without Rich."""
        with patch('builtins.print') as mock_print:
            self.cli_no_rich._print_error("Error message")
            mock_print.assert_called_once_with("ERROR: Error message")
    
    def test_print_error_with_rich(self):
        """Test error message printing with Rich."""
        self.cli_rich._print_error("Error message")
        self.mock_console.print.assert_called_once_with("[red]✗[/red] Error message")


class TestInteractiveCLIModelLoading:
    """Test CLI model and tokenizer loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            self.cli = InteractiveCLI()
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model.pt"
        self.vocab_path = Path(self.temp_dir) / "vocab.json"
        self.merges_path = Path(self.temp_dir) / "merges.txt"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_model_and_tokenizer_success(self):
        """Test successful model and tokenizer loading."""
        # Create mock checkpoint
        mock_model = create_mock_model()
        checkpoint = {
            "model_state_dict": mock_model.state_dict(),
            "model_config": {
                "d_model": 128,
                "num_heads": 4,
                "vocab_size": 1000
            },
            "iteration": 1000
        }
        torch.save(checkpoint, self.model_path)
        
        # Create mock tokenizer files
        vocab = {str(i): i for i in range(100)}
        with open(self.vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        with open(self.merges_path, 'w') as f:
            f.write("a b\\nc d\\n")
        
        # Mock the tokenizer and transformer classes
        with patch('cs336_basics.ui.cli.bpe.Tokenizer') as mock_tokenizer_class:
            with patch('cs336_basics.ui.cli.my_tf.transformer.Transformer') as mock_model_class:
                mock_tokenizer = Mock()
                mock_tokenizer_class.from_files.return_value = mock_tokenizer
                
                mock_model_instance = Mock()
                mock_model_class.return_value = mock_model_instance
                
                # Load model and tokenizer
                self.cli._load_model_and_tokenizer(
                    str(self.model_path),
                    str(self.vocab_path),
                    str(self.merges_path)
                )
                
                # Verify loading was attempted
                mock_tokenizer_class.from_files.assert_called_once()
                assert self.cli.tokenizer == mock_tokenizer
    
    def test_load_model_missing_files(self):
        """Test model loading with missing files."""
        # Don't create the files, so loading should fail gracefully
        self.cli._load_model_and_tokenizer(
            "/nonexistent/model.pt",
            "/nonexistent/vocab.json",
            "/nonexistent/merges.txt"
        )
        
        # Should handle gracefully without crashing
        assert self.cli.model is None
        assert self.cli.tokenizer is None
    
    def test_load_model_invalid_checkpoint(self):
        """Test loading invalid model checkpoint."""
        # Create invalid checkpoint file
        with open(self.model_path, 'w') as f:
            f.write("invalid checkpoint data")
        
        self.cli._load_model_and_tokenizer(str(self.model_path))
        
        # Should handle gracefully
        assert self.cli.model is None


class TestInteractiveCLIModelInfo:
    """Test CLI model information display."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', True):
            with patch('cs336_basics.ui.cli.Console') as mock_console_class:
                self.mock_console = Mock()
                mock_console_class.return_value = self.mock_console
                self.cli = InteractiveCLI(use_rich=True)
        
        # Create mock model for parameter counting
        self.cli.model = create_mock_model()
    
    def test_show_model_info_with_rich(self):
        """Test model information display with Rich interface."""
        model_config = {
            "d_model": 512,
            "num_heads": 8,
            "d_ff": 2048,
            "num_layers": 6,
            "vocab_size": 50000,
            "context_length": 1024
        }
        
        checkpoint = {
            "iteration": 10000,
            "epoch": 5
        }
        
        with patch('cs336_basics.ui.cli.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            self.cli._show_model_info(model_config, checkpoint)
            
            # Verify table was created and populated
            mock_table_class.assert_called_once_with(title="Model Information")
            
            # Check that add_row was called for various parameters
            assert mock_table.add_row.call_count >= 8  # At least 8 rows of info
            
            # Verify console.print was called
            assert self.mock_console.print.call_count >= 2  # Table + empty line
    
    def test_show_model_info_no_rich(self):
        """Test model information display without Rich (should do nothing)."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli_no_rich = InteractiveCLI()
            
            # Should complete without error and without doing anything
            cli_no_rich._show_model_info({}, {})
    
    def test_show_model_info_parameter_count(self):
        """Test parameter counting in model info display."""
        model_config = {"d_model": 128}
        checkpoint = {}
        
        with patch('cs336_basics.ui.cli.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            self.cli._show_model_info(model_config, checkpoint)
            
            # Check that parameter count was included
            param_calls = [call for call in mock_table.add_row.call_args_list 
                          if "Parameters" in str(call)]
            assert len(param_calls) > 0


class TestInteractiveCLIWelcome:
    """Test CLI welcome message display."""
    
    def test_show_welcome_with_rich(self):
        """Test welcome message with Rich interface."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', True):
            with patch('cs336_basics.ui.cli.Console') as mock_console_class:
                with patch('cs336_basics.ui.cli.Panel') as mock_panel_class:
                    with patch('cs336_basics.ui.cli.Markdown') as mock_markdown_class:
                        mock_console = Mock()
                        mock_console_class.return_value = mock_console
                        mock_panel = Mock()
                        mock_panel_class.return_value = mock_panel
                        mock_markdown = Mock()
                        mock_markdown_class.return_value = mock_markdown
                        
                        cli = InteractiveCLI(use_rich=True)
                        cli._show_welcome()
                        
                        # Verify Rich components were used
                        mock_markdown_class.assert_called_once()
                        mock_panel_class.assert_called_once()
                        mock_console.print.assert_called_once_with(mock_panel)
    
    def test_show_welcome_no_rich(self):
        """Test welcome message without Rich interface."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI(use_rich=False)
            
            with patch('builtins.print') as mock_print:
                cli._show_welcome()
                
                # Verify basic print statements were made
                assert mock_print.call_count >= 3
                
                # Check for key welcome message components
                call_args = [str(call) for call in mock_print.call_args_list]
                welcome_found = any("CS336 Transformer Chat Interface" in arg for arg in call_args)
                assert welcome_found


class TestInteractiveCLIMocking:
    """Test CLI with comprehensive mocking for external dependencies."""
    
    def test_cli_without_rich_import(self):
        """Test CLI behavior when Rich is not available."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI()
            
            assert cli.use_rich is False
            assert cli.console is None
            
            # Test that message functions work without Rich
            with patch('builtins.print'):
                cli._print_info("Test")
                cli._print_success("Test")
                cli._print_warning("Test")
                cli._print_error("Test")
                cli._show_welcome()
    
    def test_cli_import_error_handling(self):
        """Test CLI behavior when imports fail."""
        # This test verifies graceful handling of import errors
        # Since the actual import errors are handled at module level,
        # we mainly test that the CLI can be created without those dependencies
        
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli = InteractiveCLI()
            assert cli is not None
            assert cli.use_rich is False
    
    def test_cli_device_detection_edge_cases(self):
        """Test device detection edge cases."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            # Test when both CUDA and MPS are unavailable
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.backends.mps.is_available', return_value=False):
                    cli = InteractiveCLI()
                    assert cli.device.type == "cpu"
            
            # Test when MPS module doesn't exist
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.backends', spec=[]):  # Remove mps from backends
                    cli = InteractiveCLI()
                    assert cli.device.type == "cpu"


class TestInteractiveCLIIntegration:
    """Integration tests for CLI components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            self.cli = InteractiveCLI()
    
    def test_full_initialization_chain(self):
        """Test complete initialization chain."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "model": {"d_model": 256},
                "generation": {"temperature": 0.5}
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                cli = InteractiveCLI(
                    config_path=config_path,
                    device="cpu"
                )
                
                # Verify all initialization steps completed
                assert cli.config == config
                assert cli.device == torch.device("cpu")
                assert cli.generation_config is not None
                assert cli.session_stats is not None
                assert cli.conversation_history == []
        finally:
            Path(config_path).unlink()
    
    def test_message_display_consistency(self):
        """Test message display consistency across Rich/no-Rich modes."""
        test_message = "Test message"
        
        # Test without Rich
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            cli_no_rich = InteractiveCLI()
            
            with patch('builtins.print') as mock_print:
                cli_no_rich._print_info(test_message)
                cli_no_rich._print_success(test_message)
                cli_no_rich._print_warning(test_message)
                cli_no_rich._print_error(test_message)
                
                # All message types should have been printed
                assert mock_print.call_count == 4
        
        # Test with Rich
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', True):
            with patch('cs336_basics.ui.cli.Console') as mock_console_class:
                mock_console = Mock()
                mock_console_class.return_value = mock_console
                
                cli_rich = InteractiveCLI(use_rich=True)
                cli_rich._print_info(test_message)
                cli_rich._print_success(test_message)
                cli_rich._print_warning(test_message)
                cli_rich._print_error(test_message)
                
                # All message types should have been printed to console
                assert mock_console.print.call_count == 4
    
    def test_configuration_loading_robustness(self):
        """Test configuration loading under various conditions."""
        test_configs = [
            {},  # Empty config
            {"model": {"d_model": 512}},  # Partial config
            {"invalid_key": "invalid_value"},  # Invalid keys
            {"model": {"d_model": 512}, "generation": {"temperature": 0.7}}  # Full config
        ]
        
        for config in test_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
                    cli = InteractiveCLI(config_path=config_path)
                    assert cli.config == config
            finally:
                Path(config_path).unlink()
    
    def test_error_resilience(self):
        """Test CLI resilience to various error conditions."""
        with patch('cs336_basics.ui.cli.RICH_AVAILABLE', False):
            # Test with various invalid paths
            invalid_paths = [
                None,
                "",
                "/nonexistent/path",
                "/dev/null/impossible"
            ]
            
            for invalid_path in invalid_paths:
                # Should not crash
                cli = InteractiveCLI(
                    model_path=invalid_path,
                    config_path=invalid_path
                )
                assert cli is not None