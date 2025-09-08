import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import argparse

from cs336_basics.interactive import (
    find_model_files,
    validate_files,
    show_welcome,
    create_main_parser,
    launch_cli,
    launch_web,
    launch_both,
    main
)


class TestFindModelFiles:
    """Test model file discovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_find_model_files_empty_directory(self):
        """Test file discovery in empty directory."""
        discovered = find_model_files(self.temp_dir)
        
        expected_keys = ["model", "vocab", "merges", "config"]
        for key in expected_keys:
            assert key in discovered
            assert discovered[key] is None
    
    def test_find_model_files_with_model(self):
        """Test discovery of model checkpoint files."""
        # Create model files with different patterns
        model_files = ["model.pt", "checkpoint_best.pt", "other.pth"]
        
        for model_file in model_files:
            (self.base_path / model_file).touch()
        
        discovered = find_model_files(self.temp_dir)
        
        # Should find one of the model files
        assert discovered["model"] is not None
        assert any(mf in discovered["model"] for mf in model_files)
    
    def test_find_model_files_with_tokenizer(self):
        """Test discovery of tokenizer files."""
        # Create tokenizer files
        vocab_file = self.base_path / "vocab.json"
        merges_file = self.base_path / "merges.txt"
        
        vocab_file.touch()
        merges_file.touch()
        
        discovered = find_model_files(self.temp_dir)
        
        assert discovered["vocab"] is not None
        assert "vocab.json" in discovered["vocab"]
        assert discovered["merges"] is not None
        assert "merges.txt" in discovered["merges"]
    
    def test_find_model_files_with_config(self):
        """Test discovery of configuration files."""
        # Create config files
        config_files = ["config.yaml", "model_m4.yml", "settings.yaml"]
        
        for config_file in config_files:
            (self.base_path / config_file).touch()
        
        discovered = find_model_files(self.temp_dir)
        
        assert discovered["config"] is not None
        # Should prefer "m4" in name
        if "model_m4.yml" in str(discovered["config"]):
            assert "model_m4.yml" in discovered["config"]
    
    def test_find_model_files_nested_directories(self):
        """Test discovery in nested directories."""
        # Create nested directory structure
        nested_dir = self.base_path / "checkpoints" / "models"
        nested_dir.mkdir(parents=True)
        
        model_file = nested_dir / "model.pt"
        model_file.touch()
        
        discovered = find_model_files(self.temp_dir)
        
        assert discovered["model"] is not None
        assert "model.pt" in discovered["model"]
    
    def test_find_model_files_timestamp_ordering(self):
        """Test that newer files are preferred."""
        import time
        
        # Create older file
        old_model = self.base_path / "old_model.pt"
        old_model.touch()
        
        time.sleep(0.1)  # Ensure different timestamps
        
        # Create newer file
        new_model = self.base_path / "new_model.pt"
        new_model.touch()
        
        discovered = find_model_files(self.temp_dir)
        
        # Should prefer the newer file
        assert "new_model.pt" in discovered["model"]
    
    def test_find_model_files_all_types(self):
        """Test discovery of all file types."""
        # Create all types of files
        (self.base_path / "model.pt").touch()
        (self.base_path / "vocab.json").touch()
        (self.base_path / "merges.txt").touch()
        (self.base_path / "config.yaml").touch()
        
        discovered = find_model_files(self.temp_dir)
        
        assert all(discovered[key] is not None for key in discovered.keys())
        
        # Verify file names
        assert "model.pt" in discovered["model"]
        assert "vocab.json" in discovered["vocab"]
        assert "merges.txt" in discovered["merges"]
        assert "config.yaml" in discovered["config"]


class TestValidateFiles:
    """Test file validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        
        # Create test files
        self.model_path = self.base_path / "model.pt"
        self.vocab_path = self.base_path / "vocab.json"
        self.merges_path = self.base_path / "merges.txt"
        
        self.model_path.touch()
        self.vocab_path.touch()
        self.merges_path.touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_files_all_exist(self):
        """Test validation when all files exist."""
        result = validate_files(
            str(self.model_path),
            str(self.vocab_path),
            str(self.merges_path)
        )
        assert result == True
    
    def test_validate_files_model_missing(self):
        """Test validation when model file is missing."""
        result = validate_files(
            "/nonexistent/model.pt",
            str(self.vocab_path),
            str(self.merges_path)
        )
        assert result == False
    
    def test_validate_files_vocab_missing(self):
        """Test validation when vocab file is missing."""
        result = validate_files(
            str(self.model_path),
            "/nonexistent/vocab.json",
            str(self.merges_path)
        )
        assert result == False
    
    def test_validate_files_merges_missing(self):
        """Test validation when merges file is missing."""
        result = validate_files(
            str(self.model_path),
            str(self.vocab_path),
            "/nonexistent/merges.txt"
        )
        assert result == False
    
    def test_validate_files_none_paths(self):
        """Test validation with None paths."""
        result = validate_files(None, None, None)
        assert result == False
        
        result = validate_files(str(self.model_path), None, str(self.merges_path))
        assert result == False
    
    def test_validate_files_empty_paths(self):
        """Test validation with empty string paths."""
        result = validate_files("", "", "")
        assert result == False
        
        result = validate_files(str(self.model_path), "", str(self.merges_path))
        assert result == False


class TestShowWelcome:
    """Test welcome message display."""
    
    def test_show_welcome(self):
        """Test welcome message output."""
        with patch('builtins.print') as mock_print:
            show_welcome()
            
            # Should print welcome message
            mock_print.assert_called_once()
            
            # Check that the printed message contains key elements
            printed_message = mock_print.call_args[0][0]
            assert "CS336 Transformer Chat Interface" in printed_message
            assert "🤖" in printed_message
            assert "✨" in printed_message or "sampling strategies" in printed_message


class TestCreateMainParser:
    """Test argument parser creation."""
    
    def test_create_main_parser_basic(self):
        """Test basic parser creation."""
        parser = create_main_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "CS336 Transformer Interactive Interface"
    
    def test_parser_interface_choices(self):
        """Test interface argument choices."""
        parser = create_main_parser()
        
        # Test valid choices
        valid_choices = ['cli', 'web', 'both']
        for choice in valid_choices:
            args = parser.parse_args([choice])
            assert args.interface == choice
    
    def test_parser_interface_invalid(self):
        """Test invalid interface choice."""
        parser = create_main_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['invalid_interface'])
    
    def test_parser_optional_arguments(self):
        """Test optional arguments parsing."""
        parser = create_main_parser()
        
        args = parser.parse_args([
            'cli', 
            '--model', 'model.pt',
            '--vocab', 'vocab.json',
            '--merges', 'merges.txt',
            '--config', 'config.yaml',
            '--device', 'cuda'
        ])
        
        assert args.interface == 'cli'
        assert args.model == 'model.pt'
        assert args.vocab == 'vocab.json'
        assert args.merges == 'merges.txt'
        assert args.config == 'config.yaml'
        assert args.device == 'cuda'
    
    def test_parser_device_choices(self):
        """Test device argument choices."""
        parser = create_main_parser()
        
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        for device in valid_devices:
            args = parser.parse_args(['cli', '--device', device])
            assert args.device == device
    
    def test_parser_device_invalid(self):
        """Test invalid device choice."""
        parser = create_main_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['cli', '--device', 'invalid_device'])
    
    def test_parser_auto_discover_flag(self):
        """Test auto-discover flag."""
        parser = create_main_parser()
        
        # Without flag
        args = parser.parse_args(['cli'])
        assert args.auto_discover == False
        
        # With flag
        args = parser.parse_args(['cli', '--auto-discover'])
        assert args.auto_discover == True
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = create_main_parser()
        args = parser.parse_args(['cli'])
        
        assert args.device == 'auto'
        assert args.auto_discover == False
        assert args.model is None
        assert args.vocab is None
        assert args.merges is None
        assert args.config is None


class TestLaunchFunctions:
    """Test interface launching functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.mock_args = Mock()
        self.mock_args.model = "model.pt"
        self.mock_args.vocab = "vocab.json"
        self.mock_args.merges = "merges.txt"
        self.mock_args.config = "config.yaml"
        self.mock_args.device = "cpu"
        self.mock_args.host = "localhost"
        self.mock_args.port = 5000
        self.mock_args.debug = False
    
    @patch('cs336_basics.interactive.InteractiveCLI')
    def test_launch_cli_success(self, mock_cli_class):
        """Test successful CLI launch."""
        mock_cli = Mock()
        mock_cli.run.return_value = None
        mock_cli_class.return_value = mock_cli
        
        result = launch_cli(self.mock_args)
        
        assert result == 0
        mock_cli_class.assert_called_once()
        mock_cli.run.assert_called_once()
    
    @patch('cs336_basics.interactive.InteractiveCLI')
    def test_launch_cli_keyboard_interrupt(self, mock_cli_class):
        """Test CLI launch with keyboard interrupt."""
        mock_cli = Mock()
        mock_cli.run.side_effect = KeyboardInterrupt()
        mock_cli_class.return_value = mock_cli
        
        result = launch_cli(self.mock_args)
        
        assert result == 0  # Should handle gracefully
    
    @patch('cs336_basics.interactive.InteractiveCLI')
    def test_launch_cli_exception(self, mock_cli_class):
        """Test CLI launch with exception."""
        mock_cli_class.side_effect = RuntimeError("CLI failed")
        
        result = launch_cli(self.mock_args)
        
        assert result == 1  # Should return error code
    
    @patch('cs336_basics.interactive.create_web_app')
    def test_launch_web_success(self, mock_create_web_app):
        """Test successful web launch."""
        mock_app = Mock()
        mock_create_web_app.return_value = mock_app
        
        result = launch_web(self.mock_args)
        
        assert result == 0
        mock_create_web_app.assert_called_once()
        mock_app.run.assert_called_once_with(
            host=self.mock_args.host,
            port=self.mock_args.port,
            debug=self.mock_args.debug
        )
    
    @patch('cs336_basics.interactive.create_web_app')
    def test_launch_web_keyboard_interrupt(self, mock_create_web_app):
        """Test web launch with keyboard interrupt."""
        mock_app = Mock()
        mock_app.run.side_effect = KeyboardInterrupt()
        mock_create_web_app.return_value = mock_app
        
        result = launch_web(self.mock_args)
        
        assert result == 0  # Should handle gracefully
    
    @patch('cs336_basics.interactive.create_web_app')
    def test_launch_web_exception(self, mock_create_web_app):
        """Test web launch with exception."""
        mock_create_web_app.side_effect = RuntimeError("Web app failed")
        
        result = launch_web(self.mock_args)
        
        assert result == 1  # Should return error code
    
    @patch('cs336_basics.interactive.launch_cli')
    @patch('cs336_basics.interactive.launch_web')
    @patch('cs336_basics.interactive.threading.Thread')
    @patch('cs336_basics.interactive.time.sleep')
    @patch('builtins.print')
    def test_launch_both(self, mock_print, mock_sleep, mock_thread, mock_launch_web, mock_launch_cli):
        """Test launching both interfaces."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        mock_launch_cli.return_value = 0
        
        result = launch_both(self.mock_args)
        
        assert result == 0
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        mock_sleep.assert_called_once_with(2)  # Wait for web interface
        mock_launch_cli.assert_called_once_with(self.mock_args)
        
        # Check that informational messages were printed
        assert mock_print.call_count >= 2


class TestMainFunction:
    """Test main function integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        
        # Create required files
        (self.base_path / "model.pt").touch()
        (self.base_path / "vocab.json").touch()
        (self.base_path / "merges.txt").touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cs336_basics.interactive.launch_cli')
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_cli_interface(self, mock_print, mock_show_welcome, mock_launch_cli):
        """Test main function with CLI interface."""
        mock_launch_cli.return_value = 0
        
        with patch('sys.argv', ['interactive.py', 'cli', '--auto-discover']):
            with patch('cs336_basics.interactive.find_model_files') as mock_find_files:
                mock_find_files.return_value = {
                    "model": str(self.base_path / "model.pt"),
                    "vocab": str(self.base_path / "vocab.json"),
                    "merges": str(self.base_path / "merges.txt"),
                    "config": None
                }
                
                result = main()
                
                assert result == 0
                mock_show_welcome.assert_called_once()
                mock_find_files.assert_called_once()
                mock_launch_cli.assert_called_once()
    
    @patch('cs336_basics.interactive.launch_web')
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_web_interface(self, mock_print, mock_show_welcome, mock_launch_web):
        """Test main function with web interface."""
        mock_launch_web.return_value = 0
        
        test_args = [
            'interactive.py', 'web',
            '--model', str(self.base_path / "model.pt"),
            '--vocab', str(self.base_path / "vocab.json"),
            '--merges', str(self.base_path / "merges.txt")
        ]
        
        with patch('sys.argv', test_args):
            result = main()
            
            assert result == 0
            mock_show_welcome.assert_called_once()
            mock_launch_web.assert_called_once()
    
    @patch('cs336_basics.interactive.launch_both')
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_both_interfaces(self, mock_print, mock_show_welcome, mock_launch_both):
        """Test main function with both interfaces."""
        mock_launch_both.return_value = 0
        
        test_args = [
            'interactive.py', 'both',
            '--model', str(self.base_path / "model.pt"),
            '--vocab', str(self.base_path / "vocab.json"),
            '--merges', str(self.base_path / "merges.txt")
        ]
        
        with patch('sys.argv', test_args):
            result = main()
            
            assert result == 0
            mock_show_welcome.assert_called_once()
            mock_launch_both.assert_called_once()
    
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_file_validation_failure(self, mock_print, mock_show_welcome):
        """Test main function with file validation failure."""
        test_args = [
            'interactive.py', 'cli',
            '--model', '/nonexistent/model.pt',
            '--vocab', '/nonexistent/vocab.json',
            '--merges', '/nonexistent/merges.txt'
        ]
        
        with patch('sys.argv', test_args):
            result = main()
            
            assert result == 1  # Should return error code
            mock_show_welcome.assert_called_once()
    
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_auto_discover_no_files(self, mock_print, mock_show_welcome):
        """Test main function with auto-discover but no files found."""
        with patch('sys.argv', ['interactive.py', 'cli', '--auto-discover']):
            with patch('cs336_basics.interactive.find_model_files') as mock_find_files:
                mock_find_files.return_value = {
                    "model": None,
                    "vocab": None,
                    "merges": None,
                    "config": None
                }
                
                result = main()
                
                assert result == 1  # Should fail validation
                mock_show_welcome.assert_called_once()
                mock_find_files.assert_called_once()
    
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_unknown_interface(self, mock_print, mock_show_welcome):
        """Test main function with unknown interface type."""
        # This test simulates an unknown interface that somehow passes argparse
        # (which shouldn't happen with proper validation, but tests edge case)
        
        test_args = [
            'interactive.py', 'cli',
            '--model', str(self.base_path / "model.pt"),
            '--vocab', str(self.base_path / "vocab.json"),
            '--merges', str(self.base_path / "merges.txt")
        ]
        
        with patch('sys.argv', test_args):
            with patch('cs336_basics.interactive.create_main_parser') as mock_parser:
                mock_args = Mock()
                mock_args.auto_discover = False
                mock_args.model = str(self.base_path / "model.pt")
                mock_args.vocab = str(self.base_path / "vocab.json")
                mock_args.merges = str(self.base_path / "merges.txt")
                mock_args.config = None
                mock_args.device = 'auto'
                mock_args.interface = 'unknown'  # Unknown interface type
                
                mock_parser_instance = Mock()
                mock_parser_instance.parse_args.return_value = mock_args
                mock_parser.return_value = mock_parser_instance
                
                result = main()
                
                assert result == 1  # Should return error code


class TestInteractiveIntegration:
    """Integration tests for interactive module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_file_discovery_chain(self):
        """Test complete file discovery and validation chain."""
        # Create all types of files in nested structure
        models_dir = self.base_path / "checkpoints"
        models_dir.mkdir(parents=True)
        
        tokenizer_dir = self.base_path / "tokenizer"
        tokenizer_dir.mkdir(parents=True)
        
        (models_dir / "best_model.pt").touch()
        (tokenizer_dir / "vocab.json").touch()
        (tokenizer_dir / "merges.txt").touch()
        (self.base_path / "config_m4.yaml").touch()
        
        # Discover files
        discovered = find_model_files(str(self.base_path))
        
        # All files should be found
        assert all(discovered[key] is not None for key in discovered.keys())
        
        # Validate discovered files
        validation_result = validate_files(
            discovered["model"],
            discovered["vocab"],
            discovered["merges"]
        )
        
        assert validation_result == True
    
    def test_argument_parsing_integration(self):
        """Test argument parsing with realistic scenarios."""
        parser = create_main_parser()
        
        # Test realistic command line scenarios
        test_cases = [
            ['cli', '--auto-discover'],
            ['web', '--model', 'model.pt', '--vocab', 'vocab.json', '--merges', 'merges.txt'],
            ['both', '--device', 'cuda', '--config', 'config.yaml'],
            ['cli', '--model', 'model.pt', '--vocab', 'vocab.json', '--merges', 'merges.txt', '--device', 'cpu']
        ]
        
        for args in test_cases:
            parsed_args = parser.parse_args(args)
            
            assert parsed_args.interface in ['cli', 'web', 'both']
            assert parsed_args.device in ['auto', 'cpu', 'cuda', 'mps']
            
            # Validate argument consistency
            if hasattr(parsed_args, 'auto_discover'):
                assert isinstance(parsed_args.auto_discover, bool)
    
    @patch('cs336_basics.interactive.launch_cli')
    @patch('cs336_basics.interactive.show_welcome')
    @patch('builtins.print')
    def test_main_function_error_resilience(self, mock_print, mock_show_welcome, mock_launch_cli):
        """Test main function resilience to various errors."""
        # Create minimal valid files
        (self.base_path / "model.pt").touch()
        (self.base_path / "vocab.json").touch()
        (self.base_path / "merges.txt").touch()
        
        # Test with launch function failure
        mock_launch_cli.return_value = 1  # Error code
        
        test_args = [
            'interactive.py', 'cli',
            '--model', str(self.base_path / "model.pt"),
            '--vocab', str(self.base_path / "vocab.json"),
            '--merges', str(self.base_path / "merges.txt")
        ]
        
        with patch('sys.argv', test_args):
            result = main()
            
            assert result == 1  # Should propagate error code
            mock_show_welcome.assert_called_once()
            mock_launch_cli.assert_called_once()
    
    def test_configuration_display_integration(self):
        """Test that configuration is properly displayed."""
        # Create files
        model_file = self.base_path / "test_model.pt"
        vocab_file = self.base_path / "test_vocab.json"
        merges_file = self.base_path / "test_merges.txt"
        
        model_file.touch()
        vocab_file.touch()
        merges_file.touch()
        
        with patch('sys.argv', [
            'interactive.py', 'cli',
            '--model', str(model_file),
            '--vocab', str(vocab_file),
            '--merges', str(merges_file),
            '--device', 'cpu'
        ]):
            with patch('cs336_basics.interactive.launch_cli', return_value=0):
                with patch('cs336_basics.interactive.show_welcome'):
                    with patch('builtins.print') as mock_print:
                        main()
                        
                        # Check that configuration was printed
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        config_printed = any("Configuration:" in call for call in print_calls)
                        assert config_printed
                        
                        # Check that file paths were printed
                        model_printed = any("test_model.pt" in call for call in print_calls)
                        assert model_printed