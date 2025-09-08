import pytest
import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import torch

# Skip tests if Flask is not available
pytest_plugins = []
try:
    from cs336_basics.ui.web_app import WebChatApp, FLASK_AVAILABLE
    from cs336_basics.inference.generator import GenerationConfig
    if not FLASK_AVAILABLE:
        pytestmark = pytest.mark.skip("Flask not available")
except ImportError:
    pytestmark = pytest.mark.skip("Flask dependencies not available")

from ..fixtures.test_models import create_mock_model, MockTokenizer


class TestWebChatAppInitialization:
    """Test WebChatApp initialization and setup."""
    
    def test_init_minimal(self):
        """Test web app initialization with minimal parameters."""
        with patch('cs336_basics.ui.web_app.torch.cuda.is_available', return_value=False):
            with patch('cs336_basics.ui.web_app.torch.backends.mps.is_available', return_value=False):
                app = WebChatApp()
                
                assert app.app is not None
                assert app.device.type == "cpu"
                assert app.model is None
                assert app.tokenizer is None
                assert app.generator is None
                assert app.conversations == {}
                assert app.generation_configs == {}
                assert "total_conversations" in app.app_stats
    
    def test_init_with_config(self):
        """Test initialization with configuration file."""
        config_data = {
            "model": {"d_model": 256, "num_heads": 4},
            "generation": {"temperature": 0.7}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with patch('cs336_basics.ui.web_app.torch.cuda.is_available', return_value=False):
                with patch('cs336_basics.ui.web_app.torch.backends.mps.is_available', return_value=False):
                    app = WebChatApp(config_path=config_path)
                    assert app.config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_init_invalid_config(self):
        """Test initialization with invalid config file."""
        with patch('cs336_basics.ui.web_app.torch.cuda.is_available', return_value=False):
            with patch('cs336_basics.ui.web_app.torch.backends.mps.is_available', return_value=False):
                app = WebChatApp(config_path="/nonexistent/config.yaml")
                assert app.config == {}
    
    def test_device_setup_cuda_available(self):
        """Test device setup when CUDA is available."""
        with patch('cs336_basics.ui.web_app.torch.cuda.is_available', return_value=True):
            app = WebChatApp(device="auto")
            assert app.device.type == "cuda"
    
    def test_device_setup_mps_available(self):
        """Test device setup when MPS is available."""
        with patch('cs336_basics.ui.web_app.torch.cuda.is_available', return_value=False):
            with patch('cs336_basics.ui.web_app.torch.backends.mps.is_available', return_value=True):
                app = WebChatApp(device="auto")
                assert app.device.type == "mps"
    
    def test_device_setup_explicit_device(self):
        """Test device setup with explicit device specification."""
        app = WebChatApp(device="cpu")
        assert app.device == torch.device("cpu")
    
    def test_flask_app_configuration(self):
        """Test Flask app configuration."""
        app = WebChatApp()
        
        assert app.app.secret_key == "cs336-transformer-chat-key"
        assert app.app.template_folder is not None
        assert app.app.static_folder is not None
    
    def test_app_stats_initialization(self):
        """Test application statistics initialization."""
        app = WebChatApp()
        
        expected_keys = [
            "total_conversations", "total_messages", 
            "total_generation_time", "app_start_time"
        ]
        
        for key in expected_keys:
            assert key in app.app_stats
        
        assert app.app_stats["total_conversations"] == 0
        assert app.app_stats["total_messages"] == 0
        assert app.app_stats["total_generation_time"] == 0.0
        assert app.app_stats["app_start_time"] > 0


class TestWebChatAppConfiguration:
    """Test web app configuration loading."""
    
    def test_load_config_valid_yaml(self):
        """Test loading valid YAML configuration."""
        app = WebChatApp()
        config_data = {"model": {"d_model": 512}, "device": "cuda"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = app._load_config(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_file(self):
        """Test loading invalid configuration file."""
        app = WebChatApp()
        config = app._load_config("/nonexistent/file.yaml")
        assert config == {}
    
    def test_load_config_malformed_yaml(self):
        """Test loading malformed YAML file."""
        app = WebChatApp()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            config = app._load_config(config_path)
            assert config == {}
        finally:
            Path(config_path).unlink()


class TestWebChatAppModelLoading:
    """Test web app model and tokenizer loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = WebChatApp()
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
            }
        }
        torch.save(checkpoint, self.model_path)
        
        # Create mock tokenizer files
        vocab = {str(i): i for i in range(100)}
        with open(self.vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        with open(self.merges_path, 'w') as f:
            f.write("a b\\nc d\\n")
        
        # Mock the external dependencies
        with patch('cs336_basics.ui.web_app.bpe.Tokenizer') as mock_tokenizer_class:
            with patch('cs336_basics.ui.web_app.my_tf.transformer.Transformer') as mock_model_class:
                with patch('cs336_basics.ui.web_app.TextGenerator') as mock_generator_class:
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.from_files.return_value = mock_tokenizer
                    
                    mock_model_instance = Mock()
                    mock_model_class.return_value = mock_model_instance
                    
                    mock_generator = Mock()
                    mock_generator_class.return_value = mock_generator
                    
                    # Load model and tokenizer
                    self.app._load_model_and_tokenizer(
                        str(self.model_path),
                        str(self.vocab_path),
                        str(self.merges_path)
                    )
                    
                    # Verify loading was attempted
                    mock_tokenizer_class.from_files.assert_called_once()
                    mock_model_class.assert_called_once()
                    mock_generator_class.assert_called_once()
    
    def test_load_model_missing_files(self):
        """Test model loading with missing files."""
        # Should handle gracefully without crashing
        self.app._load_model_and_tokenizer(
            "/nonexistent/model.pt",
            "/nonexistent/vocab.json",
            "/nonexistent/merges.txt"
        )
        
        assert self.app.model is None
        assert self.app.tokenizer is None
        assert self.app.generator is None


class TestWebChatAppRoutes:
    """Test web app routes and API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = WebChatApp()
        self.app.app.testing = True
        self.client = self.app.app.test_client()
        
        # Mock the generator
        self.mock_generator = Mock()
        self.app.generator = self.mock_generator
    
    def test_index_route(self):
        """Test index route."""
        response = self.client.get('/')
        assert response.status_code in [200, 404]  # 404 if template not found
    
    def test_start_conversation_api(self):
        """Test start conversation API endpoint."""
        response = self.client.post('/api/start_conversation')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "conversation_id" in data
        assert data["status"] == "success"
        
        # Check that conversation was created
        conversation_id = data["conversation_id"]
        assert conversation_id in self.app.conversations
        assert conversation_id in self.app.generation_configs
    
    def test_send_message_api_no_conversation(self):
        """Test send message API without active conversation."""
        response = self.client.post('/api/send_message', 
                                   json={"message": "Hello"})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_send_message_api_no_message(self):
        """Test send message API without message."""
        # Start conversation first
        self.client.post('/api/start_conversation')
        
        response = self.client.post('/api/send_message', json={})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_send_message_api_empty_message(self):
        """Test send message API with empty message."""
        # Start conversation first
        self.client.post('/api/start_conversation')
        
        response = self.client.post('/api/send_message', 
                                   json={"message": ""})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_send_message_api_no_generator(self):
        """Test send message API without loaded model."""
        # Remove generator
        self.app.generator = None
        
        # Start conversation first
        self.client.post('/api/start_conversation')
        
        response = self.client.post('/api/send_message', 
                                   json={"message": "Hello"})
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_send_message_api_success(self):
        """Test successful message sending."""
        # Mock generator response
        self.mock_generator.chat.return_value = "Hello! How can I help you?"
        
        # Start conversation first
        response = self.client.post('/api/start_conversation')
        conversation_data = json.loads(response.data)
        conversation_id = conversation_data["conversation_id"]
        
        # Send message
        response = self.client.post('/api/send_message', 
                                   json={"message": "Hello"})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "response" in data
        assert "generation_time" in data
        
        # Check conversation history was updated
        conversation = self.app.conversations[conversation_id]
        assert len(conversation) == 2  # User message + assistant response
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
    
    def test_conversation_history_api_no_conversation(self):
        """Test conversation history API without active conversation."""
        response = self.client.get('/api/conversation_history')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["messages"] == []
    
    def test_conversation_history_api_with_conversation(self):
        """Test conversation history API with active conversation."""
        # Start conversation and send message
        self.mock_generator.chat.return_value = "Test response"
        
        self.client.post('/api/start_conversation')
        self.client.post('/api/send_message', json={"message": "Test"})
        
        # Get history
        response = self.client.get('/api/conversation_history')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data["messages"]) == 2  # User + assistant messages
    
    def test_config_api_get_default(self):
        """Test getting default configuration."""
        response = self.client.get('/api/config')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        # Should return default GenerationConfig values
        assert "temperature" in data
        assert "max_new_tokens" in data
        assert "do_sample" in data
    
    def test_config_api_get_with_conversation(self):
        """Test getting configuration with active conversation."""
        # Start conversation
        self.client.post('/api/start_conversation')
        
        response = self.client.get('/api/config')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, dict)
    
    def test_config_api_post_no_conversation(self):
        """Test updating configuration without active conversation."""
        response = self.client.post('/api/config', 
                                   json={"temperature": 0.5})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_config_api_post_success(self):
        """Test successful configuration update."""
        # Start conversation first
        response = self.client.post('/api/start_conversation')
        conversation_data = json.loads(response.data)
        conversation_id = conversation_data["conversation_id"]
        
        # Update config
        new_config = {"temperature": 0.5, "max_new_tokens": 150}
        response = self.client.post('/api/config', json=new_config)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["status"] == "success"
        
        # Verify config was updated
        config = self.app.generation_configs[conversation_id]
        assert config.temperature == 0.5
        assert config.max_new_tokens == 150
    
    def test_stats_api(self):
        """Test statistics API endpoint."""
        # Mock generator stats
        self.mock_generator.get_generation_stats.return_value = {
            "total_tokens_generated": 100,
            "avg_tokens_per_second": 50.0
        }
        
        response = self.client.get('/api/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "app_stats" in data
        assert "generator_stats" in data
        
        # Check app stats structure
        app_stats = data["app_stats"]
        assert "total_conversations" in app_stats
        assert "total_messages" in app_stats
    
    def test_reset_conversation_api(self):
        """Test reset conversation API endpoint."""
        # Start conversation and send message
        self.mock_generator.chat.return_value = "Test response"
        
        response = self.client.post('/api/start_conversation')
        conversation_data = json.loads(response.data)
        conversation_id = conversation_data["conversation_id"]
        
        self.client.post('/api/send_message', json={"message": "Test"})
        
        # Verify conversation has messages
        assert len(self.app.conversations[conversation_id]) > 0
        
        # Reset conversation
        response = self.client.post('/api/reset_conversation')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["status"] == "success"
        
        # Verify conversation was cleared
        assert len(self.app.conversations[conversation_id]) == 0


class TestWebChatAppStatistics:
    """Test web app statistics tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = WebChatApp()
        self.app.app.testing = True
        self.client = self.app.app.test_client()
        
        # Mock the generator
        self.mock_generator = Mock()
        self.mock_generator.chat.return_value = "Test response"
        self.mock_generator.get_generation_stats.return_value = {
            "total_tokens_generated": 100
        }
        self.app.generator = self.mock_generator
    
    def test_conversation_count_tracking(self):
        """Test that conversation count is tracked correctly."""
        initial_count = self.app.app_stats["total_conversations"]
        
        # Start multiple conversations
        self.client.post('/api/start_conversation')
        self.client.post('/api/start_conversation')
        self.client.post('/api/start_conversation')
        
        assert self.app.app_stats["total_conversations"] == initial_count + 3
    
    def test_message_count_tracking(self):
        """Test that message count is tracked correctly."""
        initial_count = self.app.app_stats["total_messages"]
        
        # Start conversation and send messages
        self.client.post('/api/start_conversation')
        self.client.post('/api/send_message', json={"message": "Message 1"})
        self.client.post('/api/send_message', json={"message": "Message 2"})
        
        assert self.app.app_stats["total_messages"] == initial_count + 2
    
    def test_generation_time_tracking(self):
        """Test that generation time is tracked correctly."""
        initial_time = self.app.app_stats["total_generation_time"]
        
        # Send message (which should update generation time)
        self.client.post('/api/start_conversation')
        
        with patch('time.time', side_effect=[0.0, 1.5]):  # Mock 1.5 second generation
            self.client.post('/api/send_message', json={"message": "Test"})
        
        assert self.app.app_stats["total_generation_time"] > initial_time


class TestWebChatAppErrorHandling:
    """Test web app error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = WebChatApp()
        self.app.app.testing = True
        self.client = self.app.app.test_client()
        
        # Mock the generator to raise exceptions
        self.mock_generator = Mock()
        self.app.generator = self.mock_generator
    
    def test_generation_error_handling(self):
        """Test handling of generation errors."""
        # Make generator raise an exception
        self.mock_generator.chat.side_effect = RuntimeError("Generation failed")
        
        # Start conversation
        self.client.post('/api/start_conversation')
        
        # Send message (should handle error gracefully)
        response = self.client.post('/api/send_message', 
                                   json={"message": "Test"})
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert "error" in data
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON requests."""
        response = self.client.post('/api/send_message',
                                   data="invalid json",
                                   content_type='application/json')
        # Should handle gracefully (specific status code depends on Flask version)
        assert response.status_code in [400, 500]
    
    def test_invalid_config_update(self):
        """Test handling of invalid configuration updates."""
        # Start conversation
        self.client.post('/api/start_conversation')
        
        # Try to update with invalid config values
        invalid_config = {"temperature": -1.0, "max_new_tokens": -10}
        response = self.client.post('/api/config', json=invalid_config)
        
        # Should still process (validation happens in GenerationConfig)
        assert response.status_code == 200


class TestWebChatAppIntegration:
    """Integration tests for web app components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = WebChatApp()
        self.app.app.testing = True
        self.client = self.app.app.test_client()
        
        # Mock the generator with realistic behavior
        self.mock_generator = Mock()
        self.mock_generator.chat.return_value = "Hello! How can I help you?"
        self.mock_generator.get_generation_stats.return_value = {
            "total_tokens_generated": 100,
            "avg_tokens_per_second": 25.0
        }
        self.app.generator = self.mock_generator
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow."""
        # Start conversation
        response = self.client.post('/api/start_conversation')
        assert response.status_code == 200
        conversation_data = json.loads(response.data)
        conversation_id = conversation_data["conversation_id"]
        
        # Send multiple messages
        messages = ["Hello", "How are you?", "Tell me a joke"]
        
        for msg in messages:
            response = self.client.post('/api/send_message', 
                                       json={"message": msg})
            assert response.status_code == 200
        
        # Check conversation history
        response = self.client.get('/api/conversation_history')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        # Should have 6 messages (3 user + 3 assistant)
        assert len(data["messages"]) == 6
        
        # Verify conversation structure
        conversation = self.app.conversations[conversation_id]
        for i in range(0, len(conversation), 2):
            assert conversation[i]["role"] == "user"
            assert conversation[i + 1]["role"] == "assistant"
    
    def test_configuration_persistence_across_requests(self):
        """Test that configuration persists across requests."""
        # Start conversation
        response = self.client.post('/api/start_conversation')
        conversation_data = json.loads(response.data)
        conversation_id = conversation_data["conversation_id"]
        
        # Update configuration
        new_config = {"temperature": 0.7, "max_new_tokens": 200}
        response = self.client.post('/api/config', json=new_config)
        assert response.status_code == 200
        
        # Send message (should use updated config)
        response = self.client.post('/api/send_message', 
                                   json={"message": "Test"})
        assert response.status_code == 200
        
        # Verify generator was called with updated config
        call_args = self.mock_generator.chat.call_args
        assert call_args is not None
        # The config should be passed as keyword arguments
        called_config = call_args[1]  # kwargs
        assert called_config["temperature"] == 0.7
        assert called_config["max_new_tokens"] == 200
    
    def test_session_isolation(self):
        """Test that different sessions have isolated conversations."""
        # This is a simplified test since we can't easily simulate different sessions
        # in the same test client, but we can test the data structure isolation
        
        # Start two conversations (simulating different sessions)
        response1 = self.client.post('/api/start_conversation')
        conversation1_id = json.loads(response1.data)["conversation_id"]
        
        response2 = self.client.post('/api/start_conversation')
        conversation2_id = json.loads(response2.data)["conversation_id"]
        
        # Verify they have different IDs
        assert conversation1_id != conversation2_id
        
        # Verify they exist as separate conversations
        assert conversation1_id in self.app.conversations
        assert conversation2_id in self.app.conversations
        assert conversation1_id in self.app.generation_configs
        assert conversation2_id in self.app.generation_configs
    
    def test_app_statistics_consistency(self):
        """Test that application statistics are consistent."""
        initial_stats = dict(self.app.app_stats)
        
        # Perform various operations
        self.client.post('/api/start_conversation')
        self.client.post('/api/start_conversation')
        self.client.post('/api/send_message', json={"message": "Test 1"})
        self.client.post('/api/send_message', json={"message": "Test 2"})
        
        # Check stats were updated correctly
        final_stats = self.app.app_stats
        
        assert final_stats["total_conversations"] == initial_stats["total_conversations"] + 2
        assert final_stats["total_messages"] == initial_stats["total_messages"] + 2
        assert final_stats["total_generation_time"] > initial_stats["total_generation_time"]
    
    def test_error_recovery(self):
        """Test that the app recovers gracefully from errors."""
        # Start conversation
        self.client.post('/api/start_conversation')
        
        # Cause an error in generation
        self.mock_generator.chat.side_effect = RuntimeError("Test error")
        
        response = self.client.post('/api/send_message', 
                                   json={"message": "Test"})
        assert response.status_code == 500
        
        # Reset generator and try again
        self.mock_generator.chat.side_effect = None
        self.mock_generator.chat.return_value = "Recovered response"
        
        response = self.client.post('/api/send_message', 
                                   json={"message": "Test again"})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data["response"] == "Recovered response"