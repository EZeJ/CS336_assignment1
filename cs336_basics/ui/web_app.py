"""
Web Interface for Text Generation

Flask-based web application providing a browser interface for chatting
with transformer models with real-time generation and configuration.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    from flask import Flask, render_template, request, jsonify, session, redirect, url_for
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. Web interface not available.")

try:
    from ..inference import TextGenerator, GenerationConfig
    from ..optimization import compile_transformer
    import cs336_basics.Transformers_cs336 as my_tf
    import cs336_basics.Tokenizers.BPE_tokenizer as bpe
    import torch
    import yaml
except ImportError as e:
    print(f"Error importing modules: {e}")

logger = logging.getLogger(__name__)


class WebChatApp:
    """Web-based chat application for transformer models."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_vocab_path: Optional[str] = None,
        tokenizer_merges_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize web chat application.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_vocab_path: Path to tokenizer vocabulary
            tokenizer_merges_path: Path to tokenizer merges
            config_path: Path to configuration file
            device: Device for generation
        """
        
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / "templates"),
                        static_folder=str(Path(__file__).parent / "static"))
        self.app.secret_key = "cs336-transformer-chat-key"  # Change in production
        
        # Enable CORS for API endpoints
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})
        
        # Configuration
        self.config = self._load_config(config_path) if config_path else {}
        self.device = self._setup_device(device)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Session storage for conversations
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.generation_configs: Dict[str, GenerationConfig] = {}
        
        # App statistics
        self.app_stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "total_generation_time": 0.0,
            "app_start_time": time.time(),
        }
        
        # Load model if paths provided
        if model_path:
            self._load_model_and_tokenizer(model_path, tokenizer_vocab_path, tokenizer_merges_path)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_device(self, device: Optional[str]):
        """Setup computation device."""
        if device and device != "auto":
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_model_and_tokenizer(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        merges_path: Optional[str] = None,
    ):
        """Load model and tokenizer."""
        try:
            logger.info("Loading model and tokenizer for web interface...")
            
            # Load tokenizer
            if vocab_path and merges_path:
                self.tokenizer = bpe.Tokenizer.from_files(
                    vocab_filepath=vocab_path,
                    merges_filepath=merges_path,
                    special_tokens=["<|endoftext|>"]
                )
                logger.info("Tokenizer loaded")
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get("model_config", self.config.get("model", {}))
            
            self.model = my_tf.transformer.Transformer(
                d_model=model_config.get("d_model", 512),
                num_heads=model_config.get("num_heads", 8),
                d_ff=model_config.get("d_ff", 1024),
                vocab_size=model_config.get("vocab_size", 10000),
                context_length=model_config.get("context_length", 256),
                num_layers=model_config.get("num_layers", 4),
                max_seq_len=model_config.get("context_length", 256),
                theta=model_config.get("rope_theta", 10000.0),
                device=self.device,
            )
            
            # Load weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize model
            try:
                self.model = compile_transformer(self.model, disable_on_unsupported=True)
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
            
            # Create generator
            if self.tokenizer:
                self.generator = TextGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    use_cache=True,
                )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main chat interface."""
            return render_template('chat.html')
        
        @self.app.route('/api/new_conversation', methods=['POST'])
        def new_conversation():
            """Start a new conversation."""
            conversation_id = str(uuid.uuid4())
            session['conversation_id'] = conversation_id
            
            self.conversations[conversation_id] = []
            self.generation_configs[conversation_id] = GenerationConfig(
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                sampling_strategy="top_p",
                do_sample=True,
            )
            
            self.app_stats["total_conversations"] += 1
            
            return jsonify({
                "conversation_id": conversation_id,
                "status": "success"
            })
        
        @self.app.route('/api/send_message', methods=['POST'])
        def send_message():
            """Send message and get response."""
            data = request.get_json()
            
            if not data or 'message' not in data:
                return jsonify({"error": "No message provided"}), 400
            
            conversation_id = session.get('conversation_id')
            if not conversation_id or conversation_id not in self.conversations:
                return jsonify({"error": "No active conversation"}), 400
            
            if not self.generator:
                return jsonify({"error": "Model not loaded"}), 500
            
            user_message = data['message'].strip()
            if not user_message:
                return jsonify({"error": "Empty message"}), 400
            
            try:
                # Generate response
                start_time = time.time()
                conversation = self.conversations[conversation_id]
                
                # Build messages context
                messages = []
                for msg in conversation[-5:]:  # Last 5 messages for context
                    messages.append(msg)
                messages.append({"role": "user", "content": user_message})
                
                # Get generation config
                gen_config = self.generation_configs[conversation_id]
                
                # Generate
                response = self.generator.chat(messages, **gen_config.to_dict())
                generation_time = time.time() - start_time
                
                # Save to conversation
                user_msg = {"role": "user", "content": user_message, "timestamp": time.time()}
                assistant_msg = {
                    "role": "assistant", 
                    "content": response, 
                    "timestamp": time.time(),
                    "generation_time": generation_time,
                }
                
                conversation.extend([user_msg, assistant_msg])
                
                # Update stats
                self.app_stats["total_messages"] += 1
                self.app_stats["total_generation_time"] += generation_time
                
                return jsonify({
                    "response": response,
                    "generation_time": generation_time,
                    "status": "success"
                })
            
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/conversation_history')
        def conversation_history():
            """Get conversation history."""
            conversation_id = session.get('conversation_id')
            if not conversation_id or conversation_id not in self.conversations:
                return jsonify({"messages": []})
            
            return jsonify({"messages": self.conversations[conversation_id]})
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def config():
            """Get or update generation configuration."""
            conversation_id = session.get('conversation_id')
            
            if request.method == 'GET':
                if conversation_id and conversation_id in self.generation_configs:
                    config = self.generation_configs[conversation_id]
                else:
                    config = GenerationConfig()
                
                return jsonify(config.to_dict())
            
            else:  # POST
                if not conversation_id or conversation_id not in self.generation_configs:
                    return jsonify({"error": "No active conversation"}), 400
                
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No configuration provided"}), 400
                
                # Update configuration
                config = self.generation_configs[conversation_id]
                
                for key, value in data.items():
                    if hasattr(config, key):
                        try:
                            setattr(config, key, value)
                        except Exception as e:
                            logger.warning(f"Failed to set {key}={value}: {e}")
                
                return jsonify({"status": "success", "config": config.to_dict()})
        
        @self.app.route('/api/stats')
        def stats():
            """Get application statistics."""
            uptime = time.time() - self.app_stats["app_start_time"]
            
            stats = {
                **self.app_stats,
                "uptime_seconds": uptime,
                "active_conversations": len(self.conversations),
                "device": str(self.device),
                "model_loaded": self.generator is not None,
            }
            
            # Add generation stats if available
            if self.generator:
                gen_stats = self.generator.get_generation_stats()
                stats.update(gen_stats)
            
            return jsonify(stats)
        
        @self.app.route('/api/export_conversation')
        def export_conversation():
            """Export current conversation."""
            conversation_id = session.get('conversation_id')
            if not conversation_id or conversation_id not in self.conversations:
                return jsonify({"error": "No active conversation"}), 400
            
            export_data = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": self.conversations[conversation_id],
                "config": self.generation_configs[conversation_id].to_dict(),
            }
            
            return jsonify(export_data)
        
        @self.app.route('/api/clear_conversation', methods=['POST'])
        def clear_conversation():
            """Clear current conversation."""
            conversation_id = session.get('conversation_id')
            if conversation_id and conversation_id in self.conversations:
                self.conversations[conversation_id] = []
                return jsonify({"status": "success"})
            
            return jsonify({"error": "No active conversation"}), 400
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the web application."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Cannot start web interface.")
            return
        
        if not self.generator:
            print("Warning: Model not loaded. Web interface may not work properly.")
        
        print(f"Starting web interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_web_app(
    model_path: Optional[str] = None,
    tokenizer_vocab_path: Optional[str] = None,
    tokenizer_merges_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> WebChatApp:
    """
    Create a web chat application.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_vocab_path: Path to tokenizer vocabulary
        tokenizer_merges_path: Path to tokenizer merges
        config_path: Path to configuration file
        device: Device for generation
    
    Returns:
        Configured WebChatApp instance
    """
    
    return WebChatApp(
        model_path=model_path,
        tokenizer_vocab_path=tokenizer_vocab_path,
        tokenizer_merges_path=tokenizer_merges_path,
        config_path=config_path,
        device=device,
    )