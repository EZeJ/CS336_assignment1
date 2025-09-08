#!/usr/bin/env python3
"""
CS336 Transformer Interactive Interface

Main script for launching interactive interfaces to chat with transformer models.
Supports both command-line and web-based interfaces with model loading and configuration.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from .ui.cli import InteractiveCLI, create_cli_parser
    from .ui.web_app import create_web_app
    from .optimization import compile_transformer
    import torch
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def find_model_files(base_dir: str = ".") -> dict:
    """
    Auto-discover model and tokenizer files in the project directory.
    
    Args:
        base_dir: Base directory to search in
    
    Returns:
        Dictionary with discovered file paths
    """
    base_path = Path(base_dir)
    discovered = {
        "model": None,
        "vocab": None,
        "merges": None,
        "config": None,
    }
    
    # Look for model checkpoints
    model_patterns = ["*.pt", "*.pth", "model*.pt", "checkpoint*.pt"]
    for pattern in model_patterns:
        model_files = list(base_path.rglob(pattern))
        if model_files:
            # Prefer files with "best" or recent timestamps
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            discovered["model"] = str(model_files[0])
            logger.info(f"Found model: {discovered['model']}")
            break
    
    # Look for tokenizer files
    vocab_files = list(base_path.rglob("vocab*.json"))
    if vocab_files:
        discovered["vocab"] = str(vocab_files[0])
        logger.info(f"Found vocab: {discovered['vocab']}")
    
    merges_files = list(base_path.rglob("merges*.txt")) + list(base_path.rglob("merges*.json"))
    if merges_files:
        discovered["merges"] = str(merges_files[0])
        logger.info(f"Found merges: {discovered['merges']}")
    
    # Look for config files
    config_files = list(base_path.rglob("*.yaml")) + list(base_path.rglob("*.yml"))
    if config_files:
        # Prefer config files with "m4" or similar in name
        config_files.sort(key=lambda x: ("m4" in x.name.lower(), x.stat().st_mtime), reverse=True)
        discovered["config"] = str(config_files[0])
        logger.info(f"Found config: {discovered['config']}")
    
    return discovered


def validate_files(model_path: Optional[str], vocab_path: Optional[str], 
                  merges_path: Optional[str]) -> bool:
    """
    Validate that required files exist.
    
    Args:
        model_path: Path to model checkpoint
        vocab_path: Path to vocabulary file
        merges_path: Path to merges file
    
    Returns:
        True if files are valid
    """
    if not model_path or not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not vocab_path or not Path(vocab_path).exists():
        logger.error(f"Vocabulary file not found: {vocab_path}")
        return False
    
    if not merges_path or not Path(merges_path).exists():
        logger.error(f"Merges file not found: {merges_path}")
        return False
    
    return True


def show_welcome():
    """Show welcome message with ASCII art."""
    welcome_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🤖 CS336 Transformer Chat Interface 🤖          ║
    ║                                                              ║
    ║  A powerful interactive interface for chatting with your     ║
    ║  custom-trained transformer models using modern features:    ║
    ║                                                              ║
    ║  ✨ Advanced sampling strategies (Top-p, Top-k, Temperature) ║
    ║  🚀 KV-cache for fast generation                            ║
    ║  💾 Mixed precision training support                        ║
    ║  🌐 Both CLI and web interfaces available                   ║
    ║  📊 Real-time generation statistics                         ║
    ║  💬 Conversation history and export                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(welcome_text)


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="CS336 Transformer Interactive Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover model files and launch CLI
  python -m cs336_basics.interactive cli

  # Launch web interface with specific model
  python -m cs336_basics.interactive web --model checkpoints/model.pt --vocab vocab.json --merges merges.txt

  # Launch CLI with custom configuration
  python -m cs336_basics.interactive cli --model model.pt --config config.yaml --device cuda
        """
    )
    
    # Interface type
    parser.add_argument(
        'interface',
        choices=['cli', 'web', 'both'],
        help='Interface type to launch'
    )
    
    # Model files
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--vocab', '-v',
        type=str,
        help='Path to tokenizer vocabulary file'
    )
    
    parser.add_argument(
        '--merges',
        type=str,
        help='Path to tokenizer merges file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to model configuration YAML file'
    )
    
    # Runtime options
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for generation'
    )
    
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Automatically discover model files in current directory'
    )
    
    # Web interface options
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host for web interface (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web interface (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for web interface'
    )
    
    # CLI options
    parser.add_argument(
        '--no-rich',
        action='store_true',
        help='Disable rich terminal interface for CLI'
    )
    
    return parser


def launch_cli(args) -> int:
    """Launch the CLI interface."""
    logger.info("Starting CLI interface...")
    
    try:
        cli = InteractiveCLI(
            model_path=args.model,
            tokenizer_vocab_path=args.vocab,
            tokenizer_merges_path=args.merges,
            config_path=args.config,
            device=args.device if args.device != 'auto' else None,
            use_rich=not args.no_rich,
        )
        
        cli.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("CLI interface interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"CLI interface failed: {e}")
        return 1


def launch_web(args) -> int:
    """Launch the web interface."""
    logger.info(f"Starting web interface on http://{args.host}:{args.port}")
    
    try:
        app = create_web_app(
            model_path=args.model,
            tokenizer_vocab_path=args.vocab,
            tokenizer_merges_path=args.merges,
            config_path=args.config,
            device=args.device if args.device != 'auto' else None,
        )
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
        return 0
        
    except KeyboardInterrupt:
        logger.info("Web interface interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Web interface failed: {e}")
        return 1


def launch_both(args) -> int:
    """Launch both interfaces (web in background, CLI in foreground)."""
    import threading
    import time
    
    logger.info("Starting both CLI and web interfaces...")
    
    # Start web interface in a separate thread
    web_thread = threading.Thread(
        target=lambda: launch_web(args),
        daemon=True
    )
    web_thread.start()
    
    # Give web interface time to start
    time.sleep(2)
    
    print(f"\n🌐 Web interface available at: http://{args.host}:{args.port}")
    print("💻 CLI interface starting below...")
    print("-" * 60)
    
    # Start CLI interface in main thread
    return launch_cli(args)


def main():
    """Main entry point."""
    parser = create_main_parser()
    args = parser.parse_args()
    
    # Show welcome message
    show_welcome()
    
    # Auto-discover files if requested or if no files specified
    if args.auto_discover or not (args.model and args.vocab and args.merges):
        logger.info("Auto-discovering model files...")
        discovered = find_model_files()
        
        # Use discovered files if not specified
        args.model = args.model or discovered["model"]
        args.vocab = args.vocab or discovered["vocab"] 
        args.merges = args.merges or discovered["merges"]
        args.config = args.config or discovered["config"]
    
    # Validate files
    if not validate_files(args.model, args.vocab, args.merges):
        logger.error("Required files not found. Please check paths or use --auto-discover.")
        return 1
    
    # Show configuration
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Vocab: {args.vocab}")
    print(f"   Merges: {args.merges}")
    print(f"   Config: {args.config or 'None'}")
    print(f"   Device: {args.device}")
    print(f"   Interface: {args.interface}")
    
    if args.interface == 'web':
        print(f"   Web Host: {args.host}")
        print(f"   Web Port: {args.port}")
    
    print()
    
    # Launch appropriate interface
    if args.interface == 'cli':
        return launch_cli(args)
    elif args.interface == 'web':
        return launch_web(args)
    elif args.interface == 'both':
        return launch_both(args)
    else:
        logger.error(f"Unknown interface type: {args.interface}")
        return 1


if __name__ == "__main__":
    sys.exit(main())