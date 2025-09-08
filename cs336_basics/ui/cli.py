"""
Interactive Command Line Interface for Text Generation

Provides a rich terminal interface for chatting with transformer models
with features like conversation history, parameter adjustment, and export.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import torch
import yaml

# Rich terminal interface
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not installed. Using basic terminal interface.")

# Import our modules
try:
    from ..inference import TextGenerator, GenerationConfig, create_sampler
    from ..optimization import compile_transformer
    import cs336_basics.Transformers_cs336 as my_tf
    import cs336_basics.Tokenizers.BPE_tokenizer as bpe
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class InteractiveCLI:
    """Interactive command-line interface for text generation."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_vocab_path: Optional[str] = None,
        tokenizer_merges_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        use_rich: bool = True,
    ):
        """
        Initialize the CLI interface.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_vocab_path: Path to tokenizer vocabulary
            tokenizer_merges_path: Path to tokenizer merges
            config_path: Path to model configuration
            device: Device to use for generation
            use_rich: Whether to use rich terminal interface
        """
        
        self.use_rich = use_rich and RICH_AVAILABLE
        
        if self.use_rich:
            self.console = Console()
        else:
            self.console = None
        
        # Configuration
        self.config = self._load_config(config_path) if config_path else {}
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Chat state
        self.conversation_history = []
        self.generation_config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            sampling_strategy="top_p",
        )
        
        # Stats
        self.session_stats = {
            "messages_generated": 0,
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "session_start_time": time.time(),
        }
        
        # Load model and tokenizer
        if model_path:
            self._load_model_and_tokenizer(model_path, tokenizer_vocab_path, tokenizer_merges_path)
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        return torch.device(device)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._print_info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self._print_error(f"Failed to load config: {e}")
            return {}
    
    def _load_model_and_tokenizer(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        merges_path: Optional[str] = None,
    ):
        """Load model and tokenizer from files."""
        
        self._print_info("Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            if vocab_path and merges_path:
                self.tokenizer = bpe.Tokenizer.from_files(
                    vocab_filepath=vocab_path,
                    merges_filepath=merges_path,
                    special_tokens=["<|endoftext|>"]
                )
                self._print_success("Tokenizer loaded successfully")
            else:
                self._print_warning("No tokenizer paths provided - text generation may not work")
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration from checkpoint or config file
            model_config = checkpoint.get("model_config", self.config.get("model", {}))
            
            if not model_config:
                self._print_error("No model configuration found")
                return
            
            # Create model
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
            
            # Load model weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Apply optimizations
            try:
                self.model = compile_transformer(self.model, mode="default", disable_on_unsupported=True)
            except Exception as e:
                self._print_warning(f"Model compilation failed: {e}")
            
            # Create text generator
            if self.tokenizer:
                self.generator = TextGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    use_cache=True,
                )
            
            self._print_success(f"Model loaded successfully on {self.device}")
            
            # Show model info
            if self.use_rich:
                self._show_model_info(model_config, checkpoint)
        
        except Exception as e:
            self._print_error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_model_info(self, model_config: Dict[str, Any], checkpoint: Dict[str, Any]):
        """Display model information."""
        if not self.use_rich:
            return
        
        table = Table(title="Model Information")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        # Model architecture
        table.add_row("Model Dimension", str(model_config.get("d_model", "Unknown")))
        table.add_row("Attention Heads", str(model_config.get("num_heads", "Unknown")))
        table.add_row("Feed Forward Dim", str(model_config.get("d_ff", "Unknown")))
        table.add_row("Number of Layers", str(model_config.get("num_layers", "Unknown")))
        table.add_row("Vocabulary Size", str(model_config.get("vocab_size", "Unknown")))
        table.add_row("Context Length", str(model_config.get("context_length", "Unknown")))
        
        # Training info
        if "iteration" in checkpoint:
            table.add_row("Training Iteration", str(checkpoint["iteration"]))
        if "epoch" in checkpoint:
            table.add_row("Training Epoch", str(checkpoint["epoch"]))
        
        # Device info
        table.add_row("Device", str(self.device))
        
        # Model size
        if self.model:
            num_params = sum(p.numel() for p in self.model.parameters())
            table.add_row("Parameters", f"{num_params:,}")
        
        self.console.print(table)
        self.console.print()
    
    def _print_info(self, message: str):
        """Print info message."""
        if self.use_rich:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            print(f"INFO: {message}")
    
    def _print_success(self, message: str):
        """Print success message."""
        if self.use_rich:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            print(f"SUCCESS: {message}")
    
    def _print_warning(self, message: str):
        """Print warning message."""
        if self.use_rich:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            print(f"WARNING: {message}")
    
    def _print_error(self, message: str):
        """Print error message."""
        if self.use_rich:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            print(f"ERROR: {message}")
    
    def _show_welcome(self):
        """Show welcome message."""
        if self.use_rich:
            welcome_text = """
# 🤖 CS336 Transformer Chat Interface

Welcome to the interactive chat interface! You can:
- Chat with the transformer model
- Adjust generation parameters with `/config`
- View conversation history with `/history`
- Export conversations with `/export`
- Get help with `/help`
- Quit with `/quit`

Type your message and press Enter to start chatting!
            """
            self.console.print(Panel(
                Markdown(welcome_text),
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("=== CS336 Transformer Chat Interface ===")
            print("Type '/help' for available commands")
            print("Type '/quit' to exit")
            print()
    
    def _show_help(self):
        """Show help information."""
        if self.use_rich:
            help_table = Table(title="Available Commands")
            help_table.add_column("Command", style="cyan")
            help_table.add_column("Description", style="white")
            
            commands = [
                ("/help", "Show this help message"),
                ("/config", "Show/modify generation parameters"),
                ("/history", "Show conversation history"),
                ("/export", "Export conversation to file"),
                ("/stats", "Show generation statistics"),
                ("/clear", "Clear conversation history"),
                ("/reset", "Reset model cache and conversation"),
                ("/quit", "Exit the chat interface"),
            ]
            
            for cmd, desc in commands:
                help_table.add_row(cmd, desc)
            
            self.console.print(help_table)
        else:
            print("Available Commands:")
            print("  /help - Show this help")
            print("  /config - Configuration")  
            print("  /history - Show history")
            print("  /export - Export conversation")
            print("  /stats - Show statistics")
            print("  /clear - Clear history")
            print("  /reset - Reset session")
            print("  /quit - Exit")
    
    def _show_config(self):
        """Show current configuration."""
        if self.use_rich:
            config_table = Table(title="Generation Configuration")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_dict = self.generation_config.to_dict()
            for key, value in config_dict.items():
                if value is not None:
                    config_table.add_row(key, str(value))
            
            self.console.print(config_table)
        else:
            print("Current Configuration:")
            config_dict = self.generation_config.to_dict()
            for key, value in config_dict.items():
                if value is not None:
                    print(f"  {key}: {value}")
    
    def _modify_config(self):
        """Interactively modify configuration."""
        if self.use_rich:
            self.console.print("[yellow]Configuration Modification[/yellow]")
            
            # Temperature
            new_temp = Prompt.ask(
                "Temperature (0.1-2.0)",
                default=str(self.generation_config.temperature)
            )
            try:
                self.generation_config.temperature = float(new_temp)
            except ValueError:
                pass
            
            # Max new tokens
            new_tokens = Prompt.ask(
                "Max new tokens",
                default=str(self.generation_config.max_new_tokens)
            )
            try:
                self.generation_config.max_new_tokens = int(new_tokens)
            except ValueError:
                pass
            
            # Top-p
            new_top_p = Prompt.ask(
                "Top-p (0.0-1.0, or 'none')",
                default=str(self.generation_config.top_p) if self.generation_config.top_p else "none"
            )
            if new_top_p.lower() == "none":
                self.generation_config.top_p = None
            else:
                try:
                    self.generation_config.top_p = float(new_top_p)
                except ValueError:
                    pass
            
            self.console.print("[green]Configuration updated![/green]")
        else:
            print("Configuration modification not available in basic mode")
    
    def _show_stats(self):
        """Show generation statistics."""
        if self.generator:
            gen_stats = self.generator.get_generation_stats()
        else:
            gen_stats = {}
        
        session_time = time.time() - self.session_stats["session_start_time"]
        
        if self.use_rich:
            stats_table = Table(title="Session Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Session Time", f"{session_time:.1f}s")
            stats_table.add_row("Messages Generated", str(self.session_stats["messages_generated"]))
            stats_table.add_row("Total Tokens", str(gen_stats.get("total_tokens_generated", 0)))
            stats_table.add_row("Avg Tokens/sec", f"{gen_stats.get('avg_tokens_per_second', 0):.1f}")
            
            if "cache_memory" in gen_stats:
                cache_mb = gen_stats["cache_memory"]["total_memory_mb"]
                stats_table.add_row("Cache Memory", f"{cache_mb:.1f} MB")
            
            self.console.print(stats_table)
        else:
            print("Session Statistics:")
            print(f"  Session Time: {session_time:.1f}s")
            print(f"  Messages: {self.session_stats['messages_generated']}")
            print(f"  Tokens: {gen_stats.get('total_tokens_generated', 0)}")
    
    def _export_conversation(self):
        """Export conversation to file."""
        if not self.conversation_history:
            self._print_warning("No conversation to export")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        
        export_data = {
            "timestamp": timestamp,
            "conversation": self.conversation_history,
            "generation_config": self.generation_config.to_dict(),
            "session_stats": self.session_stats,
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self._print_success(f"Conversation exported to {filename}")
        except Exception as e:
            self._print_error(f"Failed to export conversation: {e}")
    
    def _generate_response(self, user_input: str) -> str:
        """Generate response to user input."""
        if not self.generator:
            return "Error: Model not loaded properly"
        
        start_time = time.time()
        
        # Build conversation context
        messages = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages for context
            messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        
        try:
            if self.use_rich:
                with self.console.status("[bold green]Generating response..."):
                    response = self.generator.chat(messages, **self.generation_config.to_dict())
            else:
                print("Generating...")
                response = self.generator.chat(messages, **self.generation_config.to_dict())
        
            generation_time = time.time() - start_time
            
            # Update stats
            self.session_stats["messages_generated"] += 1
            self.session_stats["total_generation_time"] += generation_time
            
            return response
        
        except Exception as e:
            self._print_error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def run(self):
        """Run the interactive CLI."""
        if not self.generator:
            self._print_error("Cannot start chat - model not loaded properly")
            return
        
        self._show_welcome()
        
        try:
            while True:
                # Get user input
                if self.use_rich:
                    user_input = Prompt.ask("[bold blue]You[/bold blue]")
                else:
                    user_input = input("You: ")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower().strip()
                    
                    if command == '/quit' or command == '/exit':
                        break
                    elif command == '/help':
                        self._show_help()
                    elif command == '/config':
                        self._show_config()
                        modify = Confirm.ask("Modify configuration?") if self.use_rich else False
                        if modify:
                            self._modify_config()
                    elif command == '/history':
                        self._show_conversation_history()
                    elif command == '/export':
                        self._export_conversation()
                    elif command == '/stats':
                        self._show_stats()
                    elif command == '/clear':
                        self.conversation_history.clear()
                        self._print_success("Conversation history cleared")
                    elif command == '/reset':
                        self.conversation_history.clear()
                        if self.generator:
                            self.generator.cache_manager.reset()
                        self._print_success("Session reset")
                    else:
                        self._print_error(f"Unknown command: {command}")
                    
                    continue
                
                # Generate response
                response = self._generate_response(user_input)
                
                # Display response
                if self.use_rich:
                    self.console.print(f"[bold green]Assistant[/bold green]: {response}")
                else:
                    print(f"Assistant: {response}")
                
                # Save to conversation history
                self.conversation_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                
                print()  # Add spacing
        
        except KeyboardInterrupt:
            if self.use_rich:
                self.console.print("\\n[yellow]Chat interrupted by user[/yellow]")
            else:
                print("\\nChat interrupted by user")
        
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
        
        finally:
            self._print_info("Thanks for chatting! Goodbye!")
    
    def _show_conversation_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            self._print_info("No conversation history")
            return
        
        if self.use_rich:
            for i, msg in enumerate(self.conversation_history):
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    self.console.print(f"[bold blue]You[/bold blue]: {content}")
                else:
                    self.console.print(f"[bold green]Assistant[/bold green]: {content}")
                
                if i < len(self.conversation_history) - 1:
                    self.console.print()
        else:
            for msg in self.conversation_history:
                role = msg["role"].title()
                content = msg["content"]
                print(f"{role}: {content}")
                print()


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for CS336 Transformer Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to model checkpoint file"
    )
    
    parser.add_argument(
        "--vocab", "-v",
        type=str,
        help="Path to tokenizer vocabulary file"
    )
    
    parser.add_argument(
        "--merges",
        type=str,
        help="Path to tokenizer merges file"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for generation"
    )
    
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich terminal interface"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup device
    device = None if args.device == "auto" else args.device
    
    # Create CLI
    cli = InteractiveCLI(
        model_path=args.model,
        tokenizer_vocab_path=args.vocab,
        tokenizer_merges_path=args.merges,
        config_path=args.config,
        device=device,
        use_rich=not args.no_rich,
    )
    
    # Run CLI
    cli.run()


if __name__ == "__main__":
    main()