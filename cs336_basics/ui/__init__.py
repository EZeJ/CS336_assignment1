"""
User Interface modules for interactive text generation.

This module provides both command-line and web-based interfaces for
interacting with transformer models for text generation.
"""

from .cli import InteractiveCLI, create_cli_parser
from .web_app import create_web_app

__all__ = [
    "InteractiveCLI",
    "create_cli_parser", 
    "create_web_app",
]