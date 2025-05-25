import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .transformer_block import *
from .transformer_model import *