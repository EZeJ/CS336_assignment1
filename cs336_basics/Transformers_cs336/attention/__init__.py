import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .attention_fn import *
from .multihead_self_attention import *
