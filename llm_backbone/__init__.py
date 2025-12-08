import importlib.metadata

try:
    __version__ = importlib.metadata.version("llm_backbone")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .Transformers.modules.tools import *
