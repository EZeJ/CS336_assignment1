import importlib.metadata

try:
    __version__ = importlib.metadata.version("llm_backbone")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import modules
from . import attention
from . import transformer
