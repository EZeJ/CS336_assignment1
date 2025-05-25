import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from . import modules
from . import attention
from . import transformer
