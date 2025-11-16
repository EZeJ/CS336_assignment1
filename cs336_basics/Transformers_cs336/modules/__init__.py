import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .linear import Linear
from .embedding import Embedding
from .RMSNorm import RMSNorm
from .SwiGLU import SwiGLU
from .SwiGLU import *
from .RoPE import RotaryPositionalEmbedding
from .loss import *
from .optimizer import (
    SGD,
    AdamW,
    get_lr_cosine_schedule,
    get_gradient_clipping,
)

from .tools import *
