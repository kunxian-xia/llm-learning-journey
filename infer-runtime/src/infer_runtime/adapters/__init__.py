"""Optional model adapters."""

from .qwen25 import Qwen25Decoder
from .qwen3_coder import Qwen3CoderDecoder

__all__ = ["Qwen25Decoder", "Qwen3CoderDecoder"]
