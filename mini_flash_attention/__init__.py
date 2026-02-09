"""Mini Flash Attention - A minimal Flash Attention v2 implementation."""

__version__ = "0.1.0"

from mini_flash_attention.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
