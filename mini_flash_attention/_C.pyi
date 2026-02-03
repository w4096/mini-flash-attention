"""Type stubs for mini_flash_attention C++ extension module."""

import torch
from typing import List

def mini_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Flash Attention v2 forward pass.
    
    Args:
        q: Query tensor of shape (batch, seqlen_q, heads, head_dim)
        k: Key tensor of shape (batch, seqlen_k, heads_k, head_dim)
        v: Value tensor of shape (batch, seqlen_k, heads_k, head_dim)
    
    Returns:
        List containing output tensor of shape (batch, seqlen_q, heads, head_dim)
    """
    ...
