import torch
from typing import Tuple, Optional

def mini_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int
) -> torch.Tensor:
    """
    Flash Attention forward pass.
    
    Args:
        q: Query tensor of shape (batch, seqlen_q, heads, head_dim)
        k: Key tensor of shape (batch, seqlen_k, heads_k, head_dim)
        v: Value tensor of shape (batch, seqlen_k, heads_k, head_dim)
        causal: Whether to apply causal mask (default: False)
    
    Returns:
        List containing output tensor of shape (batch, seqlen_q, heads, head_dim)
    """
    ...



def mini_flash_attention_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int
) -> torch.Tensor:
    """
    Mini Flash Attention forward pass for variable-length sequences.
    
    Args:
        q: Query tensor (total_seqlen_q, heads, head_dim)
        k: Key tensor (total_seqlen_k, heads_k, head_dim)
        v: Value tensor (total_seqlen_k, heads_k, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries (#seq + 1)
        cu_seqlens_k: Cumulative sequence lengths for keys/values (#seq + 1)
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys/values
        causal: Whether to apply causal mask (default: False)
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    """
    ...