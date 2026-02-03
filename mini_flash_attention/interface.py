from typing import List, Tuple
import torch
import mini_flash_attention._C as _C  # type: ignore[import-not-found]


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Mini Flash Attention forward pass.
    
    Args:
        q: Query tensor (batch, seqlen_q, heads, head_dim)
        k: Key tensor (batch, seqlen_k, heads_k, head_dim)
        v: Value tensor (batch, seqlen_k, heads_k, head_dim)
        causal: Whether to apply causal mask (default: False)
    
    Returns:
        Output tensor (batch, seqlen_q, heads, head_dim)
    
    Example:
        >>> q = torch.randn(1, 4096, 8, 128, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(1, 4096, 8, 128, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(1, 4096, 8, 128, device='cuda', dtype=torch.float16)
        >>> out = flash_attn_func(q, k, v, causal=True)
    """
    result: List[torch.Tensor] = _C.mini_flash_attention_forward(q, k, v, causal)
    return result[0]


