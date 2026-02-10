from typing import List, Tuple, Optional, Union
import torch
import mini_flash_attention._C as _C  # type: ignore[import-not-found]


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Mini Flash Attention forward pass.
    
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in K, V.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    
    Examples:
        >>> q = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> out = flash_attn_func(q, k, v, causal=True)
    """
    return _C.mini_flash_attention_forward(
        q, k, v, None, causal, window_size[0], window_size[1]
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Mini Flash Attention forward pass for variable-length sequences (continuous batching).
    
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in K, V.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    
    Arguments:
        q: (total_q, nheads, headdim), where total_q = sum of all sequence lengths in the batch.
        k: (total_k, nheads_k, headdim), where total_k = sum of all sequence lengths in the batch.
        v: (total_k, nheads_k, headdim), where total_k = sum of all sequence lengths in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key/value sequence length in the batch.
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    
    Return:
        out: (total_q, nheads, headdim).
    
    Examples:
        >>> # Variable-length batch with different sequence lengths
        >>> seqlens = [128, 256, 512]
        >>> total_q = sum(seqlens)
        >>> q = torch.randn(total_q, 8, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(total_q, 8, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(total_q, 8, 64, device='cuda', dtype=torch.float16)
        >>> cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device='cuda').cumsum(0, dtype=torch.int32)
        >>> max_seqlen = max(seqlens)
        >>> out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
    """
    return _C.mini_flash_attention_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size[0], window_size[1]
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    block_table: Optional[torch.Tensor] = None,
    causal=False,
    num_splits=0,
):
    """
    Mini Flash Attention with KV cache for auto-regressive decoding.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k_cache: (num_blocks, block_size, nheads_k, headdim)
        v_cache: (num_blocks, block_size, nheads_k, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim). If provided, used to prefill the cache.
        v: (batch_size, seqlen_k, nheads_k, headdim). If provided, used to prefill the cache.
        block_table: (batch_size, max_blocks_per_seq), dtype torch.int32. If provided,
           uses the block table to index into the KV cache for non-contiguous attention.
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        num_splits: int. Number of splits for large sequence lengths to reduce memory usage.
    """
    assert q.size(1) == 1, "flash_attn_with_kvcache currently only supports seqlen_q=1 for decoding"
    
    return _C.mini_flash_attention_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens,
        block_table,
        causal,
        num_splits
    )
    
    