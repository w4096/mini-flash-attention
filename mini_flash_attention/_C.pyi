import torch
from typing import Tuple, Optional, Union

def mini_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int
) -> torch.Tensor:
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
    ...
    
def mini_flash_attention_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: Optional[Union[int, torch.Tensor]],
    block_table: Optional[torch.Tensor],
    causal: bool,
    num_splits: int,
) -> torch.Tensor:
    ...