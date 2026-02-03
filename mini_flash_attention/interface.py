from typing import Optional, Tuple

import torch
import mini_flash_attention._C as mini_flash_attn


@torch.library.custom_op("mini_flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, softmax_lse = mini_flash_attn.flash_attention_v2(q, k, v)
    return out, softmax_lse


wrapped_flash_attn_forward = torch.ops.mini_flash_attn._flash_attn_forward

def mini_flash_attn_func(
        q,
        k,
        v,
):
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return wrapped_flash_attn_forward(
        q,
        k,
        v,
    )

