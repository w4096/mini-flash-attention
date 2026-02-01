from typing import Optional, Tuple

import torch
import mini_flash_attention._C as mini_flash_attn_gpu


@torch.library.custom_op("mini_flash_attn::_flash_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _flash_attn_varlen_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
) -> Tuple[torch.Tensor]:
    return mini_flash_attn_gpu.flash_attention_v2(q, k, v)
    # out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(
    #     q,
    #     k,
    #     v,
    #     None,
    #     cu_seqlens_q,
    #     cu_seqlens_k,
    #     seqused_k,
    #     leftpad_k,
    #     block_table,
    #     alibi_slopes,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     dropout_p,
    #     softmax_scale,
    #     zero_tensors,
    #     causal,
    #     window_size_left,
    #     window_size_right,
    #     softcap,
    #     return_softmax,
    #     None,
    # )
    # # if out.isnan().any() or softmax_lse.isnan().any():
    # #     breakpoint()
    # return out, softmax_lse, S_dmask, rng_state



wrapped_flash_attn_forward = torch.ops.mini_flash_attn._flash_attn_varlen_forward

def flash_attn_varlen_func(
        q,
        k,
        v,
        # cu_seqlens_q,
        # cu_seqlens_k,
        # max_seqlen_q,
        # max_seqlen_k,
        # dropout_p=0.0,
        # softmax_scale=None,
        # causal=False,
        # window_size=(-1, -1),  # -1 means infinite context window
        # softcap=0.0, # 0.0 means deactivated
        # alibi_slopes=None,
        # deterministic=False,
        # return_attn_probs=False,
        # block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return wrapped_flash_attn_forward(
        q,
        k,
        v,
        # cu_seqlens_q,
        # cu_seqlens_k,
        # max_seqlen_q,
        # max_seqlen_k,
        # dropout_p,
        # softmax_scale,
        # causal,
        # window_size,
        # softcap,
        # alibi_slopes,
        # deterministic,
        # return_attn_probs,
        # block_table,
    )

