#pragma once

#include <torch/types.h>

namespace mfa {

// clang-format off
at::Tensor flash_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor> out_,
    bool is_causal,
    int window_size_left,
    int window_size_right);


at::Tensor flash_attention_varlen_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    bool is_causal,
    int window_size_left,
    int window_size_right);


at::Tensor mha_fwd_kvcache(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const std::optional<torch::Tensor>& seqlens_k_,
    const std::optional<torch::Tensor>& block_table_,
    bool causal,
    int num_splits);

// clang-format on

} // namespace mfa