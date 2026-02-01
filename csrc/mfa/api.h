#pragma once

#include <torch/types.h>

namespace mfa {


std::vector<at::Tensor> flash_attention_v2(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

std::vector<at::Tensor> flash_decoding_v2(
    const torch::Tensor& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor& k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                            // page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor& v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                            // page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor& cu_seqlens_q, // b+1
    const torch::Tensor& cu_seqlens_k, // b+1
    std::optional<torch::Tensor>&
        seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<torch::Tensor>& block_table_, // batch_size x max_num_blocks_per_seq
    int max_seqlen_q, int max_seqlen_k, float softmax_scale, bool zero_tensors, bool is_causal, int window_size_left,
    int window_size_right, float softcap, int num_splits, std::optional<at::Generator> gen_);

} // namespace mfa