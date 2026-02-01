
#include <iostream>
#include "api.h"
#include <torch/types.h>


torch::Tensor self_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto s = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(q.size(-1));
    auto attn = torch::softmax(s, -1);
    auto out = torch::matmul(attn, v);
    return out;
}

 int main0() {
     const int batch_size = 1;
     const int heads = 1;
     const int seq_len = 64;
     const int head_dim = 128;

     torch::Tensor q = torch::arange(seq_len * head_dim, torch::kHalf).reshape({1, seq_len, 1, head_dim}).cuda() / (seq_len * head_dim) + 1.0;
     torch::Tensor k = torch::arange(seq_len * head_dim, torch::kHalf).reshape({1, seq_len, 1, head_dim}).cuda() / (seq_len * head_dim) + 2.0;
     torch::Tensor v = torch::ones({batch_size, seq_len, heads, head_dim}, torch::kHalf).cuda() / 32;

     q = q.transpose(1, 2);
     k = k.transpose(1, 2);
     v = v.transpose(1, 2);

     std::cout << v.sizes() << std::endl;

     auto out = mfa::flash_attention_v2(q, k, v);
     std::cout << "Output shape: " << out[0] << std::endl;

    auto ref = self_attention(q, k, v);
    std::cout << "Reference shape: " << ref.sizes() << std::endl;
    std::cout << "Reference shape: " << ref.view({seq_len, head_dim}) << std::endl;
    return 0;
}

