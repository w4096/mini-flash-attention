#include "api.h"
#include "flash.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <vector>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                                                                            \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace mfa {

void forward_params_init(ForwardParams& params,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    
    const size_t batch,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t heads,
    const size_t head_dim,
    const size_t kv_heads,

    void *cu_seqlens_q,
    void *cu_seqlens_k,

    int window_size_left,
    int window_size_right
) {
    
    
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();

    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.o_row_stride = o.stride(-3);

    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_head_stride = o.stride(-2);


    if (cu_seqlens_q == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = o.stride(0);
    }

    params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q);
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k);


    params.batch = batch;
    params.head_dim = head_dim;
    params.heads = heads;
    params.kv_heads = kv_heads;
    params.kv_group_size = heads / kv_heads;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;

    params.is_causal = window_size_left < 0 && window_size_right == 0;
    if (window_size_left < 0 && window_size_right >= 0) {
        window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0) {
        window_size_right = seqlen_k;
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;


    params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    params.softmax_scale_log2 = params.softmax_scale * M_LOG2E;
}

/**
 * Flash Attention v2 forward
 *
 * @param q  total_q x num_heads x head_size
 * @param k  total_k x num_heads_k x head_size
 * @param v  total_k x num_heads_k x head_size
 * @param is_causal  whether to apply causal mask
 * @return 
 */
at::Tensor flash_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor> out_,
    bool is_causal,
    int window_size_left,
    int window_size_right)
{

    torch::cuda::CUDAGuard guard(q.device());

    auto dtype = q.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_dim = sizes[3];

    const int seqlen_k = k.size(1);
    const int kv_num_heads = k.size(2);

    TORCH_CHECK(k.size(0) == batch, "batch size of q and k must be the same");
    TORCH_CHECK(v.size(0) == batch, "batch size of q and v must be the same");
    TORCH_CHECK(k.size(1) == seqlen_k, "sequence length of k must be the same as v");
    TORCH_CHECK(v.size(1) == seqlen_k, "sequence length of k must be the same as v");
    TORCH_CHECK(head_dim <= 256, "head dimension must be less than or equal to 256");
    TORCH_CHECK(num_heads % kv_num_heads == 0, "number of key/value heads must be divisible by number of query heads");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }


    CHECK_SHAPE(q, batch, seqlen_q, num_heads, head_dim);
    CHECK_SHAPE(k, batch, seqlen_k, kv_num_heads, head_dim);
    CHECK_SHAPE(v, batch, seqlen_k, kv_num_heads, head_dim);

    torch::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch, seqlen_q, num_heads, head_dim);
    } else {
        out = torch::empty_like(q);
    }


    ForwardParams params{};
    forward_params_init(params,
        q, k, v, out,
        batch, seqlen_q, seqlen_k, num_heads, head_dim, kv_num_heads,
        nullptr, nullptr,
        window_size_left, window_size_right
    );

    auto stream = torch::cuda::getCurrentCUDAStream();
    run_flash_attention_forward(params, stream);
    torch::cuda::stream_synchronize(stream);

    return out;
}



at::Tensor flash_attention_varlen_forward(
    const torch::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const torch::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b}
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    bool is_causal,
    int window_size_left,
    int window_size_right
){
    torch::cuda::CUDAGuard guard(q.device());

    auto dtype = q.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    torch::Tensor out = torch::empty_like(q);

    const auto sizes = q.sizes();
    const int batch = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_dim = sizes[2];
    const int kv_num_heads = k.size(1);
    std::cout << "batch: " << batch << ", num_heads: " << num_heads << ", head_dim: " << head_dim << ", kv_num_heads: " << kv_num_heads << std::endl;
    TORCH_CHECK(num_heads % kv_num_heads == 0, "number of key/value heads must be divisible by number of query heads");

    if (is_causal) { window_size_right = 0; }

    ForwardParams params{};
    forward_params_init(params,
        q, k, v, out,
        batch, max_seqlen_q, max_seqlen_k, num_heads, head_dim, kv_num_heads,
        (void*)cu_seqlens_q.data_ptr<int>(), (void*)cu_seqlens_k.data_ptr<int>(),
        window_size_left, window_size_right
    );

    auto stream = torch::cuda::getCurrentCUDAStream();
    run_flash_attention_forward(params, stream);
    torch::cuda::stream_synchronize(stream);

    return out;
}


} // namespace mfa
