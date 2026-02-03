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

void forward_params_init(ForwardParams& params, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
    if (q.dim() == 3) {
        q = q.unsqueeze(0);
        k = k.unsqueeze(0);
        v = v.unsqueeze(0);
        o = o.unsqueeze(0);
    }
    // q, k, v, o: (batch, heads, seqlen, head_dim)
    TORCH_CHECK(q.dim() == 4, "q tensor must be 3D or 4D");
    TORCH_CHECK(k.dim() == 4, "k tensor must be 3D or 4D");
    TORCH_CHECK(v.dim() == 4, "v tensor must be 3D or 4D");
    TORCH_CHECK(o.dim() == 4, "o tensor must be 3D or 4D");

    TORCH_CHECK(q.stride(-1) == 1, "q tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v tensor must have contiguous last dimension");
    TORCH_CHECK(o.stride(-1) == 1, "o tensor must have contiguous last dimension");

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();

    params.batch = static_cast<int>(q.size(-4));
    params.q_batch_stride = q.stride(-4);
    params.k_batch_stride = k.stride(-4);
    params.v_batch_stride = v.stride(-4);
    params.o_batch_stride = o.stride(-4);

    params.heads = static_cast<int>(q.size(-2));
    params.kv_heads = static_cast<int>(k.size(-2));
    params.kv_group_size = params.heads / params.kv_heads;
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_head_stride = o.stride(-2);

    params.seqlen_q = static_cast<int>(q.size(-3));
    params.seqlen_k = static_cast<int>(k.size(-3));
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.o_row_stride = o.stride(-3);

    params.head_dim = static_cast<int>(q.size(-1));
    params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    params.softmax_scale_log2 = params.softmax_scale * M_LOG2E;

    params.is_bf16 = q.dtype() == torch::kBFloat16;
}

/**
 * Flash Attention v2 API
 *
 * @param q  total_q x num_heads x head_size
 * @param k  total_k x num_heads_k x head_size
 * @param v  total_k x num_heads_k x head_size
 * @return 
 */
std::vector<at::Tensor> flash_attention_v2(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
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

    torch::Tensor out = torch::empty_like(q);
    torch::Tensor lse;  // TODO

    ForwardParams params{};
    forward_params_init(params, q, k, v, out);

    auto stream = torch::cuda::getCurrentCUDAStream();
    run_flash_attention_forward(params, stream);
    torch::cuda::stream_synchronize(stream);

    return {out, lse};
}

} // namespace mfa
