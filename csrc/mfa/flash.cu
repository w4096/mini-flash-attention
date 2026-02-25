#include "mfa/flash.h"
#include "mfa/traits.h"
#include "mfa/prefill.cuh"
#include "mfa/decode.cuh"
#include "mfa/static_switch.h"

namespace mfa {



template<typename KernelTraits>
void run_mha_prefill(const ForwardParams& params, cudaStream_t stream) {
    dim3 grid, block;
    block = dim3(KernelTraits::kNThreads);
    
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);
    grid = dim3(m_blocks, params.heads, params.batch);

    auto kernel = flash_attention_fwd_kernel<KernelTraits>;

    using Element = typename KernelTraits::Element;
    static const int kBlockN = KernelTraits::kBlockN;
    static const int kHeadDim = KernelTraits::kHeadDim;
    static const int kBlockM = KernelTraits::kBlockM;

    const int smem_size = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(Element);
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

template<typename KernelTraits, bool Split>
void run_mha_decode(const ForwardParams& params, cudaStream_t stream) {
    dim3 grid, block;
    block = dim3(KernelTraits::kNThreads);

    if constexpr(Split) {
        // For multi-split decoding, we need to launch one kernel per split to handle the different KV cache pointers
        grid = dim3(params.num_splits, params.heads, params.batch);
    } else {
        grid = dim3(params.heads, params.batch);
    }

    auto kernel = flash_attention_fwd_split_kv_kernel<KernelTraits, Split>;

    using Element = typename KernelTraits::Element;
    static const int kBlockN = KernelTraits::kBlockN;
    static const int kHeadDim = KernelTraits::kHeadDim;

    int smem_size = kHeadDim * sizeof(Element); // Q
    smem_size += kBlockN * kHeadDim * 2 * sizeof(Element); // KV

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);


    if constexpr(Split) {
        dim3 grid_combine(params.batch, params.heads);
        smem_size = kHeadDim * params.num_splits * sizeof(float) + params.num_splits * sizeof(float); // O + softmax accumulators
        flash_attention_fwd_split_kv_combine_kernel<KernelTraits><<<grid_combine, KernelTraits::kNThreads, smem_size, stream>>>(params);
    }
}


void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEAD_DIM_SWITCH(params.head_dim, [&] {
            BOOL_SWITCH(params.cu_seqlens_k != nullptr, IsVarlen, [&] {
                run_mha_prefill<ForwardKernelTraits<elem_type, kHeadDim, 64, 64, 4, IsVarlen>>(params, stream);
            });
        });
    });
}

// Decoding dispatch: specialize BlockM for single-token queries to reduce wasted threads
void run_flash_attention_with_kv_cache(ForwardParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEAD_DIM_SWITCH(params.head_dim, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                run_mha_decode<ForwardKernelTraits<elem_type, kHeadDim, 64, 64, 4>, Split>(params, stream);
            });
        });
    });
}



} // namespace mfa
