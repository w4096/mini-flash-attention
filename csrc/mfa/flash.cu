#include "mfa/flash.h"
#include "mfa/traits.h"
#include "mfa/fwd.cuh"
#include "mfa/static_switch.h"

namespace mfa {



template<typename KernelTraits>
void run_mha_fwd(const ForwardParams& params, cudaStream_t stream) {
    dim3 grid, block;
    block = dim3(KernelTraits::kNThreads);
    
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);
    grid = dim3(m_blocks, params.heads, params.batch);

    auto kernel = flash_attention_fwd_kernel<KernelTraits>;

    // Set shared memory limit for large head dimensions
    // SM80 (Ampere) supports up to 164KB per SM, but 48KB per block by default
    // For head_dim >= 128, we need more than 48KB
    constexpr int smem_size = KernelTraits::smem_size;
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}


void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEAD_DIM_SWITCH(params.head_dim, [&] {
            run_mha_fwd<ForwardKernelTraits<elem_type, kHeadDim, 64, 64, 4>>(params, stream);
        });
    });
}

}
