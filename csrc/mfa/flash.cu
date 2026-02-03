#include "mfa/flash.h"
#include "mfa/traits.h"
#include "mfa/fwd.cuh"
#include "mfa/static_switch.h"

namespace mfa {


template<typename T, int HeadDim>
void run_mha_fwd(ForwardParams &params, cudaStream_t stream) {
    compute_attn<ForwardKernelTraits<T, HeadDim, 64, 64, 4>>(params, stream);
}

void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEAD_DIM_SWITCH(params.head_dim, [&] {
            run_mha_fwd<elem_type, kHeadDim>(params, stream);
        });
    });
}

}
