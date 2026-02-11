#pragma once

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <bit>

namespace mfa {

template<typename Element_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool IsVarlen_ = false>
struct ForwardKernelTraits {
    using Element = Element_;
    using ElementAccum = float;
    using index_t = int64_t;

    using MMA = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::SM80_16x8x16_F32F16F16F32_TN,
        cute::SM80_16x8x16_F32BF16BF16F32_TN>;

    static constexpr bool CpAsyncSupported = true;
    static constexpr bool is_varlen = IsVarlen_;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockSize = kNThreads;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == kNWarps * 16); // each warp handles 16 rows


    // Swizzle<BBits, MBase, SShift>
    // Choose swizzle pattern based on head dimension to optimize shared memory access
    // and avoid bank conflicts. The swizzle should match the row size in bytes.
    using SmemSwizzle = std::conditional_t<
        kHeadDim <= 64,
        cute::Swizzle<3, 3, 3>,    // For head_dim=32,64: row size = 64-128 bytes
        std::conditional_t<
            kHeadDim == 128,
            cute::Swizzle<3, 3, 4>,  // For head_dim=128: row size = 256 bytes
            cute::Swizzle<3, 3, 5>   // For head_dim>=256: row size >= 512 bytes
        >
    >;

    static constexpr int smem_size = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(Element);
};


}
