#pragma once

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <bit>

namespace mfa {

template<typename Element_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct ForwardKernelTraits {
    using Element = Element_;
    using ElementAccum = float;
    using index_t = int64_t;

    using MMA = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::SM80_16x8x16_F32F16F16F32_TN,
        cute::SM80_16x8x16_F32BF16BF16F32_TN>;

    static constexpr bool CpAsyncSupported = true;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockSize = kNThreads;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == kNWarps * 16); // each warp handles 16 rows


    // Swizzle<BBits, MBase, SShift>
    using SmemSwizzle = cute::Swizzle<3, 3, 4>;

    static constexpr int smem_size = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(Element);
};


}
