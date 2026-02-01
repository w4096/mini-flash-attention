#pragma once

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

namespace mfa {

template<typename Element_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct ForwardKernelTraits {
    using Element = Element_;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr bool CpAsyncSupported = true;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == kNWarps * 16); // each warp handles 16 rows

    // kBlockK is the width of QKV tile loaded per iteration into shared memory
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    // the shape of the tile of Q is (kBlockM, kHeadDim)
    using SmemLayoutQ = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutKV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

    static constexpr int kThreadsPerRow = 8;
    static constexpr int kRowsPerThread = kBlockM / (kNThreads / kThreadsPerRow);



    // how many elements will be loaded/stored by each thread
    static constexpr int kGmemElementsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElementsPerLoad == 0);

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElementsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0);

    // using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, LayoutRight>;
    //
    // using QKVCopyStruct = std::conditional_t<
    //     CpAsyncSupported,
    //     SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    //     DefaultCopy
    // >;
    // using GmemTiledCopyQKV = decltype(
    //     make_tiled_copy(
    //         Copy_Atom<QKVCopyStruct, Element>{},
    //         GmemLayoutAtom{},
    //         Layout<Shape<_1, _8>>{}
    //     )
    // );


    // static constexpr int heads = 1;
    // static constexpr int tokens = 64;

    // static constexpr int head_stride = tokens * head_dim;
    // static constexpr int batch_stride = heads * head_stride;
    // static constexpr int q_block_stride = block_m * head_dim;
    // static constexpr int kv_lock_stride = block_n * head_dim;

    static constexpr int q_smem_size = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kv_smem_size = size(SmemLayoutKV{}) * sizeof(Element) * 2;
    static constexpr int smem_size = 64 * 128 * 2 * 3;
};



}
