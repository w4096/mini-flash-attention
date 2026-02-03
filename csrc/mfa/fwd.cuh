#pragma once

#include "mfa/flash.h"
#include "mfa/utils.cuh"
#include <cute/tensor.hpp>

using namespace cute;

namespace mfa {

template<typename KernelTraits>
class Query {
 public:
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemShape = Shape<Int<KernelTraits::kBlockM>, Int<KernelTraits::kHeadDim>>;
    using SmemLayout = Layout<SmemShape, Stride<Int<KernelTraits::kHeadDim>, _1>>;
    using SmemTensor = decltype(make_tensor(make_smem_ptr<Element>(nullptr), SmemLayout{}));

    using GmemShape = SmemShape;
    using GmemStride = Stride<index_t, _1>; // the row stride is dynamic
    using GmemLayout = Layout<GmemShape, GmemStride>;
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), GmemLayout{}));

    explicit __device__ Query(const ForwardParams& params, Element* smem) {
        const size_t offset = blockIdx.z * params.q_batch_stride + blockIdx.y * params.q_head_stride +
                              blockIdx.x * KernelTraits::kBlockM * params.q_row_stride;
        Element* gmem_ptr = static_cast<Element*>(params.q_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.q_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem() {
        // every thread cp 8 elements per iteration (16 bytes)
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, true);
        }
    }

    __forceinline__ __device__ const Element& smem(unsigned int row, unsigned int col) const {
        typename KernelTraits::SmemSwizzle swizzle;
        auto swizzled_col = swizzle(row * size<1>(smem_) + col) % size<1>(smem_);
        return smem_(row, swizzled_col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }

 private:
    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KernelTraits>
class Key {
 public:
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemShape = Shape<Int<KernelTraits::kBlockN>, Int<KernelTraits::kHeadDim>>;
    using SmemLayout = Layout<SmemShape, Stride<Int<KernelTraits::kHeadDim>, _1>>;
    using SmemTensor = decltype(make_tensor(make_smem_ptr<Element>(nullptr), SmemLayout{}));

    using GmemShape = Shape<Int<KernelTraits::kBlockN>, Int<KernelTraits::kHeadDim>>;
    using GmemStride = Stride<index_t, _1>; // the row stride is dynamic
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), make_layout(GmemShape{}, GmemStride{})));

    explicit __device__ Key(const ForwardParams& params, Element* smem) {
        const size_t offset =
            blockIdx.z * params.k_batch_stride + (blockIdx.y / params.kv_group_size) * params.k_head_stride;
        Element* gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.k_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void advance(const ForwardParams& params) {
        const size_t offset = KernelTraits::kBlockN * params.k_row_stride;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_.data().get() + offset), gmem_.layout());
    }

    __device__ void copy_gmem_to_smem() {
        for (unsigned int i = 0; i < size(smem_); i += blockDim.x * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, true);
        }
    }

    __forceinline__ __device__ const Element& smem(unsigned int row, unsigned int col) const {
        typename KernelTraits::SmemSwizzle swizzle;
        auto swizzled_col = swizzle(row * size<1>(smem_) + col) % size<1>(smem_);
        return smem_(row, swizzled_col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }


 private:
    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KernelTraits>
struct Value {
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemLayout = Key<KernelTraits>::SmemLayout;
    using SmemTensor = Key<KernelTraits>::SmemTensor;

    using GmemShape = Key<KernelTraits>::GmemShape;
    using GmemTensor = Key<KernelTraits>::GmemTensor;

    __device__ explicit Value(const ForwardParams& params, Element* smem) {
        const index_t offset =
            blockIdx.z * params.v_batch_stride + (blockIdx.y / params.kv_group_size) * params.v_head_stride;
        Element* gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.v_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void advance(const ForwardParams& params) {
        const size_t offset = KernelTraits::kBlockN * params.v_row_stride;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_.data().get() + offset), gmem_.layout());
    }

    __device__ void copy_gmem_to_smem() {
        for (unsigned int i = 0; i < size(smem_); i += blockDim.x * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, true);
        }
    }

    __forceinline__ __device__ const Element& smem(unsigned int row, unsigned int col) const {
        using Swizzle = KernelTraits::SmemSwizzle;
        Swizzle swizzle;
        auto swizzled_col = swizzle(row * size<1>(smem_) + col) % size<1>(smem_);
        return smem_(row, swizzled_col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }

 private:
    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KernelTraits>
class Score {
 public:
    using ElementAccum = KernelTraits::ElementAccum;

    // the MMA tile size
    static constexpr int MMA_m = 16;
    static constexpr int MMA_n = 8;
    static constexpr int MMA_k = 16;

    // tiles in M, N, K dimensions
    static constexpr int m_tiles = KernelTraits::kBlockM / MMA_m;
    static constexpr int n_tiles = KernelTraits::kBlockN / MMA_n;
    static constexpr int k_tiles = KernelTraits::kHeadDim / MMA_k;
    static_assert(m_tiles == KernelTraits::kNWarps, "m_tiles must equal to kNWarps");
    static_assert(KernelTraits::kBlockN % MMA_n == 0, "BlockN must be multiple of MMA_n");

    // for every tile in C, each thread has to save 4 float values
    using ScoreTensorLayout = Layout<Shape<Int<n_tiles>, _4>, Stride<_4, _1>>;
    using ScoreTensor = decltype(cute::make_tensor<ElementAccum>(ScoreTensorLayout{}));

    __device__ Score(const Query<KernelTraits>& query, const Key<KernelTraits>& key) {
        score_ = cute::make_tensor<ElementAccum>(ScoreTensorLayout{});

        const int warp_idx = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;

        const int m_smem_idx = warp_idx * MMA_m;
        for (int n = 0; n < n_tiles; n++) {
            const int n_smem_idx = n * MMA_n;

            float c0 = 0.0f;
            float c1 = 0.0f;
            float c2 = 0.0f;
            float c3 = 0.0f;

            for (int k = 0; k < k_tiles; k++) {
                const int k_smem_idx = k * MMA_k;
                uint32_t a0, a1, a2, a3;
                uint32_t b0, b1;

                // the address within 16x16 tile of A matrix
                int a_row = m_smem_idx + (lane % 16);
                int a_col = k_smem_idx + (lane / 16) * 8;
                const auto a_addr = reinterpret_cast<const uint128_t*>(&query.smem(a_row, a_col));
                cute::SM75_U32x4_LDSM_N::copy(*a_addr, a0, a1, a2, a3);

                // the address within 16x8 tile of B matrix
                int b_row = n_smem_idx + (lane % 8);
                int b_col = k_smem_idx + (lane / 8) * 8;
                const auto b_addr = reinterpret_cast<const uint128_t*>(&key.smem(b_row, b_col));
                cute::SM75_U32x2_LDSM_N::copy(*b_addr, b0, b1);

                // Perform MMA operation
                // clang-format off
                KernelTraits::MMA::fma(
                    c0, c1, c2, c3,
                    a0, a1, a2, a3,
                    b0, b1,
                    c0, c1, c2, c3);
                // clang-format on
            }

            score_(n, 0) = c0;
            score_(n, 1) = c1;
            score_(n, 2) = c2;
            score_(n, 3) = c3;
        }
    }

    template<typename Tensor0>
    __device__ void compute_row_max(Tensor0& row_max) const {
        // each thread compute local max for row 0 and row 8 in a 16x8 block
        ElementAccum row_0_max = cuda::std::numeric_limits<ElementAccum>::min();
        ElementAccum row_1_max = cuda::std::numeric_limits<ElementAccum>::min();
        for (int n = 0; n < size<0>(score_); n++) {
            row_0_max = max(row_0_max, max(score_(n, 0), score_(n, 1)));
            row_1_max = max(row_1_max, max(score_(n, 2), score_(n, 3)));
        }
        row_max(0) = warp_reduce_max<ElementAccum, 4>(row_0_max);
        row_max(1) = warp_reduce_max<ElementAccum, 4>(row_1_max);
    }

    __device__ const ScoreTensor& score() const {
        return score_;
    }

    __device__ __forceinline__ const ElementAccum& operator()(int n, int idx) const {
        return score_(n, idx);
    }

    __device__ __forceinline__ ElementAccum& operator()(int n, int idx) {
        return score_(n, idx);
    }

 private:
    ScoreTensor score_;
};

template<typename KernelTraits>
class Softmax {
 public:
    using ElementAccum = typename KernelTraits::ElementAccum;

    // in 16x8 block, each thread handle `row` and `row + 8`
    using RowMaxTensorLayout = Layout<Shape<_2>, Stride<_1>>;
    using RowMaxTensor = decltype(cute::make_tensor<ElementAccum>(RowMaxTensorLayout{}));

    // exp(old_max - new_max)
    using RescaleTensorLayout = RowMaxTensorLayout;
    using RescaleTensor = decltype(cute::make_tensor<ElementAccum>(RescaleTensorLayout{}));

    using ExpSumTensorLayout = RowMaxTensorLayout;
    using ExpSumTensor = decltype(cute::make_tensor<ElementAccum>(ExpSumTensorLayout{}));

    __device__ Softmax() {
        row_max_ = cute::make_tensor<ElementAccum>(RowMaxTensorLayout{});
        rescale_ = cute::make_tensor<ElementAccum>(RescaleTensorLayout{});
        expsum_ = cute::make_tensor<ElementAccum>(ExpSumTensorLayout{});

        cute::fill(expsum_, ElementAccum{0});
        cute::fill(row_max_, -cuda::std::numeric_limits<ElementAccum>::infinity());
    }

    __device__ void update(Score<KernelTraits>& score, float softmax_scale_log2, bool first) {
        // compute max per row
        auto new_row_max = cute::make_tensor<ElementAccum>(RowMaxTensorLayout{});
        score.compute_row_max(new_row_max);

        #pragma unroll
        for (int i = 0; i < size<0>(row_max_); i++) {
            ElementAccum new_max = max(new_row_max(i), row_max_(i));
            rescale_(i) = exp2f((row_max_(i) - new_max) * softmax_scale_log2);
            row_max_(i) = new_max;
        }

        // exponentiate the scores with updated row_max
        float sum_0 = 0.0f, sum_1 = 0.0f;
        float max_0 = row_max_(0) * softmax_scale_log2;
        float max_1 = row_max_(1) * softmax_scale_log2;
        #pragma unroll
        for (int n = 0; n < size<0>(score.score()); n++) {
            score(n, 0) = exp2f(score(n, 0) * softmax_scale_log2 - max_0);
            score(n, 1) = exp2f(score(n, 1) * softmax_scale_log2 - max_0);
            score(n, 2) = exp2f(score(n, 2) * softmax_scale_log2 - max_1);
            score(n, 3) = exp2f(score(n, 3) * softmax_scale_log2 - max_1);
            sum_0 += score(n, 0) + score(n, 1);
            sum_1 += score(n, 2) + score(n, 3);
        }

        // update exp sum with rescaling
        if (first) {
            expsum_(0) = sum_0;
            expsum_(1) = sum_1;
        } else {
            expsum_(0) = __fmaf_rn(expsum_(0), rescale_(0), sum_0);
            expsum_(1) = __fmaf_rn(expsum_(1), rescale_(1), sum_1);
        }
    }

    __device__ void reduce_sum_expsum() {
        // final reduction across rows
        expsum_(0) = warp_reduce_sum<ElementAccum, 4>(expsum_(0));
        expsum_(1) = warp_reduce_sum<ElementAccum, 4>(expsum_(1));
    }

    __device__ const ExpSumTensor& expsum() const {
        return expsum_;
    }

    __device__ const RescaleTensor& rescale() const {
        return rescale_;
    }

    __device__ const RowMaxTensor& row_max() const {
        return row_max_;
    }

 private:
    ExpSumTensor expsum_;
    RowMaxTensor row_max_;
    RescaleTensor rescale_;
};

template<typename KernelTraits>
struct Output {
    using Element = KernelTraits::Element;
    using ElementAccum = KernelTraits::ElementAccum;
    using index_t = KernelTraits::index_t;

    static constexpr int MMA_m = 16;
    static constexpr int MMA_n = 8;
    static constexpr int MMA_k = 16;

    static constexpr int m_tiles = KernelTraits::kBlockM / MMA_m;
    static constexpr int n_tiles = KernelTraits::kHeadDim / MMA_n;
    static constexpr int k_tiles = KernelTraits::kBlockN / MMA_k;

    // each thread saves 4 elements per tile
    using RmemLayout = Layout<Shape<Int<n_tiles>, _4>, Stride<_4, _1>>;
    using RmemTensor = decltype(cute::make_tensor<ElementAccum>(RmemLayout{}));

    using GmemShape = Query<KernelTraits>::GmemShape;
    using GmemLayout = Query<KernelTraits>::GmemLayout;
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), GmemLayout{}));

    using SmemLayout = Query<KernelTraits>::SmemLayout;
    using SmemTensor = Query<KernelTraits>::SmemTensor;

    explicit __device__ Output(const ForwardParams& params, Element* smem) {
        rmem_ = make_tensor<ElementAccum>(RmemLayout{});
        cute::fill(rmem_, .0f);

        const index_t offset = blockIdx.z * params.o_batch_stride + blockIdx.y * params.o_head_stride +
                               blockIdx.x * KernelTraits::kBlockM * params.o_row_stride;
        Element* gmem_ptr = static_cast<Element*>(params.o_ptr) + offset;
        auto layout = make_layout(GmemShape{}, make_stride(params.o_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr<Element>(smem), SmemLayout{});
    }

    template<typename Tensor0>
    __device__ void accum(const Softmax<KernelTraits>::RescaleTensor& rescale, Tensor0 const& score,
                          const Value<KernelTraits>& value, bool first) {
        
        for (int n = 0; n < n_tiles; n++) {
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

            for (int k = 0; k < k_tiles; k++) {
                uint32_t a0, a1, a2, a3;

                constexpr bool fp16 = std::is_same_v<typename KernelTraits::Element, half_t>;
                if constexpr (fp16) {
                    half2 h2_0 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k, 0)));
                    half2 h2_1 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k, 2)));
                    half2 h2_2 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 0)));
                    half2 h2_3 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 2)));
                    a0 = *reinterpret_cast<uint32_t*>(&h2_0);
                    a1 = *reinterpret_cast<uint32_t*>(&h2_1);
                    a2 = *reinterpret_cast<uint32_t*>(&h2_2);
                    a3 = *reinterpret_cast<uint32_t*>(&h2_3);
                } else {
                    nv_bfloat162 b2_0 = __float22bfloat162_rn(*reinterpret_cast<const float2*>(&score(2 * k, 0)));
                    nv_bfloat162 b2_1 = __float22bfloat162_rn(*reinterpret_cast<const float2*>(&score(2 * k, 2)));
                    nv_bfloat162 b2_2 = __float22bfloat162_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 0)));
                    nv_bfloat162 b2_3 = __float22bfloat162_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 2)));
                    a0 = *reinterpret_cast<uint32_t*>(&b2_0);
                    a1 = *reinterpret_cast<uint32_t*>(&b2_1);
                    a2 = *reinterpret_cast<uint32_t*>(&b2_2);
                    a3 = *reinterpret_cast<uint32_t*>(&b2_3);
                }

                int v_row = k * 16 + (threadIdx.x % 16);
                int v_col = n * 8;
                auto addr = reinterpret_cast<const uint128_t*>(&value.smem(v_row, v_col));
                uint32_t b0, b1;
                cute::SM75_U16x4_LDSM_T::copy(*addr, b0, b1);

                // clang-format off
                KernelTraits::MMA::fma(
                    c0, c1, c2, c3,
                    a0, a1, a2, a3,
                    b0, b1,
                    c0, c1, c2, c3);
                // clang-format on
            }

            if (first) {
                rmem_(n, 0) = c0;
                rmem_(n, 1) = c1;
                rmem_(n, 2) = c2;
                rmem_(n, 3) = c3;
            } else {
                rmem_(n, 0) = __fmaf_rn(rmem_(n, 0), rescale(0), c0);
                rmem_(n, 1) = __fmaf_rn(rmem_(n, 1), rescale(0), c1);
                rmem_(n, 2) = __fmaf_rn(rmem_(n, 2), rescale(1), c2);
                rmem_(n, 3) = __fmaf_rn(rmem_(n, 3), rescale(1), c3);
            }
        }
    }

    template<typename Tensor>
    __device__ void normalize(const Tensor& expsum) {
        // normalize with 1 / exp_sum
        float sum0 = expsum(0);
        float sum1 = expsum(1);
        auto r0 = (sum0 == 0.0f || sum0 != sum0) ? 1.f : 1.f / sum0;
        auto r1 = (sum1 == 0.0f || sum1 != sum1) ? 1.f : 1.f / sum1;
        for (int n = 0; n < size<0>(rmem_); n++) {
            rmem_(n, 0) *= r0;
            rmem_(n, 1) *= r0;
            rmem_(n, 2) *= r1;
            rmem_(n, 3) *= r1;
        }
    }

    __device__ void copy_smem_to_gmem() {
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            // vectorize
            *reinterpret_cast<uint4*>(&gmem_(row, col)) = *reinterpret_cast<uint4*>(&smem(row, col));
        }
    }

    __forceinline__ __device__ void copy_rmem_to_smem() {
        int wrap_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;

        for (int n = 0; n < n_tiles; n++) {
            // the row and col within 16x8 tile of output matrix
            int row = wrap_id * MMA_m + lane / 4;
            int col = n * MMA_n + (lane % 4) * 2;

            // convert float to half/bfloat16 and store to shared memory
            if constexpr (std::is_same_v<Element, half_t>) {
                half2 *h2 = reinterpret_cast<half2*>(&smem(row, col));
                float2 *f2 = reinterpret_cast<float2*>(&rmem_(n, 0));
                *h2 = __float22half2_rn(*f2);

                h2 = reinterpret_cast<half2*>(&smem(row + 8, col));
                f2 = reinterpret_cast<float2*>(&rmem_(n, 2));
                *h2 = __float22half2_rn(*f2);
            } else {
                nv_bfloat162 *b2 = reinterpret_cast<nv_bfloat162*>(&smem(row, col));
                float2 *f2 = reinterpret_cast<float2*>(&rmem_(n, 0));
                *b2 = __float22bfloat162_rn(*f2);

                b2 = reinterpret_cast<nv_bfloat162*>(&smem(row + 8, col));
                f2 = reinterpret_cast<float2*>(&rmem_(n, 2));
                *b2 = __float22bfloat162_rn(*f2);
            }
        }
    }

    __forceinline__ __device__ const Element& smem(unsigned int row, unsigned int col) const {
        typename KernelTraits::SmemSwizzle swizzle;
        auto swizzled_col = swizzle(row * size<1>(smem_) + col) % size<1>(smem_);
        return smem_(row, swizzled_col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }

    
 private:
    RmemTensor rmem_;
    GmemTensor gmem_;
    SmemTensor smem_;
};

template<typename KernelTraits>
__global__ void flash_attention_fwd_kernel(__grid_constant__ const ForwardParams params) {
    using Element = KernelTraits::Element;

    // init shared memory block tensors
    extern __shared__ char smem_data[];
    Query<KernelTraits> Q(params, reinterpret_cast<Element*>(smem_data));
    uint32_t offset = size(typename Query<KernelTraits>::SmemLayout{});
    Key<KernelTraits> K(params, reinterpret_cast<Element*>(smem_data) + offset);
    offset += size(typename Key<KernelTraits>::SmemLayout{});
    Value<KernelTraits> V(params, reinterpret_cast<Element*>(smem_data) + offset);
    Output<KernelTraits> O(params, reinterpret_cast<Element*>(smem_data + 0)); // reuse Q's smem

    Softmax<KernelTraits> softmax{};

    Q.copy_gmem_to_smem();
    cute::cp_async_fence();

    constexpr int kBlockN = KernelTraits::kBlockN;
    const int n_blocks = params.seqlen_k / kBlockN;
    for (int nbi = 0; nbi < n_blocks; nbi++) {
        K.copy_gmem_to_smem();
        cute::cp_async_fence();

        cute::cp_async_wait<0>();

        V.copy_gmem_to_smem();
        cute::cp_async_fence();

        Score<KernelTraits> score(Q, K);
        softmax.update(score, params.softmax_scale_log2, nbi == 0);

        cute::cp_async_wait<0>();
        O.accum(softmax.rescale(), score.score(), V, nbi == 0);

        K.advance(params);
        V.advance(params);
    }

    softmax.reduce_sum_expsum();

    O.normalize(softmax.expsum());

    O.copy_rmem_to_smem();

    __syncthreads();
        
    O.copy_smem_to_gmem();

}

template<typename KernelTraits>
void compute_attn(const ForwardParams& params, cudaStream_t stream) {
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);

    dim3 grid(m_blocks, params.heads, params.batch);
    dim3 block(KernelTraits::kNThreads);

    flash_attention_fwd_kernel<KernelTraits><<<grid, block, KernelTraits::smem_size, stream>>>(params);
}

} // namespace mfa