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
        const size_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, blockIdx.x * KernelTraits::kBlockM);
        auto gmem_ptr = static_cast<Element*>(params.q_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.q_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const ForwardParams& params) {
        // every thread cp 8 elements per iteration (16 bytes)
        const int row_offset = blockIdx.x * KernelTraits::kBlockM;
        const int max_row = params.seqlen_q;
        
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            
            // Check boundary: only copy if within valid range
            bool valid = (row_offset + row) < max_row;
            
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, valid);
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

    __device__ index_t get_gmem_offset(const ForwardParams& params, int batch, int head, int row) const {
        return batch * params.q_batch_stride + head * params.q_head_stride + row * params.q_row_stride;
    };

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
        const size_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, 0);

        auto gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.k_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const ForwardParams& params, int nbidx) {
        set_gmem_address(params, nbidx);
        
        const int row_offset = nbidx * KernelTraits::kBlockN;
        const int max_row = params.seqlen_k;

        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            
            // Check boundary: only copy if within valid range
            bool valid = (row_offset + row) < max_row;
            
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, valid);
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
    __forceinline__ __device__ void set_gmem_address(const ForwardParams& params, int nbidx) {
        int row = nbidx * KernelTraits::kBlockN;
        const size_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, row);
        auto gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), gmem_.layout());
    }

    __device__ index_t get_gmem_offset(const ForwardParams& params, int batch, int head, int row) const {
        head = head / params.kv_group_size;
        return batch * params.k_batch_stride + head * params.k_head_stride + row * params.k_row_stride;
    };

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
        const index_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, 0);
        auto gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.v_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const ForwardParams& params, int nbidx) {
        set_gmem_address(params, nbidx);
        
        const int row_offset = nbidx * KernelTraits::kBlockN;
        const int max_row = params.seqlen_k;
        
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            
            // Check boundary: only copy if within valid range
            bool valid = (row_offset + row) < max_row;
            
            auto dst_ptr = reinterpret_cast<int4*>(&smem(row, col));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(row, col));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, valid);
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

    __forceinline__ __device__ void set_gmem_address(const ForwardParams& params, int nbidx) {
        int row = nbidx * KernelTraits::kBlockN;
        const size_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, row);
        auto gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), gmem_.layout());
    }

    __device__ index_t get_gmem_offset(const ForwardParams& params, int batch, int head, int row) const {
        head = head / params.kv_group_size;
        return batch * params.v_batch_stride + head * params.v_head_stride + row * params.v_row_stride;
    };

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
                const int a_row = m_smem_idx + (lane % 16);
                const int a_col = k_smem_idx + (lane / 16) * 8;
                const auto a_addr = reinterpret_cast<const uint128_t*>(&query.smem(a_row, a_col));
                cute::SM75_U32x4_LDSM_N::copy(*a_addr, a0, a1, a2, a3);

                // the address within 16x8 tile of B matrix
                const int b_row = n_smem_idx + (lane % 8);
                const int b_col = k_smem_idx + (lane / 8) * 8;
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

    // Set causal mask: set scores to -inf where col_idx > row_idx
    // Unified mask: handles both sequence length mask and causal mask
    // If is_causal=true: masks where col > row OR col >= seqlen_k
    // If is_causal=false: masks only where col >= seqlen_k
    __device__ void set_mask(int row_offset, int col_offset, int seqlen_k, bool is_causal) {
        const int warp_idx = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        
        // Each thread handles 2 rows (row and row+8) in a 16x8 tile
        const int m_base = row_offset + warp_idx * MMA_m;
        
        for (int n = 0; n < n_tiles; n++) {
            const int n_base = col_offset + n * MMA_n;
            
            // For each of the 4 elements this thread owns in the tile
            // Elements (0,1) are for row m_base + (lane/4)
            // Elements (2,3) are for row m_base + (lane/4) + 8
            
            int row_0 = m_base + (lane / 4);
            int row_1 = m_base + (lane / 4) + 8;
            
            // Each element covers 2 columns
            int col_0 = n_base + (lane % 4) * 2;
            int col_1 = col_0 + 1;
            
            static constexpr auto _inf = -cuda::std::numeric_limits<ElementAccum>::infinity();

            // Apply unified mask
            if (is_causal) {
                // Causal mode: mask if col > row OR col >= seqlen_k
                if (col_0 > row_0 || col_0 >= seqlen_k) score_(n, 0) = _inf;
                if (col_1 > row_0 || col_1 >= seqlen_k) score_(n, 1) = _inf;
                if (col_0 > row_1 || col_0 >= seqlen_k) score_(n, 2) = _inf;
                if (col_1 > row_1 || col_1 >= seqlen_k) score_(n, 3) = _inf;
            } else {
                // Non-causal mode: mask only if col >= seqlen_k
                if (col_0 >= seqlen_k) score_(n, 0) = _inf;
                if (col_1 >= seqlen_k) score_(n, 1) = _inf;
                if (col_0 >= seqlen_k) score_(n, 2) = _inf;
                if (col_1 >= seqlen_k) score_(n, 3) = _inf;
            }
        }
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

    __device__ void update(Score<KernelTraits>& score, float softmax_scale_log2) {
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
        const float max_scaled_0 = row_max_(0) * softmax_scale_log2;
        const float max_scaled_1 = row_max_(1) * softmax_scale_log2;

        #pragma unroll
        for (int n = 0; n < size<0>(score.score()); n++) {
            score(n, 0) = exp2f(score(n, 0) * softmax_scale_log2 - max_scaled_0);
            score(n, 1) = exp2f(score(n, 1) * softmax_scale_log2 - max_scaled_0);
            score(n, 2) = exp2f(score(n, 2) * softmax_scale_log2 - max_scaled_1);
            score(n, 3) = exp2f(score(n, 3) * softmax_scale_log2 - max_scaled_1);
            sum_0 += score(n, 0) + score(n, 1);
            sum_1 += score(n, 2) + score(n, 3);
        }

        // update exp sum with rescaling
        expsum_(0) = __fmaf_rn(expsum_(0), rescale_(0), sum_0);
        expsum_(1) = __fmaf_rn(expsum_(1), rescale_(1), sum_1);
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

        const index_t offset = get_gmem_offset(params, blockIdx.z, blockIdx.y, blockIdx.x * KernelTraits::kBlockM);
        auto gmem_ptr = static_cast<Element*>(params.o_ptr) + offset;
        auto layout = make_layout(GmemShape{}, make_stride(params.o_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr<Element>(smem), SmemLayout{});
    }

    template<typename Tensor0>
    __device__ void accum(const Softmax<KernelTraits>::RescaleTensor& rescale, Tensor0 const& score,
                          const Value<KernelTraits>& value) {
        
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

            rmem_(n, 0) = __fmaf_rn(rmem_(n, 0), rescale(0), c0);
            rmem_(n, 1) = __fmaf_rn(rmem_(n, 1), rescale(0), c1);
            rmem_(n, 2) = __fmaf_rn(rmem_(n, 2), rescale(1), c2);
            rmem_(n, 3) = __fmaf_rn(rmem_(n, 3), rescale(1), c3);
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

    __device__ void copy_smem_to_gmem(const ForwardParams& params) {
        const int row_offset = blockIdx.x * KernelTraits::kBlockM;
        const int max_row = params.seqlen_q;
        
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            auto row = idx / size<1>(smem_);
            auto col = idx % size<1>(smem_);
            
            // Check boundary: only write if within valid range
            if ((row_offset + row) < max_row) {
                // vectorize
                *reinterpret_cast<uint4*>(&gmem_(row, col)) = *reinterpret_cast<uint4*>(&smem(row, col));
            }
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
    __device__ index_t get_gmem_offset(const ForwardParams& params, int batch, int head, int row) const {
        return batch * params.o_batch_stride + head * params.o_head_stride + row * params.o_row_stride;
    };

    RmemTensor rmem_;
    GmemTensor gmem_;
    SmemTensor smem_;
};

template<typename KernelTraits>
__global__ void flash_attention_fwd_kernel(__grid_constant__ const ForwardParams params) {
    using Element = KernelTraits::Element;

    auto mbidx = blockIdx.x;

    // init shared memory block tensors
    extern __shared__ char smem_data[];
    Query<KernelTraits> Q(params, reinterpret_cast<Element*>(smem_data));
    uint32_t offset = size(typename Query<KernelTraits>::SmemLayout{});
    Key<KernelTraits> K(params, reinterpret_cast<Element*>(smem_data) + offset);
    offset += size(typename Key<KernelTraits>::SmemLayout{});
    Value<KernelTraits> V(params, reinterpret_cast<Element*>(smem_data) + offset);
    Output<KernelTraits> O(params, reinterpret_cast<Element*>(smem_data + 0)); // reuse Q's smem

    Softmax<KernelTraits> softmax{};

    Q.copy_gmem_to_smem(params);
    cute::cp_async_fence();

    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kBlockM = KernelTraits::kBlockM;
    // blocks in N dimension
    const int n_blocks = cute::ceil_div(params.seqlen_k, kBlockN);
    const int row_offset_base = mbidx * kBlockM;
    
    // Determine which K/V blocks to process
    // For causal: only process blocks where some elements are not masked
    // Skip block if: row_offset_base > col_offset_end
    // In other words: skip if the first row of Q is after the last column of K
    const int n_block_min = 0;
    const int n_block_max = params.is_causal 
        ? cute::ceil_div(row_offset_base + kBlockM, kBlockN)  // Only process up to diagonal
        : n_blocks;
    


    for (int nbidx = n_block_min; nbidx < n_block_max; nbidx++) {
        K.copy_gmem_to_smem(params, nbidx);
        cute::cp_async_fence();

        cute::cp_async_wait<0>();
        __syncthreads();  // Ensure K is fully loaded before computing scores

        V.copy_gmem_to_smem(params, nbidx);
        cute::cp_async_fence();

        Score<KernelTraits> score(Q, K);

        // Apply unified mask (handles both seqlen and causal masks)
        const int col_offset = nbidx * kBlockN;
        
        // Skip masking if entire block is within valid range
        // This is common when seqlen_k is a multiple of kBlockN
        const bool need_mask = params.is_causal || (col_offset + kBlockN > params.seqlen_k);
        
        if (need_mask) {
            score.set_mask(row_offset_base, col_offset, params.seqlen_k, params.is_causal);
        }
        
        softmax.update(score, params.softmax_scale_log2);

        cute::cp_async_wait<0>();
        __syncthreads();  // Ensure V is fully loaded before accumulation
        O.accum(softmax.rescale(), score.score(), V);
    }

    softmax.reduce_sum_expsum();

    O.normalize(softmax.expsum());
    
    O.copy_rmem_to_smem();

    // Ensure all threads have written to shared memory before reading
    __syncthreads();
        
    O.copy_smem_to_gmem(params);
}

template<typename KernelTraits>
void compute_attn(const ForwardParams& params, cudaStream_t stream) {
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);

    dim3 grid(m_blocks, params.heads, params.batch);
    dim3 block(KernelTraits::kNThreads);

    // Set shared memory limit for large head dimensions
    // SM80 (Ampere) supports up to 164KB per SM, but 48KB per block by default
    // For head_dim >= 128, we need more than 48KB
    constexpr int smem_size = KernelTraits::smem_size;
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_attention_fwd_kernel<KernelTraits>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    flash_attention_fwd_kernel<KernelTraits><<<grid, block, smem_size, stream>>>(params);
}

} // namespace mfa