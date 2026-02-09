#pragma once

#include "mfa/flash.h"
#include "mfa/utils.cuh"
#include <cute/tensor.hpp>

using namespace cute;

namespace mfa {

namespace decode {

template<typename KernelTraits>
class Context {
 public:
    __device__ Context(const ForwardParams& params) {
        constexpr int kBlockN = KernelTraits::kBlockN;

        if (params.num_splits > 1) {
            split_idx_ = blockIdx.x;
            head_idx_ = blockIdx.y;
            batch_idx_ = blockIdx.z;
            
            actual_seqlen_k = params.seqlens_k[batch_idx_];
            const int n_blocks_per_split = cute::ceil_div(cute::ceil_div(params.seqlen_k, kBlockN), params.num_splits);

            n_block_min = split_idx_ * n_blocks_per_split;
            n_block_max = std::min(cute::ceil_div(actual_seqlen_k, kBlockN), (split_idx_ + 1) * n_blocks_per_split);
        } else {
            batch_idx_ = blockIdx.z;
            head_idx_ = blockIdx.y;
            actual_seqlen_k = params.seqlens_k[batch_idx_];

            n_block_min = 0;
            n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
        }
    }

    __device__ size_t get_q_gmem_offset(const ForwardParams& params) const {
        // For decode, Q is [batch, 1, heads, dim], always process the single query token
        return batch_idx_ * params.q_batch_stride + head_idx_ * params.q_head_stride;
    }

    __device__ size_t get_k_gmem_offset(const ForwardParams& params, int block_n_idx) const {
        const int kv_head_idx = head_idx_ / params.kv_group_size;
        if (params.block_table) {
            // Paged attention: use block table to find KV cache blocks
            const int* block_table_ptr = static_cast<const int*>(params.block_table) + batch_idx_ * params.block_table_batch_stride;
            const int physical_block_idx = block_table_ptr[block_n_idx];
            // K cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
            return physical_block_idx * params.k_batch_stride + kv_head_idx * params.k_head_stride;
        } else {
            // Direct KV cache access
            return batch_idx_ * params.k_batch_stride + kv_head_idx * params.k_head_stride
                   + block_n_idx * KernelTraits::kBlockN * params.k_row_stride;
        }
    }

    __device__ size_t get_v_gmem_offset(const ForwardParams& params, int block_n_idx) const {
        const int kv_head_idx = head_idx_ / params.kv_group_size;
        if (params.block_table) {
            const int* block_table_ptr = static_cast<const int*>(params.block_table) + batch_idx_ * params.block_table_batch_stride;
            const int physical_block_idx = block_table_ptr[block_n_idx];
            return physical_block_idx * params.v_batch_stride + kv_head_idx * params.v_head_stride;
        } else {
            return batch_idx_ * params.v_batch_stride + kv_head_idx * params.v_head_stride
                   + block_n_idx * KernelTraits::kBlockN * params.v_row_stride;
        }
    }

    __device__ void* get_o_gmem_ptr(const ForwardParams& params) const {
        size_t offset;
        if (params.num_splits > 1) {
            offset = ((split_idx_ * params.batch + batch_idx_) * params.heads + head_idx_) * params.head_dim;
            return static_cast<void*>(static_cast<float*>(params.oaccum_ptr) + offset);
        } else {
            offset = batch_idx_ * params.o_batch_stride + head_idx_ * params.o_head_stride;
            return static_cast<void*>(static_cast<half*>(params.o_ptr) + offset);
        }
    }

    int actual_seqlen_q;
    int actual_seqlen_k;

    int batch_idx_;
    int head_idx_;
    int n_block_min;
    int n_block_max;
    int split_idx_;
};


template<typename KernelTraits>
class Query {
 public:
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemShape = Shape<Int<KernelTraits::kHeadDim>>;
    using SmemLayout = Layout<SmemShape, Stride<_1>>;
    using SmemTensor = decltype(make_tensor(make_smem_ptr<Element>(nullptr), SmemLayout{}));

    using GmemLayout = SmemLayout;
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), GmemLayout{}));

    explicit __device__ Query(const Context<KernelTraits>& ctx, const ForwardParams& params, Element* smem) {
        const size_t offset = ctx.get_q_gmem_offset(params);
        auto gmem_ptr = static_cast<Element*>(params.q_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), GmemLayout{});

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const Context<KernelTraits>& ctx, const ForwardParams& params) {
        for (unsigned int i = 0; i < size(smem_); i += KernelTraits::kBlockSize * 8) {
            auto idx = i + threadIdx.x * 8;
            if (idx >= size(smem_)) break; // boundary check

            auto dst_ptr = reinterpret_cast<int4*>(&smem(idx));
            auto src_ptr = reinterpret_cast<int4*>(&gmem_(idx));
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(*src_ptr, *dst_ptr, true);
        }
    }

    __forceinline__ __device__ const Element& smem(unsigned int col) const {
        return smem_(col);
    }

    __forceinline__ __device__ Element& smem(unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(col));
    }

    __forceinline__ __device__ const SmemTensor& smem() const {
        return smem_;
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

    explicit __device__ Key(const Context<KernelTraits>& ctx, const ForwardParams& params, Element* smem) {
        const size_t offset = ctx.get_k_gmem_offset(params, 0);
        auto gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.k_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const Context<KernelTraits>& ctx, const ForwardParams& params, int nbidx) {
        set_gmem_address(ctx, params, nbidx);
        
        const int row_offset = nbidx * KernelTraits::kBlockN;
        const int max_row = ctx.actual_seqlen_k;

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
        return smem_(row, col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }

    __device__ const SmemTensor& smem() const {
        return smem_;
    }


 private:
    __forceinline__ __device__ void set_gmem_address(const Context<KernelTraits>& ctx, const ForwardParams& params, int nbidx) {
        const size_t offset = ctx.get_k_gmem_offset(params, nbidx);
        auto gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), gmem_.layout());
    }

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

    __device__ explicit Value(const Context<KernelTraits>& ctx, const ForwardParams& params, Element* smem) {
        const index_t offset = ctx.get_v_gmem_offset(params, 0);
        auto gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.v_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem(const Context<KernelTraits>& ctx, const ForwardParams& params, int nbidx) {
        set_gmem_address(ctx, params, nbidx);
        
        const int row_offset = nbidx * KernelTraits::kBlockN;
        const int max_row = ctx.actual_seqlen_k;
        
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
        return smem_(row, col);
    }

    __forceinline__ __device__ Element& smem(unsigned int row, unsigned int col) {
        return const_cast<Element&>(std::as_const(*this).smem(row, col));
    }

    __device__ const SmemTensor& smem() const {
        return smem_;
    }

 private:

    __forceinline__ __device__ void set_gmem_address(const Context<KernelTraits>& ctx, const ForwardParams& params, int nbidx) {
        const size_t offset = ctx.get_v_gmem_offset(params, nbidx);
        auto gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), gmem_.layout());
    }

    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KernelTraits>
class Score {
 public:
    using ElementAccum = KernelTraits::ElementAccum;
    using Element = KernelTraits::Element;

    // MMA tile dimensions

    // a warp compute Q (1 row) against multiple rows of K (rows_per_warp rows)
    static constexpr int rows_per_warp = KernelTraits::kBlockN / KernelTraits::kNWarps;

    using ScoreTensorLayout = Layout<Shape<Int<rows_per_warp>>, Stride<_1>>;
    using ScoreTensor = decltype(cute::make_tensor<ElementAccum>(ScoreTensorLayout{}));

    __device__ Score(const Query<KernelTraits>& query, const Key<KernelTraits>& key) {
        score_ = cute::make_tensor<ElementAccum>(ScoreTensorLayout{});
        cute::fill(score_, ElementAccum{0});

        const int lane = threadIdx.x % 32;
        const int warp_idx = threadIdx.x / 32;

        for (int n = 0; n < rows_per_warp; n++) {
            int row = n * KernelTraits::kNWarps + warp_idx;
            ElementAccum sum = 0;
            for (int k = 0; k < KernelTraits::kHeadDim; k += 64) {
                int col = k + lane * 2; // each thread loads 2 elements (4 bytes) for the query and key
                if constexpr (std::is_same_v<Element, half_t>) {
                    float2 q2 = __half22float2(*reinterpret_cast<const half2*>(&query.smem(col)));
                    float2 k2 = __half22float2(*reinterpret_cast<const half2*>(&key.smem(row, col)));
                    sum += q2.x * k2.x + q2.y * k2.y;
                } else {
                    float2 q2 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&query.smem(col)));
                    float2 k2 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&key.smem(row, col)));
                    sum += q2.x * k2.x + q2.y * k2.y;
                }
            }
            score_(n) = warp_reduce_sum<ElementAccum, 32>(sum);
        }
    }


    __device__ void copy_to_smem(ElementAccum* smem_ptr) {
        int lane = threadIdx.x % 32;
        if (lane == 0) {
            int warp_idx = threadIdx.x / 32;
            for (int i = 0; i < size(score_); i++) {
                smem_ptr[i * 4 + warp_idx] = score_(i);
            }
        }
    }

    __device__ const ScoreTensor& score() const {
        return score_;
    }

    __device__ __forceinline__ const ElementAccum& operator()(int idx) const {
        return score_(idx);
    }

    __device__ __forceinline__ ElementAccum& operator()(int idx) {
        return score_(idx);
    }

    // If is_causal=true: masks where col > row OR col >= seqlen_k
    // If is_causal=false: masks only where col >= seqlen_k
    // __device__ void set_mask(int row_offset, int col_offset, int seqlen_k, bool is_causal) {
    //     const int warp_idx = threadIdx.x / 32;
    //     const int lane = threadIdx.x % 32;
        
    //     // Each thread handles 2 rows (row and row+8) in a 16x8 tile
    //     const int m_base = row_offset + warp_idx;
        
    //     for (int n = 0; n < 1; n++) {
    //         const int n_base = col_offset;
            
    //         // For each of the 4 elements this thread owns in the tile
    //         // Elements (0,1) are for row m_base + (lane/4)
    //         // Elements (2,3) are for row m_base + (lane/4) + 8
            
    //         int row_0 = m_base + (lane / 4);
    //         int row_1 = m_base + (lane / 4) + 8;
            
    //         // Each element covers 2 columns
    //         int col_0 = n_base + (lane % 4) * 2;
    //         int col_1 = col_0 + 1;
            
    //         // Apply unified mask
    //         // Unified mask: check causal condition OR seqlen condition
    //         if ((is_causal && col_0 > row_0) || col_0 >= seqlen_k) score_(0) = -INFINITY;
    //         if ((is_causal && col_1 > row_0) || col_1 >= seqlen_k) score_(1) = -INFINITY;
    //         if ((is_causal && col_0 > row_1) || col_0 >= seqlen_k) score_(2) = -INFINITY;
    //         if ((is_causal && col_1 > row_1) || col_1 >= seqlen_k) score_(3) = -INFINITY;
    //     }
    // }

 private:
    ScoreTensor score_;
};

template<typename KernelTraits>
class Softmax {
 public:
    using ElementAccum = typename KernelTraits::ElementAccum;
    static constexpr int rows_per_warp = Score<KernelTraits>::rows_per_warp;

    __device__ Softmax() {
        max_ = -INFINITY;
        rescale_ = 1.0f;
        expsum_ = 0;
    }

    __device__ void update(Score<KernelTraits>& score, float softmax_scale_log2) {
        ElementAccum new_max = -INFINITY;
        for (int n = 0; n < size<0>(score.score()); n++) {
            new_max = max(new_max, score(n));
        }

        rescale_ = exp2f((max_ - new_max) * softmax_scale_log2);
        max_ = max(max_, new_max);

        // exponentiate the scores with updated row_max
        const float max_scaled = max_ * softmax_scale_log2;
        float sum = 0.f;
        for (int n = 0; n < size<0>(score.score()); n++) {
            score(n) = exp2f(score(n) * softmax_scale_log2 - max_scaled);
            sum += score(n);
        }
        expsum_ = expsum_ * rescale_ + sum;
    }

    __device__ float expsum() const {
        return expsum_;
    }

    __device__ float rescale() const {
        return rescale_;
    }

    __device__ float rowmax() const {
        return max_;
    }


 private:
    float expsum_;
    float max_;
    float rescale_;
};

template<typename KernelTraits, bool Split>
struct Output {
    using index_t = KernelTraits::index_t;

    constexpr static int d_elements_per_thread = KernelTraits::kHeadDim / 32;
    using RmemShape = Shape<Int<d_elements_per_thread>>;
    using RmemLayout = Layout<RmemShape, Stride<_1>>;
    using RmemTensor = decltype(cute::make_tensor<float>(RmemLayout{}));

    explicit __device__ Output(const Context<KernelTraits>& ctx, const ForwardParams& params) {
        rmem_ = make_tensor<float>(RmemLayout{});
        cute::fill(rmem_, .0f);
    }

    template<typename Tensor0>
    __device__ void accum(float rescale, const Tensor0& score, const Value<KernelTraits>& value) {
        const int warp_idx = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;

        for (int i = 0; i < size(rmem_); i++) {
            rmem_(i) *= rescale;
        }

        constexpr int rows_per_warp = Score<KernelTraits>::rows_per_warp;
        for (int n = 0; n < rows_per_warp; n++) {
            int row = n * KernelTraits::kNWarps + warp_idx;
            float s = score(n);
            for (int d = 0, idx = 0; d < KernelTraits::kHeadDim; d += 32 * 2, idx += 2) {
                int col = d + lane * 2;
                float2 v0;
                if constexpr (std::is_same_v<typename KernelTraits::Element, half_t>) {
                    v0 = __half22float2(*reinterpret_cast<const half2*>(&value.smem(row, col)));
                } else {
                    v0 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&value.smem(row, col)));
                }
                rmem_(idx) += s * v0.x;
                rmem_(idx + 1) += s * v0.y;
            }
        }
    }

    __device__ void normalize(float global_expsum, float rescale) {
        const float inv_expsum = 1.0f / global_expsum;
        for (int i = 0; i < size(rmem_); i++) {
            rmem_(i) = rmem_(i) * rescale * inv_expsum;
        }
    }

    template<typename Tensor0>
    __device__ void copy_smem_to_gmem(const Context<KernelTraits>& ctx, const ForwardParams& params, const Tensor0& smem) {
        using DataType = typename Tensor0::value_type;
        static_assert(std::is_same_v<DataType, std::conditional_t<Split, float, typename KernelTraits::Element>>, "DataType must match expected output type based on Split");

        void* gmem_ptr = ctx.get_o_gmem_ptr(params);
        using GmemShape = Shape<Int<KernelTraits::kHeadDim>>;
        auto layout = make_layout(GmemShape{}, make_stride(_1{}));
        auto gmem = make_tensor(make_gmem_ptr<DataType>(gmem_ptr), layout);

        for (unsigned int i = 0; i < size(smem); i += KernelTraits::kBlockSize * 2) {
            auto idx = i + threadIdx.x * 2;
            if (idx >= size(smem)) break; // boundary check

            if constexpr (std::is_same_v<DataType, float>) {
                *reinterpret_cast<float2*>(&gmem(idx)) = *reinterpret_cast<const float2*>(&smem(idx));
            } else {
                *reinterpret_cast<uint32_t*>(&gmem(idx)) = *reinterpret_cast<const uint32_t*>(&smem(idx));
            }
        }
    }

    template<typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void reduce_sum_warp_out(const Tensor0& warp_out, Tensor1& out) {
        using OutType = typename Tensor1::value_type;
        constexpr int BlockSize = KernelTraits::kBlockSize;

        // sum up all rows
        for (int i = 0; i < size<1>(warp_out); i += BlockSize) {
            int col = i + threadIdx.x;
            if (col >= size<1>(warp_out)) break;
            float sum = 0;
            for (int row = 0; row < size<0>(warp_out); row++) {
                sum += warp_out(row, col);
            }
            out(col) = static_cast<OutType>(sum);
        }
    }


    template<typename Tensor0>
    __device__ void copy_rmem_to_smem(Tensor0& smem) {
        static_assert(std::is_same_v<typename Tensor0::value_type, float>, "Expected smem tensor to be of type float");

        const int lane = threadIdx.x % 32;
        const int warp_idx = threadIdx.x / 32;
        for (int d = 0, idx = 0; d < KernelTraits::kHeadDim; d += 64, idx += 2) {
            int col = d + lane * 2;
            smem(warp_idx, col) = rmem_(idx);
            smem(warp_idx, col + 1) = rmem_(idx + 1);
        }
    }

 private:
    RmemTensor rmem_;
};


} // namespace decode

template<typename KernelTraits, bool Split>
__global__ void flash_attention_fwd_split_kv_kernel(__grid_constant__ const ForwardParams params) {
    using Element = KernelTraits::Element;
    using ElementAccum = typename KernelTraits::ElementAccum;
    using namespace decode;

    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kHeadDim = KernelTraits::kHeadDim;
    
    Context<KernelTraits> ctx(params);

    // init shared memory block tensors
    extern __shared__ char smem_data[];
    Element* smem_ptr = reinterpret_cast<Element*>(smem_data);
    Query<KernelTraits> Q(ctx, params, smem_ptr);

    uint32_t offset = size(typename Query<KernelTraits>::SmemLayout{});
    Key<KernelTraits> K(ctx, params, smem_ptr + offset);
    
    offset += size(typename Key<KernelTraits>::SmemLayout{});
    Value<KernelTraits> V(ctx, params, smem_ptr + offset);

    Output<KernelTraits, Split> O(ctx, params); // reuse the same smem space

    // softmax keep track of row max and exp sum
    Softmax<KernelTraits> softmax{};

    // blocks in N dimension
    const int n_blocks = cute::ceil_div(ctx.actual_seqlen_k, kBlockN);
    

    Q.copy_gmem_to_smem(ctx, params);
    K.copy_gmem_to_smem(ctx, params, ctx.n_block_min);
    cute::cp_async_fence();

    for (int nbidx = ctx.n_block_min; nbidx < ctx.n_block_max; nbidx++) {
        cute::cp_async_wait<0>();
        __syncthreads();  // Ensure K is fully loaded before computing scores

        V.copy_gmem_to_smem(ctx, params, nbidx);
        cute::cp_async_fence();

        Score<KernelTraits> score(Q, K);

        softmax.update(score, params.softmax_scale_log2);

        cute::cp_async_wait<0>();
        __syncthreads();  // Ensure V is fully loaded before accumulation

        if (nbidx + 1 < ctx.n_block_max) {
            K.copy_gmem_to_smem(ctx, params, nbidx + 1);
            cute::cp_async_fence();
        }

        O.accum(softmax.rescale(), score.score(), V);
    }


    // After processing all blocks, we need to do a block-wide reduction to find the global max and exp sum for the softmax normalization.
    ElementAccum * score_smem = reinterpret_cast<ElementAccum*>(smem_data);
    float *warp_max_val = reinterpret_cast<float*>(score_smem); // reuse score_smem for storing per-warp max values
    float *warp_expsum_val = warp_max_val + KernelTraits::kNWarps;

    if (threadIdx.x % 32 == 0) {
        // Each warp writes its local max and exp sum to shared memory
        warp_max_val[threadIdx.x / 32] = softmax.rowmax();
        warp_expsum_val[threadIdx.x / 32] = softmax.expsum();
    }
    __syncthreads();

    float gloabl_max_val = -INFINITY;
    for (int i = 0; i < KernelTraits::kNWarps; i++) {
        gloabl_max_val = max(gloabl_max_val, warp_max_val[i]);
    }
    float gloabl_expsum_val = 0.f;
    for (int i = 0; i < KernelTraits::kNWarps; i++) {
        // Need to rescale each warp's exp sum by the difference between its local max and the global max
        const float rescale = exp2f((warp_max_val[i] - gloabl_max_val) * params.softmax_scale_log2);
        gloabl_expsum_val += warp_expsum_val[i] * rescale;
    }

    // Now we have the global max and exp sum, we can normalize the output.
    const float rescale = exp2f((softmax.rowmax() - gloabl_max_val) * params.softmax_scale_log2);
    O.normalize(gloabl_expsum_val, rescale);

    // Copy the normalized output from register to shared memory for reduction
    using WarpOutLayout = Layout<Shape<Int<KernelTraits::kNWarps>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    Tensor warp_output = make_tensor(make_smem_ptr<float>(smem_ptr), WarpOutLayout{});
    O.copy_rmem_to_smem(warp_output);

    __syncthreads();


    // Perform a rueduce sum across warps to get the final output for this block
    using ReduceSumLayout = Layout<Shape<Int<kHeadDim>>, Stride<_1>>;
    using OutputType = std::conditional_t<Split, float, typename KernelTraits::Element>;
    Tensor reduce_sum_output = make_tensor(make_smem_ptr<OutputType>(smem_ptr), ReduceSumLayout{});
    O.reduce_sum_warp_out(warp_output, reduce_sum_output);

    __syncthreads();


    // Copy the reduced output from shared memory to global memory
    O.copy_smem_to_gmem(ctx, params, reduce_sum_output);

    // write the LSE value to global memory for this split, batch, head
    if (threadIdx.x == 0 && params.num_splits > 1) {
        float lse = gloabl_max_val * params.softmax_scale + __logf(gloabl_expsum_val);

        auto layout = make_layout(make_shape(params.num_splits, params.batch, params.heads), make_stride(params.batch * params.heads, params.heads, _1{}));
        auto lse_tensor = make_tensor(make_gmem_ptr<float>(params.softmax_lseaccum_ptr), layout);
        lse_tensor(ctx.split_idx_, ctx.batch_idx_, ctx.head_idx_) = lse;
    }
}



template<typename KernelTraits>
__global__ void flash_attention_fwd_split_kv_combine_kernel(__grid_constant__ const ForwardParams params) {
    using Element = KernelTraits::Element;
    using namespace decode;
    constexpr int kHeadDim = KernelTraits::kHeadDim;

    const int total_blocks = params.batch * params.heads;
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int linear_idx = head_idx * gridDim.x + batch_idx;


    if (linear_idx >= total_blocks) return;
    

    extern __shared__ float smem_[]; // one max per block (head)
    const int num_splits = params.num_splits;
    
    // Pointers to accumulated outputs and LSE values from each split
    float* lse_accum = static_cast<float*>(params.softmax_lseaccum_ptr);
    float* o_accum = static_cast<float*>(params.oaccum_ptr);
    float* o_output = static_cast<float*>(params.o_ptr);


    auto o_accum_layout = make_layout(make_shape(params.num_splits, params.batch, params.heads, Int<kHeadDim>{}), LayoutRight{});
    auto o_accum_tensor = make_tensor(make_gmem_ptr<float>(o_accum), o_accum_layout);

    auto o_accum_smem_layout = make_layout(make_shape(num_splits, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, _1{}));
    auto o_accum_smem_tensor = make_tensor(make_smem_ptr<float>(smem_), o_accum_smem_layout);

    auto lse_gmem_layout = make_layout(make_shape(params.num_splits, params.batch, params.heads), make_stride(params.batch * params.heads, params.heads, _1{}));
    auto lse_gmem_tensor = make_tensor(make_gmem_ptr<float>(lse_accum), lse_gmem_layout);

    auto lse_smem_layout = make_layout(make_shape(num_splits), make_stride(_1{}));
    auto lse_smem_tensor = make_tensor(make_smem_ptr<float>(smem_ + KernelTraits::kHeadDim * num_splits), lse_smem_layout);

    // copy accumulated outputs from global memory to shared memory for reduction
    for (int split = 0; split < num_splits; split++) {
        for (int d = 0; d < kHeadDim; d += blockDim.x) {
            int col = d + threadIdx.x;
            if (col < kHeadDim) {
                o_accum_smem_tensor(split, col) = o_accum_tensor(split, batch_idx, head_idx, col);
            }
        }
    }

    for (int i = 0; i < num_splits; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if (idx < num_splits) {
            lse_smem_tensor(idx) = lse_gmem_tensor(idx, batch_idx, head_idx);
        }
    }
    __syncthreads();


    float global_lse = 0;
    for (int split = 0; split < num_splits; split++) {
        float split_lse = lse_smem_tensor(split);
        global_lse += expf(split_lse);
    }
    global_lse = __logf(global_lse);


    // compute the final output by summing over splits and applying the softmax normalization using the global LSE
    auto output_layout = make_layout(make_shape(params.batch, params.heads, params.head_dim), make_stride(params.heads * params.head_dim, params.head_dim, 1));
    auto output_tensor = make_tensor(make_gmem_ptr<Element>(o_output), output_layout);
    
    for (int d = 0; d < kHeadDim; d += blockDim.x) {
        int col = d + threadIdx.x;
        float o_accum_sum = 0;
        if (col < kHeadDim) {
            for (int split = 0; split < num_splits; split++) {
                const int row = split;
                float split_lse = lse_smem_tensor(split);
                o_accum_sum += o_accum_smem_tensor(row, col) * expf(split_lse - global_lse);
            }
            output_tensor(batch_idx, head_idx, col) = static_cast<Element>(o_accum_sum);
        }
    }
}

} // namespace mfa
