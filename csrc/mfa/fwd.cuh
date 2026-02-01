#pragma once

#include "mfa/flash.h"
#include "mfa/traits.h"
#include "mfa/utils.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
namespace mfa {


#define KERNEL_TRAITS KernelTraits


template<typename KERNEL_TRAITS>
class Query {
 public:
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemShape = Shape<Int<KernelTraits::kBlockM>, Int<KernelTraits::kHeadDim>>;
    using SmemLayout = Layout<SmemShape, Stride<Int<KernelTraits::kHeadDim>, _1>>;
    using SmemTensor = decltype(make_tensor(make_smem_ptr<Element>(nullptr), SmemLayout{}));
    static constexpr index_t smem_offset = 0;
    static constexpr size_t smem_size = size(SmemLayout{});

    using GmemShape = SmemShape;
    using GmemStride = Stride<index_t, _1>; // the row stride is dynamic
    using GmemLayout = Layout<GmemShape, GmemStride>;
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), GmemLayout{}));

    explicit __device__ Query(const ForwardParams &params, void *smem) {
        const size_t offset = blockIdx.z * params.q_batch_stride + blockIdx.y * params.q_head_stride +
                        blockIdx.x * KernelTraits::kBlockM * params.q_row_stride;
        Element* gmem_ptr = static_cast<Element*>(params.q_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.q_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        smem_ = make_tensor(make_smem_ptr<Element>(smem), SmemLayout{});
    }

    __device__ void copy_gmem_to_smem() {
        using namespace cute;
        for (unsigned int i = 0; i < size(smem_); i += blockDim.x) {
            if (auto idx = i + threadIdx.x; idx < size(smem_)) {
                smem_[idx] = gmem_[idx];
            }
        }
    }

    __device__ const SmemTensor& smem() const {
        return smem_;
    }

    __device__ const GmemTensor& gmem() const {
        return gmem_;
    }

private:
    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KERNEL_TRAITS>
class Key {
 public:
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemLayout = KernelTraits::SmemLayoutKV;
    using SmemTensor = decltype(make_tensor(make_smem_ptr<Element>(nullptr), SmemLayout{}));
    static constexpr index_t smem_offset = Query<KernelTraits>::smem_offset + Query<KernelTraits>::smem_size;
    static constexpr size_t smem_size = size(SmemLayout{});

    using GmemShape = Shape<Int<KernelTraits::kBlockN>, Int<KernelTraits::kHeadDim>>;
    using GmemStride = Stride<index_t, _1>; // the row stride
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), make_layout(GmemShape{}, GmemStride{})));

    explicit __device__ Key(const ForwardParams &params, void *smem) {
        const size_t offset = blockIdx.z * params.k_batch_stride +
            (blockIdx.y / params.kv_group_size) * params.k_head_stride;

        Element* gmem_ptr = static_cast<Element*>(params.k_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.k_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        Element * smem_ptr = static_cast<Element*>(smem) + smem_offset;
        smem_ = make_tensor(make_smem_ptr<Element>(smem_ptr), SmemLayout{});
    }

    __device__ void advance(const int row_stride) {
        const size_t offset = KernelTraits::kBlockN * row_stride;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_.data().get() + offset), gmem_.layout());
    }

    __device__ void copy_gmem_to_smem() {
        for (unsigned int i = 0; i < size(smem_); i += blockDim.x) {
            if (auto idx = i + threadIdx.x; idx < size(smem_)) {
                smem_[idx] = gmem_[idx];
            }
        }
    }

    __device__ const SmemTensor& smem() const {
        return smem_;
    }

    __device__ const GmemTensor& gmem() const {
        return gmem_;
    }

private:
    SmemTensor smem_;
    GmemTensor gmem_;
};

template<typename KERNEL_TRAITS>
struct Value {
    using Element = KernelTraits::Element;
    using index_t = KernelTraits::index_t;

    using SmemLayout = Key<KernelTraits>::SmemLayout;
    using SmemTensor = Key<KernelTraits>::SmemTensor;
    static constexpr index_t smem_offset = Key<KernelTraits>::smem_offset + Key<KernelTraits>::smem_size;

    using GmemShape = Key<KernelTraits>::GmemShape;
    using GmemTensor = Key<KernelTraits>::GmemTensor;

    __device__ explicit Value(const ForwardParams& params, Element* smem) {
        const index_t offset = blockIdx.z * params.v_batch_stride +
            (blockIdx.y / params.kv_group_size) * params.v_head_stride;
        Element* gmem_ptr = static_cast<Element*>(params.v_ptr) + offset;
        const auto layout = make_layout(GmemShape{}, make_stride(params.k_row_stride, _1{}));
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_ptr), layout);

        Element * smem_ptr = smem + smem_offset;
        smem_ = make_tensor(make_smem_ptr(smem_ptr), SmemLayout{});
    }

    __device__ void advance(const ForwardParams& params) {
        const size_t offset = KernelTraits::kBlockN * params.k_row_stride;
        gmem_ = make_tensor(make_gmem_ptr<Element>(gmem_.data().get() + offset), gmem_.layout());
    }

    __device__ void copy_gmem_to_smem() {
        using namespace cute;
        for (signed int i = 0; i < size(smem_); i += blockDim.x) {
            if (auto idx = i + threadIdx.x; idx < size(smem_)) {
                smem_[idx] = gmem_[idx];
            }
        }
    }

    __device__ const SmemTensor& smem() const {
        return smem_;
    }

    __device__ const GmemTensor& gmem() const {
        return gmem_;
    }

private:
    SmemTensor smem_;
    GmemTensor gmem_;
};


CUTE_HOST_DEVICE static void
abccopy(void* smem_src,
     uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
}

template<typename KERNEL_TRAITS>
class Score {
 public:
    using ElementAccum = KernelTraits::ElementAccum;
    using Element = KernelTraits::Element;

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

    template<typename Tensor0, typename Tensor1>
    __device__ Score(Tensor0 const& q, Tensor1 const& k, const float scale) {
        // static_assert(size<0>(q) == KernelTraits::kBlockM, "Q row dim must match kBlockM");
        // static_assert(size<1>(q) == KernelTraits::kHeadDim, "Q head dim must match kHeadDim");
        // static_assert(size<0>(k) == KernelTraits::kBlockN, "K row dim must match kBlockN");
        // static_assert(size<1>(k) == KernelTraits::kHeadDim, "K head dim must match kHeadDim");

        score_ = cute::make_tensor<ElementAccum>(ScoreTensorLayout{});

        const int warp_idx = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;

        const int m_smem_idx = warp_idx * MMA_m;
        for (int n_idx = 0; n_idx < n_tiles; n_idx++) {
            const int n_smem_idx = n_idx * MMA_n;

            float c0 = 0.0f;
            float c1 = 0.0f;
            float c2 = 0.0f;
            float c3 = 0.0f;

            for (int k_idx = 0; k_idx < k_tiles; k_idx++) {
                const int k_smem_idx = k_idx * MMA_k;
                uint32_t a0, a1, a2, a3;
                uint32_t b0 = 0, b1 = 0;

                // the address within 16x16 tile of A matrix
                int a_row = m_smem_idx + (lane % 16);
                int a_col = k_smem_idx + (lane / 16) * 8;
                const auto a_addr = reinterpret_cast<uint128_t*>(&q(a_row, a_col));
                cute::SM75_U32x4_LDSM_N::copy(*a_addr, a0, a1, a2, a3);

                // the address within 16x8 tile of B matrix
                int b_row = n_smem_idx + (lane % 8);
                int b_col = k_smem_idx + (lane / 8) * 8;
                const auto b_addr = reinterpret_cast<uint128_t*>(&k(b_row, b_col));
                cute::SM75_U32x2_LDSM_N::copy(*b_addr, b0, b1);

                // Perform MMA operation
                cute::SM80_16x8x16_F32F16F16F32_TN::fma(
                c0, c1, c2, c3,
                    a0, a1, a2, a3,
                    b0, b1,
                    c0, c1, c2, c3
                );
            }

            score_(n_idx, 0) = c0;
            score_(n_idx, 1) = c1;
            score_(n_idx, 2) = c2;
            score_(n_idx, 3) = c3;
        }

        // scale
        for (int i = 0; i < size(score_); i++) {
            score_(i) = score_(i) * static_cast<ElementAccum>(scale);
        }
    }

    template<typename Tensor0>
    __device__ void compute_row_max(Tensor0& row_max) const {
        // static_assert(size<0>(row_max) == 2, "max tensor size must be 2");

        ElementAccum row_0_max = cuda::std::numeric_limits<ElementAccum>::min();
        ElementAccum row_8_max = cuda::std::numeric_limits<ElementAccum>::min();
        for (int n = 0; n < size<1>(score_); n++) {
            row_0_max = max(row_0_max, max(score_(n, 0), score_(n, 1)));
            row_8_max = max(row_8_max, max(score_(n, 2), score_(n, 3)));
        }
        row_max(0) = WarpReduce<4, MaxOp<float>>::run(row_0_max);
        row_max(1) = WarpReduce<4, MaxOp<float>>::run(row_8_max);
    }

    template<typename Tensor0, typename Tensor1>
    __device__ void compute_exp_and_accum_expsum(Tensor0& row_max, Tensor1& exp_sum) {
        // static_assert(size<0>(row_max) == 2, "max tensor size must be 2");
        // static_assert(size<0>(exp_sum) == 2, "exp_sum tensor size must be 2");

        ElementAccum sum = 0;
        for (int n = 0; n < size<0>(score_); n++) {
            score_(n, 0) = expf(static_cast<float>(score_(n, 0) - row_max(0)));
            score_(n, 1) = expf(static_cast<float>(score_(n, 1) - row_max(0)));
            score_(n, 2) = expf(static_cast<float>(score_(n, 2) - row_max(1)));
            score_(n, 3) = expf(static_cast<float>(score_(n, 3) - row_max(1)));
            exp_sum(0) += score_(n, 0);
            exp_sum(0) += score_(n, 1);
            exp_sum(1) += score_(n, 2);
            exp_sum(1) += score_(n, 3);
        }
    }

    __device__ const ScoreTensor& score() const {
        return score_;
    }

private:
    ScoreTensor score_;
};

template<typename KERNEL_TRAITS>
class Softmax {
 public:
    using Element = typename KernelTraits::Element;
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

    __device__ void update(Score<KernelTraits>& score) {
        // compute max per row
        auto new_row_max = cute::make_tensor<ElementAccum>(RowMaxTensorLayout{});
        score.compute_row_max(new_row_max);
        // if (thread0()) {
        //     print_tensor(score.score());
        //     printf("max row before update: %f, %f\n", new_row_max(0), new_row_max(1));
        // }
        // 1. update row_max
        // 2. compute rescale
        // 3. update exp_sum
        #pragma unroll
        for (int i = 0; i < size<0>(row_max_); i++) {
            ElementAccum m = new_row_max(i) > row_max_(i) ? new_row_max(i) : row_max_(i);
            rescale_(i) = expf(row_max_(i) - m);
            row_max_(i) = m;
            expsum_(i) *= static_cast<ElementAccum>(rescale_(i));
        }

        score.compute_exp_and_accum_expsum(row_max_, expsum_);
    }

    __device__ void reduce_sum_expsum() {
        // final reduction across rows
        expsum_(0) = WarpReduce<4, SumOp<float>>::run(expsum_(0));
        expsum_(1) = WarpReduce<4, SumOp<float>>::run(expsum_(1));
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


template<typename KERNEL_TRAITS>
struct Output {
    using Element = KernelTraits::Element;
    using ElementAccum = KernelTraits::ElementAccum;
    using index_t = KernelTraits::index_t;

    static constexpr int m_tiles = KernelTraits::kBlockM / 16;
    static constexpr int n_tiles = KernelTraits::kHeadDim / 8;
    static constexpr int k_tiles = KernelTraits::kBlockN / 16;

    // each thread saves 4 elements per tile
    using RmemLayout = Layout<Shape<Int<n_tiles>, _4>, Stride<_4, _1>>;
    using RmemTensor = decltype(cute::make_tensor<ElementAccum>(RmemLayout{}));

    using GmemShape = Query<KernelTraits>::GmemShape;
    using GmemLayout = Query<KernelTraits>::GmemLayout;
    using GmemTensor = decltype(make_tensor(make_gmem_ptr<Element>(nullptr), GmemLayout{}));

    using SmemLayout = Query<KernelTraits>::SmemLayout;
    using SmemTensor = Query<KernelTraits>::SmemTensor;

    explicit __device__ Output(const ForwardParams &params, Element* smem) {
        rmem_ = make_tensor<ElementAccum>(RmemLayout{});
        cute::fill(rmem_, .0f);

        const index_t offset = blockIdx.z * params.o_batch_stride + blockIdx.y * params.o_head_stride +
                        blockIdx.x * KernelTraits::kBlockM * params.o_row_stride;
        Element *gmem_ptr = static_cast<Element*>(params.o_ptr) + offset;
        gmem_ = make_tensor(make_gmem_ptr(gmem_ptr), make_layout(GmemShape{}, make_stride(params.o_row_stride, _1{})));

        smem_ = make_tensor(make_smem_ptr<Element>(smem), SmemLayout{});
    }

    template<typename Tensor0, typename Tensor1>
    __device__ void accum(Tensor0 const& score, Tensor1 const& v) {
        // static_assert(size<0>(score) == Score<KernelTraits>::n_tiles);
        // static_assert(size<1>(score) == 4);
        // static_assert(size<0>(v) == KernelTraits::kBlockN);
        // static_assert(size<1>(v) == KernelTraits::kHeadDim);

        for (int n = 0; n < n_tiles; n++) {
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (int k = 0; k < k_tiles; k++) {
                // convert float to half
                half2 a0 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k, 0)));
                half2 a1 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k, 2)));
                half2 a2 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 0)));
                half2 a3 = __float22half2_rn(*reinterpret_cast<const float2*>(&score(2 * k + 1, 2)));

                // if (thread0() && k == 0) {
                //     print_tensor(score);
                //     printf("half2 a0: %f, %f\n", __half2float(a0.x), __half2float(a0.y));
                //     printf("half2 a1: %f, %f\n", __half2float(a1.x), __half2float(a1.y));
                //     printf("half2 a2: %f, %f\n", __half2float(a2.x), __half2float(a2.y));
                //     printf("half2 a3: %f, %f\n", __half2float(a3.x), __half2float(a3.y));
                // }

                // half2 to uint32_t
                uint32_t a0_u = *reinterpret_cast<uint32_t*>(&a0);
                uint32_t a1_u = *reinterpret_cast<uint32_t*>(&a1);
                uint32_t a2_u = *reinterpret_cast<uint32_t*>(&a2);
                uint32_t a3_u = *reinterpret_cast<uint32_t*>(&a3);

                int v_row = k * 16 + (threadIdx.x % 16);
                int v_col = n * 8;
                auto addr = reinterpret_cast<uint128_t*>(&v(v_row, v_col));
                uint32_t v_regs[2];
                cute::SM75_U16x4_LDSM_T::copy(*addr, v_regs[0], v_regs[1]);

                cute::SM80_16x8x16_F32F16F16F32_TN::fma(
                    acc0, acc1, acc2, acc3,
                    a0_u, a1_u, a2_u, a3_u,
                    v_regs[0], v_regs[1],
                    acc0, acc1, acc2, acc3
                );
            }

            rmem_(n, 0) += acc0;
            rmem_(n, 1) += acc1;
            rmem_(n, 2) += acc2;
            rmem_(n, 3) += acc3;
        }
    }

    __device__ void rescale(const Softmax<KernelTraits>::RescaleTensor& rescale) {
        for (int n = 0; n < size<0>(rmem_); n++) {
            rmem_(n, 0) *= rescale(0);
            rmem_(n, 1) *= rescale(0);
            rmem_(n, 2) *= rescale(1);
            rmem_(n, 3) *= rescale(1);
        }
    }

    template <typename Tensor>
    __device__ void normalize(const Tensor &expsum) {
        for (int n = 0; n < size<0>(rmem_); n++) {
            rmem_(n, 0) /= expsum(0);
            rmem_(n, 1) /= expsum(0);
            rmem_(n, 2) /= expsum(1);
            rmem_(n, 3) /= expsum(1);
        }
    }

    __device__ void copy_smem_to_gmem() {
        for (unsigned int i = 0; i < size(smem_); i += blockDim.x) {
            auto idx = i + threadIdx.x;
            gmem_[idx] = smem_[idx];
        }
    }

    __forceinline__ __device__ void copy_rmem_to_smem() {
        int wrap_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;

        for (int n = 0; n < n_tiles; n++) {
            // the column within 16x8 tile of output matrix
            int row = wrap_id * 16 + lane / 4;
            int col = n * 8 + (lane % 4) * 2;

            if constexpr (std::is_same_v<Element, half_t>) {
                *reinterpret_cast<half2*>(&smem_(row, col)) = __float22half2_rn(
                    *reinterpret_cast<float2*>(&rmem_(n, 0))
                );
                *reinterpret_cast<half2*>(&smem_(row + 8, col)) = __float22half2_rn(
                    *reinterpret_cast<float2*>(&rmem_(n, 2))
                );
            } else {
                *reinterpret_cast<nv_bfloat162*>(&smem_(row, col)) = __float22bfloat162_rn(
                    *reinterpret_cast<float2*>(&rmem_(n, 0))
                );
                *reinterpret_cast<nv_bfloat162*>(&smem_(row + 8, col)) = __float22bfloat162_rn(
                    *reinterpret_cast<float2*>(&rmem_(n, 2))
                );
            }
        }
    }

    __device__ const GmemTensor& gmem() const {
        return gmem_;
    }

    __device__ const RmemTensor& rmem() const {
        return rmem_;
    }

private:
    RmemTensor rmem_;
    GmemTensor gmem_;
    SmemTensor smem_;
};




template<typename Layout>
__device__ void print_layout(const Layout& layout) {
    using namespace cute;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Layout: ");
        print(layout);
    }
}

template<typename... Args>
__device__ void print_tensors(const Args&... args) {
    using namespace cute;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((print(args.layout()), printf("\n")), ...);
    }
}

template<typename KERNEL_TRAITS>
__global__ void flash_attention_fwd_kernel(__grid_constant__ const ForwardParams params) {
    using Element = KernelTraits::Element;
    // using ElementAccum = KernelTraits::ElementAccum;
    // using index_t = typename KernelTraits::index_t;

    constexpr int kBlockM = KernelTraits::kBlockM;
    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kHeadDim = KernelTraits::kHeadDim;
    constexpr int kNWarps = KernelTraits::kNWarps;

    // init shared memory block tensors
    extern __shared__ __align__(128) uint8_t smem_data[];
    Query<KernelTraits> Q(params, reinterpret_cast<Element*>(smem_data));
    Key<KernelTraits> K(params, reinterpret_cast<Element*>(smem_data));
    Value<KernelTraits> V(params, reinterpret_cast<Element*>(smem_data));
    Output<KernelTraits> O(params, reinterpret_cast<Element*>(smem_data));

    if (cute::thread0()) {
        printf("Q/K/V/O gmem:\n");
        print_tensors(Q.gmem(), K.gmem(), V.gmem(), O.gmem());

        printf("Q/K/V smem:\n");
        print_tensors(Q.smem(), K.smem(), V.smem());

        printf("warps: %d, blockDim.x: %d\n", kNWarps, blockDim.x);
    }

    Q.copy_gmem_to_smem();
    Softmax<KernelTraits> softmax{};

    const int n_blocks = params.seqlen_k / kBlockN;
    for (int nbi = 0; nbi < n_blocks; nbi++) {
        K.copy_gmem_to_smem();
        __syncthreads();

        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;

        Score<KernelTraits> score(Q.smem(), K.smem(), params.softmax_scale);


        softmax.update(score);

        // Tensor aa = make_tensor(make_smem_ptr<Element>(smem_data), Layout<Shape<Int<64>, Int<64>>, Stride<_64, _1>>{});
        // for (int i = 0; i < size<0>(score.score()); i++) {
        //     int row = (warp_id * 16) + (lane / 4);
        //     int col = i * 8 + (lane % 4) * 2;
        //     aa(row, col + 0) = static_cast<Element>(score.score()(i, 0));
        //     aa(row, col + 1) = static_cast<Element>(score.score()(i, 1));
        //     aa(row + 8, col + 0) = static_cast<Element>(score.score()(i, 2));
        //     aa(row + 8, col + 1) = static_cast<Element>(score.score()(i, 3));
        // }

        // __syncthreads();
        // if (thread0()) {
        //     printf("score:==========================\n");
        //     print_tensor(aa);
        // }

        // if (thread0()) {
        //     printf("score before softmax:==========================\n");
        //     print_tensor(score.score());
        // }

        V.copy_gmem_to_smem();
        //
        if (cute::thread0()) {
            print("exp score\n");
            print_tensor(score.score());
            printf("rowmax:\n");
            print_tensor(softmax.row_max());

        }
        //
        O.rescale(softmax.rescale());
        O.accum(score.score(), V.smem());
        //
        K.advance(params.k_row_stride);
        V.advance(params);
    }

    softmax.reduce_sum_expsum();

    if (thread0()) {
        printf("expsum:\n");
        print_tensor(softmax.expsum());
    }

    O.normalize(softmax.expsum());

    O.copy_rmem_to_smem();
    O.copy_smem_to_gmem();
}

template<typename KERNEL_TRAITS>
void compute_attn(const ForwardParams& params, cudaStream_t stream) {
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);

    dim3 grid(m_blocks, params.heads, params.batch);
    dim3 block(KernelTraits::kNThreads);

    flash_attention_fwd_kernel<KernelTraits><<<grid, block, KernelTraits::smem_size, stream>>>(params);
}

} // namespace mfa