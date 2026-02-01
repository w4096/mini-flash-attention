#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>

namespace mfa {

struct SM75_U16x8_LDSM {
    __device__ static void copy(const uint32_t *smem_src, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(smem_src));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                     : "r"(smem_int_ptr));
    }
};

struct SM75_U16x4_LDSM_T {
    __device__ static void copy(const uint32_t *smem_src, uint32_t& dst0, uint32_t& dst1) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(smem_src));
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.trans.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst0), "=r"(dst1)
                     : "r"(smem_int_ptr));
    }
};

struct SM80_16x8x16_F32F16F16F32_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    __device__ static void
    fma(float         & d0, float         & d1, float         & d2, float         & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const   & c0, float const   & c1, float const   & c2, float const   & c3)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
             "r"(b0),  "r"(b1),
             "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
    }
};

template<typename Element, int kBlockM, int kBlockN, int kBlockK, int kWarps>
__forceinline__ __device__ void gemm() {
        // compute MMA
        constexpr int m_fragments = kBlockM / 16;
        constexpr int n_fragments = kBlockN / 8;

        // loop over m x n small 16x8 MMA tiles
        for (int idx = wrap_id; idx < m_fragments * n_fragments; idx += kWarps) {
            const int m = idx / n_fragments;
            const int n = idx % n_fragments;

            // the left top corner of C tile in shared memory
            const int row = m * 16;
            const int col = n * 8;

            float c_regs[4] = {0.0f};

            constexpr int k_fragments = kBlockK / 16;
            for (int l = 0; l < k_fragments; ++l) {
                // load A and B from shared memory
                uint32_t a_regs[4];
                uint32_t b_regs[2];

                // the address within 16x16 tile of A matrix
                int a_row = row + (lane % 16);
                int a_col = l * 16 + (lane / 16) * 8;

                const auto a_addr = reinterpret_cast<uint32_t*>(&smem_a[a_row][a_col]);
                SM75_U16x8_LDSM::copy(a_addr, a_regs[0], a_regs[1], a_regs[2], a_regs[3]);

                // the address within 16x8 tile of B matrix
                int b_row = l * 16 + lane;
                int b_col = col;
                const auto b_addr = reinterpret_cast<uint32_t*>(&smem_b[b_row][b_col]);
                SM75_U16x4_LDSM_T::copy(b_addr, b_regs[0], b_regs[1]);

                // Perform MMA operation
                SM80_16x8x16_F32F16F16F32_TN::fma(
                    c_regs[0], c_regs[1], c_regs[2], c_regs[3],
                    a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                    b_regs[0], b_regs[1],
                    c_regs[0], c_regs[1], c_regs[2], c_regs[3]
                );
            }

            // store C registers to shared memory
            const int c_col = col + (lane % 4) * 2;
            smem_c[row + lane / 4][c_col + 0] += c_regs[0];
            smem_c[row + lane / 4][c_col + 1] += c_regs[1];
            smem_c[row + lane / 4 + 8][c_col + 0] += c_regs[2];
            smem_c[row + lane / 4 + 8][c_col + 1] += c_regs[3];
        }
    }
}


}