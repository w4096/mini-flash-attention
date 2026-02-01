#include <cstdint>
#include <cuda_runtime.h>

namespace mfa {

template<typename T>
struct MaxOp {
    __device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
    // This is slightly faster
    __device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
    __device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS, typename Operator>
struct WarpReduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T>
    static __device__ inline T run(T x) {
        for (int offset = THREADS / 2; offset > 0; offset /= 2) {
            x = Operator()(x, __shfl_xor_sync(0xffffffff, x, offset));
        }
        return x;
    }
};

}