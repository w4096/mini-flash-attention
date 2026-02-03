#pragma once

#define FP16_SWITCH(COND, ...)                                                                                         \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            using elem_type = cutlass::half_t;                                                                         \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            using elem_type = cutlass::bfloat16_t;                                                                     \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                             \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            constexpr static bool CONST_NAME = true;                                                                   \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            constexpr static bool CONST_NAME = false;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define HEAD_DIM_SWITCH(HEADDIM, ...)                                                                                  \
    [&] {                                                                                                              \
        if (HEADDIM <= 32) {                                                                                           \
            constexpr static int kHeadDim = 32;                                                                        \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 64) {                                                                                    \
            constexpr static int kHeadDim = 64;                                                                        \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 96) {                                                                                    \
            constexpr static int kHeadDim = 96;                                                                        \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 128) {                                                                                   \
            constexpr static int kHeadDim = 128;                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 160) {                                                                                   \
            constexpr static int kHeadDim = 160;                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 192) {                                                                                   \
            constexpr static int kHeadDim = 192;                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 224) {                                                                                   \
            constexpr static int kHeadDim = 224;                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 256) {                                                                                   \
            constexpr static int kHeadDim = 256;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()
