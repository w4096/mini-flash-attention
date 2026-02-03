#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace mfa {

struct ForwardParams {
    using index_t = int64_t;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;

    index_t q_batch_stride;
    index_t q_head_stride;
    index_t q_row_stride;

    index_t k_batch_stride;
    index_t k_head_stride;
    index_t k_row_stride;

    index_t v_batch_stride;
    index_t v_head_stride;
    index_t v_row_stride;

    index_t o_batch_stride;
    index_t o_head_stride;
    index_t o_row_stride;

    bool is_causal;

    int heads;          // number of heads
    int kv_heads;       // number of key/value heads
    int kv_group_size;  // ratio of heads / kv_heads

    int batch;  // batch size
    int seqlen_q;
    int seqlen_k;

    int head_dim;
    float softmax_scale;
    float softmax_scale_log2;

    bool is_bf16;

    void *reserved[8];
};


void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream);


}