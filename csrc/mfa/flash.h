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
    int window_size_left;
    int window_size_right;

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

    // For variable-length sequences (continuous batching)
    int* __restrict__ cu_seqlens_q;  // cumulative sequence lengths for Q [batch+1]
    int* __restrict__ cu_seqlens_k;  // cumulative sequence lengths for K [batch+1]
    int max_seqlen_q;  // maximum sequence length in the batch for Q
    int max_seqlen_k;  // maximum sequence length in the batch for K


    void * __restrict__ block_table; // optional block table for flash attention with KV caching
    int block_table_batch_stride;
    int page_block_size;
    size_t k_cache_block_stride; // stride between blocks in the KV cache
    size_t v_cache_block_stride;


    int* __restrict__ seqlens_k;

    int num_splits; // number of splits for KV caching

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    void * __restrict__ oaccum_ptr;
};


void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream);
void run_flash_attention_with_kv_cache(ForwardParams& params, cudaStream_t stream);


}