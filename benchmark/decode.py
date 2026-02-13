import torch
from mini_flash_attention import flash_attn_func, flash_attn_with_kvcache
from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_reference


device = torch.device("cuda:0")
dtype = torch.float16

head_dim = 128
batch_size = 96
seqlen_q = 1  # Decoding: one token at a time
seqlen_kv = 4096
num_heads = 48

# Query: (batch_size, 1, nheads, headdim)
q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)

k_cache = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
v_cache = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)

cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)

for _ in range(3):
    flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)
    flash_attn_with_kvcache_reference(q, k_cache, v_cache, cache_seqlens=cache_seqlens)

with torch.profiler.profile() as prof:
    output = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
    )
key_averages = prof.key_averages()
print("Mini Flash Attention with KV Cache Profiling Results:")
print(key_averages.table(sort_by="cuda_time_total", row_limit=10))

assert output.shape == q.shape, f"Output shape {output.shape} != input shape {q.shape}"
assert output.dtype == dtype, f"Output dtype {output.dtype} != expected {dtype}"
assert output.device == device, "Output device mismatch"

with torch.profiler.profile() as prof:
    flash_attn_out, lse = flash_attn_with_kvcache_reference(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        return_softmax_lse=True
    )

key_averages = prof.key_averages()
print("Flash Attention with KV Cache Profiling Results:")
print(key_averages.table(sort_by="cuda_time_total", row_limit=10))


max_diff = torch.abs(output - flash_attn_out).max().item()
mean_diff = torch.abs(output - flash_attn_out).mean().item()
print(f"Max difference between mini flash attention and reference: {max_diff:.6f}")
print(f"Mean difference between mini flash attention and reference: {mean_diff:.6f}")
assert max_diff < 0.02, f"Max difference too large: {max_diff:.6f}"
assert mean_diff < 0.002, f"Mean difference too large: {mean_diff:.6f}"
