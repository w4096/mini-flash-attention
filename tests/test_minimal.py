import torch
from mini_flash_attention import flash_attn_with_kvcache
from flash_attn.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_ref

def test_minimal():
    """Minimal test case: batch_size=2, 1 head."""
    device = "cuda"
    dtype = torch.float16
    batch_size = 4
    num_heads = 4  # Increase to 4 heads
    head_dim = 128
    block_size = 256
    seqlen = 256
    
    # Allocate cache
    num_blocks_per_seq = 2
    num_blocks = batch_size * num_blocks_per_seq
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # Create block table
    block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for i in range(num_blocks_per_seq):
            block_table[b, i] = b * num_blocks_per_seq + i
    
    print(f"Block table:\n{block_table}")
    
    # Fill cache with SAME random data
    torch.manual_seed(12345)
    for b in range(batch_size):
        block_idx = block_table[b, 0].item()  # Only first block for seqlen=256
        k_cache[block_idx, :seqlen] = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v_cache[block_idx, :seqlen] = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
    
    # Test: All batches together
    print("\n=== All batches together ===")
    torch.manual_seed(999)
    q_all = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor([seqlen] * batch_size, dtype=torch.int32, device=device)
    
    print("Calling mini_flash_attention...")
    output_ours = flash_attn_with_kvcache(
        q_all, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=False
    )
    
    print("Calling flash_attn_ref...")
    output_ref = flash_attn_with_kvcache_ref(
        q_all, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=False
    )
    
    print(f"\nOurs - Batch 0: {output_ours[0, 0, 0, :8]}")
    print(f"Ours - Batch 1: {output_ours[1, 0, 0, :8]}")
    print(f"Ref  - Batch 0: {output_ref[0, 0, 0, :8]}")
    print(f"Ref  - Batch 1: {output_ref[1, 0, 0, :8]}")
    
    for b in range(batch_size):
        diff = torch.abs(output_ours[b] - output_ref[b])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\nBatch {b}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff > 0.02:
            max_idx = diff.argmax()
            max_idx_tuple = torch.unravel_index(max_idx, diff.shape)
            s, h, d = max_idx_tuple[0].item(), max_idx_tuple[1].item(), max_idx_tuple[2].item()
            print(f"  ERROR at (seq={s}, head={h}, dim={d})")
            print(f"    ours: {output_ours[b, s, h, d].item():.6f}")
            print(f"    ref:  {output_ref[b, s, h, d].item():.6f}")
    
    # Test: Each batch alone
    print("\n=== Each batch alone ===")
    for b in range(batch_size):
        torch.manual_seed(999)  # Keep same seed
        q_idx = b  # Use different q for each batch as in the combined case
        q_single = q_all[q_idx:q_idx+1]
        torch.manual_seed(999 + q_idx)  # Adjust seed to match the indexing in q_all
        cache_seqlens_single = cache_seqlens[b:b+1]
        block_table_single = block_table[b:b+1]
        
        output_ours_single = flash_attn_with_kvcache(
            q_single, k_cache, v_cache,
            cache_seqlens=cache_seqlens_single,
            block_table=block_table_single,
            causal=False
        )
        
        output_ref_single = flash_attn_with_kvcache_ref(
            q_single, k_cache, v_cache,
            cache_seqlens=cache_seqlens_single,
            block_table=block_table_single,
            causal=False
        )
        
        max_diff_ours_ref = torch.abs(output_ours_single - output_ref_single).max().item()
        max_diff_ours_vs_all = torch.abs(output_ours_single[0] - output_ours[b]).max().item()
        print(f"Batch {b} solo: ours vs ref={max_diff_ours_ref:.6f}, ours_solo vs ours_all={max_diff_ours_vs_all:.6f}")

if __name__ == "__main__":
    test_minimal()
