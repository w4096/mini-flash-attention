import torch
from mini_flash_attention import flash_attn_with_kvcache
from flash_attn.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_ref

def test_compare_outputs():
    """Compare outputs in different configurations."""
    device = "cuda"
    dtype = torch.float16
    batch_size = 4
    num_heads = 8
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
    
    # Fill cache with SAME random data for all tests
    torch.manual_seed(12345)
    for b in range(batch_size):
        for i in range(num_blocks_per_seq):
            block_idx = block_table[b, i].item()
            start_pos = i * block_size
            end_pos = min(start_pos + block_size, seqlen)
            if start_pos < seqlen:
                length = end_pos - start_pos
                k_cache[block_idx, :length] = torch.randn(length, num_heads, head_dim, device=device, dtype=dtype)
                v_cache[block_idx, :length] = torch.randn(length, num_heads, head_dim, device=device, dtype=dtype)
    
    # Test 1: All batches together - OUR implementation
    print("=== Test 1: All 4 batches together (ours) ===")
    torch.manual_seed(999)
    q_all = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
    cache_seqlens_all = torch.tensor([seqlen] * batch_size, dtype=torch.int32, device=device)
    
    output_ours_all = flash_attn_with_kvcache(
        q_all, k_cache, v_cache,
        cache_seqlens=cache_seqlens_all,
        block_table=block_table,
        causal=False
    )
    print(f"Output shape: {output_ours_all.shape}")
    print(f"Batch 0, head 0, first 8 dims: {output_ours_all[0, 0, 0, :8]}")
    print(f"Batch 1, head 0, first 8 dims: {output_ours_all[1, 0, 0, :8]}")
    
    # Test 2: Batch 0 alone - OUR implementation
    print("\n=== Test 2: Batch 0 alone (ours) ===")
    torch.manual_seed(999)  # Same seed
    q_b0 = q_all[0:1]
    cache_seqlens_b0 = cache_seqlens_all[0:1]
    block_table_b0 = block_table[0:1]
    
    output_ours_b0 = flash_attn_with_kvcache(
        q_b0, k_cache, v_cache,
        cache_seqlens=cache_seqlens_b0,
        block_table=block_table_b0,
        causal=False
    )
    print(f"Output shape: {output_ours_b0.shape}")
    print(f"Batch 0, head 0, first 8 dims: {output_ours_b0[0, 0, 0, :8]}")
    
    # Test 3: All batches together - REFERENCE implementation
    print("\n=== Test 3: All 4 batches together (ref) ===")
    torch.manual_seed(999)  # Same seed
    output_ref_all = flash_attn_with_kvcache_ref(
        q_all, k_cache, v_cache,
        cache_seqlens=cache_seqlens_all,
        block_table=block_table,
        causal=False
    )
    print(f"Output shape: {output_ref_all.shape}")
    print(f"Batch 0, head 0, first 8 dims: {output_ref_all[0, 0, 0, :8]}")
    print(f"Batch 1, head 0, first 8 dims: {output_ref_all[1, 0, 0, :8]}")
    
    # Compare
    print("\n=== Comparisons ===")
    diff_all_b0_ours = torch.abs(output_ours_all[0] - output_ours_b0[0]).max().item()
    print(f"Ours: all[0] vs b0_alone: max_diff={diff_all_b0_ours:.6f}")
    
    diff_ours_ref_all = torch.abs(output_ours_all - output_ref_all).max().item()
    print(f"All batches: ours vs ref: max_diff={diff_ours_ref_all:.6f}")
    
    diff_ours_ref_b0_from_all = torch.abs(output_ours_all[0] - output_ref_all[0]).max().item()
    print(f"Batch 0 from all: ours vs ref: max_diff={diff_ours_ref_b0_from_all:.6f}")

if __name__ == "__main__":
    test_compare_outputs()
