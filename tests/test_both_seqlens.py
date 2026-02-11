import torch
from mini_flash_attention import flash_attn_with_kvcache
from flash_attn.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_ref

def test_both_seqlens():
    """Test both seqlen=256 and 257 with batch_size=4, num_heads=8."""
    device = "cuda"
    dtype = torch.float16
    batch_size = 4
    num_heads = 8
    head_dim = 128
    block_size = 256
    
    for seqlen in [256, 257]:
        print(f"\n{'='*60}")
        print(f"Testing seqlen={seqlen}")
        print('='*60)
        
        # Allocate cache
        num_blocks_per_seq = (seqlen + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_table[b, i] = b * num_blocks_per_seq + i
        
        # Fill cache with random data
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
        
        # Test all batches together
        torch.manual_seed(999)
        q_all = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        cache_seqlens = torch.tensor([seqlen] * batch_size, dtype=torch.int32, device=device)
        
        output_ours = flash_attn_with_kvcache(
            q_all, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        output_ref = flash_attn_with_kvcache_ref(
            q_all, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        # Check each batch
        all_pass = True
        for b in range(batch_size):
            diff = torch.abs(output_ours[b] - output_ref[b])
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            status = "PASS" if max_diff < 0.02 else "FAIL"
            print(f"Batch {b}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
            if max_diff >= 0.02:
                all_pass = False
                # Find max difference location
                max_idx = diff.argmax()
                max_idx_tuple = torch.unravel_index(max_idx, diff.shape)
                s, h, d = max_idx_tuple[0].item(), max_idx_tuple[1].item(), max_idx_tuple[2].item()
                print(f"  ERROR at (seq={s}, head={h}, dim={d})")
                print(f"    ours: {output_ours[b, s, h, d].item():.6f}")
                print(f"    ref:  {output_ref[b, s, h, d].item():.6f}")
        
        print(f"\nOverall: {'✓ ALL PASS' if all_pass else '✗ FAILED'}")

if __name__ == "__main__":
    test_both_seqlens()
