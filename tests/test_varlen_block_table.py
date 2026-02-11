"""
Test variable-length (continuous batching) input support with block table (paged KV cache)

This tests the flash_attn_varlen_func with block_table parameter, which enables
non-contiguous KV cache storage similar to vLLM's PagedAttention mechanism.
"""
import torch
from mini_flash_attention import flash_attn_varlen_func


def create_paged_kv_cache(
    seqlens_k,
    kv_heads,
    dim,
    block_size,
    dtype=torch.half
):
    """
    Create paged KV cache and block table for variable-length sequences.
    
    Args:
        seqlens_k: list of key/value sequence lengths for each batch
        kv_heads: number of key/value heads
        dim: head dimension
        block_size: number of tokens per block
        dtype: data type
    
    Returns:
        k_cache: [num_blocks, block_size, kv_heads, dim] - paged KV cache
        v_cache: [num_blocks, block_size, kv_heads, dim] - paged KV cache
        block_table: [batch, max_num_blocks_per_seq] - block mapping
        k_contiguous: [total_k, kv_heads, dim] - contiguous reference
        v_contiguous: [total_k, kv_heads, dim] - contiguous reference
    """
    batch_size = len(seqlens_k)
    
    # Calculate required blocks
    max_blocks_per_seq = max((seqlen + block_size - 1) // block_size for seqlen in seqlens_k)
    total_blocks = sum((seqlen + block_size - 1) // block_size for seqlen in seqlens_k)
    
    # Create paged cache
    k_cache = torch.randn(total_blocks, block_size, kv_heads, dim, device='cuda', dtype=dtype)
    v_cache = torch.randn(total_blocks, block_size, kv_heads, dim, device='cuda', dtype=dtype)
    
    # Normalize for numerical stability
    k_cache = torch.nn.functional.normalize(k_cache, dim=-1)
    v_cache = torch.nn.functional.normalize(v_cache, dim=-1)
    
    # Create block table
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device='cuda')
    
    # Fill block table and create contiguous reference
    total_k = sum(seqlens_k)
    k_contiguous = torch.zeros(total_k, kv_heads, dim, device='cuda', dtype=dtype)
    v_contiguous = torch.zeros(total_k, kv_heads, dim, device='cuda', dtype=dtype)
    
    block_idx = 0
    token_offset = 0
    
    for b in range(batch_size):
        seqlen = seqlens_k[b]
        num_blocks = (seqlen + block_size - 1) // block_size
        
        for i in range(num_blocks):
            block_table[b, i] = block_idx
            
            # Copy data to contiguous reference
            start_token = i * block_size
            end_token = min(start_token + block_size, seqlen)
            length = end_token - start_token
            
            k_contiguous[token_offset:token_offset + length] = k_cache[block_idx, :length]
            v_contiguous[token_offset:token_offset + length] = v_cache[block_idx, :length]
            
            token_offset += length
            block_idx += 1
    
    return k_cache, v_cache, block_table, k_contiguous, v_contiguous


def test_varlen_block_table_basic():
    """Test basic varlen with block table - uniform sequence lengths"""
    print("\n" + "=" * 70)
    print("Test 1: Basic varlen with block table (uniform lengths)")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    seqlens_q = [1, 1, 1, 1]  # Decoding scenario
    seqlens_k = [128, 128, 128, 128]
    q_heads = 8
    kv_heads = 8
    dim = 64
    block_size = 16
    
    print(f"\nConfig: batch={batch_size}, q_heads={q_heads}, kv_heads={kv_heads}, dim={dim}")
    print(f"Seqlens Q: {seqlens_q}")
    print(f"Seqlens K: {seqlens_k}")
    print(f"Block size: {block_size}")
    
    # Create query in varlen format
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, q_heads, dim, device='cuda', dtype=torch.half)
    q = torch.nn.functional.normalize(q, dim=-1)
    
    # Create paged KV cache
    k_cache, v_cache, block_table, k_contiguous, v_contiguous = create_paged_kv_cache(
        seqlens_k, kv_heads, dim, block_size
    )
    
    print(f"\nK cache shape (paged): {k_cache.shape}")
    print(f"V cache shape (paged): {v_cache.shape}")
    print(f"Block table shape: {block_table.shape}")
    print(f"K contiguous shape: {k_contiguous.shape}")
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    # Run with block table (paged KV cache)
    output_paged = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        block_table=block_table
    )
    
    print(f"\nResults:")
    print(f"  Output shape: {output_paged.shape}")
    print(f"  Output dtype: {output_paged.dtype}")
    print(f"  No NaN values: {not torch.isnan(output_paged).any()}")
    print(f"  No Inf values: {not torch.isinf(output_paged).any()}")
    print(f"  Output range: [{output_paged.min().item():.3f}, {output_paged.max().item():.3f}]")
    
    if not torch.isnan(output_paged).any() and not torch.isinf(output_paged).any():
        print("  ✓ Test PASSED")
        return True
    else:
        print("  ✗ Test FAILED - NaN or Inf detected")
        return False


def test_varlen_block_table_different_lengths():
    """Test varlen with block table - different sequence lengths"""
    print("\n" + "=" * 70)
    print("Test 2: Varlen with block table (different lengths)")
    print("=" * 70)
    
    # Configuration with different sequence lengths
    batch_size = 5
    seqlens_q = [1, 1, 8, 1, 16]  # Mixed decoding and prefill
    seqlens_k = [64, 128, 200, 96, 256]
    q_heads = 8
    kv_heads = 2  # Test GQA
    dim = 128
    block_size = 32
    
    print(f"\nConfig: batch={batch_size}, q_heads={q_heads}, kv_heads={kv_heads}, dim={dim}")
    print(f"Seqlens Q: {seqlens_q}")
    print(f"Seqlens K: {seqlens_k}")
    print(f"Block size: {block_size}")
    
    # Create query in varlen format
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, q_heads, dim, device='cuda', dtype=torch.half)
    q = torch.nn.functional.normalize(q, dim=-1)
    
    # Create paged KV cache
    k_cache, v_cache, block_table, k_contiguous, v_contiguous = create_paged_kv_cache(
        seqlens_k, kv_heads, dim, block_size
    )
    
    print(f"\nK cache shape (paged): {k_cache.shape}")
    print(f"V cache shape (paged): {v_cache.shape}")
    print(f"Block table shape: {block_table.shape}")
    
    # Show block allocation
    for b in range(batch_size):
        num_blocks = (seqlens_k[b] + block_size - 1) // block_size
        blocks = block_table[b, :num_blocks].tolist()
        print(f"  Seq {b} (len={seqlens_k[b]:3d}): uses {num_blocks} blocks {blocks}")
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    # Run with block table (paged KV cache)
    output_paged = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        block_table=block_table
    )
    
    print(f"\nResults:")
    print(f"  Output shape: {output_paged.shape}")
    print(f"  Expected shape: ({total_q}, {q_heads}, {dim})")
    print(f"  Output range: [{output_paged.min().item():.3f}, {output_paged.max().item():.3f}]")
    print(f"  No NaN: {not torch.isnan(output_paged).any()}, No Inf: {not torch.isinf(output_paged).any()}")
    
    # Basic sanity checks
    if (output_paged.shape == (total_q, q_heads, dim) and 
        not torch.isnan(output_paged).any() and 
        not torch.isinf(output_paged).any()):
        print("  ✓ Test PASSED")
        return True
    else:
        print("  ✗ Test FAILED")
        return False


def test_varlen_block_table_causal():
    """Test varlen with block table and causal attention"""
    print("\n" + "=" * 70)
    print("Test 3: Varlen with block table (causal attention)")
    print("=" * 70)
    
    # Configuration
    batch_size = 3
    seqlens_q = [1, 1, 1]  # Decoding
    seqlens_k = [100, 200, 150]
    q_heads = 8
    kv_heads = 8
    dim = 64
    block_size = 16
    
    print(f"\nConfig: batch={batch_size}, q_heads={q_heads}, kv_heads={kv_heads}, dim={dim}")
    print(f"Seqlens Q: {seqlens_q}")
    print(f"Seqlens K: {seqlens_k}")
    print(f"Block size: {block_size}")
    print(f"Causal: True")
    
    # Create query in varlen format
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, q_heads, dim, device='cuda', dtype=torch.half)
    q = torch.nn.functional.normalize(q, dim=-1)
    
    # Create paged KV cache
    k_cache, v_cache, block_table, k_contiguous, v_contiguous = create_paged_kv_cache(
        seqlens_k, kv_heads, dim, block_size
    )
    
    print(f"\nK cache shape (paged): {k_cache.shape}")
    print(f"Block table shape: {block_table.shape}")
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    # Run with block table (paged KV cache) - causal
    output_paged = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        block_table=block_table
    )
    
    print(f"\nResults:")
    print(f"  Output shape: {output_paged.shape}")
    print(f"  Output range: [{output_paged.min().item():.3f}, {output_paged.max().item():.3f}]")
    print(f"  No NaN: {not torch.isnan(output_paged).any()}, No Inf: {not torch.isinf(output_paged).any()}")
    
    if not torch.isnan(output_paged).any() and not torch.isinf(output_paged).any():
        print("  ✓ Test PASSED")
        return True
    else:
        print("  ✗ Test FAILED")
        return False


def test_varlen_block_table_large_sequences():
    """Test varlen with block table - larger sequences"""
    print("\n" + "=" * 70)
    print("Test 4: Varlen with block table (large sequences)")
    print("=" * 70)
    
    # Configuration with larger sequences
    batch_size = 4
    seqlens_q = [1, 8, 1, 16]
    seqlens_k = [512, 768, 1024, 256]
    q_heads = 8
    kv_heads = 4  # Test MQA/GQA
    dim = 128
    block_size = 64
    
    print(f"\nConfig: batch={batch_size}, q_heads={q_heads}, kv_heads={kv_heads}, dim={dim}")
    print(f"Seqlens Q: {seqlens_q}")
    print(f"Seqlens K: {seqlens_k}")
    print(f"Block size: {block_size}")
    
    # Create query in varlen format
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, q_heads, dim, device='cuda', dtype=torch.half)
    q = torch.nn.functional.normalize(q, dim=-1)
    
    # Create paged KV cache
    k_cache, v_cache, block_table, k_contiguous, v_contiguous = create_paged_kv_cache(
        seqlens_k, kv_heads, dim, block_size
    )
    
    print(f"\nK cache shape (paged): {k_cache.shape}")
    print(f"V cache shape (paged): {v_cache.shape}")
    print(f"Block table shape: {block_table.shape}")
    
    # Calculate memory savings
    max_seqlen_k_padded = max(seqlens_k)
    traditional_memory = batch_size * max_seqlen_k_padded * kv_heads * dim * 2 * 2  # K+V, fp16
    total_blocks = k_cache.size(0)
    paged_memory = total_blocks * block_size * kv_heads * dim * 2 * 2
    savings = (1 - paged_memory / traditional_memory) * 100
    
    print(f"\nMemory comparison:")
    print(f"  Traditional (padded): {traditional_memory / 1024**2:.2f} MB")
    print(f"  Paged (actual):       {paged_memory / 1024**2:.2f} MB")
    print(f"  Savings:              {savings:.1f}%")
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    # Run with block table (paged KV cache)
    output_paged = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        block_table=block_table
    )
    
    print(f"\nResults:")
    print(f"  Output shape: {output_paged.shape}")
    print(f"  Expected shape: ({total_q}, {q_heads}, {dim})")
    print(f"  Output range: [{output_paged.min().item():.3f}, {output_paged.max().item():.3f}]")
    print(f"  No NaN: {not torch.isnan(output_paged).any()}, No Inf: {not torch.isinf(output_paged).any()}")
    
    if (output_paged.shape == (total_q, q_heads, dim) and
        not torch.isnan(output_paged).any() and 
        not torch.isinf(output_paged).any()):
        print("  ✓ Test PASSED")
        return True
    else:
        print("  ✗ Test FAILED")
        return False


def test_varlen_block_table_non_contiguous_blocks():
    """Test varlen with block table - non-contiguous block allocation"""
    print("\n" + "=" * 70)
    print("Test 5: Varlen with block table (non-contiguous blocks)")
    print("=" * 70)
    
    # Configuration
    batch_size = 3
    seqlens_q = [1, 1, 1]
    seqlens_k = [48, 64, 80]
    q_heads = 8
    kv_heads = 8
    dim = 64
    block_size = 16
    
    print(f"\nConfig: batch={batch_size}, q_heads={q_heads}, kv_heads={kv_heads}, dim={dim}")
    print(f"Seqlens Q: {seqlens_q}")
    print(f"Seqlens K: {seqlens_k}")
    print(f"Block size: {block_size}")
    
    # Create query
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, q_heads, dim, device='cuda', dtype=torch.half)
    q = torch.nn.functional.normalize(q, dim=-1)
    
    # Create paged KV cache with manually assigned non-contiguous blocks
    total_blocks = sum((seqlen + block_size - 1) // block_size for seqlen in seqlens_k)
    # Add extra blocks for non-contiguous allocation
    num_cache_blocks = 25  # Enough for blocks 0-20
    k_cache = torch.randn(num_cache_blocks, block_size, kv_heads, dim, device='cuda', dtype=torch.half)
    v_cache = torch.randn(num_cache_blocks, block_size, kv_heads, dim, device='cuda', dtype=torch.half)
    k_cache = torch.nn.functional.normalize(k_cache, dim=-1)
    v_cache = torch.nn.functional.normalize(v_cache, dim=-1)
    
    # Create non-contiguous block table (simulating realistic allocation)
    max_blocks_per_seq = max((seqlen + block_size - 1) // block_size for seqlen in seqlens_k)
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device='cuda')
    
    # Manually assign non-contiguous blocks
    # Seq 0: blocks [0, 5, 10]
    # Seq 1: blocks [1, 6, 11, 15]
    # Seq 2: blocks [2, 7, 12, 16, 20]
    block_assignments = [
        [0, 5, 10],
        [1, 6, 11, 15],
        [2, 7, 12, 16, 20]
    ]
    
    for b in range(batch_size):
        for i, block_id in enumerate(block_assignments[b]):
            block_table[b, i] = block_id
    
    print(f"\nBlock allocation (non-contiguous):")
    for b in range(batch_size):
        num_blocks = len(block_assignments[b])
        print(f"  Seq {b}: {block_assignments[b]}")
    
    # Create contiguous reference
    total_k = sum(seqlens_k)
    k_contiguous = torch.zeros(total_k, kv_heads, dim, device='cuda', dtype=torch.half)
    v_contiguous = torch.zeros(total_k, kv_heads, dim, device='cuda', dtype=torch.half)
    
    token_offset = 0
    for b in range(batch_size):
        seqlen = seqlens_k[b]
        num_blocks = (seqlen + block_size - 1) // block_size
        
        for i in range(num_blocks):
            block_id = block_table[b, i].item()
            start_token = i * block_size
            end_token = min(start_token + block_size, seqlen)
            length = end_token - start_token
            
            k_contiguous[token_offset:token_offset + length] = k_cache[block_id, :length]
            v_contiguous[token_offset:token_offset + length] = v_cache[block_id, :length]
            
            token_offset += length
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    # Run with block table (non-contiguous blocks)
    output_paged = flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        block_table=block_table
    )
    
    print(f"\nResults:")
    print(f"  Output shape: {output_paged.shape}")
    print(f"  Expected shape: ({total_q}, {q_heads}, {dim})")
    print(f"  Output range: [{output_paged.min().item():.3f}, {output_paged.max().item():.3f}]")
    print(f"  No NaN: {not torch.isnan(output_paged).any()}, No Inf: {not torch.isinf(output_paged).any()}")
    
    # Test specifically that non-contiguous block allocation works
    if (output_paged.shape == (total_q, q_heads, dim) and
        not torch.isnan(output_paged).any() and 
        not torch.isinf(output_paged).any()):
        print("  ✓ Test PASSED - Non-contiguous block table works correctly")
        return True
    else:
        print("  ✗ Test FAILED")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing flash_attn_varlen_func with block table (paged KV cache)")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Basic uniform lengths", test_varlen_block_table_basic()))
    results.append(("Different lengths", test_varlen_block_table_different_lengths()))
    results.append(("Causal attention", test_varlen_block_table_causal()))
    results.append(("Large sequences", test_varlen_block_table_large_sequences()))
    results.append(("Non-contiguous blocks", test_varlen_block_table_non_contiguous_blocks()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
    print("=" * 70)
