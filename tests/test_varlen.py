"""
Test variable-length (continuous batching) input support
"""
import torch
from mini_flash_attention import flash_attn_varlen_func as mini_flash_attn_varlen_func
from flash_attn import flash_attn_varlen_func

def create_varlen_batch(seqlens_q, seqlens_k, q_heads, kv_heads, dim, dtype=torch.half):
    """
    Create variable-length batch with different sequence lengths
    
    Args:
        seqlens_q: list of query sequence lengths for each batch
        seqlens_k: list of key sequence lengths for each batch
        q_heads: number of query heads
        kv_heads: number of key/value heads
        dim: head dimension
    
    Returns:
        q, k, v: [total_tokens, heads, dim]
        cu_seqlens_q, cu_seqlens_k: cumulative sequence lengths
        max_seqlen_q, max_seqlen_k: maximum sequence lengths
    """
    batch_size = len(seqlens_q)
    assert len(seqlens_k) == batch_size
    
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    
    # Create tensors in varlen format: [total_tokens, heads, dim]
    q = torch.randn((total_q, q_heads, dim), device='cuda', dtype=dtype)
    k = torch.randn((total_k, kv_heads, dim), device='cuda', dtype=dtype)
    v = torch.randn((total_k, kv_heads, dim), device='cuda', dtype=dtype)
    
    # Normalize for better numerical stability
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    
    return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


def test_varlen_basic():
    """Test basic variable-length input with uniform sequence lengths"""
    print("\n=== Test 1: Basic varlen with uniform lengths ===")
    
    # Create batch with same sequence length (should match fixed-length behavior)
    seqlens_q = [512, 512, 512, 512]
    seqlens_k = [512, 512, 512, 512]
    q_heads = 8
    kv_heads = 8
    dim = 64
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Q shape: {q.shape}, cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"K shape: {k.shape}, cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"Max seqlen - Q: {max_seqlen_q}, K: {max_seqlen_k}")
    
    # Run mini flash attention (varlen)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    print(f"Output shape: {mini_out.shape}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_different_lengths():
    """Test variable-length input with different sequence lengths per batch"""
    print("\n=== Test 2: Varlen with different lengths ===")
    
    # Create batch with different sequence lengths (continuous batching scenario)
    seqlens_q = [128, 256, 512, 1024, 64]
    seqlens_k = [128, 256, 512, 1024, 64]
    q_heads = 12
    kv_heads = 12
    dim = 128
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Batch seqlens_q: {seqlens_q}")
    print(f"Batch seqlens_k: {seqlens_k}")
    print(f"Total tokens - Q: {q.shape[0]}, K: {k.shape[0]}")
    print(f"cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"cu_seqlens_k: {cu_seqlens_k.tolist()}")
    
    # Run mini flash attention (varlen)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_causal():
    """Test variable-length input with causal masking"""
    print("\n=== Test 3: Varlen with causal masking ===")
    
    seqlens_q = [256, 512, 128, 1024]
    seqlens_k = [256, 512, 128, 1024]
    q_heads = 16
    kv_heads = 16
    dim = 64
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Batch sizes: {seqlens_q}")
    print(f"Testing with causal=True")
    
    # Run mini flash attention (varlen + causal)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True
    )
    
    # Run reference flash attention (varlen + causal)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_gqa():
    """Test variable-length input with Grouped Query Attention (GQA)"""
    print("\n=== Test 4: Varlen with GQA ===")
    
    seqlens_q = [200, 400, 600]
    seqlens_k = [200, 400, 600]
    q_heads = 24
    kv_heads = 8  # GQA: q_heads / kv_heads = 3
    dim = 128
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Q heads: {q_heads}, KV heads: {kv_heads} (GQA ratio: {q_heads // kv_heads})")
    print(f"Batch sizes: {seqlens_q}")
    
    # Run mini flash attention (varlen + GQA)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen + GQA)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_short_sequences():
    """Test variable-length input with very short sequences"""
    print("\n=== Test 5: Varlen with short sequences ===")
    
    # Short sequences to test edge cases
    seqlens_q = [16, 32, 48, 8]
    seqlens_k = [16, 32, 48, 8]
    q_heads = 8
    kv_heads = 8
    dim = 64
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Short sequences: {seqlens_q}")
    
    # Run mini flash attention (varlen)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_large_batch():
    """Test variable-length input with large batch size"""
    print("\n=== Test 6: Varlen with large batch ===")
    
    # Simulate continuous batching with many requests
    import random
    random.seed(42)
    batch_size = 16
    seqlens_q = [random.randint(64, 512) for _ in range(batch_size)]
    seqlens_k = seqlens_q.copy()
    q_heads = 8
    kv_heads = 8
    dim = 64
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Total tokens: {q.shape[0]}")
    print(f"Seqlen range: [{min(seqlens_q)}, {max(seqlens_q)}]")
    
    # Run mini flash attention (varlen)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out - flash_out).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 0.1, f"Difference too large: {diff}"
    print("✓ Test passed")


def test_varlen_bf16():
    """Test variable-length input with bfloat16 dtype"""
    print("\n=== Test 7: Varlen with bfloat16 ===")
    
    seqlens_q = [128, 256, 512]
    seqlens_k = [128, 256, 512]
    q_heads = 8
    kv_heads = 8
    dim = 128
    
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = create_varlen_batch(
        seqlens_q, seqlens_k, q_heads, kv_heads, dim, dtype=torch.bfloat16
    )
    
    print(f"Testing with dtype: bfloat16")
    print(f"Batch sizes: {seqlens_q}")
    
    # Run mini flash attention (varlen)
    mini_out = mini_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    # Run reference flash attention (varlen)
    flash_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False
    )
    
    diff = torch.abs(mini_out.float() - flash_out.float()).max().item()
    print(f"Max difference: {diff:.6e}")
    
    assert diff < 1e-2, f"Difference too large: {diff}"  # BF16 has lower precision
    print("✓ Test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Variable-Length Input Support")
    print("=" * 60)
    
    try:
        test_varlen_basic()
        test_varlen_different_lengths()
        test_varlen_causal()
        test_varlen_gqa()
        test_varlen_short_sequences()
        test_varlen_large_batch()
        test_varlen_bf16()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! (causal mode skipped)")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        raise
