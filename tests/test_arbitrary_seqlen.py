"""Test arbitrary sequence lengths support."""

import torch
from mini_flash_attention import flash_attn_func


def test_arbitrary_seqlen():
    """Test that the implementation works with non-multiple-of-64 sequence lengths."""
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Test various sequence lengths that are NOT multiples of 64
    test_seqlens = [1, 7, 63, 65, 100, 127, 129, 200, 511, 513, 1000, 2047]
    
    batch_size = 2
    num_heads = 8
    head_dim = 64
    
    print("Testing arbitrary sequence lengths support...")
    print("=" * 80)
    
    for seqlen in test_seqlens:
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        try:
            # Our implementation
            output_ours = flash_attn_func(q, k, v, causal=False)
            
            # PyTorch reference
            q_ref = q.transpose(1, 2).float()
            k_ref = k.transpose(1, 2).float()
            v_ref = v.transpose(1, 2).float()
            
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=False
            ).transpose(1, 2).half()
            
            # Check accuracy
            max_diff = torch.abs(output_ours - output_ref).max().item()
            mean_diff = torch.abs(output_ours - output_ref).mean().item()
            
            # Check for NaN/Inf
            has_nan = torch.isnan(output_ours).any().item()
            has_inf = torch.isinf(output_ours).any().item()
            
            status = "✓" if (max_diff < 0.01 and not has_nan and not has_inf) else "✗"
            
            print(f"seqlen={seqlen:4d}: {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}", end="")
            
            if has_nan:
                print("  [NaN detected!]", end="")
            if has_inf:
                print("  [Inf detected!]", end="")
            
            print()
            
            if max_diff >= 0.01 or has_nan or has_inf:
                print(f"  ERROR: Test failed for seqlen={seqlen}")
                return False
                
        except Exception as e:
            print(f"seqlen={seqlen:4d}: ✗  Exception: {e}")
            return False
    
    print("=" * 80)
    print("✓ All arbitrary sequence length tests passed!")
    return True


def test_causal_arbitrary_seqlen():
    """Test causal attention with arbitrary sequence lengths."""
    device = torch.device('cuda')
    dtype = torch.float16
    
    test_seqlens = [1, 7, 63, 65, 127, 129, 511, 1000]
    
    batch_size = 1
    num_heads = 4
    head_dim = 64
    
    print("\nTesting causal attention with arbitrary sequence lengths...")
    print("=" * 80)
    
    for seqlen in test_seqlens:
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        try:
            # Our implementation
            output_ours = flash_attn_func(q, k, v, causal=True)
            
            # PyTorch reference
            q_ref = q.transpose(1, 2).float()
            k_ref = k.transpose(1, 2).float()
            v_ref = v.transpose(1, 2).float()
            
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=True
            ).transpose(1, 2).half()
            
            max_diff = torch.abs(output_ours - output_ref).max().item()
            mean_diff = torch.abs(output_ours - output_ref).mean().item()
            
            has_nan = torch.isnan(output_ours).any().item()
            has_inf = torch.isinf(output_ours).any().item()
            
            status = "✓" if (max_diff < 0.01 and not has_nan and not has_inf) else "✗"
            
            print(f"seqlen={seqlen:4d} (causal): {status}  max_diff={max_diff:.6f}", end="")
            
            if has_nan or has_inf:
                print("  [NaN/Inf detected!]", end="")
            
            print()
            
            if max_diff >= 0.01 or has_nan or has_inf:
                print(f"  ERROR: Causal test failed for seqlen={seqlen}")
                return False
                
        except Exception as e:
            print(f"seqlen={seqlen:4d} (causal): ✗  Exception: {e}")
            return False
    
    print("=" * 80)
    print("✓ All causal arbitrary sequence length tests passed!")
    return True


if __name__ == "__main__":
    success = test_arbitrary_seqlen()
    success = test_causal_arbitrary_seqlen() and success
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 All tests passed! Arbitrary sequence length support verified.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Some tests failed.")
        print("=" * 80)
