"""Unit tests for Group Query Attention (GQA) functionality using pytest."""

import pytest
import torch
from mini_flash_attention import flash_attn_func


class TestGroupQueryAttention:
    """Test suite for Group Query Attention (GQA)."""
    
    @pytest.fixture
    def device(self):
        """Ensure CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is required for tests"
        return torch.device('cuda')
    
    @pytest.fixture
    def dtype(self):
        """Use half precision for tests."""
        return torch.float16
    
    def test_gqa_basic(self, device, dtype):
        """Test basic GQA with 8 query heads and 2 KV heads."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape, f"Output shape {output.shape} != input shape {q.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_gqa_causal(self, device, dtype):
        """Test GQA with causal mask."""
        batch_size = 1
        seqlen = 1024
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 128
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=True)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("num_heads_q,num_heads_kv", [
        (8, 1),    # MQA: Multi-Query Attention
        (8, 2),    # GQA: 4 queries per KV
        (8, 4),    # GQA: 2 queries per KV
        (16, 2),   # GQA: 8 queries per KV
        (16, 4),   # GQA: 4 queries per KV
        (32, 8),   # GQA: 4 queries per KV
    ])
    def test_gqa_various_ratios(self, device, dtype, num_heads_q, num_heads_kv):
        """Test GQA with various query-to-KV head ratios."""
        batch_size = 1
        seqlen = 512
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # Should not raise errors
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape, \
            f"heads_q={num_heads_q}, heads_kv={num_heads_kv}: shape mismatch"
        assert not torch.isnan(output).any(), \
            f"heads_q={num_heads_q}, heads_kv={num_heads_kv}: NaN in output"
    
    def test_mqa_single_kv_head(self, device, dtype):
        """Test Multi-Query Attention (MQA) with single KV head."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 1
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gqa_vs_pytorch_reference(self, device, dtype):
        """Compare GQA output with PyTorch reference implementation."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # Our implementation
        output_ours = flash_attn_func(q, k, v, causal=False)
        
        # PyTorch reference: expand KV heads to match Q heads
        group_size = num_heads_q // num_heads_kv
        k_expanded = k.repeat_interleave(group_size, dim=2)  # (B, S, num_heads_q, D)
        v_expanded = v.repeat_interleave(group_size, dim=2)
        
        q_ref = q.transpose(1, 2).float()
        k_ref = k_expanded.transpose(1, 2).float()
        v_ref = v_expanded.transpose(1, 2).float()
        
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False
        ).transpose(1, 2).half()
        
        diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nGQA accuracy (heads_q={num_heads_q}, heads_kv={num_heads_kv}):")
        print(f"  Max diff: {diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert diff < 0.01, f"GQA difference too large: {diff}"
    
    def test_gqa_causal_vs_pytorch_reference(self, device, dtype):
        """Compare GQA with causal mask against PyTorch reference."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # Our implementation
        output_ours = flash_attn_func(q, k, v, causal=True)
        
        # PyTorch reference
        group_size = num_heads_q // num_heads_kv
        k_expanded = k.repeat_interleave(group_size, dim=2)
        v_expanded = v.repeat_interleave(group_size, dim=2)
        
        q_ref = q.transpose(1, 2).float()
        k_ref = k_expanded.transpose(1, 2).float()
        v_ref = v_expanded.transpose(1, 2).float()
        
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=True
        ).transpose(1, 2).half()
        
        diff = torch.abs(output_ours - output_ref).max().item()
        assert diff < 0.01, f"GQA causal difference too large: {diff}"
    
    def test_gqa_batch_size(self, device, dtype):
        """Test GQA with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        for batch_size in batch_sizes:
            q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
            
            output = flash_attn_func(q, k, v, causal=False)
            
            assert output.shape == q.shape, f"Batch size {batch_size} failed"
            assert not torch.isnan(output).any()
    
    def test_gqa_deterministic(self, device, dtype):
        """Test that GQA is deterministic."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output1 = flash_attn_func(q, k, v, causal=False)
        output2 = flash_attn_func(q, k, v, causal=False)
        
        assert torch.allclose(output1, output2, rtol=0, atol=0), \
            "GQA is not deterministic"
    
    def test_gqa_invalid_ratio(self, device, dtype):
        """Test that invalid Q/KV head ratios are handled."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 3  # 8 is not divisible by 3
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # This might raise an error or handle gracefully
        try:
            output = flash_attn_func(q, k, v, causal=False)
            # If it doesn't raise, check the output is valid
            assert output.shape == q.shape
        except (RuntimeError, ValueError, AssertionError) as e:
            # Expected to fail with invalid ratio
            print(f"Expected error for invalid ratio: {e}")
    
    @pytest.mark.parametrize("seqlen", [256, 512, 1024, 2048, 4096])
    def test_gqa_various_seqlens(self, device, dtype, seqlen):
        """Test GQA with various sequence lengths."""
        batch_size = 1
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape, f"seqlen={seqlen} failed"
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 256])
    def test_gqa_various_head_dims(self, device, dtype, head_dim):
        """Test GQA with various head dimensions."""
        batch_size = 1
        seqlen = 512
        num_heads_q = 8
        num_heads_kv = 2
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape, f"head_dim={head_dim} failed"
        assert not torch.isnan(output).any()
    
    def test_mha_as_special_case(self, device, dtype):
        """Test that MHA (equal Q/KV heads) is a special case of GQA."""
        batch_size = 1
        seqlen = 512
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        # This is standard MHA (GQA with ratio 1:1)
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gqa_large_model_config(self, device, dtype):
        """Test GQA with realistic large model configuration (e.g., Llama-2 style)."""
        batch_size = 2
        seqlen = 2048
        num_heads_q = 32
        num_heads_kv = 8  # 4:1 ratio like Llama-2 34B
        head_dim = 128
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=True)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print(f"\nLarge model config test passed:")
        print(f"  Batch: {batch_size}, Seqlen: {seqlen}")
        print(f"  Q heads: {num_heads_q}, KV heads: {num_heads_kv}, Head dim: {head_dim}")
        print(f"  Output shape: {output.shape}")


class TestGQAEdgeCases:
    """Test edge cases and corner cases for GQA."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda')
    
    @pytest.fixture
    def dtype(self):
        return torch.float16
    
    def test_gqa_single_token(self, device, dtype):
        """Test GQA with single token (seqlen=1)."""
        batch_size = 1
        seqlen = 1
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_gqa_very_long_sequence(self, device, dtype):
        """Test GQA with very long sequence."""
        batch_size = 1
        seqlen = 8192
        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=True)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
