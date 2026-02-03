"""Unit tests for standard Multi-Head Attention (MHA) using pytest."""

import pytest
import torch
from mini_flash_attention import flash_attn_func


class TestMultiHeadAttention:
    """Test suite for standard Multi-Head Attention."""
    
    @pytest.fixture
    def device(self):
        """Ensure CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is required for tests"
        return torch.device('cuda:0')
    
    @pytest.fixture
    def dtype(self):
        """Use half precision for tests."""
        return torch.float16
    
    def test_mha_basic(self, device, dtype):
        """Test basic MHA functionality."""
        batch_size = 2
        seqlen = 512
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape, f"Output shape {output.shape} != input shape {q.shape}"
        assert output.dtype == dtype, f"Output dtype {output.dtype} != expected {dtype}"
        assert output.device == device, "Output device mismatch"
    
    def test_mha_output_not_nan_or_inf(self, device, dtype):
        """Test that MHA output doesn't contain NaN or Inf."""
        batch_size = 1
        seqlen = 256
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_mha_vs_pytorch_reference(self, device, dtype):
        """Compare MHA output with PyTorch reference implementation."""
        batch_size = 1
        seqlen = 512
        num_heads = 8
        head_dim = 64
        
        # Normalize inputs for better numerical stability
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        
        # Our implementation
        output_ours = flash_attn_func(q, k, v, causal=False)
        
        # PyTorch reference
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False
        ).transpose(1, 2).half()
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nMHA Accuracy:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.01, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.001, f"Mean difference too large: {mean_diff}"
    
    def test_mha_deterministic(self, device, dtype):
        """Test that MHA produces deterministic results."""
        batch_size = 1
        seqlen = 256
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output1 = flash_attn_func(q, k, v)
        output2 = flash_attn_func(q, k, v)
        
        assert torch.equal(output1, output2), "MHA is not deterministic"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_mha_various_batch_sizes(self, device, dtype, batch_size):
        """Test MHA with various batch sizes."""
        seqlen = 256
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape, f"Batch size {batch_size}: shape mismatch"
        assert not torch.isnan(output).any(), f"Batch size {batch_size}: NaN in output"
    
    @pytest.mark.parametrize("seqlen", [64, 128, 256, 512, 1024, 2048])
    def test_mha_various_seqlens(self, device, dtype, seqlen):
        """Test MHA with various sequence lengths."""
        batch_size = 1
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape, f"Seqlen {seqlen}: shape mismatch"
        assert not torch.isnan(output).any(), f"Seqlen {seqlen}: NaN in output"
    
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_mha_various_num_heads(self, device, dtype, num_heads):
        """Test MHA with various number of heads."""
        batch_size = 1
        seqlen = 256
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape, f"Num heads {num_heads}: shape mismatch"
        assert not torch.isnan(output).any(), f"Num heads {num_heads}: NaN in output"
    
    @pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 256])
    def test_mha_various_head_dims(self, device, dtype, head_dim):
        """Test MHA with various head dimensions."""
        batch_size = 1
        seqlen = 256
        num_heads = 4
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape, f"Head dim {head_dim}: shape mismatch"
        assert not torch.isnan(output).any(), f"Head dim {head_dim}: NaN in output"
    
    def test_mha_batch_independence(self, device, dtype):
        """Test that batch elements are processed independently."""
        batch_size = 4
        seqlen = 64
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        # Process full batch
        output_batch = flash_attn_func(q, k, v)
        
        # Process each element separately
        for i in range(batch_size):
            output_single = flash_attn_func(q[i:i+1], k[i:i+1], v[i:i+1])
            
            diff = torch.abs(output_batch[i:i+1] - output_single).max().item()
            assert diff < 1e-5, f"Batch element {i} differs: {diff}"
    
    def test_mha_input_normalization(self, device, dtype):
        """Test MHA with normalized inputs."""
        batch_size = 1
        seqlen = 256
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        # Normalize along head dimension
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mha_single_head(self, device, dtype):
        """Test MHA with single head (equivalent to standard attention)."""
        batch_size = 1
        seqlen = 256
        num_heads = 1
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_mha_long_sequence(self, device, dtype):
        """Test MHA with long sequence."""
        batch_size = 1
        seqlen = 4096
        num_heads = 8
        head_dim = 128
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mha_realistic_config(self, device, dtype):
        """Test MHA with realistic model configuration (e.g., GPT-2 style)."""
        batch_size = 2
        seqlen = 1024
        num_heads = 12
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        
        print(f"\nRealistic config test passed:")
        print(f"  Batch: {batch_size}, Seqlen: {seqlen}")
        print(f"  Heads: {num_heads}, Head dim: {head_dim}")
        print(f"  Total dims: {num_heads * head_dim}")


class TestMHAEdgeCases:
    """Test edge cases for Multi-Head Attention."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda')
    
    @pytest.fixture
    def dtype(self):
        return torch.float16
    
    def test_mha_single_token(self, device, dtype):
        """Test MHA with single token (seqlen=1)."""
        batch_size = 1
        seqlen = 1
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_mha_very_large_batch(self, device, dtype):
        """Test MHA with large batch size."""
        batch_size = 32
        seqlen = 128
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
