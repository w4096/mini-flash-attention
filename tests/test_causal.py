"""Unit tests for causal attention mask functionality using pytest."""

import pytest
import torch
from mini_flash_attention import flash_attn_func


class TestCausalAttention:
    """Test suite for causal attention mask."""
    
    @pytest.fixture
    def device(self):
        """Ensure CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is required for tests"
        return torch.device('cuda')
    
    @pytest.fixture
    def dtype(self):
        """Use half precision for tests."""
        return torch.float16
    
    @pytest.fixture
    def simple_inputs(self, device, dtype):
        """Create simple test inputs."""
        batch_size = 1
        seqlen = 512
        heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        return q, k, v
    
    @pytest.fixture
    def large_inputs(self, device, dtype):
        """Create larger test inputs."""
        batch_size = 2
        seqlen = 2048
        heads = 8
        head_dim = 128
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        return q, k, v
    
    def test_causal_output_shape(self, simple_inputs):
        """Test that causal attention produces correct output shape."""
        q, k, v = simple_inputs
        output = flash_attn_func(q, k, v, causal=True)
        
        assert output.shape == q.shape, f"Output shape {output.shape} doesn't match input shape {q.shape}"
        assert output.dtype == q.dtype, f"Output dtype {output.dtype} doesn't match input dtype {q.dtype}"
        assert output.device == q.device, "Output device doesn't match input device"
    
    def test_non_causal_output_shape(self, simple_inputs):
        """Test that non-causal attention produces correct output shape."""
        q, k, v = simple_inputs
        output = flash_attn_func(q, k, v, causal=False)
        
        assert output.shape == q.shape
        assert output.dtype == q.dtype
        assert output.device == q.device
    
    def test_causal_vs_non_causal_different(self, simple_inputs):
        """Test that causal and non-causal produce different results."""
        q, k, v = simple_inputs
        
        out_causal = flash_attn_func(q, k, v, causal=True)
        out_non_causal = flash_attn_func(q, k, v, causal=False)
        
        # They should be different
        diff = torch.abs(out_causal - out_non_causal).max().item()
        assert diff > 1e-3, f"Causal and non-causal outputs are too similar (diff={diff})"
    
    def test_causal_vs_pytorch_reference(self, simple_inputs):
        """Test causal attention against PyTorch reference implementation."""
        q, k, v = simple_inputs
        
        # Our implementation
        out_ours = flash_attn_func(q, k, v, causal=True)
        
        # PyTorch reference
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=True
        ).transpose(1, 2).half()
        
        # Check accuracy
        diff = torch.abs(out_ours - out_ref).max().item()
        assert diff < 0.01, f"Difference with PyTorch reference too large: {diff}"
    
    def test_non_causal_vs_pytorch_reference(self, simple_inputs):
        """Test non-causal attention against PyTorch reference implementation."""
        q, k, v = simple_inputs
        
        # Our implementation
        out_ours = flash_attn_func(q, k, v, causal=False)
        
        # PyTorch reference
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False
        ).transpose(1, 2).half()
        
        # Check accuracy
        diff = torch.abs(out_ours - out_ref).max().item()
        assert diff < 0.01, f"Difference with PyTorch reference too large: {diff}"
    
    @pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 2048])
    def test_causal_various_seqlens(self, device, dtype, seqlen):
        """Test causal attention with various sequence lengths."""
        batch_size = 1
        heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        # Should not raise any errors
        output = flash_attn_func(q, k, v, causal=True)
        assert output.shape == q.shape
        
        # Verify against PyTorch
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=True
        ).transpose(1, 2).half()
        
        diff = torch.abs(output - out_ref).max().item()
        assert diff < 0.01, f"seqlen={seqlen}: difference {diff} too large"
    
    @pytest.mark.parametrize("heads", [1, 4, 8, 16])
    def test_causal_various_heads(self, device, dtype, heads):
        """Test causal attention with various number of heads."""
        batch_size = 1
        seqlen = 512
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=True)
        assert output.shape == q.shape
    
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_causal_various_head_dims(self, device, dtype, head_dim):
        """Test causal attention with various head dimensions."""
        batch_size = 1
        seqlen = 512
        heads = 4
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        output = flash_attn_func(q, k, v, causal=True)
        assert output.shape == q.shape
    
    def test_causal_no_nan_or_inf(self, simple_inputs):
        """Test that causal attention doesn't produce NaN or Inf."""
        q, k, v = simple_inputs
        
        output = flash_attn_func(q, k, v, causal=True)
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_causal_deterministic(self, simple_inputs):
        """Test that causal attention is deterministic."""
        q, k, v = simple_inputs
        
        output1 = flash_attn_func(q, k, v, causal=True)
        output2 = flash_attn_func(q, k, v, causal=True)
        
        assert torch.allclose(output1, output2, rtol=0, atol=0), \
            "Causal attention is not deterministic"
    
    def test_causal_batch_independence(self, device, dtype):
        """Test that different batch elements are processed independently."""
        batch_size = 4
        seqlen = 512
        heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        # Run full batch
        output_batch = flash_attn_func(q, k, v, causal=True)
        
        # Run each batch element separately
        for i in range(batch_size):
            output_single = flash_attn_func(
                q[i:i+1], k[i:i+1], v[i:i+1], causal=True
            )
            
            diff = torch.abs(output_batch[i:i+1] - output_single).max().item()
            assert diff < 1e-5, f"Batch element {i} differs: {diff}"
    
    def test_causal_large_inputs(self, large_inputs):
        """Test causal attention with larger inputs."""
        q, k, v = large_inputs
        
        output = flash_attn_func(q, k, v, causal=True)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_default_is_non_causal(self, simple_inputs):
        """Test that default behavior is non-causal."""
        q, k, v = simple_inputs
        
        # Default call (no causal parameter)
        output_default = flash_attn_func(q, k, v)
        
        # Explicit non-causal
        output_non_causal = flash_attn_func(q, k, v, causal=False)
        
        # They should be the same
        assert torch.allclose(output_default, output_non_causal, rtol=0, atol=0), \
            "Default behavior is not non-causal"


class TestCausalBlockSkipping:
    """Test suite for block-level skipping optimization in causal attention."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda')
    
    @pytest.fixture
    def dtype(self):
        return torch.float16
    
    def test_block_skipping_correctness(self, device, dtype):
        """Verify that block skipping doesn't affect correctness."""
        batch_size = 1
        seqlen = 4096  # Large enough to have multiple blocks
        heads = 8
        head_dim = 128
        
        q = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, heads, head_dim, device=device, dtype=dtype)
        
        # Our implementation (with block skipping)
        output = flash_attn_func(q, k, v, causal=True)
        
        # PyTorch reference
        q_ref = q.transpose(1, 2).float()
        k_ref = k.transpose(1, 2).float()
        v_ref = v.transpose(1, 2).float()
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=True
        ).transpose(1, 2).half()
        
        diff = torch.abs(output - output_ref).max().item()
        assert diff < 0.01, f"Block skipping affects correctness: diff={diff}"
