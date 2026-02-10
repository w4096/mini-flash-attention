"""Unit tests for Flash Decoding using flash_attn_with_kvcache."""

import pytest
import torch
from mini_flash_attention import flash_attn_func, flash_attn_with_kvcache

# Import reference implementation
try:
    from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_ref
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not installed, skipping reference comparison tests")


class TestFlashDecoding:
    """Test suite for Flash Decoding with KV cache."""
    
    @pytest.fixture
    def device(self):
        """Ensure CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is required for tests"
        return torch.device('cuda:0')
    
    @pytest.fixture
    def dtype(self):
        """Use half precision for tests."""
        return torch.float16
    
    def test_decoding_basic(self, device, dtype):
        """Test basic flash decoding functionality."""
        batch_size = 2
        seqlen_q = 1  # Decoding: one token at a time
        seqlen_kv = 512  # Context length in cache
        num_heads = 8
        head_dim = 64
        block_size = 256
        
        # Query: (batch_size, 1, nheads, headdim)
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        # Prepare contiguous KV cache
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format: (num_blocks, block_size, nheads, headdim)
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        output = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        assert output.shape == q.shape, f"Output shape {output.shape} != input shape {q.shape}"
        assert output.dtype == dtype, f"Output dtype {output.dtype} != expected {dtype}"
        assert output.device == device, "Output device mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_decoding_vs_flash_attn_contiguous(self, device, dtype):
        """Compare flash decoding with official flash_attn (contiguous cache)."""
        batch_size = 4
        seqlen_q = 1
        seqlen_kv = 512
        num_heads = 8
        head_dim = 128
        block_size = 256
        
        # Prepare query
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        # Prepare contiguous KV
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        # Our implementation
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        # Reference implementation
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (contiguous):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_decoding_vs_flash_attn_with_block_table(self, device, dtype):
        """Compare flash decoding with official flash_attn (non-contiguous cache with block table)."""
        batch_size = 4
        seqlen_q = 1
        num_heads = 8
        head_dim = 128
        block_size = 256
        num_blocks = 32
        max_blocks_per_seq = 8
        
        # Query
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        # KV cache (shared pool of blocks)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k_cache = torch.nn.functional.normalize(k_cache, dim=-1)
        # v_cache = torch.nn.functional.normalize(v_cache, dim=-1)
        
        # Block table: each batch entry points to different blocks
        block_table = torch.randint(
            0, num_blocks, 
            (batch_size, max_blocks_per_seq), 
            dtype=torch.int32,
            device=device
        )
        
        # Cache sequence lengths (different for each batch)
        cache_seqlens_list = [192, 256, 384, 128]
        cache_seqlens = torch.tensor(cache_seqlens_list, dtype=torch.int32, device=device)
        
        # Our implementation
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        print(output_ours)
        
        # Reference implementation
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (block table):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_decoding_vs_flash_attn_gqa(self, device, dtype):
        """Compare flash decoding with official flash_attn (Grouped Query Attention)."""
        batch_size = 2
        seqlen_q = 1
        seqlen_kv = 512
        num_heads_q = 8
        num_heads_kv = 2  # GQA: fewer KV heads
        head_dim = 128
        block_size = 256
        
        # Query has more heads than KV
        q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype)
        
        # KV with fewer heads
        k = torch.randn(batch_size, seqlen_kv, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads_kv, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads_kv, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        # Our implementation
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        # Reference implementation
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (GQA):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_decoding_vs_flash_attn_causal(self, device, dtype):
        """Compare flash decoding with official flash_attn (causal masking)."""
        batch_size = 2
        seqlen_q = 1
        seqlen_kv = 512
        num_heads = 8
        head_dim = 64
        block_size = 256
        
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        # Test with causal=True
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True
        )
        
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (causal):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    @pytest.mark.parametrize("seqlen_kv", [256, 512, 1024, 2048])
    def test_decoding_vs_flash_attn_various_seqlens(self, device, dtype, seqlen_kv):
        """Compare flash decoding with official flash_attn for various context lengths."""
        batch_size = 2
        seqlen_q = 1
        num_heads = 8
        head_dim = 128
        block_size = 256
        
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        # Use splits for large sequences
        num_splits = 2 if seqlen_kv > 1024 else 0
        
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False,
            num_splits=num_splits
        )
        
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (seqlen={seqlen_kv}):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Seqlen {seqlen_kv}: Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Seqlen {seqlen_kv}: Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_decoding_vs_flash_attn_various_head_dims(self, device, dtype, head_dim):
        """Compare flash decoding with official flash_attn for various head dimensions."""
        batch_size = 2
        seqlen_q = 1
        seqlen_kv = 512
        num_heads = 8
        block_size = 256
        
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"\nFlash Decoding vs flash_attn (head_dim={head_dim}):")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.02, f"Head dim {head_dim}: Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Head dim {head_dim}: Mean difference too large: {mean_diff}"
    
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_decoding_cross_block_boundary(self, device, dtype):
        """Test decoding when seqlen crosses block boundary (e.g., 256 -> 257)."""
        batch_size = 1
        num_heads = 8
        head_dim = 128
        block_size = 256
        
        # Test seqlen = 257 (just crosses the block boundary)
        seqlen_kv = 257
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size  # Should be 2
        num_blocks = batch_size * num_blocks_per_seq
        
        q = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
        
        # Use the same random data for both implementations
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Prepare paged cache
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv], dtype=torch.int32, device=device)
        
        print(f"\nTesting cross-block boundary: seqlen={seqlen_kv}, num_blocks={num_blocks_per_seq}")
        print(f"  Block 0: tokens 0-255 ({block_size} tokens)")
        print(f"  Block 1: token 256 (1 token)")
        
        # Our implementation
        output_ours = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        # Reference implementation
        output_ref = flash_attn_with_kvcache_ref(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        max_diff = torch.abs(output_ours - output_ref).max().item()
        mean_diff = torch.abs(output_ours - output_ref).mean().item()
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff > 0.02:
            print(f"\n  ERROR: Large difference detected!")
            print(f"  Output ours (first 8 values): {output_ours[0, 0, 0, :8]}")
            print(f"  Output ref  (first 8 values): {output_ref[0, 0, 0, :8]}")
        
        assert max_diff < 0.02, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.002, f"Mean difference too large: {mean_diff}"
    
    def test_decoding_generation_loop(self, device, dtype):
        """Test flash decoding in a generation loop, simulating autoregressive generation."""
        batch_size = 4  # Increase to test concurrent batches
        num_heads = 8
        head_dim = 128
        block_size = 256
        initial_seqlen = 256  # Initial context length
        num_generation_steps = 10  # Generate 10 tokens
        max_seqlen = initial_seqlen + num_generation_steps
        
        # Allocate cache with enough space
        num_blocks_per_seq = (max_seqlen + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_table[b, i] = b * num_blocks_per_seq + i
        
        # Fill initial cache with random data
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, initial_seqlen)
                if start_idx < initial_seqlen:
                    length = end_idx - start_idx
                    k_cache[block_idx, :length] = torch.randn(length, num_heads, head_dim, device=device, dtype=dtype)
                    v_cache[block_idx, :length] = torch.randn(length, num_heads, head_dim, device=device, dtype=dtype)
        
        # Cache for reference implementation (contiguous format)
        k_ref = torch.zeros(batch_size, max_seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v_ref = torch.zeros(batch_size, max_seqlen, num_heads, head_dim, device=device, dtype=dtype)
        
        # Copy initial cache to reference
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, initial_seqlen)
                if start_idx < initial_seqlen:
                    length = end_idx - start_idx
                    k_ref[b, start_idx:end_idx] = k_cache[block_idx, :length]
                    v_ref[b, start_idx:end_idx] = v_cache[block_idx, :length]
        
        # Initialize cache lengths
        cache_seqlens = torch.tensor([initial_seqlen] * batch_size, dtype=torch.int32, device=device)
        
        print(f"\nGeneration loop test: initial_seqlen={initial_seqlen}, steps={num_generation_steps}")
        
        # Generation loop
        for step in range(num_generation_steps):
            # Generate new K, V for the next token
            new_k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
            new_v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
            
            # Update cache with new K, V
            current_seqlen = initial_seqlen + step
            for b in range(batch_size):
                pos_in_seq = current_seqlen  # Position where to insert new token
                block_idx_in_table = pos_in_seq // block_size
                pos_in_block = pos_in_seq % block_size
                
                if block_idx_in_table < num_blocks_per_seq:
                    block_idx = block_table[b, block_idx_in_table].item()
                    k_cache[block_idx, pos_in_block] = new_k[b, 0]
                    v_cache[block_idx, pos_in_block] = new_v[b, 0]
                    
                    # Update reference cache
                    k_ref[b, pos_in_seq] = new_k[b, 0]
                    v_ref[b, pos_in_seq] = new_v[b, 0]
            
            # Increment cache lengths
            cache_seqlens += 1
            current_seqlen += 1  # Now includes the newly added token
            
            # Generate new query token for next iteration
            q = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)
            
            # Our implementation
            output_ours = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=False
            )
            
            # Reference implementation (using the same paged cache)
            output_ref = flash_attn_with_kvcache_ref(
                q, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=False
            )
            
            # Compare outputs
            max_diff = torch.abs(output_ours - output_ref).max().item()
            mean_diff = torch.abs(output_ours - output_ref).mean().item()
            
            print(f"  Step {step + 1}/{num_generation_steps} (seqlen={current_seqlen}): "
                  f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            
            assert max_diff < 0.02, f"Step {step}: Max difference too large: {max_diff}"
            assert mean_diff < 0.002, f"Step {step}: Mean difference too large: {mean_diff}"
        
        print("  ✓ All generation steps passed!")
    
    def test_decoding_deterministic(self, device, dtype):
        """Test that flash decoding produces deterministic results."""
        batch_size = 2
        seqlen_q = 1
        seqlen_kv = 512
        num_heads = 8
        head_dim = 128
        block_size = 256
        
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        
        k = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_kv, num_heads, head_dim, device=device, dtype=dtype)
        
        # Reshape into paged cache format
        num_blocks_per_seq = (seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * num_blocks_per_seq
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        
        # Fill cache and create block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            for i in range(num_blocks_per_seq):
                block_idx = b * num_blocks_per_seq + i
                block_table[b, i] = block_idx
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, seqlen_kv)
                length = end_idx - start_idx
                k_cache[block_idx, :length] = k[b, start_idx:end_idx]
                v_cache[block_idx, :length] = v[b, start_idx:end_idx]
        
        cache_seqlens = torch.tensor([seqlen_kv] * batch_size, dtype=torch.int32, device=device)
        
        # Run twice with same inputs
        output1 = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        output2 = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=False
        )
        
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5), \
            "Flash decoding is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
