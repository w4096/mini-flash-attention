import math

import torch
from mini_flash_attention import flash_attn_func as mini_flash_attn_func
from flash_attn import flash_attn_func, flash_attn_varlen_func

def flash_attention(q, k, v, causal=False):
    return flash_attn_func(q, k, v, causal=causal)


def torch_attention(q, k, v, causal=False):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    return o.transpose(1, 2)

def mini_flash_attention(q, k, v, causal=False):
    return mini_flash_attn_func(q, k, v, causal=causal)
    
def test_flash_attn_forward():

    # Define input parameters
    batch_size = 48
    seqlen = 4096
    dim = 128
    heads = 24
    is_causal = False
    

    # Create random tensors for q, k, v
    q = torch.randn((batch_size, seqlen, heads, dim), device='cuda', dtype=torch.half)
    k = torch.randn((batch_size, seqlen, heads, dim), device='cuda', dtype=torch.half)
    v = torch.randn((batch_size, seqlen, heads, dim), device='cuda', dtype=torch.half)
    

    # normalize q, k, v for better numerical stability
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)


    for _ in range(3):
        mini_flash_attention(q, k, v, causal=is_causal)
        torch_attention(q, k, v, causal=is_causal)
        flash_attention(q, k, v, causal=is_causal)


    with torch.profiler.profile() as prof:
        mini_flash_attn_out = mini_flash_attention(q, k, v, causal=is_causal)
    key_averages = prof.key_averages()
    print("Mini Flash Attention Profiling Results:")
    print(key_averages.table(sort_by="cuda_time_total", row_limit=10))
    
    with torch.profiler.profile() as prof:
        torch_out = torch_attention(q, k, v, causal=is_causal)
    key_averages = prof.key_averages()
    print("Torch Attention Profiling Results:")
    print(key_averages.table(sort_by="cuda_time_total", row_limit=10))

    with torch.profiler.profile() as prof:
        flash_attn_out = flash_attention(q, k, v, causal=is_causal)
    key_averages = prof.key_averages()
    print("Flash Attention Profiling Results:")
    print(key_averages.table(sort_by="cuda_time_total", row_limit=10))    



    print("shape:", mini_flash_attn_out.shape, flash_attn_out.shape, torch_out.shape)

    diff = torch.abs(mini_flash_attn_out - flash_attn_out).max().item()
    print("Max difference between mini-flash-attn and flash-attn:", diff)
    
    torch_flash_attn_diff = torch.abs(torch_out - flash_attn_out).max().item()
    print("Max difference between torch and flash-attn:", torch_flash_attn_diff)

    torch_mini_flash_attn_diff = torch.abs(torch_out - mini_flash_attn_out).max().item()
    print("Max difference between torch and mini-flash-attn:", torch_mini_flash_attn_diff)


if __name__ == "__main__":
    test_flash_attn_forward()
