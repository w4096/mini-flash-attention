import argparse
from typing import Callable, Dict, List

import matplotlib
import torch
from flash_attn import flash_attn_with_kvcache as flash_attn_official
from mini_flash_attention import flash_attn_with_kvcache as flash_attn_mini

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt  # noqa: E402


def measure_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    """Return mean latency in milliseconds for the callable."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def benchmark_one(
    seqlen_kv: int,
    seqlen_q: int,
    batch_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    torch.manual_seed(0)
    q = torch.randn(batch_size, seqlen_q, heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_kv, heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_kv, heads, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.full((batch_size,), seqlen_kv, device=device, dtype=torch.int32)

    def mini_call():
        return flash_attn_mini(q, k_cache, v_cache, cache_seqlens=cache_seqlens)

    def official_call():
        return flash_attn_official(q, k_cache, v_cache, cache_seqlens=cache_seqlens)

    with torch.inference_mode():
        mini_ms = measure_ms(mini_call, warmup, iters)
        official_ms = measure_ms(official_call, warmup, iters)

    return {"mini_ms": mini_ms, "official_ms": official_ms}


def plot(results: Dict[int, Dict[str, float]], output: str) -> None:
    seqlens: List[int] = list(results.keys())
    mini_vals = [results[s]["mini_ms"] for s in seqlens]
    official_vals = [results[s]["official_ms"] for s in seqlens]

    width = 0.35
    xs = range(len(seqlens))

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - width / 2 for x in xs], mini_vals, width, label="mini-flash-attn decode")
    ax.bar([x + width / 2 for x in xs], official_vals, width, label="flash-attn decode (official)")

    ax.set_xlabel("KV cache sequence length")
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(s) for s in seqlens])
    ax.legend()
    ax.set_title("Flash Attention decode latency across KV lengths")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_seqlens(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mini-flash-attn vs flash-attn decode across KV lengths")
    parser.add_argument("--seqlens", type=str, default="512,1024,2048,4096,8192", help="Comma-separated KV cache sequence lengths")
    parser.add_argument("--seqlen-q", type=int, default=1, help="Decode query length (usually 1)")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="dtype for qkv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=str, default="benchmark/flash_attn_decode.png", help="Path to save the bar chart")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    seqlens = parse_seqlens(args.seqlens)
    results: Dict[int, Dict[str, float]] = {}

    for seqlen_kv in seqlens:
        res = benchmark_one(
            seqlen_kv=seqlen_kv,
            seqlen_q=args.seqlen_q,
            batch_size=args.batch_size,
            heads=args.heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        speedup = res["official_ms"] / res["mini_ms"]
        print(f"kv_len={seqlen_kv:5d} | mini={res['mini_ms']:.3f} ms | official={res['official_ms']:.3f} ms | speedup={speedup:.3f}x")
        results[seqlen_kv] = res

    plot(results, args.output)
    print(f"Saved bar chart to {args.output}")


if __name__ == "__main__":
    main()
