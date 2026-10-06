"""Benchmark (L, W) x (W, B) matrix multiplication in CuPy and report the best L, W, B.

Usage: python benchmark_matmul.py [--dtype float32] [--sizes 256 512 ...] [--repeat 20]
Performance is reported as TFLOP/s (2*L*W*B flops per matmul).
"""
import argparse
import itertools

import cupy as cp


def bench(L, W, B, dtype, repeat, warmup=3):
    a = cp.random.random((L, W)).astype(dtype)
    b = cp.random.random((W, B)).astype(dtype)
    for _ in range(warmup):
        cp.matmul(a, b)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeat):
        cp.matmul(a, b)
    end.record()
    end.synchronize()
    sec = cp.cuda.get_elapsed_time(start, end) / 1e3 / repeat
    return sec, 2 * L * W * B / sec / 1e12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="float32")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()

    free, _ = cp.cuda.Device().mem_info
    itemsize = cp.dtype(args.dtype).itemsize
    results = []
    for L, W, B in itertools.product(args.sizes, repeat=3):
        if (L * W + W * B + L * B) * itemsize * 1.2 > free:
            continue
        sec, tflops = bench(L, W, B, args.dtype, args.repeat)
        results.append((tflops, L, W, B, sec))
        print(f"L={L:6d} W={W:6d} B={B:6d}  {sec * 1e3:10.4f} ms  {tflops:8.3f} TFLOP/s")
        cp.get_default_memory_pool().free_all_blocks()

    tflops, L, W, B, sec = max(results)
    print(f"\nBest: L={L} W={W} B={B} -> {tflops:.3f} TFLOP/s ({sec * 1e3:.4f} ms) on "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")


if __name__ == "__main__":
    main()
