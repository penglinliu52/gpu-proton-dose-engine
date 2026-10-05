# -*- coding: utf-8 -*-
"""GPU 微基准：显存带宽 / L2 带宽 / FP32 峰值，用于 roofline 分析。"""
import os
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch

dev = "cuda"
print(torch.cuda.get_device_name(0))
out = {}


def bench(fn, warm=5, rep=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


# ---- DRAM 带宽：大张量拷贝（>> L2） ----
n = 512 * 1024 * 1024 // 4          # 512 MB
a = torch.empty(n, device=dev); b = torch.empty(n, device=dev)
ms = bench(lambda: b.copy_(a))
bw = 2 * n * 4 / (ms * 1e-3) / 1e9
print(f"DRAM copy 512MB : {ms:.3f} ms -> {bw:.1f} GB/s (read+write)")
out["dram_bw_gbs"] = bw

# ---- L2 带宽：小张量反复拷贝（< L2） ----
n2 = 8 * 1024 * 1024 // 4           # 8 MB
a2 = torch.empty(n2, device=dev); b2 = torch.empty(n2, device=dev)
ms2 = bench(lambda: b2.copy_(a2), rep=200)
bw2 = 2 * n2 * 4 / (ms2 * 1e-3) / 1e9
print(f"L2   copy   8MB : {ms2:.4f} ms -> {bw2:.1f} GB/s")
out["l2_bw_gbs"] = bw2

# ---- FP32 GEMM 峰值（大矩阵，cuBLAS） ----
for N in (2048, 4096):
    A = torch.randn(N, N, device=dev); B = torch.randn(N, N, device=dev)
    ms3 = bench(lambda: A @ B, warm=3, rep=10)
    tf = 2 * N**3 / (ms3 * 1e-3) / 1e12
    print(f"FP32 GEMM {N}^3 : {ms3:.3f} ms -> {tf:.1f} TFLOP/s")
    out[f"gemm_fp32_{N}_tflops"] = tf

# ---- 小批量 bmm（本项目的实际形状） ----
for batch in (128, 5120):
    A = torch.randn(batch, 41, 41, device=dev); B = torch.randn(batch, 41, 128, device=dev)
    ms4 = bench(lambda: torch.bmm(A, B), warm=10, rep=50)
    tf = 2 * batch * 41 * 41 * 128 / (ms4 * 1e-3) / 1e12
    print(f"bmm batch={batch:5d} (41x41x128): {ms4:.3f} ms -> {tf:.2f} TFLOP/s")
    out[f"bmm_batch{batch}_ms"] = ms4
    out[f"bmm_batch{batch}_tflops"] = tf
    out[f"bmm_batch{batch}_bytes"] = batch * (41 * 41 + 41 * 128 + 41 * 128) * 4

os.makedirs(r"<REPO_ROOT>\results", exist_ok=True)
json.dump(out, open(r"<REPO_ROOT>\results\gpu_microbench.json", "w"),
          indent=1)
print("saved")
