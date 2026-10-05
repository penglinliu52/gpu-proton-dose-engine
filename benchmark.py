"""
pbdose.benchmark — 计时框架与全尺寸基准测试
============================================

计时方法学（对应性能报告"测量方法"一节）
----------------------------------------
* GPU 计时使用 ``torch.cuda.Event(enable_timing=True)``，避免主机-设备同步误差；
  CPU 计时使用 ``time.perf_counter``。
* 每次测量前做 ``warmup`` 次预热（触发 JIT 编译、cuBLAS 算法选择、显存分配池预热、
  GPU 升频），再取 ``repeat`` 次的中位数与最小值。
* 每次测量前后 ``torch.cuda.synchronize()``；报告 min（最优性能）与 median（稳健性）。
* 显存峰值用 ``torch.cuda.max_memory_allocated`` 记录，用于 VRAM 约束分析。

算术强度分析（roofline 视角）
-----------------------------
* 朴素 gather ：每体素每束流搬运 ~24 B（位置/权重/LUT），仅 2 FLOP -> ~0.08 FLOP/B
* 窗口 gather ：K=11 时每体素搬运 121*(24 B) 但 LUT 命中 -> ~0.3 FLOP/B
* 可分离 bmm ：稠密小矩阵乘，算术强度 >> 1，属 compute-bound/tensor-core 友好
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------
def time_callable(fn: Callable, warmup: int = 3, repeat: int = 10,
                  device: str = "cpu") -> dict:
    """返回 {'min_ms','median_ms','mean_ms','std_ms','n'}。"""
    use_cuda = str(device).startswith("cuda") and HAS_TORCH and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    ts = []
    if use_cuda:
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    else:
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)

    ts = np.array(ts, dtype=np.float64)
    return {"min_ms": float(ts.min()), "median_ms": float(statistics.median(ts)),
            "mean_ms": float(ts.mean()), "std_ms": float(ts.std()), "n": repeat}


def gpu_mem_mb() -> float:
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


@dataclass
class BenchRow:
    name: str
    device: str
    dtype: str
    algorithm: str
    min_ms: float
    median_ms: float
    speedup_vs_cpu_naive: float = 0.0
    speedup_vs_prev: float = 0.0
    max_rel_err: float = 0.0
    rms_rel_err: float = 0.0
    flops: float = 0.0
    arithmetic_intensity: float = 0.0
    mem_mb: float = 0.0
    note: str = ""

    def as_dict(self):
        return asdict(self)


def roofline_estimate(problem, algorithm: str) -> dict:
    """
    估算各算法的访存量与算术强度（用于性能报告的 roofline 分析）。
    以"每体素需搬运的字节数"为口径。
    """
    g, L = problem.grid, problem.lattice
    nvox = g.n_vox
    if algorithm == "naive":
        # 每体素循环 M 次：每次读 spot x,y,w,layer (4*4B=16B) + idd,sigma 标量 (8B)
        bytes_per_vox = L.n_spots * 24.0
        flops_per_vox = L.n_spots * 2.0
    elif algorithm == "gather_local":
        K = int(2 * np.ceil(4.0 * problem.sigma.max() / L.dx_spot) + 1)
        bytes_per_vox = K * K * 24.0
        flops_per_vox = K * K * 2.0
    else:  # separable
        # 两次 pass：读 A (nsy*nsx) + 写/读中间量 (nsx*ny) + 写输出
        b1 = L.n_sy * L.n_sx * 4.0 + L.n_sx * g.ny * 4.0 * 2
        b2 = L.n_sx * g.ny * 4.0 + g.nx * 4.0 + L.n_sx * 4.0
        bytes_per_vox = (b1 + b2) / g.nx
        flops_per_vox = (L.n_sx * L.n_sy * g.ny + g.nx * L.n_sx * g.ny) * 2.0 / (g.nx * g.ny)

    total_bytes = bytes_per_vox * nvox
    total_flops = flops_per_vox * nvox
    return {
        "algorithm": algorithm,
        "bytes_per_voxel": bytes_per_vox,
        "flops_per_voxel": flops_per_vox,
        "total_bytes": total_bytes,
        "total_flops": total_flops,
        "arithmetic_intensity": flops_per_vox / max(bytes_per_vox, 1e-12),
    }


def achieved_bandwidth(total_bytes: float, ms: float) -> float:
    """由耗时反推等效带宽 [GB/s]。"""
    return total_bytes / (ms * 1e-3) / 1e9


def achieved_tflops(total_flops: float, ms: float) -> float:
    return total_flops / (ms * 1e-3) / 1e12


def save_results(rows, path: str, extra: Optional[dict] = None):
    payload = {"rows": [r.as_dict() if isinstance(r, BenchRow) else r for r in rows]}
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def rows_to_latex(rows) -> str:
    """生成 Markdown 表格（性能报告用）。"""
    hdr = ("| 实现 | 设备 | 精度 | 中位耗时 | 加速比 | 最大相对误差 | 说明 |\n"
           "|---|---|---|---:|---:|---:|---|\n")
    body = ""
    for r in rows:
        r = r.as_dict() if isinstance(r, BenchRow) else r
        body += (f"| {r['name']} | {r['device']} | {r['dtype']} | "
                 f"{r['median_ms']:.2f} ms | {r['speedup_vs_cpu_naive']:.1f}x | "
                 f"{r['max_rel_err']:.2e} | {r.get('note','')} |\n")
    return hdr + body
