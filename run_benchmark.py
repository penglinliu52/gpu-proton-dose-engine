# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py — 全尺寸 IMPT 场的加速链基准测试

运行：
    python scripts/run_benchmark.py            # 完整基准（含朴素内核，较慢）
    python scripts/run_benchmark.py --quick    # 跳过最慢的两个内核
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pbdose import physics as ph
from pbdose.model import DoseGrid, gaussian_impt_field, clinical_layer_energies
from pbdose import engines as E
from pbdose import benchmark as B
from pbdose import triton_kernels as TK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def build_problem(n_grid=128, dvox=0.2, n_layers=40, n_spot=41, spacing=0.5,
                  r_min=3.0, r_max=25.5, seed=0):
    grid = DoseGrid.centered(n_grid, dvox)
    energies = clinical_layer_energies(n_layers, r_min, r_max)
    return gaussian_impt_field(energies, grid, n_spot=n_spot,
                               spot_spacing=spacing, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过朴素内核的完整规模测量")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    print("=" * 78)
    print("GPU 强化 Pencil Beam 剂量重构 —— 加速链基准测试")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"SM={torch.cuda.get_device_properties(0).multi_processor_count}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}  "
          f"arch={torch.cuda.get_device_capability(0)}")
    print()

    prob = build_problem()
    print(prob.summary())
    print()

    ops = E.operation_counts(prob)
    print("算法复杂度（乘加次数）:")
    for k, v in ops.items():
        if isinstance(v, (int, float)) and k in ("naive", "separable_exact",
                                                 "gather_local", "amplitude_map"):
            print(f"   {k:<18}: {v:>16,.0f}   ({v/ops['naive']*100:6.2f}% of naive)")
    print(f"   理论算法加速比 (naive/separable) = {ops['naive']/ops['separable_exact']:.1f}x")
    print()

    rows = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---------------- float64 CPU 参考解（精度基准） ----------------
    print("[1/9] NumPy float64 可分离参考解 (CPU) ...")
    d_ref, t_ref = E.separable_numpy(prob, per_layer=True, return_timing=True)
    print(f"      完成，耗时 {t_ref:.3f} s，峰值剂量 {d_ref.max():.6e}")
    ref_np = d_ref

    # ---------------- 朴素 CPU：缩小规模 + 外推 ----------------
    print("[2/9] 朴素 NumPy 逐束流累加 (CPU, 缩小规模用于标度外推) ...")
    g_small = DoseGrid(64, 64, 64, 0.4, 0.4, 0.4, -12.8, -12.8, 0.0)
    prob_small = build_problem(n_grid=64, dvox=0.4, n_layers=10, n_spot=21,
                               spacing=1.0, r_min=6.0, r_max=24.0)
    m_sub = 200
    t0 = time.perf_counter()
    _ = E.naive_numpy_per_beam(prob_small, max_spots=m_sub)
    t_sub = time.perf_counter() - t0
    per_beam_vox = t_sub / (m_sub * prob_small.grid.n_vox)
    t_naive_full_est = per_beam_vox * prob.lattice.n_spots * prob.grid.n_vox
    print(f"      缩小规模: {m_sub} beams x {prob_small.grid.n_vox:,} voxels "
          f"= {t_sub:.3f} s")
    print(f"      外推至全尺寸 ({prob.lattice.n_spots:,} beams x "
          f"{prob.grid.n_vox:,} voxels) ≈ {t_naive_full_est:,.0f} s "
          f"({t_naive_full_est/3600:.2f} h)")
    rows.append(B.BenchRow(
        name="朴素 CPU 逐束流 (外推)", device="CPU (Ryzen 9 9955HX)",
        dtype="float64", algorithm="naive",
        min_ms=t_naive_full_est * 1e3, median_ms=t_naive_full_est * 1e3,
        max_rel_err=0.0, rms_rel_err=0.0,
        note=f"由 {m_sub} beams 实测线性外推 (O(M*Nvox))"))

    # ---------------- NumPy float64 可分离 ----------------
    tm = B.time_callable(lambda: E.separable_numpy(prob, per_layer=True),
                         warmup=1, repeat=3, device="cpu")
    rows.append(B.BenchRow(
        name="NumPy 可分离 (逐层精确)", device="CPU 单线程", dtype="float64",
        algorithm="separable", min_ms=tm["min_ms"], median_ms=tm["median_ms"],
        max_rel_err=0.0, rms_rel_err=0.0, note="float64 参考真值"))
    print(f"[3/9] NumPy 可分离 CPU: {tm['median_ms']:.1f} ms")

    # ---------------- Torch CPU ----------------
    tm = B.time_callable(lambda: E.torch_separable(prob, "cpu", per_layer=True),
                         warmup=2, repeat=args.repeat, device="cpu")
    d = E.torch_separable(prob, "cpu", per_layer=True).numpy()
    r = E.relative_error(d, ref_np)
    rows.append(B.BenchRow(
        name="PyTorch bmm (逐层精确)", device="CPU 多线程", dtype="float64",
        algorithm="separable", min_ms=tm["min_ms"], median_ms=tm["median_ms"],
        max_rel_err=r["max_rel_err_global"], rms_rel_err=r["rms_rel_global"],
        note=f"{torch.get_num_threads()} threads"))
    print(f"[4/9] PyTorch bmm CPU: {tm['median_ms']:.1f} ms")

    # ---------------- GPU: bmm 逐层精确 f32 ----------------
    tm = B.time_callable(lambda: E.torch_separable(prob, "cuda", per_layer=True),
                         warmup=5, repeat=args.repeat, device="cuda")
    d = E.torch_separable(prob, "cuda", per_layer=True).double().cpu().numpy()
    r = E.relative_error(d, ref_np)
    gpu_bytes = B.roofline_estimate(prob, "separable")["total_bytes"]
    rows.append(B.BenchRow(
        name="PyTorch bmm 逐层精确 (FP32)", device="RTX 5070 Ti", dtype="float32",
        algorithm="separable", min_ms=tm["min_ms"], median_ms=tm["median_ms"],
        max_rel_err=r["max_rel_err_global"], rms_rel_err=r["rms_rel_global"],
        note=f"等效带宽 {B.achieved_bandwidth(gpu_bytes, tm['min_ms']):.0f} GB/s"))
    print(f"[5/9] PyTorch bmm GPU 逐层精确: {tm['median_ms']:.2f} ms  "
          f"(max_rel_err {r['max_rel_err_global']:.2e})")

    # ---------------- GPU: bmm 逐层 TF32 ----------------
    tm = B.time_callable(lambda: E.torch_separable(prob, "cuda", per_layer=True,
                                                   allow_tf32=True),
                         warmup=5, repeat=args.repeat, device="cuda")
    d = E.torch_separable(prob, "cuda", per_layer=True, allow_tf32=True)\
        .double().cpu().numpy()
    r = E.relative_error(d, ref_np)
    rows.append(B.BenchRow(
        name="PyTorch bmm 逐层精确 (TF32 张量核)", device="RTX 5070 Ti",
        dtype="tf32", algorithm="separable", min_ms=tm["min_ms"],
        median_ms=tm["median_ms"], max_rel_err=r["max_rel_err_global"],
        rms_rel_err=r["rms_rel_global"], note="张量核加速，精度下降"))
    print(f"[6/9] PyTorch bmm GPU TF32: {tm['median_ms']:.2f} ms  "
          f"(max_rel_err {r['max_rel_err_global']:.2e})")

    # ---------------- GPU: 单 sigma 近似 ----------------
    tm = B.time_callable(lambda: E.torch_separable(prob, "cuda", per_layer=False),
                         warmup=5, repeat=args.repeat, device="cuda")
    d = E.torch_separable(prob, "cuda", per_layer=False).double().cpu().numpy()
    r = E.relative_error(d, ref_np)
    rows.append(B.BenchRow(
        name="PyTorch bmm 单 sigma 近似 (FP32)", device="RTX 5070 Ti",
        dtype="float32", algorithm="separable-approx", min_ms=tm["min_ms"],
        median_ms=tm["median_ms"], max_rel_err=r["max_rel_err_global"],
        rms_rel_err=r["rms_rel_global"], note="跨层合并 sigma，牺牲精度换速度"))
    print(f"[7/9] PyTorch bmm GPU 单 sigma: {tm['median_ms']:.2f} ms  "
          f"(max_rel_err {r['max_rel_err_global']:.2e})")

    # ---------------- GPU: 局部窗口 gather ----------------
    tm = B.time_callable(lambda: E.torch_gather_local(prob, "cuda"),
                         warmup=3, repeat=args.repeat, device="cuda")
    d = E.torch_gather_local(prob, "cuda").double().cpu().numpy()
    r = E.relative_error(d, ref_np)
    rl = B.roofline_estimate(prob, "gather_local")
    rows.append(B.BenchRow(
        name="PyTorch 局部窗口 Gather (FP32)", device="RTX 5070 Ti",
        dtype="float32", algorithm="gather_local", min_ms=tm["min_ms"],
        median_ms=tm["median_ms"], max_rel_err=r["max_rel_err_global"],
        rms_rel_err=r["rms_rel_global"],
        arithmetic_intensity=rl["arithmetic_intensity"],
        note=f"K={int(2*np.ceil(4*prob.sigma.max()/0.5)+1)}, ~0.3 FLOP/B"))
    print(f"[8/9] PyTorch 局部窗口 Gather GPU: {tm['median_ms']:.2f} ms  "
          f"(max_rel_err {r['max_rel_err_global']:.2e})")

    # ---------------- GPU: Triton 自定义内核 ----------------
    # 8a. 窗口 gather + LUT（逐层精确）
    tm = B.time_callable(lambda: TK.triton_window(prob, kx=6, ky=6, block_x=64),
                         warmup=2, repeat=args.repeat, device="cuda")
    d = TK.triton_window(prob, kx=6, ky=6, block_x=64).double().cpu().numpy()
    r = E.relative_error(d, ref_np)
    rows.append(B.BenchRow(
        name="Triton 窗口 Gather + LUT (逐层精确)", device="RTX 5070 Ti",
        dtype="float32", algorithm="triton_gather", min_ms=tm["min_ms"],
        median_ms=tm["median_ms"], max_rel_err=r["max_rel_err_global"],
        rms_rel_err=r["rms_rel_global"], note="自定义 kernel；shared/LUT 缓存"))
    print(f"      Triton 窗口 Gather: {tm['median_ms']:.2f} ms  "
          f"(max_rel_err {r['max_rel_err_global']:.2e})")

    # 8b. Triton 可分离卷积
    tm = B.time_callable(lambda: TK.triton_separable(prob, 64, 64, 32, 32),
                         warmup=2, repeat=args.repeat, device="cuda")
    d = TK.triton_separable(prob, 64, 64, 32, 32).double().cpu().numpy()
    d_approx = E.torch_separable(prob, "cuda", per_layer=False).double().cpu().numpy()
    r = E.relative_error(d, d_approx)
    rows.append(B.BenchRow(
        name="Triton 可分离卷积 (融合高斯 GEMM)", device="RTX 5070 Ti",
        dtype="float32", algorithm="triton_conv", min_ms=tm["min_ms"],
        median_ms=tm["median_ms"], max_rel_err=r["max_rel_err_global"],
        rms_rel_err=r["rms_rel_global"], note="自定义 kernel；核矩阵不落显存"))
    print(f"      Triton 可分离卷积: {tm['median_ms']:.2f} ms  "
          f"(vs 单sigma参考 max_rel_err {r['max_rel_err_global']:.2e})")

    # 8c. Triton 朴素 gather（缩小 M 标度 + 全尺寸尝试）
    if not args.quick:
        print("      Triton 朴素 Gather —— 标度测量 ...")
        scale = []
        for m_lim in (256, 1024, 4096):
            t = B.time_callable(lambda m=m_lim: TK.triton_naive(prob, 32, m_limit=m),
                                warmup=1, repeat=3, device="cuda")
            scale.append((m_lim, t["min_ms"]))
            print(f"         M={m_lim:>6}: {t['min_ms']:9.2f} ms")
        ms_per_beam = np.polyfit([s[0] for s in scale], [s[1] for s in scale], 1)
        t_naive_full = float(np.polyval(ms_per_beam, prob.lattice.n_spots))
        print(f"         线性外推至 M={prob.lattice.n_spots:,}: "
              f"{t_naive_full:,.0f} ms ({t_naive_full/1000:.1f} s)")
        rows.append(B.BenchRow(
            name="Triton 朴素 Gather (全 M 外推)", device="RTX 5070 Ti",
            dtype="float32", algorithm="triton_naive",
            min_ms=t_naive_full, median_ms=t_naive_full,
            max_rel_err=0.0, rms_rel_err=0.0,
            arithmetic_intensity=B.roofline_estimate(prob, "naive")["arithmetic_intensity"],
            note="每体素循环全部 M 条 beam；由 M<=4096 线性外推"))

    # ---------------- 汇总 ----------------
    baseline = rows[0].median_ms
    prev = None
    for r_ in rows:
        r_.speedup_vs_cpu_naive = baseline / r_.median_ms
        if prev is not None:
            r_.speedup_vs_prev = prev / r_.median_ms
        prev = r_.median_ms

    print()
    print("=" * 100)
    print(f"{'实现':<40}{'设备':<18}{'中位[ms]':>12}{'加速比':>12}{'max_rel':>12}")
    print("-" * 100)
    for r_ in rows:
        print(f"{r_.name:<40}{r_.device:<18}{r_.median_ms:>12.2f}"
              f"{r_.speedup_vs_cpu_naive:>11.1f}x{r_.max_rel_err:>12.2e}")
    print("=" * 100)

    out = B.save_results(rows, os.path.join(RES, f"benchmark_{args.tag}.json"),
                         extra={"problem": {
                             "grid": [prob.grid.nx, prob.grid.ny, prob.grid.nz],
                             "voxel_cm": prob.grid.dx,
                             "n_layers": prob.lattice.n_layers,
                             "n_spot_per_layer": prob.lattice.n_sx * prob.lattice.n_sy,
                             "M": prob.lattice.n_spots,
                             "n_vox": prob.grid.n_vox,
                             "energy_range_MeV": [float(prob.lattice.energies.min()),
                                                  float(prob.lattice.energies.max())],
                         },
                         "ops": ops,
                         "roofline": {a: B.roofline_estimate(prob, a)
                                      for a in ("naive", "gather_local", "separable")}})
    print(f"\n结果已保存: {out}")

    # 保存参考剂量场供后续验证/作图
    np.save(os.path.join(RES, "dose_reference_f64.npy"), ref_np)
    np.save(os.path.join(RES, "dose_torch_gpu_f32.npy"),
            E.torch_separable(prob, "cuda", per_layer=True).double().cpu().numpy())
    print(f"参考剂量场已保存到 {RES}")


if __name__ == "__main__":
    main()
