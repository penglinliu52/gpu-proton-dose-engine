# -*- coding: utf-8 -*-
"""Triton 自定义内核正确性与速度验证。"""
import os
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from pbdose import physics as ph
from pbdose.model import DoseGrid, SpotLattice, PBProblem
from pbdose import engines as E
from pbdose import triton_kernels as TK

grid = DoseGrid(32, 32, 40, 0.25, 0.25, 0.25, -4.0, -4.0, 0.0)
energies = np.array([90.0, 140.0, 190.0])
rng = np.random.default_rng(3)
w = rng.random((3, 5, 5))
lat = SpotLattice(5, 5, 3, 0.5, 0.5, energies=energies, weights=w)
z = grid.z
idd = np.stack([ph.idd_pristine(z, e)[0] for e in energies])
sig = np.stack([ph.fermi_eyges_moments(z, e, sigma_x0=0.3)["sigma_x"] for e in energies])
prob = PBProblem(grid, lat, idd, sig)
print(prob.summary())

ref = E.separable_numpy(prob, per_layer=True)          # float64 精确参考
ref_fast = E.separable_numpy(prob, per_layer=False)    # 单 sigma 近似

def check(name, d, t, refarr):
    d = d.double().cpu().numpy() if torch.is_tensor(d) else np.asarray(d)
    r = E.relative_error(d, refarr)
    print(f"{name:<34}{t*1e3:9.2f} ms{ r['max_rel_err_global']:14.3e}"
          f"{r['rms_rel_global']:14.3e}")

print(f"\n{'内核':<34}{'耗时':>12}{'max_rel_err':>14}{'rms_rel':>14}")

# --- 内核 1: 朴素 gather (全部 spot) ---
for _ in range(2):
    t0 = time.perf_counter(); d = TK.triton_naive(prob, block_x=32); torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0
check("triton_naive (all spots)", d, t_naive, ref)

# --- 内核 2: 窗口 gather ---
for _ in range(2):
    t0 = time.perf_counter(); d = TK.triton_window(prob, kx=6, ky=6, block_x=64); torch.cuda.synchronize()
    t_win = time.perf_counter() - t0
check("triton_window (K=6, LUT)", d, t_win, ref)

# --- 内核 3: Triton 可分离 (单 sigma) ---
for _ in range(2):
    t0 = time.perf_counter(); d = TK.triton_separable(prob, 32, 32, 32, 32); torch.cuda.synchronize()
    t_sep = time.perf_counter() - t0
check("triton_separable (single-sigma)", d, t_sep, ref_fast)

# --- 对照: PyTorch bmm ---
for _ in range(2):
    t0 = time.perf_counter(); d = E.torch_separable(prob, "cuda", per_layer=True); torch.cuda.synchronize()
    t_bmm = time.perf_counter() - t0
check("torch bmm per-layer (exact)", d, t_bmm, ref)
for _ in range(2):
    t0 = time.perf_counter(); d = E.torch_separable(prob, "cuda", per_layer=False); torch.cuda.synchronize()
    t_bmmf = time.perf_counter() - t0
check("torch bmm single-sigma", d, t_bmmf, ref_fast)
for _ in range(2):
    t0 = time.perf_counter(); d = E.torch_gather_local(prob, "cuda"); torch.cuda.synchronize()
    t_gl = time.perf_counter() - t0
check("torch gather_local (window)", d, t_gl, ref)

print("\n单 sigma 近似 vs 逐层精确参考:")
r = E.relative_error(ref_fast, ref)
print(f"   max_rel_global = {r['max_rel_err_global']:.3e}   rms_rel_global = {r['rms_rel_global']:.3e}")
