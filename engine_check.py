# -*- coding: utf-8 -*-
"""引擎正确性交叉验证：naive 三重循环 vs 逐束流 NumPy vs 可分离矩阵 vs PyTorch(CPU/GPU)。"""
import os
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pbdose import physics as ph
from pbdose.model import DoseGrid, SpotLattice, PBProblem
from pbdose import engines as E

np.set_printoptions(precision=4, suppress=True)

# --- 小规模问题：3 层 x 3x3 spots = 27 beams, 24x24x32 网格 ---
grid = DoseGrid(24, 24, 32, 0.25, 0.25, 0.25, -3.0, -3.0, 0.0)
energies = np.array([80.0, 130.0, 180.0])
rng = np.random.default_rng(7)
w = rng.random((3, 3, 3))
lat = SpotLattice(3, 3, 3, 0.5, 0.5, energies=energies, weights=w)
z = grid.z
idd = np.stack([ph.idd_pristine(z, e)[0] for e in energies])
sig = np.stack([ph.fermi_eyges_moments(z, e, sigma_x0=0.3)["sigma_x"] for e in energies])
prob = PBProblem(grid, lat, idd, sig, meta={"规模": "small"})
print(prob.summary())

t = time.perf_counter(); d_naive = E.naive_python_triple_loop(prob); t_naive = time.perf_counter() - t
t = time.perf_counter(); d_perbeam = E.naive_numpy_per_beam(prob); t_pb = time.perf_counter() - t
t = time.perf_counter(); d_sep = E.separable_numpy(prob, per_layer=True); t_sep = time.perf_counter() - t
t = time.perf_counter(); d_sep_fast = E.separable_numpy(prob, per_layer=False); t_self = time.perf_counter() - t
t = time.perf_counter(); d_torch_cpu = E.torch_separable(prob, "cpu", None, per_layer=True).numpy(); t_tc = time.perf_counter() - t
t = time.perf_counter(); d_torch_gpu = E.torch_separable(prob, "cuda", None, per_layer=True).double().cpu().numpy(); t_tg = time.perf_counter() - t
try:
    t = time.perf_counter(); d_gather = E.torch_gather_local(prob, "cuda", None).double().cpu().numpy(); t_gl = time.perf_counter() - t
except Exception as ex:
    d_gather = None; t_gl = float("nan"); print("gather FAILED:", ex)

print()
print(f"{'实现':<28}{'耗时[s]':>10}{'max_rel_err(global)':>22}{'rms_rel(global)':>18}")
ref = d_naive
for name, d, tt in [("naive_python_triple_loop", d_naive, t_naive),
                    ("naive_numpy_per_beam", d_perbeam, t_pb),
                    ("separable_numpy(per_layer)", d_sep, t_sep),
                    ("separable_numpy(single-sigma)", d_sep_fast, t_self),
                    ("torch_separable CPU f64", d_torch_cpu, t_tc),
                    ("torch_separable GPU f32", d_torch_gpu, t_tg),
                    ("torch_gather_local GPU f32", d_gather, t_gl)]:
    if d is None:
        continue
    r = E.relative_error(d, ref)
    print(f"{name:<28}{tt:10.4f}{r['max_rel_err_global']:22.3e}{r['rms_rel_global']:18.3e}")

print()
print("单 sigma 近似 vs 逐层精确 的偏差（量化'近似换速度'的代价）:")
r = E.relative_error(d_sep_fast, d_sep)
for k, v in r.items():
    print(f"   {k:<22}: {v:.4e}" if isinstance(v, float) else f"   {k:<22}: {v}")

print()
print("最大剂量位置/数值检查: naive=%.6e  separable=%.6e  torch_gpu=%.6e"
      % (ref.max(), d_sep.max(), d_torch_gpu.max()))
print("峰值体素索引: ", np.unravel_index(np.argmax(ref), ref.shape))
