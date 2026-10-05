# -*- coding: utf-8 -*-
"""Gamma 失败体素的空间分布分析：入口 / 野边缘 / 内部。"""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

g = np.load(r"<REPO_ROOT>\results\gamma_bohr.npy")
mc = np.load(r"<REPO_ROOT>\results\dose_mc_bohr.npy")
gp = np.load(r"<REPO_ROOT>\results\dose_gpu_bohr.npy")
nz, ny, nx = g.shape
dz = 0.2
z = np.arange(nz) * dz
x = -15.9 + np.arange(nx) * dz
y = -15.9 + np.arange(ny) * dz

ref_max = mc.max()
mask = mc >= 0.10 * ref_max           # 与验证脚本一致的 10% 阈值
fail = mask & (g > 1.0)
print(f"评估体素 {mask.sum():,}   失败体素 {fail.sum():,}   "
      f"通过率 {(1 - fail.sum()/mask.sum())*100:.2f}%")

XX, YY = np.meshgrid(x, y, indexing="xy")
rr = np.hypot(XX, YY)                  # 到束轴距离（平顶野半宽 5 cm）

regions = [
    ("入口 z < 1.5 cm", np.broadcast_to((z < 1.5)[:, None, None], g.shape)),
    ("野内 r < 4.5 cm", np.broadcast_to((rr < 4.5)[None, :, :], g.shape)),
    ("边缘 4.5 <= r < 6 cm", np.broadcast_to(
        ((rr >= 4.5) & (rr < 6.0))[None, :, :], g.shape)),
    ("野外 r >= 6 cm", np.broadcast_to((rr >= 6.0)[None, :, :], g.shape)),
    ("远端 z > 24 cm", np.broadcast_to((z > 24.0)[:, None, None], g.shape)),
]
print(f"\n{'区域':<24}{'该区评估体素':>14}{'失败':>10}{'该区失败占比':>14}"
      f"{'占总失败比例':>14}")
tot_fail = fail.sum()
for name, m in regions:
    mm = m & mask
    ff = m & fail
    if mm.sum() == 0:
        continue
    print(f"{name:<24}{mm.sum():>14,}{ff.sum():>10,}"
          f"{ff.sum()/mm.sum()*100:>13.2f}%{ff.sum()/max(tot_fail,1)*100:>13.1f}%")

print()
print("分层通过率（按深度）：")
for z0, z1 in [(0, 1.5), (1.5, 5), (5, 10), (10, 15), (15, 20), (20, 24), (24, 26)]:
    k0, k1 = int(z0/dz), int(z1/dz)
    m = mask[k0:k1]
    # 需要与 gamma 同切片
    gg = g[k0:k1][m]
    if gg.size == 0:
        continue
    print(f"  z = {z0:4.1f}-{z1:4.1f} cm : 体素 {gg.size:>8,}  "
          f"通过率 {(gg<=1).mean()*100:6.2f}%  gamma_mean {gg.mean():5.3f}")

print()
print("剂量差异分区域 RMS（相对参考峰值）：")
rel = (gp - mc) / ref_max * 100.0
for name, m in regions:
    mm = np.broadcast_to(m, g.shape) & mask
    if mm.sum():
        print(f"  {name:<24} RMS = {np.sqrt((rel[mm]**2).mean()):6.2f}%   "
              f"max|.| = {np.abs(rel[mm]).max():6.2f}%")
