# -*- coding: utf-8 -*-
"""验证 Fermi-Eyges 网格分辨率修正 + B=2A2 恒等式 + 蒙特卡洛交叉对比。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pbdose import physics as ph

print("=== 网格无关性检验: sigma_x(10 cm, 200 MeV, sigma_x0=0) [cm] ===")
for zg in ([0.0, 10.0], np.linspace(0, 25, 8), np.linspace(0, 25, 51),
           np.linspace(0, 25, 501), np.linspace(0, 25, 2001)):
    m = ph.fermi_eyges_moments(np.asarray(zg), 200.0)
    v = m["sigma_x"][np.argmin(np.abs(np.asarray(zg) - 10.0))]
    print(f"  n_grid={len(zg):5d}  sigma_x(10cm) = {v:.6f} cm   (B={m['B'][np.argmin(np.abs(np.asarray(zg)-10.0))]:.6e})")
print("  MC 自测参考值                     = 0.124628 cm")

print()
print("=== B(z) = 2*A2(z) 恒等式 ===")
z = np.linspace(0.5, 25.0, 30)
m = ph.fermi_eyges_moments(z, 200.0)
for zi, B, A2 in zip(z[::6], m["B"][::6], m["A2"][::6]):
    print(f"  z={zi:5.2f} cm   B={B:.6e}   2*A2={2*A2:.6e}   ratio={B/(2*A2):.6f}")

print()
print("=== 完整 sigma_x 表 (sigma_x0 = 3 mm) [cm] ===")
zt = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0])
for E0 in (100.0, 150.0, 200.0):
    mm = ph.fermi_eyges_moments(zt, E0, sigma_x0=0.30)
    print(f"  {E0:5.0f} MeV:", np.array2string(mm["sigma_x"], precision=4))
print()
print("=== 纯散射 sigma_x (sigma_x0 = 0) [cm] 用于与 MC 对比 ===")
for E0 in (100.0, 150.0, 200.0):
    mm = ph.fermi_eyges_moments(zt, E0)
    print(f"  {E0:5.0f} MeV:", np.array2string(mm["sigma_x"], precision=5))
print()
print("Highland 积分 theta0(200 MeV, 10 cm) = %.4f mrad" %
      (ph.sigma_highland_integrated(200.0, 10.0) * 1e3))
print("Fermi-Eyges sqrt(A0)(10 cm)          = %.4f mrad" %
      (np.sqrt(ph.fermi_eyges_moments(np.array([10.0]), 200.0)["A0"][0]) * 1e3))
