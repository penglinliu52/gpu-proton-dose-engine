# -*- coding: utf-8 -*-
"""对比已保存的 MC / GPU 剂量场深度剖面，定位 40 层场的不一致。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

mc = np.load(r"<REPO_ROOT>\results\dose_mc_bohr.npy")
gp = np.load(r"<REPO_ROOT>\results\dose_gpu_bohr.npy")
print("shapes", mc.shape, gp.shape)
z = np.arange(mc.shape[0]) * 0.2

pm = mc.max(axis=(1, 2)); pg = gp.max(axis=(1, 2))
sm = mc.sum(axis=(1, 2)); sg = gp.sum(axis=(1, 2))
pm = pm / pm.max(); pg = pg / pg.max()
sm = sm / sm.max(); sg = sg / sg.max()

idx = np.arange(0, mc.shape[0], 8)
print(f"{'z[cm]':>7} {'MC_max':>9} {'GPU_max':>9} | {'MC_sum':>9} {'GPU_sum':>9}")
for i in idx:
    print(f"{z[i]:7.1f} {pm[i]:9.4f} {pg[i]:9.4f} | {sm[i]:9.4f} {sg[i]:9.4f}")

print()
print("横向剖面 @ z=3cm (归一化到各自峰值)")
j = 15
k = int(3.0 / 0.2)
a = gp[k, j, :]; a = a / a.max()
b = mc[k, j, :]; b = b / b.max()
print("  x[cm] :", " ".join(f"{(-16 + i*0.2):6.1f}" for i in range(0, 160, 16)))
print("  GPU   :", " ".join(f"{a[i]:6.3f}" for i in range(0, 160, 16)))
print("  MC    :", " ".join(f"{b[i]:6.3f}" for i in range(0, 160, 16)))

print()
print("各层能量层核验（解析 IDD 在自身峰位的剂量，MeV/cm）:")
from pbdose import physics as ph
from pbdose.model import clinical_layer_energies
en = clinical_layer_energies(40, 3.0, 25.5)
zz = np.arange(0, 31, 0.02)
for l in (0, 10, 20, 30, 39):
    o = ph.idd_pristine(zz, en[l], return_components=True, normalize=False)
    print(f"  layer {l:2d}  E={en[l]:6.1f} MeV  R0={o['R0']:5.2f} cm  "
          f"IDD峰值={o['idd'].max():6.2f} MeV/cm  在 z=3cm 处 IDD={np.interp(3.0, zz, o['idd']):6.2f}")
