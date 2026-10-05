# -*- coding: utf-8 -*-
"""最小受控对比：单层 / 双层，解析 vs 蒙特卡洛，逐层核对深度剂量。"""
import os
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch
from pbdose import physics as ph
from pbdose.model import DoseGrid, flat_impt_field
from pbdose import engines as E
import pbdose.montecarlo as MC

grid = DoseGrid(96, 96, 160, 0.2, 0.2, 0.2, -9.5, -9.5, 0.0)
cfg = MC.MCConfig(); cfg.energy_straggling = True; cfg.straggling_model = "bohr"


def run(energies, label):
    p = flat_impt_field(np.array(energies), grid, n_spot=21, spot_spacing=0.5,
                        field_half_width=4.0, fluence_normalized=True)
    d = E.torch_separable(p, "cuda", per_layer=True).double().cpu().numpy()
    lat = dict(n_sx=21, n_sy=21, n_layers=len(energies), dx_spot=0.5, dy_spot=0.5,
               x0=float(p.lattice.spot_x[0]), y0=float(p.lattice.spot_y[0]),
               energies=np.asarray(energies), weights=p.lattice.ensure_weights())
    t0 = time.time()
    mc = MC.mc_impt_field(lat, grid.as_dict(), histories_per_spot=400000,
                          config=cfg, device="cuda", seed=5)
    dm = mc["dose"].double().cpu().numpy()
    z = grid.z
    pa = d.sum(axis=(1, 2)); pm = dm.sum(axis=(1, 2))
    pa = pa / pa.max(); pm = pm / pm.max()
    ia50 = np.where(pa >= 0.5)[0]; im50 = np.where(pm >= 0.5)[0]
    ia90 = np.where(pa >= 0.9)[0]; im90 = np.where(pm >= 0.9)[0]
    print(f"--- {label}  ({time.time()-t0:.0f}s MC) ---")
    print(f"  解析: R_peak={z[pa.argmax()]:5.2f}  R90={z[ia90[-1]] if ia90.size else np.nan:5.2f}"
          f"  R50={z[ia50[-1]] if ia50.size else np.nan:5.2f}  峰/入射={pa.max()/pa[:20].mean():.3f}")
    print(f"  MC  : R_peak={z[pm.argmax()]:5.2f}  R90={z[im90[-1]] if im90.size else np.nan:5.2f}"
          f"  R50={z[im50[-1]] if im50.size else np.nan:5.2f}  峰/入射={pm.max()/pm[:20].mean():.3f}")
    # 面积（每层的总能量）核对
    print(f"  深度积分  解析={pa.sum():.3f} (归一化)   MC={pm.sum():.3f} (归一化)")
    return z, pa, pm


for ens, lab in [([ph.energy_from_range(20.0)], "单层 R0=20cm"),
                 ([ph.energy_from_range(6.0)], "单层 R0=6cm"),
                 ([ph.energy_from_range(6.0), ph.energy_from_range(20.0)], "双层 6+20cm")]:
    z, pa, pm = run(ens, lab)
    idx = np.linspace(0, len(z) - 1, 12).astype(int)
    print("   z[cm] :", " ".join(f"{z[i]:6.1f}" for i in idx))
    print("   解析  :", " ".join(f"{pa[i]:6.3f}" for i in idx))
    print("   MC    :", " ".join(f"{pm[i]:6.3f}" for i in idx))
