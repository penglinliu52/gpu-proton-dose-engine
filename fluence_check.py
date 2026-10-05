# -*- coding: utf-8 -*-
"""验证注量型 IDD 归一化修复：比较峰值归一化 vs 注量归一化的场，并给出单 sigma 误差。"""
import os
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pbdose import physics as ph
from pbdose.model import DoseGrid, flat_impt_field, clinical_layer_energies
from pbdose import engines as E

grid = DoseGrid(160, 160, 155, 0.2, 0.2, 0.2, -15.9, -15.9, 0.0)
en = clinical_layer_energies(40, 3.0, 25.5)

for fn in (True, False):
    t0 = time.perf_counter()
    p = flat_impt_field(en, grid, n_spot=41, spot_spacing=0.5,
                        field_half_width=5.0, fluence_normalized=fn)
    d = E.torch_separable(p, "cuda", per_layer=True).double().cpu().numpy()
    da = E.torch_separable(p, "cuda", per_layer=False).double().cpu().numpy()
    z = grid.z
    prof = d.max(axis=(1, 2)); prof = prof / prof.max()
    m = ph.idd_peak_metrics(z, prof)
    r = E.relative_error(da, d)
    i50 = np.where(prof >= 0.5)[0]
    print(f"fluence_normalized={fn}:  R_peak={m['R_peak']:.1f}  R50={m['R50']:.2f}  "
          f"R80={m['R80']:.2f}  峰/入射={m['peak_to_entrance']:.2f}  "
          f"单sigma max_rel={r['max_rel_err_global']:.4f}   "
          f"IDD峰值[MeV/cm] 首层={p.idd[0].max():.1f} 末层={p.idd[-1].max():.1f}  "
          f"({time.perf_counter()-t0:.1f}s)")
