# -*- coding: utf-8 -*-
"""
scripts/run_validation.py — 独立蒙特卡洛参考验证与 Gamma Index 分析

流程
----
1. 构造临床风格平顶 IMPT 场（40 能量层，41x41 spots @ 5 mm，M = 67,240）
2. 用 GPU 加速的凝聚历史蒙特卡洛引擎独立输运，得到参考剂量场
3. 用解析 + GPU 引擎重构剂量场（逐层精确）
4. 归一化后做 Gamma Index 分析（3%/3mm，TG-218 口径）
5. 多判据扫描 + 剖面图 + 图表输出
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
from pbdose.model import DoseGrid, flat_impt_field, clinical_layer_energies
from pbdose import engines as E
from pbdose import gamma as G
from pbdose import viz
import pbdose.montecarlo as MC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def build(n_grid, dvox, n_layers, n_spot, spacing, half_width, model="bohr",
          field="flat", nz=None, sigma_x0=0.30):
    """非立方网格：横向 n_grid x n_grid，深度 nz（默认 n_grid）。"""
    nx = ny = n_grid
    nz = nz or n_grid
    grid = DoseGrid(nx, ny, nz, dvox, dvox, dvox,
                    -(nx - 1) * dvox / 2.0, -(ny - 1) * dvox / 2.0, 0.0)
    energies = clinical_layer_energies(n_layers, 3.0, 25.5)
    return flat_impt_field(energies, grid, n_spot=n_spot, spot_spacing=spacing,
                           field_half_width=half_width,
                           idd_model=ph.IDDModel(straggling_model=model),
                           beam_sigma_x0=sigma_x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--histories", type=int, default=1_000_000,
                    help="每个能量层的历史数")
    ap.add_argument("--ngrid", type=int, default=160)
    ap.add_argument("--nz", type=int, default=155, help="深度体素数 (0.2 cm -> 31 cm)")
    ap.add_argument("--dvox", type=float, default=0.2)
    ap.add_argument("--nlayers", type=int, default=40)
    ap.add_argument("--nspot", type=int, default=41)
    ap.add_argument("--straggling", default="bohr")
    ap.add_argument("--sigma-x0", dest="sigma_x0", type=float, default=0.30,
                    help="入射束横向 rms 尺寸 [cm]，必须与解析模型一致")
    ap.add_argument("--tag", default="bohr")
    args = ap.parse_args()

    print("=" * 78)
    print("独立蒙特卡洛验证 + Gamma Index 分析")
    print("=" * 78)

    prob = build(args.ngrid, args.dvox, args.nlayers, args.nspot, 0.5, 5.0,
                 model=args.straggling, nz=args.nz, sigma_x0=args.sigma_x0)
    print(prob.summary())
    # ---------------- 1. 解析 + GPU 引擎 ----------------
    t0 = time.perf_counter()
    d_gpu = E.torch_separable(prob, "cuda", per_layer=True).double().cpu().numpy()
    t_an = time.perf_counter() - t0
    print(f"\n[1] 解析 GPU 引擎 (逐层精确) : {t_an*1e3:.1f} ms, peak={d_gpu.max():.4e}")

    # ---------------- 2. 蒙特卡洛参考 ----------------
    cfg = MC.MCConfig()
    if hasattr(cfg, "energy_straggling"):
        cfg.energy_straggling = True
        cfg.straggling_model = "bohr"
        print("    MC: 能量歧离已开启 (Bohr)")
    else:
        print("    !! MC 尚未支持能量歧离 —— Bragg 峰区对比将失效")

    lat = {"n_sx": prob.lattice.n_sx, "n_sy": prob.lattice.n_sy,
           "n_layers": prob.lattice.n_layers,
           "dx_spot": prob.lattice.dx_spot, "dy_spot": prob.lattice.dy_spot,
           # 注意：mc_impt_field 采用与 SpotLattice 相同的约定 —— x0/y0 是点阵**中心**
           "x0": float(prob.lattice.x0), "y0": float(prob.lattice.y0),
           "energies": prob.lattice.energies,
           "weights": prob.lattice.ensure_weights()}
    print(f"\n[2] 蒙特卡洛参考: {args.nlayers} 层 x {args.histories:,} histories "
          f"= {args.nlayers*args.histories/1e6:.0f} M histories ...")
    print(f"    入射束相空间必须与解析模型一致: sigma_x0 = "
          f"{prob.meta.get('入射束 sigma_x0','?')}")
    t0 = time.perf_counter()
    mc = MC.mc_impt_field(lat, prob.grid.as_dict(), histories_per_spot=args.histories,
                          sigma_x0=args.sigma_x0, config=cfg, device="cuda",
                          seed=20261005)
    t_mc = time.perf_counter() - t0
    d_mc = mc["dose"].double().cpu().numpy() if torch.is_tensor(mc["dose"]) else \
        np.asarray(mc["dose"], dtype=np.float64)
    print(f"    完成: {t_mc:.1f} s, peak={d_mc.max():.4e}")

    # ---------------- 3. 归一化 ----------------
    # 两者均为相对分布：各自归一化到高剂量区（>50% 最大值）内的均值，
    # 避免单点统计噪声影响归一化。
    roi = d_mc >= 0.5 * d_mc.max()
    d_mc_n = d_mc / d_mc[roi].mean()
    d_gpu_n = d_gpu / d_gpu[roi].mean()
    print(f"    归一化 ROI 体素数 = {roi.sum():,} "
          f"({roi.sum()/d_mc.size*100:.1f}% of grid)")

    # ---------------- 4. Gamma 分析 ----------------
    spacing = (prob.grid.dz, prob.grid.dy, prob.grid.dx)
    print("\n[3] Gamma Index (3%/3mm, global, 10% threshold) ...")
    t0 = time.perf_counter()
    gam = G.gamma_index(d_mc_n, d_gpu_n, spacing, 3.0, 3.0, 10.0)
    t_gam = time.perf_counter() - t0
    print(f"    通过率 = {gam['pass_rate']:.3f} %   "
          f"γ_mean = {gam['gamma_mean']:.4f}   γ_max = {gam['gamma_max']:.3f}   "
          f"({t_gam:.1f} s, {gam['n_eval']:,} voxels, {gam['n_dta_offsets']} DTA offsets)")

    dd = G.dose_difference(d_mc_n, d_gpu_n, 10.0)
    print(f"    剂量差: mean = {dd['mean_rel_pct']:+.2f}%, "
          f"RMS = {dd['rms_rel_pct']:.2f}%, max|·| = {dd['max_abs_rel_pct']:.2f}%")

    # --- 附加：排除表面建成区（前 3 个深度平面）后的通过率 ---
    #     临床 TPS QA 的标准做法：建成区（build-up）不参与 γ 统计。
    n_buildup = 3
    gam_nb = G.gamma_index(d_mc_n[n_buildup:], d_gpu_n[n_buildup:], spacing,
                           3.0, 3.0, 10.0, return_maps=False)
    print(f"    排除表面建成区 (前 {n_buildup} 层, z < "
          f"{n_buildup*prob.grid.dz:.1f} cm) 后: 通过率 = "
          f"{gam_nb['pass_rate']:.3f} %   gamma_mean = {gam_nb['gamma_mean']:.4f}")

    print("\n[4] 多判据扫描 ...")
    scan = G.gamma_pass_rate_vs_criteria(
        d_mc_n, d_gpu_n, spacing,
        criteria=((3.0, 3.0), (3.0, 2.0), (2.0, 3.0), (2.0, 2.0), (1.0, 1.0)),
        threshold_pct=10.0)
    for s in scan:
        print(f"    {s['dose_pct']:.0f}%/{s['dta_mm']:.0f}mm : "
              f"{s['pass_rate']:.2f} %   (γ_mean={s['gamma_mean']:.3f})")

    # ---------------- 5. 深度剂量与横向剖面 ----------------
    nz, ny, nx = d_mc_n.shape
    zax = prob.grid.z
    prof = {
        "z": zax.tolist(),
        "mc_central": d_mc_n[:, ny // 2, nx // 2].tolist(),
        "gpu_central": d_gpu_n[:, ny // 2, nx // 2].tolist(),
        "mc_maxslice": d_mc_n.max(axis=(1, 2)).tolist(),
        "gpu_maxslice": d_gpu_n.max(axis=(1, 2)).tolist(),
    }
    mtr_mc = ph.idd_peak_metrics(zax, d_mc_n.max(axis=(1, 2)))
    mtr_gpu = ph.idd_peak_metrics(zax, d_gpu_n.max(axis=(1, 2)))
    print("\n[5] 深度剂量指标 (max-over-slice 剖面):")
    for k in ("R_peak", "R80", "R50", "distal_falloff_80_20", "peak_to_entrance"):
        print(f"    {k:<24} MC={mtr_mc[k]:9.4f}   GPU={mtr_gpu[k]:9.4f}   "
              f"diff={mtr_gpu[k]-mtr_mc[k]:+8.4f}")

    # ---------------- 6. 图表 ----------------
    print("\n[6] 生成图表 ...")
    paths = []
    paths.append(viz.fig_gamma(gam, prob.grid, FIG, ref_map=d_mc_n,
                               eval_map=d_gpu_n))
    paths.append(viz.fig_criteria_scan(scan, FIG))
    for p in paths:
        print("   ", p)

    # ---------------- 7. 保存 ----------------
    out = {
        "problem": {"grid": [nx, ny, nz], "voxel_cm": prob.grid.dx,
                    "n_layers": args.nlayers, "n_spot_per_layer": args.nspot**2,
                    "M": prob.lattice.n_spots, "straggling": args.straggling,
                    "histories_per_layer": args.histories,
                    "total_histories": args.nlayers * args.histories},
        "timing": {"analytic_gpu_ms": t_an * 1e3, "mc_s": t_mc, "gamma_s": t_gam},
        "gamma_3_3": {k: v for k, v in gam.items()
                      if k not in ("gamma", "mask", "ref", "eval")},
        "dose_difference": {k: v for k, v in dd.items()
                            if k not in ("diff", "rel_pct", "mask")},
        "criteria_scan": scan,
        "depth_metrics": {"mc": mtr_mc, "gpu": mtr_gpu},
    }
    jp = os.path.join(RES, f"validation_{args.tag}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    np.save(os.path.join(RES, f"dose_mc_{args.tag}.npy"), d_mc_n)
    np.save(os.path.join(RES, f"dose_gpu_{args.tag}.npy"), d_gpu_n)
    np.save(os.path.join(RES, f"gamma_{args.tag}.npy"), gam["gamma"])
    print(f"\n结果已保存: {jp}")
    print("=" * 78)
    print(f"结论: 3%/3mm Gamma 通过率 = {gam['pass_rate']:.2f}%  "
          f"(目标 > 98%)  -> {'通过' if gam['pass_rate']>98 else '未达标'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
