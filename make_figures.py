# -*- coding: utf-8 -*-
"""
scripts/make_figures.py — 生成全部科研图表
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pbdose import physics as ph
from pbdose.model import DoseGrid, gaussian_impt_field, clinical_layer_energies
from pbdose import engines as E
from pbdose import viz
from scripts.run_benchmark import build_problem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    print("生成图表 ...")
    # ---------- 1. 物理验证 ----------
    sv = ph.validate_stopping_power(verbose=False)
    z = np.arange(0.0, 34.0, 0.02)
    idds = {}
    for E0 in (100.0, 150.0, 200.0, 230.0):
        idd, _ = ph.idd_pristine(z, E0)
        idds[E0] = {"z": z, "idd": idd}
    phys_res = {"stopping": sv, "range_max_relerr_pct": sv["R_max_relerr_pct"],
                "idd": idds}
    print(" ", viz.fig_physics(phys_res, FIG))

    prob = build_problem()
    print(" ", viz.fig_lateral(prob, FIG))

    # ---------- 2. 剂量分布（临床风格平顶野 10 x 10 cm） ----------
    print("  构造临床风格平顶野 (10 x 10 cm) 用于剂量分布图 ...")
    from pbdose.model import flat_impt_field
    prob_clin = flat_impt_field(
        clinical_layer_energies(40, 3.0, 25.5),
        DoseGrid.centered(128, 0.2), n_spot=41, spot_spacing=0.5,
        field_half_width=5.0)
    d_clin = E.torch_separable(prob_clin, "cuda", per_layer=True)\
        .double().cpu().numpy()
    print(" ", viz.fig_dose_maps(d_clin, prob_clin.grid, FIG,
                                 title="IMPT dose field (10x10 cm flat-top)"))
    np.save(os.path.join(RES, "dose_clinical_flat10x10.npy"), d_clin)

    # ---------- 3. 精度变体（基准算例：2.8 mm 高斯包络小野） ----------
    d_exact = E.torch_separable(prob, "cuda", per_layer=True).double().cpu().numpy()
    variants = {}
    d = E.torch_separable(prob, "cuda", per_layer=False).double().cpu().numpy()
    variants["single-sigma bmm (FP32)"] = d
    d = E.torch_separable(prob, "cuda", per_layer=True, layer_chunk=4)\
        .double().cpu().numpy()
    variants["per-layer, layer_chunk=4"] = d
    d = E.torch_gather_local(prob, "cuda").double().cpu().numpy()
    variants["local-window gather (FP32)"] = d
    d = E.torch_separable(prob, "cuda", per_layer=True, allow_tf32=True)\
        .double().cpu().numpy()
    variants["per-layer, TF32"] = d
    print(" ", viz.fig_accuracy_comparison(d_exact, variants, prob.grid, FIG))

    # ---------- 4. 性能 ----------
    bpath = os.path.join(RES, "benchmark_full.json")
    if not os.path.exists(bpath):
        bpath = os.path.join(RES, "benchmark_quick.json")
    with open(bpath, encoding="utf-8") as f:
        bench = json.load(f)
    ops = dict(bench["extra"]["ops"])
    ops["roofline"] = bench["extra"].get("roofline", {})
    print(" ", viz.fig_speedup(bench["rows"], ops, FIG))

    # 保存精确剂量供报告
    np.save(os.path.join(RES, "dose_exact_f32_gpu.npy"), d_exact)
    print("图表输出目录:", FIG)
    for f in sorted(os.listdir(FIG)):
        print("   ", f, f"{os.path.getsize(os.path.join(FIG, f))/1024:.0f} KB")


if __name__ == "__main__":
    main()
