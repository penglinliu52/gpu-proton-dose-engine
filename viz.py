"""
pbdose.viz — 科研图表生成（性能曲线 / 剂量分布 / Gamma 图）
==========================================================

统一使用英文坐标轴标签（期刊与申请材料通用惯例），高 DPI 输出 PNG。
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "figure.autolayout": False,
    # 中文字体（Windows）：Microsoft YaHei / SimHei，回退 DejaVu Sans
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

CMAP_DOSE = "inferno"
CMAP_DIFF = "coolwarm"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 物理曲线
# ---------------------------------------------------------------------------
def fig_physics(phys_results: dict, outdir: str):
    """阻止本领/射程验证 + IDD + sigma(z) 三联图。"""
    ensure_dir(outdir)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) 阻止本领 vs NIST PSTAR
    r = phys_results["stopping"]
    E, S_ref, S_bb = r["E"], r["S_ref"], r["S_bethe_bloch"]
    ax[0].loglog(E, S_ref, "o", ms=6, label="NIST PSTAR (reference)", color="k")
    ax[0].loglog(E, S_bb, "-", lw=2, label="Bethe-Bloch (this work)")
    ax[0].set_xlabel("Proton kinetic energy [MeV]")
    ax[0].set_ylabel(r"Mass stopping power [MeV cm$^2$/g]")
    ax[0].set_title("(a) Stopping power: model vs NIST PSTAR\n"
                    f"max deviation {r['S_bb_max_relerr_pct']:.2f}% (1-300 MeV)")
    ax[0].legend(fontsize=8)

    # (b) 射程-能量关系
    Eg = np.linspace(20, 300, 200)
    from . import physics as ph
    ax[1].plot(Eg, ph.csda_range(Eg), "-", lw=2, label="CSDA range (numerical)")
    ax[1].plot([200, 250, 300], [25.92, 37.90, 51.31], "ks", ms=7,
               label="NIST PSTAR anchors")
    ax[1].set_xlabel("Proton kinetic energy [MeV]")
    ax[1].set_ylabel("CSDA range in water [cm]")
    ax[1].set_title(f"(b) Range-energy relation\nmax deviation "
                    f"{phys_results['range_max_relerr_pct']:.2f}%")
    ax[1].legend(fontsize=8)

    # (c) IDD
    for E0, c in zip((100.0, 150.0, 200.0, 230.0), ("C0", "C1", "C2", "C3")):
        z, idd = phys_results["idd"][E0]["z"], phys_results["idd"][E0]["idd"]
        ax[2].plot(z, idd, color=c, lw=2, label=f"{E0:.0f} MeV")
    ax[2].set_xlabel("Depth in water [cm]")
    ax[2].set_ylabel("Relative dose (peak = 1)")
    ax[2].set_title("(c) Analytical pristine Bragg peaks\n"
                    "CSDA energy-conserving binning + Bohr straggling")
    ax[2].legend(fontsize=8)
    ax[2].set_xlim(0, 34)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_physics_validation.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_lateral(problem, outdir: str):
    """Fermi-Eyges 侧向展宽 sigma(z) 与理论曲线对比。"""
    from . import physics as ph
    ensure_dir(outdir)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    z = np.linspace(0.01, 25.0, 300)
    for E0, c in zip((100.0, 150.0, 200.0), ("C0", "C1", "C2")):
        m = ph.fermi_eyges_moments(z, E0, sigma_x0=0.3)
        R0 = float(ph.csda_range(E0))
        msk = z < R0
        ax[0].plot(z[msk], m["sigma_x"][msk] * 10, color=c, lw=2,
                   label=f"{E0:.0f} MeV  ($R_0$={R0:.1f} cm)")
    ax[0].set_xlabel("Depth in water [cm]")
    ax[0].set_ylabel(r"Lateral rms width $\sigma_x$ [mm]")
    ax[0].set_title("(a) Fermi-Eyges lateral spread\n"
                    r"$\sigma_x^2(z)=\sigma_{x0}^2+\int_0^z(z-u)^2T(u)\,du$")
    ax[0].legend(fontsize=8)

    # T(z) 与 A0(z)
    m = ph.fermi_eyges_moments(np.linspace(0.01, 25.0, 300), 200.0)
    zz = np.linspace(0.01, 25.0, 300)
    ax[1].semilogy(zz, m["T"], lw=2, color="C3", label=r"$T(z)$ [rad$^2$/cm]")
    ax[1].set_xlabel("Depth in water [cm]")
    ax[1].set_ylabel(r"Scattering power $T$ [rad$^2$/cm]", color="C3")
    ax[1].tick_params(axis="y", labelcolor="C3")
    ax2 = ax[1].twinx()
    ax2.plot(zz, m["A0"] * 1e3, lw=2, color="C0", label=r"$A_0(z)$")
    ax2.set_ylabel(r"$A_0(z)=\int_0^z T\,du$  [$10^{-3}$ rad$^2$]", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.grid(False)
    ax[1].set_title("(b) Local scattering power and Fermi-Eyges moment\n"
                    "differentiated Highland formula, 200 MeV")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_fermi_eyges.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 剂量分布
# ---------------------------------------------------------------------------
def fig_dose_maps(dose, grid, outdir: str, title="IMPT dose field"):
    """三维剂量的三个正交剖面 + 深度剂量曲线。"""
    ensure_dir(outdir)
    nz, ny, nx = dose.shape
    kz = int(0.55 * nz)
    ky = ny // 2
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    ext_yx = [grid.ox, grid.ox + (nx - 1) * grid.dx,
              grid.oy, grid.oy + (ny - 1) * grid.dy]
    im = ax[0].imshow(dose[kz], origin="lower", extent=ext_yx, cmap=CMAP_DOSE,
                      vmin=0, vmax=np.percentile(dose[kz], 99.5))
    ax[0].set_title(f"(a) Axial slice z = {grid.oz + kz*grid.dz:.1f} cm")
    ax[0].set_xlabel("x [cm]"); ax[0].set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    ext_xz = [grid.ox, grid.ox + (nx - 1) * grid.dx,
              grid.oz, grid.oz + (nz - 1) * grid.dz]
    im = ax[1].imshow(dose[:, ky, :], origin="lower", extent=ext_xz,
                      cmap=CMAP_DOSE, aspect="auto",
                      vmin=0, vmax=np.percentile(dose[:, ky, :], 99.5))
    ax[1].set_title(f"(b) Sagittal slice y = {grid.oy + ky*grid.dy:.1f} cm")
    ax[1].set_xlabel("x [cm]"); ax[1].set_ylabel("z [cm]")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    # 深度剂量（中心轴）
    zax = grid.z
    ax[2].plot(zax, dose[:, ny // 2, nx // 2], lw=2, color="C3",
               label="central axis")
    ax[2].plot(zax, np.array([dose[k].max() for k in range(nz)]), lw=1.5,
               ls="--", color="C0", label="max over slice")
    ax[2].set_xlabel("Depth z [cm]"); ax[2].set_ylabel("Dose [a.u.]")
    ax[2].set_title("(c) Depth-dose (SOBP)")
    ax[2].legend(fontsize=8)

    # 对比：逐层精确 vs 单 sigma 近似（横向剖面）
    ax[3].plot(grid.x, dose[nz // 2, ny // 2, :], lw=2, label="depth-dose slice")
    ax[3].set_xlabel("x [cm]"); ax[3].set_ylabel("Dose [a.u.]")
    ax[3].set_title(f"(d) Lateral profile at z = "
                    f"{grid.oz + (nz//2)*grid.dz:.1f} cm")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_dose_maps.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_accuracy_comparison(dose_exact, dose_variants: dict, grid, outdir: str):
    """逐层精确 vs 各近似/精度变体的差异图。"""
    ensure_dir(outdir)
    n = len(dose_variants)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 8.4))
    ax = np.atleast_2d(ax)
    nz, ny, nx = dose_exact.shape
    kz = int(0.75 * nz)
    ref_max = dose_exact.max()
    ext = [grid.ox, grid.ox + (nx - 1) * grid.dx,
           grid.oy, grid.oy + (ny - 1) * grid.dy]
    for i, (name, d) in enumerate(dose_variants.items()):
        rel = (d - dose_exact) / ref_max * 100.0
        gmax = float(np.abs(rel).max())
        v = max(float(np.percentile(np.abs(rel[kz]), 99.0)), 1e-9)
        im = ax[0, i].imshow(rel[kz], origin="lower", extent=ext, cmap=CMAP_DIFF,
                             vmin=-v, vmax=v)
        ax[0, i].set_title(f"{name}\nmax |diff| = {gmax:.2e}% of peak"
                           + ("   (bit-identical)" if gmax == 0 else ""))
        ax[0, i].set_xlabel("x [cm]"); ax[0, i].set_ylabel("y [cm]")
        plt.colorbar(im, ax=ax[0, i], fraction=0.046, label="[% of peak]")

        # 深度方向最大偏差（对全零情形加下限，避免对数轴为空）
        prof = np.maximum(np.abs(rel).max(axis=(1, 2)), 1e-12)
        ax[1, i].plot(grid.z, prof, lw=2, color="C3")
        ax[1, i].set_xlabel("Depth z [cm]")
        ax[1, i].set_ylabel("max |diff| [% of peak]")
        ax[1, i].set_yscale("log")
        ax[1, i].set_title("depth-wise maximum deviation")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_accuracy_variants.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------
def fig_speedup(rows, ops, outdir: str):
    """加速链柱状图 + 算术强度散点（roofline 视角）。"""
    ensure_dir(outdir)
    names, med, dev, alg = [], [], [], []
    for r in rows:
        r = r.as_dict() if hasattr(r, "as_dict") else r
        if r["median_ms"] <= 0:
            continue
        names.append(r["name"])
        med.append(r["median_ms"])
        dev.append(r["device"])
        alg.append(r["algorithm"])

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    order = np.argsort(med)[::-1]
    y = np.arange(len(order))
    colors = ["#c0392b" if "CPU" in dev[i] else
              ("#2980b9" if "Triton" in names[i] else "#27ae60") for i in order]
    ax[0].barh(y, [med[i] for i in order], color=colors)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([names[i] for i in order], fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Median wall time [ms] (log scale)")
    ax[0].set_title("(a) Acceleration chain: IMPT dose reconstruction\n"
                    f"M = {ops['M']:,} pencil beams, {ops['N_vox']/1e6:.2f} M voxels")
    ax[0].axvline(100, color="k", ls="--", lw=1.5)
    ax[0].text(105, len(order) - 1, "< 100 ms target", fontsize=8, rotation=90,
               va="top")
    for i, idx in enumerate(order):
        ax[0].text(med[idx] * 1.15, i, f"{med[idx]:.2f} ms", va="center", fontsize=7)

    # ---- (b) Roofline ----
    micro_path = os.path.join(os.path.dirname(outdir.rstrip("/\\")), "gpu_microbench.json")
    if not os.path.exists(micro_path):
        micro_path = os.path.join(outdir, "..", "gpu_microbench.json")
    micro = {}
    if os.path.exists(micro_path):
        import json as _json
        with open(micro_path, encoding="utf-8") as f:
            micro = _json.load(f)
    bw = micro.get("dram_bw_gbs", 552.0)          # GB/s
    peak = micro.get("gemm_fp32_4096_tflops", 14.8) * 1e3   # GFLOP/s
    small = micro.get("bmm_batch5120_tflops", 4.11) * 1e3   # 小批量 GEMM 实测

    algo2roof = {
        "naive": "naive", "triton_naive": "naive",
        "gather_local": "gather_local", "triton_gather": "gather_local",
        "separable": "separable", "separable-approx": "separable",
        "triton_conv": "separable",
    }
    roof = ops.get("roofline", {}) if isinstance(ops, dict) else {}
    markers = {"naive": "o", "gather_local": "s", "separable": "^",
               "separable-approx": "^", "triton_gather": "s", "triton_conv": "^",
               "triton_naive": "o"}
    cmap = {"CPU": "#c0392b", "GPU": "#27ae60", "Triton": "#2980b9"}

    ai_axis = np.logspace(-2, 2, 200)
    ax[1].plot(ai_axis, bw * ai_axis, "k--", lw=1.5,
               label=f"DRAM roofline ({bw:.0f} GB/s)")
    ax[1].axhline(peak, color="gray", ls=":", lw=1.5,
                  label=f"FP32 GEMM peak ({peak/1e3:.1f} TFLOP/s)")
    ax[1].axhline(small, color="#8e44ad", ls="-.", lw=1.2,
                  label=f"small-batch bmm ({small/1e3:.2f} TFLOP/s)")

    pts = []
    for r in rows:
        r = r.as_dict() if hasattr(r, "as_dict") else r
        key = algo2roof.get(r["algorithm"])
        if key is None or key not in roof or r["median_ms"] <= 0:
            continue
        info = roof[key]
        ai = info["arithmetic_intensity"]
        gflops = info["total_flops"] / (r["median_ms"] * 1e-3) / 1e9
        fam = ("Triton" if r["algorithm"].startswith("triton")
               else ("CPU" if "CPU" in r["device"] else "GPU"))
        pts.append((ai, gflops, r, fam))
    seen = set()
    for ai, gflops, r, fam in pts:
        lbl = fam if fam not in seen else None
        seen.add(fam)
        ax[1].scatter(ai, gflops, s=150, marker=markers.get(r["algorithm"], "o"),
                      color=cmap[fam], edgecolor="k", zorder=4, label=lbl)
        ax[1].annotate(r["name"].split(" (")[0][:18], (ai, gflops),
                       textcoords="offset points", xytext=(7, 5), fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax[1].set_ylabel("Achieved performance [GFLOP/s]")
    ax[1].set_title("(b) Roofline analysis\n"
                    "naive gather exceeds the DRAM roof by cache reuse")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig5_performance.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------
def fig_gamma(gamma_res: dict, grid, outdir: str, ref_map=None, eval_map=None,
              title="Gamma analysis"):
    """Gamma 图 + Gamma 直方图 + 剂量剖面。"""
    ensure_dir(outdir)
    g = gamma_res["gamma"]
    mask = gamma_res["mask"]
    nz, ny, nx = g.shape
    kz = int(0.6 * nz)
    gm = np.where(mask, g, np.nan)

    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.4))
    ext = [grid.ox, grid.ox + (nx - 1) * grid.dx,
           grid.oy, grid.oy + (ny - 1) * grid.dy]
    im = ax[0].imshow(gm[kz], origin="lower", extent=ext, cmap="jet",
                      vmin=0, vmax=2)
    ax[0].set_title(f"(a) Gamma map, z = {grid.oz + kz*grid.dz:.1f} cm "
                    f"({gamma_res['dose_crit_pct']:.0f}%/"
                    f"{gamma_res['dta_mm']:.0f}mm)")
    ax[0].set_xlabel("x [cm]"); ax[0].set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    ext_xz = [grid.ox, grid.ox + (nx - 1) * grid.dx,
              grid.oz, grid.oz + (nz - 1) * grid.dz]
    im = ax[1].imshow(gm[:, ny // 2, :], origin="lower", extent=ext_xz,
                      cmap="jet", vmin=0, vmax=2, aspect="auto")
    ax[1].set_title("(b) Gamma, sagittal slice")
    ax[1].set_xlabel("x [cm]"); ax[1].set_ylabel("z [cm]")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    gm_flat = gm[np.isfinite(gm)]
    ax[2].hist(np.clip(gm_flat, 0, 3), bins=120, color="C0", alpha=0.85)
    ax[2].axvline(1.0, color="r", ls="--", lw=2, label=r"$\gamma = 1$")
    ax[2].set_xlabel(r"$\gamma$ index")
    ax[2].set_ylabel("voxel count")
    ax[2].set_yscale("log")
    ax[2].set_title(f"(c) Gamma histogram\npass rate = "
                    f"{gamma_res['pass_rate']:.2f}%  "
                    f"($\\gamma_{{mean}}$={gamma_res['gamma_mean']:.3f})")
    ax[2].legend(fontsize=8)

    if ref_map is not None and eval_map is not None:
        zax = grid.z
        ax[3].plot(zax, ref_map[:, ny // 2, nx // 2], lw=2, label="reference (MC)")
        ax[3].plot(zax, eval_map[:, ny // 2, nx // 2], lw=2, ls="--",
                   label="analytic GPU engine")
        ax[3].set_xlabel("Depth z [cm]"); ax[3].set_ylabel("Dose [a.u.]")
        ax[3].set_title("(d) Central-axis depth dose")
        ax[3].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig6_gamma.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_criteria_scan(scan, outdir: str):
    """不同判据下的 Gamma 通过率。"""
    ensure_dir(outdir)
    labels = [f"{s['dose_pct']:.0f}%/{s['dta_mm']:.0f}mm" for s in scan]
    vals = [s["pass_rate"] for s in scan]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, vals, color=["#27ae60" if v >= 98 else
                                       ("#f39c12" if v >= 95 else "#c0392b")
                                       for v in vals])
    ax.axhline(98, color="k", ls="--", lw=1.2, label="project target 98%")
    ax.axhline(95, color="gray", ls=":", lw=1.2, label="TG-218 tolerance 95%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Gamma pass rate [%]")
    ax.set_xlabel("Dose / DTA criterion")
    ax.set_title("Gamma pass rate under multiple criteria")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig7_gamma_criteria.png")
    fig.savefig(p)
    plt.close(fig)
    return p
