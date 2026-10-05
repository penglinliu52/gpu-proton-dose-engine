"""
tools/mc_selftest.py — acceptance tests for pbdose.montecarlo
============================================================

Run:  python tools/mc_selftest.py

Checks (in order):
  1  mc_spot_dose, 200 MeV, 2e6 histories, 128^3 @ 0.2 cm grid, CUDA < 60 s
  2  R90 within 3 % of physics.csda_range(200)
  3  lateral rms at z = 10 cm within 5 % of physics.fermi_eyges_moments
     (phase-space estimator; the voxelised dose-plane value is reported too)
  4  mc_impt_field on a 4x4x3 lattice returns a finite dose tensor
  5  wide beam (sigma_x0 = 1 cm): voxelised rms vs Fermi-Eyges
  6  reproducibility for a fixed seed; CUDA never silently downgraded to CPU
  7  bookkeeping: counter closure, energy balance, step schedule
  8  grid given as a dict and as an object both work
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbdose import physics                      # noqa: E402
from pbdose import montecarlo as mc             # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def idd_r90_analytic(E0: float, lam: float = 150.0, frac: float = 0.9,
                     n: int = 400_001) -> float:
    """R90 of the analytic CSDA + attenuation IDD: D(z) = exp(-z/lam) * S(E(z)).

    Exact expectation of the Monte Carlo deposit profile (a proton alive at
    depth z deposits S(E(z))dz there, so the expectation of the scored profile
    is fluence(z)*S(E(z)) with no other correction).
    """
    R0 = float(physics.csda_range(E0))
    z = np.linspace(0.0, R0, n)
    E = physics.energy_from_range(np.clip(R0 - z, 0.0, None))
    S = physics.stopping_power_water(np.clip(E, 1e-3, None))
    d = np.exp(-z / lam) * S
    c = np.cumsum(d)
    c /= c[-1]
    return float(np.interp(frac, c, z))


def expected_binned_sigma(sigma: float, dx: float) -> float:
    """Voxelisation bias of a scored Gaussian profile (see the module helper)."""
    return mc.binned_gaussian_sigma(sigma, dx)


def fe_reference(E0: float, depths, sigma_x0: float = 0.0,
                 sigma_theta0: float = 0.0, npts: int = 1001) -> dict:
    """Fermi-Eyges sigma_x / sigma_theta interpolated at the requested depths.

    ``npts`` is deliberately larger than the module default grid: the discrete
    upper-triangle sum inside ``fermi_eyges_moments`` is a left-rectangle rule
    whose error is O(h/z) (~0.4 % on sigma_x at z = 2 cm for npts = 501, ~0.02 %
    at z = 10 cm), which would otherwise show up as an apparent MC bias.
    """
    R0 = float(physics.csda_range(E0))
    z = np.linspace(0.0, R0 * 1.0001, npts)
    m = physics.fermi_eyges_moments(z, E0, sigma_x0=sigma_x0, sigma_theta0=sigma_theta0)
    return {
        "sigma_x": {d: float(np.interp(d, z, m["sigma_x"])) for d in depths},
        "sigma_theta": {d: float(np.sqrt(np.interp(d, z, m["A0"]))) for d in depths},
    }


def main() -> int:
    t_all = time.perf_counter()
    print("pbdose.montecarlo self test")
    print(f"  torch {torch.__version__}   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  sm_{p.major}{p.minor}  "
              f"{p.total_memory / 2**30:.1f} GiB  {p.multi_processor_count} SMs")
    print(f"  physics.csda_range(200) = {float(physics.csda_range(200.0)):.5f} cm")

    # ------------------------------------------------------------------ 1-3
    banner("TEST 1/2/3 — 200 MeV single spot, 2e6 histories, 128^3 @ 0.2 cm")
    grid = dict(nx=128, ny=128, nz=128, dx=0.2, dy=0.2, dz=0.2,
                ox=-12.8, oy=-12.8, oz=0.0)
    N = 2_000_000
    t0 = time.perf_counter()
    res = mc.mc_spot_dose(200.0, grid, n_histories=N, config=mc.DEFAULT_MC,
                          device="cuda", seed=12345, chunk_size=2_000_000)
    wall = time.perf_counter() - t0
    st = res["stat"]
    R_csda = float(physics.csda_range(200.0))

    print(f"  histories            : {st['n_history']:,}")
    print(f"  wall time            : {res['wall_time_s']:.3f} s  "
          f"(outer {wall:.3f} s)")
    print(f"  histories / second   : {st['histories_per_s']:.4g}")
    print(f"  steps / chunks       : {st['n_steps']} / {st['n_chunks']}  "
          f"(planned {st['n_steps_planned']})  dtype={st['dtype']} device={st['device']}")
    print(f"  energy / primary     : {st['mean_energy_deposited']:.4f} MeV "
          f"(balance {st['energy_balance'] * 100:.3f} %)")
    print(f"  exit / nuclear / ecut: {st['n_exit']:,} / {st['n_nuclear']:,} / "
          f"{st['n_ecut']:,}  (absorbed {res['n_absorbed']:,})")
    print(f"  R90 (MC)             : {st['r90_depth']:.4f} cm")
    print(f"  csda_range(200)      : {R_csda:.4f} cm   [NIST PSTAR 25.92]")
    d90 = 100.0 * (st["r90_depth"] - R_csda) / R_csda
    print(f"  R90 deviation        : {d90:+.3f} %")
    print(f"  R80 / R50            : {st['r80_depth']:.4f} / {st['r50_depth']:.4f} cm")

    fe = fe_reference(200.0, (2.0, 5.0, 10.0, 15.0, 20.0))
    print("  lateral width, MC phase-space estimator vs Fermi-Eyges")
    print(f"    {'z[cm]':>6} {'sigma_MC':>10} {'sigma_FE':>10} {'dev[%]':>8} "
          f"{'n':>10}")
    devs = {}
    for d in (2.0, 5.0, 10.0, 15.0, 20.0):
        sm = st["sigma_x_at_depth"][d]
        sf = fe["sigma_x"][d]
        devs[d] = 100.0 * (sm - sf) / sf
        print(f"    {d:6.1f} {sm:10.6f} {sf:10.6f} {devs[d]:+8.3f} "
              f"{st['n_at_depth'][d]:10,}")
    sth_mc = st["sigma_theta_at_depth"][10.0]
    sth_fe = fe["sigma_theta"][10.0]
    print(f"    sigma_theta(10 cm): MC {sth_mc:.6f} rad  FE {sth_fe:.6f} rad  "
          f"({100.0 * (sth_mc - sth_fe) / sth_fe:+.3f} %)")
    # the scored plane [10.0, 10.2) receives deposits at z = 10.0 and 10.1, so
    # compare against the Fermi-Eyges width at the voxel centre (10.1 cm)
    sig_slab = float(np.interp(10.1, np.linspace(0.0, R_csda * 1.0001, 501),
                               physics.fermi_eyges_moments(
                                   np.linspace(0.0, R_csda * 1.0001, 501), 200.0)["sigma_x"]))
    exp_binned = expected_binned_sigma(sig_slab, 0.2)
    print(f"  voxelised dose-plane rms at 10 cm : "
          f"{st['voxelised_sigma_x_at_depth'][10.0]:.6f} cm  vs binned-FE expectation "
          f"{exp_binned:.6f} cm "
          f"({100.0 * (st['voxelised_sigma_x_at_depth'][10.0] - exp_binned) / exp_binned:+.2f} %)")
    check("3d voxelised dose-plane rms matches the binned Fermi-Eyges value",
          abs(st["voxelised_sigma_x_at_depth"][10.0] - exp_binned) / exp_binned < 0.02,
          f"{100.0 * (st['voxelised_sigma_x_at_depth'][10.0] - exp_binned) / exp_binned:+.2f} %")

    check("1  runtime < 60 s on CUDA", res["wall_time_s"] < 60.0,
          f"{res['wall_time_s']:.2f} s, {st['histories_per_s']:.3g} hist/s")
    check("1b dose on CUDA (no silent CPU fallback)", str(st["device"]).startswith("cuda"),
          str(st["device"]))
    check("2  R90 within 3 % of csda_range(200)", abs(d90) < 3.0, f"{d90:+.3f} %")
    check("3  sigma_x(10 cm) within 5 % of Fermi-Eyges", abs(devs[10.0]) < 5.0,
          f"{devs[10.0]:+.3f} %")
    check("3b sigma_x within 5 % at all recorded depths",
          all(abs(v) < 5.0 for v in devs.values()),
          ", ".join(f"{d:g}cm {v:+.2f}%" for d, v in devs.items()))
    check("3c sigma_theta(10 cm) within 5 %",
          abs(100.0 * (sth_mc - sth_fe) / sth_fe) < 5.0,
          f"{100.0 * (sth_mc - sth_fe) / sth_fe:+.3f} %")

    # 7 — bookkeeping
    n_sum = st["n_exit"] + st["n_nuclear"] + st["n_ecut"]
    check("7a counter closure n_exit+n_nuclear+n_ecut == n_primary", n_sum == N,
          f"{n_sum:,} vs {N:,}")
    print(f"  NOTE: the 128^3 grid ends at z = 25.6 cm while csda_range(200) = "
          f"{R_csda:.3f} cm, so protons leave the distal face and bank their")
    print(f"        remaining energy there (n_exit = {st['n_exit']:,}); "
          f"n_steps {st['n_steps']} < planned {st['n_steps_planned']}.")
    check("7b step schedule is an upper bound on the loop length",
          0 < st["n_steps"] <= st["n_steps_planned"] ==
          int(mc.mc_step_schedule(200.0).size),
          f"{st['n_steps']} <= {st['n_steps_planned']}")
    check("7c energy outside the scoring grid is negligible",
          st["energy_outside_scoring_grid"] <= 0.0,
          f"{st['energy_outside_scoring_grid']:.3e} MeV")
    check("7d energy balance in [0.85, 1.0]",
          0.85 <= st["energy_balance"] <= 1.0, f"{st['energy_balance']:.4f}")
    check("7e dose sum == reported total",
          abs(float(res["dose"].sum()) - st["total_energy_deposited"]) < 1e-6)
    check("7f n_absorbed == n_exit + n_nuclear",
          res["n_absorbed"] == st["n_exit"] + st["n_nuclear"])

    # 7g — phantom deep enough to stop the beam: no truncation artefact
    deep = dict(nx=128, ny=128, nz=140, dx=0.2, dy=0.2, dz=0.2,
                ox=-12.8, oy=-12.8, oz=0.0)
    dres = mc.mc_spot_dose(200.0, deep, n_histories=500_000, device="cuda",
                           seed=12345, chunk_size=500_000)
    dst = dres["stat"]
    d90_deep = 100.0 * (dst["r90_depth"] - R_csda) / R_csda
    r90_idd = idd_r90_analytic(200.0, lam=mc.DEFAULT_MC.lambda_att)
    print(f"  140-voxel-deep phantom (z to 28 cm): R90 = {dst['r90_depth']:.4f} cm "
          f"({d90_deep:+.3f} %), n_exit = {dst['n_exit']}, "
          f"n_ecut = {dst['n_ecut']:,}, steps = {dst['n_steps']} "
          f"(planned {dst['n_steps_planned']})")
    print(f"  analytic CSDA+attenuation IDD R90  : {r90_idd:.4f} cm  "
          f"({100.0 * (dst['r90_depth'] - r90_idd) / r90_idd:+.3f} %)")
    check("7g untruncated phantom: no escapes and R90 within 3 % of csda_range",
          dst["n_exit"] == 0 and abs(d90_deep) < 3.0
          and dst["n_steps"] == dst["n_steps_planned"], f"{d90_deep:+.3f} %")
    check("7h R90 matches the analytic CSDA+attenuation IDD within 1 %",
          abs(dst["r90_depth"] - r90_idd) / r90_idd < 0.01,
          f"{100.0 * (dst['r90_depth'] - r90_idd) / r90_idd:+.3f} %")

    # ------------------------------------------------------------------ 4
    banner("TEST 4 — mc_impt_field on a 4x4x3 lattice")
    n_sx, n_sy, n_layers = 4, 4, 3
    energies = np.array([120.0, 150.0, 180.0])
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.0, size=(n_layers, n_sy, n_sx))
    lattice = dict(n_sx=n_sx, n_sy=n_sy, n_layers=n_layers, dx_spot=1.0, dy_spot=1.0,
                   x0=0.0, y0=0.0, energies=energies, weights=weights)
    ln = dict(nx=96, ny=96, nz=100, dx=0.25, dy=0.25, dz=0.25,
              ox=-12.0, oy=-12.0, oz=0.0)
    t0 = time.perf_counter()
    fld = mc.mc_impt_field(lattice, ln, histories_per_spot=100_000,
                           device="cuda", seed=999, chunk_size=2_000_000)
    fwall = time.perf_counter() - t0
    fd = fld["dose"]
    print(f"  dose shape           : {tuple(fd.shape)}  dtype {fd.dtype}")
    print(f"  finite               : {fld['stat']['finite']}   "
          f"min {float(fd.min()):.3e}  max {float(fd.max()):.4e}  "
          f"sum {float(fd.sum()):.6e} MeV")
    print(f"  wall time            : {fld['wall_time_s']:.3f} s (outer {fwall:.3f} s)")
    print(f"  histories            : {fld['n_primary']:,} "
          f"({fld['n_history']:,}), spots used {fld['stat']['n_spots']}")
    for L in fld["stat"]["layers"]:
        print(f"    layer E={L['E']:6.1f} MeV  spots={L['n_spots']:2d}  "
              f"R90={L['r90_depth']:.3f} cm  kernel={L['kernel_energy_MeV']:.4g} MeV  "
              f"{L['wall_time_s']:.2f} s")
    check("4a mc_impt_field runs and is finite",
          bool(torch.isfinite(fd).all()) and float(fd.sum()) > 0.0)
    check("4b dose non-negative", float(fd.min()) >= 0.0)
    check("4c weighted field differs per layer (physics varies with E)",
          len({round(L["r90_depth"], 3) for L in fld["stat"]["layers"]}) == n_layers)

    # 4d/4e — placement must reproduce a direct transport exactly
    #       (mc_impt_field uses seed + 1000*(layer+1) for layer 0)
    one = dict(n_sx=1, n_sy=1, n_layers=1, dx_spot=1.0, dy_spot=1.0,
               x0=0.0, y0=0.0, energies=[200.0], weights=np.ones((1, 1, 1)))
    f0 = mc.mc_impt_field(one, grid, histories_per_spot=200_000, device="cuda",
                          seed=5, chunk_size=2_000_000)
    d0 = mc.mc_spot_dose(200.0, grid, n_histories=200_000, device="cuda",
                         seed=1005, chunk_size=2_000_000)
    e_f, e_d = float(f0["dose"].sum()), float(d0["dose"].sum())
    print(f"  1-spot field vs direct transport: sum {e_f:.6e} vs {e_d:.6e} MeV "
          f"(rel {abs(e_f - e_d) / e_d:.2e}), crop-clipped "
          f"{f0['stat']['kernel_crop_clipped_energy']:.2e} MeV")
    check("4d aligned single-spot field == direct transport (energy)",
          abs(e_f - e_d) / e_d < 1e-6, f"{abs(e_f - e_d) / e_d:.2e}")
    zc = np.arange(128) * 0.2 + 0.1
    xc = np.arange(128) * 0.2 - 12.8 + 0.1
    yc = xc
    p_f = f0["dose"].sum(dim=0).double().cpu().numpy()
    p_d = d0["dose"].sum(dim=0).double().cpu().numpy()
    cx_f = float((p_f * xc[None, :]).sum() / p_f.sum())
    cx_d = float((p_d * xc[None, :]).sum() / p_d.sum())
    print(f"  lateral centroid x: field {cx_f:+.5f} cm, direct {cx_d:+.5f} cm")
    check("4e aligned placement keeps the centroid", abs(cx_f) < 5e-3,
          f"{cx_f:+.5f} cm")

    one_off = dict(one, x0=0.13, y0=-0.07)
    fo = mc.mc_impt_field(one_off, grid, histories_per_spot=200_000, device="cuda",
                          seed=5, chunk_size=2_000_000)
    po = fo["dose"].sum(dim=0).double().cpu().numpy()
    cx_o = float((po * xc[None, :]).sum() / po.sum())
    cy_o = float((po * yc[:, None]).sum() / po.sum())
    e_o = float(fo["dose"].sum())
    print(f"  sub-voxel spot (0.13, -0.07): centroid ({cx_o:+.5f}, {cy_o:+.5f}) cm, "
          f"energy rel diff {abs(e_o - e_d) / e_d:.2e}")
    check("4f sub-voxel placement conserves energy",
          abs(e_o - e_d) / e_d < 1e-6, f"{abs(e_o - e_d) / e_d:.2e}")
    check("4g sub-voxel placement puts the centroid at the spot",
          abs(cx_o - 0.13) < 5e-3 and abs(cy_o + 0.07) < 5e-3,
          f"({cx_o:+.5f}, {cy_o:+.5f}) vs (0.13, -0.07)")

    # ------------------------------------------------------------------ 5
    banner("TEST 5 — wide beam (sigma_x0 = 1 cm): voxelised rms vs Fermi-Eyges")
    wide = mc.mc_spot_dose(200.0, grid, n_histories=400_000, sigma_x0=1.0,
                           device="cuda", seed=4242, chunk_size=2_000_000)
    fe_w = fe_reference(200.0, (10.0, 10.1), sigma_x0=1.0)
    sm = wide["stat"]["voxelised_sigma_x_at_depth"][10.0]
    sf = fe_w["sigma_x"][10.0]
    exp_vox = expected_binned_sigma(fe_w["sigma_x"][10.1], 0.2)
    print(f"  dose-plane sigma_x(10) : {sm:.6f} cm")
    print(f"  Fermi-Eyges            : {sf:.6f} cm   "
          f"(binned expectation at the voxel centre -> {exp_vox:.6f})")
    print(f"  phase-space estimator  : {wide['stat']['sigma_x_at_depth'][10.0]:.6f} cm")
    check("5a dose-plane rms agrees within 1 %", abs(sm - exp_vox) / exp_vox < 0.01,
          f"{100.0 * (sm - exp_vox) / exp_vox:+.3f} % vs binned FE")
    check("5b phase-space rms agrees within 2 %",
          abs(wide["stat"]["sigma_x_at_depth"][10.0] - sf) / sf < 0.02,
          f"{100.0 * (wide['stat']['sigma_x_at_depth'][10.0] - sf) / sf:+.3f} %")

    # ------------------------------------------------------------------ 6
    banner("TEST 6 — reproducibility and device handling")
    a = mc.mc_spot_dose(200.0, grid, n_histories=200_000, device="cuda", seed=777,
                        chunk_size=200_000)
    b = mc.mc_spot_dose(200.0, grid, n_histories=200_000, device="cuda", seed=777,
                        chunk_size=200_000)
    c = mc.mc_spot_dose(200.0, grid, n_histories=200_000, device="cuda", seed=778,
                        chunk_size=200_000)
    dmax_same = float((a["dose"] - b["dose"]).abs().max())
    dmax_diff = float((a["dose"] - c["dose"]).abs().max())
    print(f"  |same seed| max abs diff : {dmax_same:.3e} MeV")
    print(f"  |seed+1|   max abs diff : {dmax_diff:.3e} MeV")
    check("6a identical seed reproduces the dose", dmax_same == 0.0, f"{dmax_same:.3e}")
    check("6b different seed changes the dose", dmax_diff > 0.0)
    cpu = mc.mc_spot_dose(200.0, grid, n_histories=20_000, device="cpu", seed=1,
                          chunk_size=20_000)
    check("6c explicit device='cpu' works", str(cpu["stat"]["device"]) == "cpu",
          f"{cpu['stat']['device']}, {cpu['wall_time_s']:.2f} s")
    check("6d chunking keeps the same physics (n_steps)",
          a["stat"]["n_steps"] == res["stat"]["n_steps"],
          f"{a['stat']['n_steps']} vs {res['stat']['n_steps']}")

    # ------------------------------------------------------------------ 8
    banner("TEST 8 — grid as object, and grid geometry variants")
    gs = mc.GridSpec(nx=64, ny=64, nz=100, dx=0.4, dy=0.4, dz=0.2,
                     ox=-12.8, oy=-12.8, oz=-0.2)
    obj = mc.mc_spot_dose(150.0, gs, n_histories=200_000, device="cuda", seed=3)
    R150 = float(physics.csda_range(150.0))
    r90_150 = idd_r90_analytic(150.0, lam=mc.DEFAULT_MC.lambda_att)
    print(f"  GridSpec run: R90 = {obj['stat']['r90_depth']:.4f} cm, "
          f"csda_range(150) = {R150:.4f} cm, analytic IDD R90 = {r90_150:.4f} cm, "
          f"{obj['wall_time_s']:.2f} s")
    d150 = 100.0 * (obj["stat"]["r90_depth"] - R150) / R150
    check("8a object-style grid accepted", obj["dose"].shape == (100, 64, 64))
    check("8b 150 MeV R90 within 3 % of csda_range", abs(d150) < 3.0, f"{d150:+.3f} %")
    check("8b2 150 MeV R90 within 1 % of the analytic IDD",
          abs(obj["stat"]["r90_depth"] - r90_150) / r90_150 < 0.01,
          f"{100.0 * (obj['stat']['r90_depth'] - r90_150) / r90_150:+.3f} %")
    try:
        mc.mc_spot_dose(200.0, dict(nx=8, ny=8, nz=8, dx=0.2, dy=0.2, dz=0.2,
                                    oz=1.0), n_histories=10, device="cpu")
        raised = False
    except ValueError:
        raised = True
    check("8c grid not containing z = 0 raises ValueError", raised)

    # ------------------------------------------------------------------ 9
    banner("TEST 9 — step-size convergence, energy conservation, integrator")
    import dataclasses as _dc
    c_coarse = _dc.replace(mc.DEFAULT_MC, ds_max=0.2, step_frac=0.04, min_ds=5e-3)
    c_fine = _dc.replace(mc.DEFAULT_MC, ds_max=0.05, step_frac=0.01, min_ds=1e-3)
    c_noatt = _dc.replace(mc.DEFAULT_MC, lambda_att=float("inf"))
    c_euler = _dc.replace(mc.DEFAULT_MC, integrator="euler")
    runs = {}
    for tag, cfg in (("coarse", c_coarse), ("default", mc.DEFAULT_MC),
                     ("fine", c_fine), ("no-atten", c_noatt), ("euler", c_euler)):
        r = mc.mc_spot_dose(200.0, deep, n_histories=300_000, device="cuda",
                            seed=21, config=cfg)
        runs[tag] = r["stat"]
        print(f"  {tag:9s} steps={r['stat']['n_steps']:4d}  "
              f"R90={r['stat']['r90_depth']:.4f} cm  "
              f"sigma_x(10)={r['stat']['sigma_x_at_depth'][10.0]:.6f}  "
              f"balance={r['stat']['energy_balance']:.6f}  "
              f"{r['wall_time_s']:.2f} s")
    b = runs["default"]
    dR_coarse = 100.0 * (runs["coarse"]["r90_depth"] - b["r90_depth"]) / b["r90_depth"]
    dR_fine = 100.0 * (runs["fine"]["r90_depth"] - b["r90_depth"]) / b["r90_depth"]
    ds_coarse = 100.0 * (runs["coarse"]["sigma_x_at_depth"][10.0]
                         - b["sigma_x_at_depth"][10.0]) / b["sigma_x_at_depth"][10.0]
    ds_fine = 100.0 * (runs["fine"]["sigma_x_at_depth"][10.0]
                       - b["sigma_x_at_depth"][10.0]) / b["sigma_x_at_depth"][10.0]
    check("9a R90 converged w.r.t. step size (<0.1 % coarse vs fine)",
          abs(dR_coarse - dR_fine) < 0.1, f"coarse {dR_coarse:+.3f} %, fine {dR_fine:+.3f} %")
    check("9b sigma_x(10) step-size stable (<1 %)",
          abs(ds_coarse) < 1.0 and abs(ds_fine) < 1.0,
          f"coarse {ds_coarse:+.3f} %, fine {ds_fine:+.3f} %")
    check("9c energy exactly conserved without nuclear absorption",
          abs(runs["no-atten"]["energy_balance"] - 1.0) < 1e-9,
          f"balance = {runs['no-atten']['energy_balance']:.12f}")
    check("9d euler integrator works and stays within 1 % of the exact CSDA R90",
          abs(100.0 * (runs["euler"]["r90_depth"] - b["r90_depth"]) / b["r90_depth"]) < 1.0,
          f"euler R90 {runs['euler']['r90_depth']:.4f} vs {b['r90_depth']:.4f} cm")

    # ------------------------------------------------------------------ 10
    banner("TEST 10 — Bohr energy straggling (200 MeV pristine peak)")
    sgrid = dict(nx=64, ny=64, nz=180, dx=0.2, dy=0.2, dz=0.2,
                 ox=-6.4, oy=-6.4, oz=0.0)          # z to 36 cm
    sgs = mc.GridSpec(**sgrid)
    zc_s = sgs.centers("z")
    z_fine = np.arange(0.0, 34.0, 0.005)             # 0.5 mm analysis grid

    def peak_metrics(dose, z_src, z_out=z_fine):
        prof = mc.depth_profile(dose) if hasattr(dose, "detach") else np.asarray(dose)
        p = np.interp(z_out, z_src, prof)
        idd = p / p.max()
        entrance = float(np.mean(idd[: max(3, int(0.03 * len(idd)))]))
        def cross(level):
            idx = np.where(idd >= level)[0]
            i = idx[-1]
            y0, y1 = idd[i], idd[i + 1]
            return float(z_out[i] + (level - y0) / (y1 - y0)
                         * (z_out[i + 1] - z_out[i]))
        r80, r20 = cross(0.8), cross(0.2)
        return {"peak_to_entrance": float(idd.max() / entrance),
                "falloff_80_20_mm": (r20 - r80) * 10.0, "R80": r80, "R20": r20,
                "sigma_R_implied": (r20 - r80) / 1.683}

    sig_bohr = float(physics.range_straggling_bohr(200.0))
    sig_icru = float(physics.range_straggling_icru49(200.0))
    scale_icru = sig_icru / sig_bohr
    c_off = _dc.replace(mc.DEFAULT_MC, lambda_att=1e9)
    c_bohr = _dc.replace(c_off, energy_straggling=True, straggling_model="bohr")
    c_icru = _dc.replace(c_bohr, straggling_scale=scale_icru)
    out = {}
    for tag, cfg in (("CSDA (straggling off)", c_off),
                     ("MC bohr  (scale 1.0)", c_bohr),
                     (f"MC icru49(scale {scale_icru:.3f})", c_icru)):
        r = mc.mc_spot_dose(200.0, sgrid, n_histories=1_000_000, device="cuda",
                            seed=2024, config=cfg)
        m = peak_metrics(r["dose"], zc_s)
        m["wall"] = r["wall_time_s"]
        m["steps"] = r["stat"]["n_steps"]
        m["balance"] = r["stat"]["energy_balance"]
        m["ratio"] = r["stat"]["straggling_sampled_over_theory"]
        m["delta_rms"] = r["stat"]["straggling_delta_rms"]
        m["omega_rms"] = r["stat"]["straggling_omega_rms"]
        m["sigma_R_hist"] = r["stat"]["range_straggling_measured"]
        out[tag] = m
        print(f"  {tag:26s} peak/entr {m['peak_to_entrance']:6.3f}   "
              f"F80-20 {m['falloff_80_20_mm']:5.2f} mm   "
              f"sigma_R(hist) {m['sigma_R_hist']:.4f} cm   steps {m['steps']}  "
              f"{m['wall']:.1f} s")
    # the analytic IDD is compared with the SAME (disabled) attenuation as the MC
    ATTN = dict(attenuation_length=1e9)
    for tag, m in (("bohr   (es=0)", physics.IDDModel(straggling_model="bohr",
                                                      energy_spread=0.0, **ATTN)),
                   ("bohr   (es=.0033)", physics.IDDModel(straggling_model="bohr",
                                                          **ATTN)),
                   ("icru49 (es=0)", physics.IDDModel(straggling_model="icru49",
                                                      energy_spread=0.0, **ATTN)),
                   ("icru49 (es=.0033)", physics.IDDModel(straggling_model="icru49",
                                                          **ATTN)),
                   ("none   (es=0)", physics.IDDModel(straggling_model="none",
                                                      energy_spread=0.0, **ATTN))):
        idd_a, sR = physics.idd_pristine(z_fine, 200.0, m)
        met = physics.idd_peak_metrics(z_fine, idd_a)
        print(f"  analytic {tag:16s} peak/entr {met['peak_to_entrance']:6.3f}   "
              f"F80-20 {met['distal_falloff_80_20'] * 10:5.2f} mm   sigma_R {sR:.4f} cm")
    b_mc = out["MC bohr  (scale 1.0)"]
    b_an = peak_metrics(physics.idd_pristine(z_fine, 200.0,
                                             physics.IDDModel(straggling_model="bohr",
                                                              energy_spread=0.0,
                                                              **ATTN))[0], z_fine)
    i_mc = out[f"MC icru49(scale {scale_icru:.3f})"]
    s_off = out["CSDA (straggling off)"]
    print(f"  straggling diagnostic: delta rms {b_mc['delta_rms']:.6f} MeV vs "
          f"theory omega rms {b_mc['omega_rms']:.6f} MeV -> ratio "
          f"{b_mc['ratio']:.4f} (reflection removes "
          f"{100 * (1 - b_mc['ratio'] ** 2):.1f} % of the raw variance)")
    print(f"  analytic Bohr sigma_R = {sig_bohr:.4f} cm, ICRU-49 = {sig_icru:.4f} cm")
    check("10a straggling switches on and conserves energy",
          abs(b_mc["balance"] - 1.0) < 1e-9, f"balance {b_mc['balance']:.9f}")
    check("10b sampled fluctuation matches sqrt(BOHR_COEF*ds/beta^2) within 5 %",
          abs(b_mc["ratio"] - 1.0) < 0.05,
          f"delta_rms {b_mc['delta_rms']:.6f} vs omega_rms {b_mc['omega_rms']:.6f} MeV")
    check("10c peak/entrance in 4.5-6.9 (analytic Bohr 5.0-5.7)",
          4.5 <= b_mc["peak_to_entrance"] <= 6.9, f"{b_mc['peak_to_entrance']:.3f}")
    check("10d distal 80-20 % falloff in 4.5-9.5 mm (analytic Bohr 6.7-7.1 mm)",
          4.5 <= b_mc["falloff_80_20_mm"] <= 9.5, f"{b_mc['falloff_80_20_mm']:.2f} mm")
    check("10e MC falloff within 25 % of the analytic Bohr convolution",
          abs(b_mc["falloff_80_20_mm"] - b_an["falloff_80_20_mm"])
          / b_an["falloff_80_20_mm"] < 0.25,
          f"{b_mc['falloff_80_20_mm']:.2f} vs {b_an['falloff_80_20_mm']:.2f} mm")
    print(f"  NOTE: F80-20/1.683 = {b_mc['sigma_R_implied']:.3f} cm is only a shape-"
          f"dependent proxy; the direct stopping-depth histogram above is the test.")
    check("10e2 direct MC range spread (stopping-depth histogram) within 10 % of Bohr",
          abs(b_mc["sigma_R_hist"] - sig_bohr) / sig_bohr < 0.10,
          f"sigma_R(hist) {b_mc['sigma_R_hist']:.4f} vs {sig_bohr:.4f} cm")
    check("10e3 ICRU-49 scaled range spread within 10 % of the ICRU-49 fit",
          abs(i_mc["sigma_R_hist"] - sig_icru) / sig_icru < 0.10,
          f"sigma_R(hist) {i_mc['sigma_R_hist']:.4f} vs {sig_icru:.4f} cm")
    check("10f CSDA (off) peak is much sharper, as expected",
          s_off["falloff_80_20_mm"] < 0.6 * b_mc["falloff_80_20_mm"]
          and s_off["peak_to_entrance"] > 1.5 * b_mc["peak_to_entrance"],
          f"F80-20 {s_off['falloff_80_20_mm']:.2f} mm, "
          f"peak/entr {s_off['peak_to_entrance']:.2f}")
    check("10g ICRU-49 scaled run lands near the ICRU-49 analytic falloff",
          2.0 <= i_mc["falloff_80_20_mm"] <= 6.0, f"{i_mc['falloff_80_20_mm']:.2f} mm")

    # ------------------------------------------------------------------ 11
    banner("TEST 11 — finite incident phase space + Gamma vs the analytic engine")
    # 11a/11b: the source size must appear in the phase-space width as
    #          sqrt(sigma_x0^2 + sigma_scatter(z)^2) and be chunk-size stable
    z_ref11 = np.linspace(0.0, 26.0, 1001)
    fe11 = physics.fermi_eyges_moments(z_ref11, 200.0, sigma_x0=0.30)
    sig_fe11 = float(np.interp(10.0, z_ref11, fe11["sigma_x"]))
    scat11 = float(np.interp(10.0, z_ref11,
                             physics.fermi_eyges_moments(z_ref11, 200.0)["sigma_x"]))
    expect11 = math.hypot(0.30, scat11)
    src = {}
    for cs in (250_000, 1_000_000):
        r = mc.mc_spot_dose(200.0, deep, n_histories=1_000_000, sigma_x0=0.30,
                            device="cuda", seed=3, chunk_size=cs)
        src[cs] = r["stat"]["sigma_x_at_depth"][10.0]
    print(f"  sigma_x(10 cm) with sigma_x0 = 0.30: chunk250k {src[250_000]:.6f} cm, "
          f"chunk1M {src[1_000_000]:.6f} cm")
    print(f"    Fermi-Eyges {sig_fe11:.6f} cm, "
          f"sqrt(0.30^2 + {scat11:.6f}^2) = {expect11:.6f} cm")
    check("11a finite source size appears as sqrt(sigma_x0^2 + sigma_scatter^2)",
          abs(src[1_000_000] - expect11) / expect11 < 0.01,
          f"{src[1_000_000]:.6f} vs {expect11:.6f} cm "
          f"({100 * (src[1_000_000] - expect11) / expect11:+.3f} %)")
    check("11b phase-space width stable against chunk_size",
          abs(src[250_000] - src[1_000_000]) / src[1_000_000] < 0.005,
          f"delta {100 * (src[250_000] - src[1_000_000]) / src[1_000_000]:+.3f} %")

    try:
        from pbdose import engines as _E
        from pbdose import gamma as _G
        from pbdose import model as _M
    except Exception as exc:                                   # pragma: no cover
        print(f"  SKIP analytic comparison: {type(exc).__name__}: {exc}")
    else:
        E11 = float(physics.energy_from_range(10.0))
        d11, g11, gz11 = 0.2, 96, 160
        grid_a = _M.DoseGrid(g11, g11, gz11, d11, d11, d11,
                             ox=-(g11 - 1) * d11 / 2, oy=-(g11 - 1) * d11 / 2, oz=0.0)
        prob = _M.flat_impt_field(
            np.array([E11]), grid_a, n_spot=21, spot_spacing=0.5,
            field_half_width=4.0, fluence_normalized=True, beam_sigma_x0=0.30,
            idd_model=physics.IDDModel(straggling_model="bohr", energy_spread=0.0,
                                       attenuation_length=150.0))
        ref = _E.torch_separable(prob, "cuda", dtype=torch.float64,
                                 per_layer=True).cpu().numpy()
        lat11 = dict(n_sx=21, n_sy=21, n_layers=1, dx_spot=0.5, dy_spot=0.5,
                     x0=0.0, y0=0.0, energies=np.array([E11]),
                     weights=prob.lattice.ensure_weights())
        # voxel-centre alignment with DoseGrid.axis() = o + i*d
        grid_mc11 = dict(nx=g11, ny=g11, nz=gz11, dx=d11, dy=d11, dz=d11,
                         ox=grid_a.ox - d11 / 2, oy=grid_a.oy - d11 / 2,
                         oz=grid_a.oz - d11 / 2)
        cfg11 = _dc.replace(mc.DEFAULT_MC, energy_straggling=True,
                            straggling_model="bohr", lambda_att=150.0)
        N11 = 1_500_000
        r11 = mc.mc_impt_field(lat11, grid_mc11, histories_per_spot=N11,
                               sigma_x0=0.30, sigma_theta0=0.0, config=cfg11,
                               device="cuda", seed=12345, chunk_size=2_000_000)
        mc_d = r11["dose"].detach().cpu().double().numpy()
        m50 = mc_d > 0.5 * mc_d.max()
        # the two codes have independent absolute units here; each field is
        # normalised to its own mean over the same high-dose region (the
        # absolute ratio between the unit systems is reported as check 11f)
        ref_n = ref / float(ref[m50].mean())
        mc_n = mc_d / float(mc_d[m50].mean())
        gam = _G.gamma_index(ref_n, mc_n, spacing=(0.2, 0.2, 0.2),
                             dose_crit_pct=3.0, dta_mm=3.0, threshold_pct=10.0)
        mask = ref_n >= 0.10 * ref_n.max()
        rms11 = 100.0 * float(np.sqrt(((mc_n - ref_n)[mask] ** 2).mean())) \
            / float(ref_n.max())
        abs_ratio = float(mc_d[m50].mean()) / (float(ref[m50].mean()) * N11 * d11 ** 3)
        print(f"  layer E = {E11:.2f} MeV (R0 = 10 cm), 21x21 spots @ 0.5 cm, "
              f"flat-top +-4 cm, sigma_x0 = 0.30 cm, N = {N11:,}, "
              f"{r11['wall_time_s']:.1f} s")
        print(f"  3 %/3 mm Gamma (10 % threshold, global): {gam['pass_rate']:.2f} %  "
              f"(mean gamma {gam['gamma_mean']:.3f}, n_eval {gam['n_eval']})")
        print(f"  dose-difference RMS: {rms11:.3f} % of the reference peak")
        print(f"  absolute scale: MC per history/cm^3 vs analytic fluence-normalised "
              f"= {abs_ratio:.4f}  (1.0 = identical units)")
        row = g11 // 2
        a11 = ref_n[50, row, :] / ref_n[50, row, :].max()
        b11 = mc_n[50, row, :] / mc_n[50, row, :].max()
        cols = list(range(0, 96, 6))
        print("  lateral profile at z = 10 cm (row y = -0.1 cm), each to its own peak")
        print("    x[cm] " + " ".join(f"{grid_a.x[i]:+6.2f}" for i in cols))
        print("    anly  " + " ".join(f"{a11[i]:6.4f}" for i in cols))
        print("    MC    " + " ".join(f"{b11[i]:6.4f}" for i in cols))
        plateau = (np.abs(grid_a.x) <= 3.0) & (a11 > 0.4)
        dev_plateau = float(np.abs(b11[plateau] - a11[plateau]).max())
        check("11c single-layer 3 %/3 mm Gamma pass rate > 90 %",
              gam["pass_rate"] > 90.0, f"{gam['pass_rate']:.2f} %")
        check("11d dose-difference RMS < 3 % of the reference peak",
              rms11 < 3.0, f"{rms11:.3f} %")
        check("11e lateral profile within 4 % of the analytic in the plateau",
              dev_plateau < 0.04, f"max |delta| = {dev_plateau * 100:.2f} % of peak")
        check("11f absolute dose scale agrees within 3 % (per history vs per fluence)",
              abs(abs_ratio - 1.0) < 0.03, f"ratio {abs_ratio:.4f}")

    # ------------------------------------------------------------------
    banner("SUMMARY")
    print(f"  total self-test wall time: {time.perf_counter() - t_all:.2f} s")
    print(f"  200 MeV spot : {st['histories_per_s']:.4g} histories/s, "
          f"R90 {st['r90_depth']:.4f} cm vs csda {R_csda:.4f} cm ({d90:+.2f} %), "
          f"sigma_x(10) {st['sigma_x_at_depth'][10.0]:.6f} vs FE "
          f"{fe['sigma_x'][10.0]:.6f} cm ({devs[10.0]:+.2f} %)")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED CHECK(S):")
        for f in FAILURES:
            print(f"    - {f}")
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
