"""
pbdose.montecarlo — independent Monte-Carlo proton transport reference engine
=============================================================================

WHAT THIS IS
------------
A **home-built condensed-history (class II) proton Monte Carlo**, written from
scratch in vectorised PyTorch, used as an *independent* numerical reference for
the analytic Fermi-Eyges pencil-beam dose model of :mod:`pbdose.physics`.

    ***   THIS IS *NOT* GEANT4 / TOPAS / MCNP / FLUKA / PENELOPE.   ***

It is a few-hundred-line engine whose only purpose is to answer one question:
*does the analytic model's transport reproduce a stochastic realisation of the
very same equations?*  Agreement therefore validates the numerics and the moment
algebra of the analytic model (CSDA energy loss + Fermi-Eyges moment integration
+ attenuation).  It does **not** validate the absolute clinical accuracy of
either model, because both share the same simplified physics.

APPROXIMATIONS AND OMISSIONS (all deliberate, all important)
------------------------------------------------------------
1. **Continuous slowing down (CSDA), no energy straggling.**  The energy loss is
   deterministic along the path; the energy-loss distribution (Vavilov/Landau),
   range straggling and the beam energy spread are *not* simulated.  The
   simulated Bragg peak is consequently *sharper* than a real one and its distal
   edge is an ideal CSDA edge.  R90 is therefore expected to sit ~1.6-2 % below
   ``physics.csda_range(E0)``; that is a property of the model, not a bug.
2. **Gaussian multiple Coulomb scattering only.**  Per step a projected angle is
   drawn from ``N(0, T(E) * ds)`` in each transverse plane, with ``T`` from
   :func:`pbdose.physics.scattering_power` — *exactly* the scattering power the
   Fermi-Eyges analytic model integrates (see ``MCConfig.z_over_x0_reference``).
   Large-angle single scattering (Moliere/Bethe tails, which dominate
   out-of-field dose) is omitted; the transverse distribution is strictly
   Gaussian.
3. **No nuclear secondaries and no delta-ray transport.**  A nuclear interaction
   simply removes the proton with probability ``1 - exp(-ds / lambda_att)`` per
   step; the fragments' energy is neither transported nor deposited.  This is
   the same effective-attenuation knob the analytic IDD model exposes
   (``physics.IDDModel.attenuation_length``).  Delta rays (which carry dose
   away from the track) are ignored.
4. **Homogeneous water only** (no WED/heterogeneity), no magnetic field, and no
   lateral boundary physics other than the absorbing grid faces.
5. **Voxelised dose scoring.**  Energy is banked at the end of each step (or at
   the boundary crossing point), one deposit per history per step.  The scored
   lateral width therefore carries a voxelisation bias of ``dx^2/12``; the
   engine additionally reports *phase-space* second moments recorded at
   user-chosen depths, which are free of that bias — use those for comparisons
   with analytic moments.
6. Float64 by default; the only variance reduction is the per-layer pencil-beam
   reuse of :func:`mc_impt_field`.

HOW IT TRANSPORTS
-----------------
Every history is advanced simultaneously with PyTorch tensors; the Python loop
runs over transport **steps** (~455 for 200 MeV), never over histories.  Because
CSDA energy loss is deterministic, every live history shares the same residual
range, hence the same step length ``ds``, the same energy and the same depth
``z`` — those are Python scalars, and only ``x, y, theta_x, theta_y`` are
tensors.  Each step:

1. ``ds = min(ds_max, step_frac * R(E))``, floored at ``min_ds`` and capped by
   the residual range ``R(E)`` so that a step can never overshoot the end of
   range.
2. Draw the two projected scattering kicks
   ``kx, ky ~ N(0, T(E) * ds)`` with ``T = physics.scattering_power(E)``.
3. Kick and drift with the kick placed at the *middle* of the step,
   ``x += theta_x * ds + kx * ds/2``, which makes the discrete lateral variance
   second-order accurate against ``B(z) = int_0^z (z-u)^2 T(u) du``.
4. Deposit ``dE = E(R) - E(R - ds)`` in the voxel containing the new position
   (or at the boundary crossing point for escaping protons) with one
   ``scatter_add_`` on flattened linear indices.
5. Remove protons that left the grid (``n_exit``) or underwent a nuclear
   reaction (``n_nuclear``); repeat until the residual range is exhausted or
   ``E <= e_cut``, then deposit the remaining kinetic energy in place.

CSDA INTEGRATION
----------------
``physics.stopping_power_water`` and ``physics.csda_range`` are mutually
consistent (the range table is built by integrating ``1/S``).  The default
integrator therefore *integrates the CSDA ODE exactly* by stepping in residual
range: ``R -> R - ds``, ``E = physics.energy_from_range(R)``.  This is not a
first-order Euler approximation of ``dE/ds = -S(E)`` but its closed-form
solution for the tabulated ``S``.  ``MCConfig.integrator = "euler"`` switches to
the literal ``E -= stopping_power_water(E) * ds`` update, which loses range
systematically; it exists only to quantify that drift.

REPRODUCIBILITY
---------------
``torch.manual_seed(seed)`` is called and every draw uses a ``torch.Generator``
created on the compute device.  For a fixed ``(seed, n_histories, chunk_size,
config)`` the random stream — and hence the result — is reproducible.  The dose
deposit uses ``scatter_add_`` (float atomics), so the last bits of a voxel sum
can vary between runs at the ~1e-15 relative level; set
``MCConfig.deterministic_deposit = True`` for a sort-based (slower) deposit.

Author: research code base (pbdose).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:  # package context:  import pbdose.montecarlo
    from . import physics
except ImportError:  # script / standalone context
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from pbdose import physics  # type: ignore


__all__ = [
    "MCConfig",
    "DEFAULT_MC",
    "GridSpec",
    "mc_config_summary",
    "mc_step_schedule",
    "mc_spot_dose",
    "mc_impt_field",
    "depth_profile",
    "cumulative_depth_metric",
    "binned_gaussian_sigma",
    "straggling_docs",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class MCConfig:
    """Tunables of the condensed-history transport (clinical units: cm, MeV, rad).

    ds_max : float
        Maximum step length [cm]; caps the step in the entrance plateau.
    step_frac : float
        Adaptive rule ``ds = min(ds_max, step_frac * R(E))``; 0.02 keeps the
        energy loss per step near 2 % and resolves the Bragg peak.
    min_ds : float
        Absolute step floor [cm] used over the last ~0.1 cm of range, where the
        "2 % of range" rule would otherwise force tens of thousands of steps.
        20 um is two orders of magnitude below the 0.2 cm voxels used here.
    e_cut : float
        Kinetic-energy cut [MeV]; the remaining energy is deposited in place.
    lambda_att : float
        Effective nuclear attenuation length [cm] of the primary fluence:
        ``P(survive one step) = exp(-ds / lambda_att)``.  ``math.inf`` disables
        nuclear absorption.  150 cm ~ 0.67 %/cm, the same knob as
        ``physics.IDDModel.attenuation_length``.  Because the survival
        probability depends only on depth, the whole fate of a history is
        encoded in one exponential variate (so no per-step random draw).
    energy_straggling : bool
        ``False`` (default): pure CSDA, deterministic energy loss, every proton
        of a given energy has exactly the same range.  ``True``: sample the
        energy-loss fluctuation per proton per step (Bohr straggling), which
        makes the residual range of each history a random variable — this is
        what produces the physical width of the distal falloff.
        See :meth:`straggling_docs` for the model's assumptions.
    straggling_model : {"bohr", "none"}
        Only used when ``energy_straggling`` is True.  ``"bohr"`` uses
        ``dOmega^2/ds = BOHR_COEF / beta(E)^2`` with ``physics.BOHR_COEF``
        (= K (Z/A) rho m_e c^2 = 0.0871 MeV^2/cm in water), which is the same
        first-principles integral that :func:`pbdose.physics.range_straggling_bohr`
        evaluates analytically (sigma_R(200 MeV) = 0.511 cm).  ``"none"``
        disables the sampling.
    straggling_seed_offset : int
        The straggling variates are drawn from a *separate* ``torch.Generator``
        seeded with ``seed + straggling_seed_offset``, so the energy-loss stream
        is statistically independent of the multiple-scattering stream (and the
        scattering kicks stay bit-identical to a run with straggling disabled).
    straggling_scale : float
        Multiplier on the Bohr fluctuation width ``Omega`` (default 1.0 = first
        principles).  Because ``sigma_R`` is linear in ``Omega``, setting
        ``straggling_scale = physics.range_straggling_icru49(E0) /
        physics.range_straggling_bohr(E0)`` (0.493 at 200 MeV) makes the Monte
        Carlo reproduce the ICRU-49 empirical range-straggling fit instead, which
        is useful for bracketing the model uncertainty.
    scattering_model : str
        Passed to :func:`pbdose.physics.scattering_power`.
    z_over_x0_reference : float or None
        ``L = z/X0`` at which the Highland derivative is evaluated.  ``None``
        reproduces *exactly* the ``T(E)`` that ``physics.fermi_eyges_moments``
        integrates (that function passes no ``z_over_X0``, i.e. ``L = 1``),
        which is what makes the lateral-width agreement test meaningful.
    integrator : {"range", "euler"}
        ``"range"``: exact CSDA integration through ``csda_range`` /
        ``energy_from_range`` (recommended, self-consistent).  ``"euler"``:
        literal ``E -= stopping_power_water(E) * ds`` at each step.
    max_steps : int
        Hard cap on the Python step loop (safety valve; 4000 covers 400 MeV).
    moment_depths : tuple of float
        Depths [cm] at which the transverse phase-space second moments of the
        live population are recorded (a pure MC estimator, no voxel bias).
    deposit_escaped_energy : bool
        ``True``: a proton leaving the grid deposits its *remaining* energy at
        the crossing point.  This conserves energy and makes R90 insensitive to
        where the grid ends, as long as the boundary is distal to R90 (energy
        moved further distal cannot change a proximal cumulative fraction).
        ``False``: escaping protons are dropped and only the energy lost up to
        the crossing point (linear in the crossing fraction) is scored.
    deterministic_deposit : bool
        Use ``index_put_(accumulate=True)`` instead of ``scatter_add_``.
    dtype : {"float64", "float32"}
        Dtype of the transport state and of the dose grid.
    verbose : bool
        Print per-chunk progress and a summary line.
    """

    ds_max: float = 0.10
    step_frac: float = 0.02
    min_ds: float = 2.0e-3
    e_cut: float = 0.05
    lambda_att: float = 150.0
    energy_straggling: bool = False
    straggling_model: str = "bohr"
    straggling_seed_offset: int = 7919
    straggling_scale: float = 1.0
    scattering_model: str = "highland_local"
    z_over_x0_reference: float | None = None
    integrator: str = "range"
    max_steps: int = 4000
    moment_depths: tuple = (2.0, 5.0, 10.0, 15.0, 20.0, 24.0)
    deposit_escaped_energy: bool = True
    deterministic_deposit: bool = False
    dtype: str = "float64"
    verbose: bool = False


DEFAULT_MC = MCConfig()

_DTYPES = {"float64": torch.float64, "float32": torch.float32}

#: fallback value of ``physics.BOHR_COEF`` [MeV^2/cm] (water) if the physics
#: module ever stops exporting it:  K_BB * (Z/A) * rho * m_e c^2
BOHR_COEF_FALLBACK = 0.087107

#: proton rest energy [MeV] (fallback if physics.M_P_C2 disappears)
M_P_C2_FALLBACK = 938.27208816

STRAGGLING_DOC = """\
Bohr energy-loss straggling (MCConfig.energy_straggling=True)
------------------------------------------------------------
Per transport step of length ds the kinetic energy receives an independent
Gaussian kick in addition to the deterministic CSDA loss:

    dOmega^2 = BOHR_COEF * ds / beta(E)^2        [MeV^2],
    BOHR_COEF = K (Z/A) rho m_e c^2 = 0.0871 MeV^2/cm  (water),
    beta from the full relativistic beta(gamma) (physics.beta_gamma),
    E_new = clamp(E_CSDA(after ds) + N(0, dOmega^2), e_cut, E_before).

The proton then carries its own residual range (physics.csda_range(E_new)), so
range straggling *emerges* from the random walk instead of being imposed; in
the continuum limit the accumulated variance is

    sigma_R^2 = int BOHR_COEF / (beta(E)^2 S(E)^3) dE,

i.e. exactly the integral evaluated by physics.range_straggling_bohr()
(0.511 cm rms at 200 MeV in water, ~2x the ICRU 49 empirical fit 0.012 R^0.935).

Assumptions and known limitations of this model (all standard for Bohr):
  * free, stationary, unpolarised atomic electrons; no binding corrections and
    no density-effect correction to the straggling;
  * the finite maximum energy transfer T_max is neglected, i.e. the Fano /
    Bethe straggling correction is taken as ~1 (valid at these thicknesses; it
    is a few percent for the full track);
  * no explicit delta-ray transport: the energy lost to a hard collision stays
    in the same voxel as the proton;
  * the kick is Gaussian and independent of the multiple-scattering stream
    (separate generator, seed + straggling_seed_offset), so the two are
    uncorrelated by construction;
  * the energy is clamped to E_new <= E_before (a fluctuation can never *add*
    energy to the proton: the clamp, rather than a reflection, is used) and to
    E_new >= e_cut, which is also the stopping condition together with
    csda_range(E_new) == 0;
  * the step length is the shared CSDA schedule (driven by the largest residual
    range among the live histories); it is a numerical parameter, not physics.

With straggling enabled the 200 MeV pristine Bragg peak acquires the physical
shape: distal 80-20 % falloff ~6-7 mm and peak-to-entrance ~5 (versus ~1.2 mm
and ~12 for pure CSDA sampled on 0.2 cm voxels).
"""


def straggling_docs() -> str:
    """Text block documenting the Bohr straggling model (see module docstring)."""
    return STRAGGLING_DOC


def mc_config_summary(config: "MCConfig" = DEFAULT_MC) -> str:
    """Human-readable dump of an :class:`MCConfig`, with units and derived notes."""
    cfg = config
    if not np.isfinite(cfg.lambda_att):
        lam = "inf  (nuclear absorption disabled)"
    else:
        lam = f"{cfg.lambda_att:g} cm  (mean free path; {100.0 / cfg.lambda_att:.3f} %/cm)"
    zref = ("None  -> L = 1, the same T(E) that physics.fermi_eyges_moments uses"
            if cfg.z_over_x0_reference is None
            else f"{cfg.z_over_x0_reference:g} (z/X0)")
    integ = ("range   (exact CSDA:  R -= ds ;  E = physics.energy_from_range(R))"
             if cfg.integrator == "range"
             else "euler   (E -= physics.stopping_power_water(E) * ds)")
    if not cfg.energy_straggling:
        strg = "off     (pure CSDA: deterministic energy loss, zero range spread)"
    elif str(cfg.straggling_model).lower() == "none":
        strg = "off     (straggling_model = 'none')"
    else:
        strg = (f"on      model = {cfg.straggling_model}"
                f"  (dOmega^2/ds = BOHR_COEF/beta^2, scale"
                f" {cfg.straggling_scale:g}, separate RNG seed +"
                f" {cfg.straggling_seed_offset})")
    lines = [
        "MCConfig — pbdose.montecarlo condensed-history proton transport",
        "-" * 74,
        f"  ds_max                 = {cfg.ds_max:g} cm",
        f"  step_frac              = {cfg.step_frac:g}    (ds = min(ds_max, step_frac*R(E)))",
        f"  min_ds                 = {cfg.min_ds:g} cm  (floor over the last ~0.1 cm of range)",
        f"  e_cut                  = {cfg.e_cut:g} MeV",
        f"  lambda_att             = {lam}",
        f"  energy_straggling      = {strg}",
        f"  scattering_model       = {cfg.scattering_model}",
        f"  z_over_x0_reference    = {zref}",
        f"  integrator             = {integ}",
        f"  max_steps              = {cfg.max_steps}",
        f"  moment_depths [cm]     = {tuple(cfg.moment_depths)}",
        f"  deposit_escaped_energy = {cfg.deposit_escaped_energy}",
        f"  deterministic_deposit  = {cfg.deterministic_deposit}",
        f"  dtype                  = {cfg.dtype}",
        "-" * 74,
        "  Home-built class-II MC: CSDA + Gaussian MCS + nuclear attenuation"
        + (" + Bohr straggling." if (cfg.energy_straggling
                                     and str(cfg.straggling_model).lower() != "none")
           else "."),
        "  NOT Geant4/TOPAS: no nuclear/delta secondaries, no large-angle tails"
        + ("" if (cfg.energy_straggling
                  and str(cfg.straggling_model).lower() != "none")
           else ", no energy straggling") + ".",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# grid handling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """Rectilinear voxel grid.

    Voxel ``(k, j, i)`` spans ``[o + n*d, o + (n+1)*d)`` for ``n = i, j, k`` and
    its centre is at ``o + (n + 0.5) * d``; ``ox/oy/oz`` are the lower corners of
    voxel ``(0, 0, 0)``.  The dose tensor is indexed ``[k, j, i]`` = ``(nz, ny, nx)``.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    ox: float = 0.0
    oy: float = 0.0
    oz: float = 0.0

    @property
    def x0(self) -> float: return self.ox

    @property
    def x1(self) -> float: return self.ox + self.nx * self.dx

    @property
    def y0(self) -> float: return self.oy

    @property
    def y1(self) -> float: return self.oy + self.ny * self.dy

    @property
    def z0(self) -> float: return self.oz

    @property
    def z1(self) -> float: return self.oz + self.nz * self.dz

    @property
    def shape(self) -> tuple: return (self.nz, self.ny, self.nx)

    @property
    def n_voxels(self) -> int: return self.nx * self.ny * self.nz

    @property
    def voxel_volume(self) -> float: return self.dx * self.dy * self.dz

    def centers(self, axis: str) -> np.ndarray:
        o = {"x": self.ox, "y": self.oy, "z": self.oz}[axis]
        n = {"x": self.nx, "y": self.ny, "z": self.nz}[axis]
        d = {"x": self.dx, "y": self.dy, "z": self.dz}[axis]
        return o + (np.arange(n) + 0.5) * d


def _as_grid_spec(grid: Any) -> GridSpec:
    """Accept a :class:`GridSpec`, a mapping, or any object with nx..oz attributes."""
    if isinstance(grid, GridSpec):
        return grid
    if isinstance(grid, Mapping):
        get = lambda k, d=None: grid.get(k, d)  # noqa: E731
    else:
        get = lambda k, d=None: getattr(grid, k, d)  # noqa: E731
    try:
        spec = GridSpec(
            nx=int(get("nx")), ny=int(get("ny")), nz=int(get("nz")),
            dx=float(get("dx")), dy=float(get("dy")), dz=float(get("dz")),
            ox=float(get("ox", 0.0) or 0.0),
            oy=float(get("oy", 0.0) or 0.0),
            oz=float(get("oz", 0.0) or 0.0),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise ValueError(
            "grid must provide nx, ny, nz, dx, dy, dz (optionally ox, oy, oz)") from exc
    if min(spec.nx, spec.ny, spec.nz) < 1:
        raise ValueError("grid dimensions must be >= 1")
    if min(spec.dx, spec.dy, spec.dz) <= 0.0:
        raise ValueError("voxel sizes must be > 0")
    if not (spec.z0 <= 0.0 < spec.z1):
        raise ValueError(
            f"the beam enters at z = 0, which must lie inside the grid "
            f"[{spec.z0}, {spec.z1}); pass oz <= 0 < oz + nz*dz")
    return spec


def _resolve_device(device: Any) -> torch.device:
    """Resolve the compute device; never silently downgrade CUDA to CPU."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device!r} requested but torch.cuda.is_available() is "
                f"False (torch {torch.__version__}). Install a CUDA build or pass "
                f"device='cpu' explicitly — no silent fallback is performed.")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
    return dev


# ---------------------------------------------------------------------------
# step schedule
# ---------------------------------------------------------------------------
def mc_step_schedule(E0: float, config: "MCConfig" = DEFAULT_MC) -> np.ndarray:
    """Step lengths [cm] that the transport would use for a beam of energy ``E0``.

    Mirrors the transport loop exactly (adaptive rule, floor, clamping against
    the residual range and identical termination), so ``len(schedule)`` equals
    ``stat['n_steps']`` **for a phantom deep enough to stop the beam**.  If the
    grid's distal face is shallower than the CSDA range the transport stops
    earlier (``stat['n_exit'] > 0``) and ``n_steps < n_steps_planned``.
    Useful for predicting the cost of a run.
    """
    cfg = config
    R = float(physics.csda_range(E0))
    E = float(E0)
    out = []
    while len(out) < cfg.max_steps:
        ds = min(cfg.ds_max, cfg.step_frac * R)
        ds = min(max(ds, cfg.min_ds), R)
        if ds <= 0.0:
            out.append(0.0)
            break
        out.append(ds)
        if cfg.integrator == "euler":
            E = max(E - float(physics.stopping_power_water(E)) * ds, 0.0)
            R = float(physics.csda_range(E))
            if E <= cfg.e_cut or R <= 0.0:
                break
        else:
            R = R - ds
            if R <= 0.0 or float(physics.energy_from_range(R)) <= cfg.e_cut:
                break
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# scalar <-> tensor helpers for the straggling option
# ---------------------------------------------------------------------------
class _UniformLUT:
    """Linear interpolation on a uniform grid, evaluated on a torch device.

    ``y`` must be sampled on ``x = x0 + arange(n) * h``.  Values outside the
    grid are clamped to the end values (identical to ``numpy.interp`` behaviour),
    which is what makes ``R_of_E`` return 0 below the CSDA range-table floor and
    ``E_of_R`` return the floor energy for a non-positive residual range.
    """

    __slots__ = ("x0", "h", "n", "y", "_n2")

    def __init__(self, x0: float, h: float, y, device, dtype):
        self.x0 = float(x0)
        self.h = float(h)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float64),
                                 dtype=dtype, device=device)
        self.n = int(self.y.numel())
        self._n2 = self.n - 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = (x - self.x0) / self.h
        i = t.floor().clamp_(0.0, float(self._n2))
        f = (t - i).clamp_(0.0, 1.0)
        i = i.to(torch.int64)
        ya = self.y[i]
        yb = self.y[i + 1]
        return ya + (yb - ya) * f


class _RangeLookups:
    """``R(E)`` and ``E(R)`` as torch lookups built from the *public* physics API."""

    def __init__(self, E0: float, e_cut: float, device, dtype, n: int = 20001):
        # the lower end of the energy grid is the range-table floor: below it
        # csda_range() is identically zero, and starting the grid exactly there
        # makes R_of_E(floor) == 0 exactly (no interpolation across the kink,
        # which would otherwise leave a proton with a tiny positive range that
        # no step can ever consume).
        e_floor = float(physics.energy_from_range(0.0))
        e_hi = max(float(E0) * 1.05, float(E0) + 1.0)
        e_lo = e_floor if 0.0 < e_floor < float(E0) else min(float(e_cut), 0.5 * float(E0))
        self.e_floor = e_lo
        self.e_hi = e_hi
        E = np.linspace(e_lo, e_hi, int(n))
        R = np.asarray(physics.csda_range(E), dtype=np.float64)
        R = np.maximum.accumulate(np.clip(R, 0.0, None))   # enforce monotone
        R[E <= e_lo] = 0.0
        self._r_of_e = _UniformLUT(e_lo, (e_hi - e_lo) / (int(n) - 1), R, device, dtype)
        r_hi = float(R[-1]) if R[-1] > 0.0 else 1.0
        Rg = np.linspace(0.0, r_hi, int(n))
        Eg = np.asarray(physics.energy_from_range(Rg), dtype=np.float64)
        Eg = np.maximum.accumulate(np.clip(Eg, 0.0, None))
        self._e_of_r = _UniformLUT(0.0, r_hi / (int(n) - 1), Eg, device, dtype)
        self.r_max = r_hi

    def R_of_E(self, E: torch.Tensor) -> torch.Tensor:
        return self._r_of_e(E)

    def E_of_R(self, R: torch.Tensor) -> torch.Tensor:
        return self._e_of_r(R)


def _beta_of_E(E, m_p_c2: float):
    """Relativistic ``beta`` (same expression as ``physics.beta_gamma``)."""
    if isinstance(E, torch.Tensor):
        gam = 1.0 + E / m_p_c2
        return torch.sqrt((1.0 - 1.0 / (gam * gam)).clamp_min(1e-14))
    gam = 1.0 + float(E) / m_p_c2
    return math.sqrt(max(1.0 - 1.0 / (gam * gam), 1e-14))


def _beta_pc(E, m_p_c2: float):
    """Relativistic ``beta * p * c`` [MeV]; torch or float in, same type out.

    Identical to ``physics.scattering_power``'s internal ``beta * p_MeV``:
    ``beta*pc = m c^2 (gamma^2 - 1) / gamma``.
    """
    if isinstance(E, torch.Tensor):
        gam = 1.0 + E / m_p_c2
        return m_p_c2 * (gam * gam - 1.0) / gam
    gam = 1.0 + float(E) / m_p_c2
    return m_p_c2 * (gam * gam - 1.0) / gam


def _check_beta_consistency(m_p_c2: float) -> None:
    """Fail loudly if the local beta(E) ever drifts from ``physics.beta_gamma``."""
    for E in (0.6, 1.0, 50.0, 200.0):
        beta, gamma, _ = physics.beta_gamma(E)
        p_mev = math.sqrt(max(gamma * gamma - 1.0, 0.0)) * m_p_c2
        for name, got, ref in (("beta", _beta_of_E(E, m_p_c2), float(beta)),
                               ("beta*pc", _beta_pc(E, m_p_c2),
                                float(beta) * float(p_mev))):
            if ref > 0.0 and abs(got - ref) / ref > 1e-12:
                raise RuntimeError(
                    f"bohr straggling: {name}(E) disagrees with physics.beta_gamma "
                    f"at E={E} MeV ({got} vs {ref})")


# ---------------------------------------------------------------------------
# core transport (one pencil beam)
# ---------------------------------------------------------------------------
def _transport_pencil(
    E0: float,
    n_histories: int,
    gspec: GridSpec,
    config: MCConfig,
    *,
    sigma_x0: float = 0.0,
    sigma_theta0: float = 0.0,
    rho0: float = 0.0,
    device: torch.device,
    generator: torch.Generator,
    lateral_world: bool = True,
    record_depths: Sequence[float] = (),
    straggling_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Vectorised condensed-history transport of one monoenergetic pencil beam.

    The beam enters at ``(x, y, z) = (0, 0, 0)`` travelling in ``+z`` and is
    tracked until every history has stopped, left the grid, or been absorbed.

    Parameters
    ----------
    lateral_world : bool
        ``True``: the grid's lateral faces are the physical boundary of the
        phantom (escaping protons deposit or drop their remaining energy there).
        ``False``: the phantom is laterally infinite and the grid only *scores*;
        energy deposited outside the scoring window lands in an overflow bin
        reported as ``info['overflow_energy']`` (used for the IMPT kernel).
    record_depths : sequence of float
        Depths [cm] at which transverse phase-space second moments of the live
        population are accumulated (unbiased by voxelisation).
    straggling_generator : torch.Generator or None
        RNG for the energy-loss fluctuations; only used when
        ``config.energy_straggling`` is on.  Keeping it separate guarantees that
        the multiple-scattering stream is unchanged (bit-identical kicks)
        whether or not straggling is enabled.

    Returns
    -------
    dose_flat : (n_voxels + 1,) tensor
        Energy [MeV] per flattened voxel; the extra final entry is the overflow
        bin for energy deposited outside the scoring grid.
    info : dict
        Counters, moment sums and diagnostics.
    """
    cfg = config
    dt = _DTYPES[cfg.dtype]
    dev = device
    nx, ny, nz = gspec.nx, gspec.ny, gspec.nz
    inv_dx, inv_dy, inv_dz = 1.0 / gspec.dx, 1.0 / gspec.dy, 1.0 / gspec.dz
    gx0, gy0, gz0 = gspec.ox, gspec.oy, gspec.oz
    n_vox = gspec.n_voxels
    overflow = n_vox

    # ---- energy straggling setup (no-op unless switched on) -----------
    straggling = bool(cfg.energy_straggling) and \
        str(cfg.straggling_model).strip().lower() != "none"
    m_p_c2 = float(getattr(physics, "M_P_C2", M_P_C2_FALLBACK))
    if straggling:
        smodel = str(cfg.straggling_model).strip().lower()
        if smodel != "bohr":
            raise ValueError(
                f"unsupported straggling_model={cfg.straggling_model!r}; "
                f"use 'bohr' or 'none'")
        bohr_coef = float(getattr(physics, "BOHR_COEF", BOHR_COEF_FALLBACK))
        _check_beta_consistency(m_p_c2)
        lookups = _RangeLookups(E0, cfg.e_cut, dev, dt)
    else:
        bohr_coef = 0.0
        lookups = None

    if lateral_world:
        wx_lo, wx_hi, wy_lo, wy_hi = gspec.x0, gspec.x1, gspec.y0, gspec.y1
    else:
        wx_lo = wy_lo = -math.inf
        wx_hi = wy_hi = math.inf
    wz_hi = gspec.z1

    dose = torch.zeros(n_vox + 1, dtype=dt, device=dev)
    n0 = int(n_histories)
    empty_info = {
        "n_history": 0, "n_exit": 0, "n_nuclear": 0, "n_ecut": 0, "n_steps": 0,
        "n_steps_planned": int(mc_step_schedule(E0, cfg).size), "moments": {},
        "hit_max_steps": False, "overflow_energy": 0.0, "path_length_mean": 0.0,
        "E_final": float(E0),
    }
    if n0 <= 0:
        return dose, empty_info

    def _normal(count: int) -> torch.Tensor:
        return torch.randn(count, generator=generator, device=dev, dtype=dt)

    # ---- initial phase space at z = 0 ---------------------------------
    # x, y ~ N(0, sigma_x0^2) and theta_x, theta_y ~ N(0, sigma_theta0^2), with
    # an optional x-theta correlation rho0 (same in both planes, matching
    # physics.fermi_eyges_moments).  Drawn ONCE per history here; translating
    # the whole kernel laterally (mc_impt_field) leaves it unchanged.
    rho = float(rho0)
    if sigma_x0 > 0.0:
        g_x = _normal(n0)
        g_y = _normal(n0)
        x = g_x * sigma_x0
        y = g_y * sigma_x0
    else:
        g_x = g_y = None
        x = torch.zeros(n0, dtype=dt, device=dev)
        y = torch.zeros(n0, dtype=dt, device=dev)
    if sigma_theta0 > 0.0:
        g_tx = _normal(n0)
        g_ty = _normal(n0)
        if rho != 0.0 and g_x is not None:
            c = math.sqrt(max(1.0 - rho * rho, 0.0))
            thx = (rho * g_x + c * g_tx) * sigma_theta0
            thy = (rho * g_y + c * g_ty) * sigma_theta0
        else:
            thx = g_tx * sigma_theta0
            thy = g_ty * sigma_theta0
    else:
        thx = torch.zeros(n0, dtype=dt, device=dev)
        thy = torch.zeros(n0, dtype=dt, device=dev)

    # ---- nuclear absorption: survival depends on depth only, so a single
    #      exponential variate per history encodes its entire fate.
    if np.isfinite(cfg.lambda_att) and cfg.lambda_att > 0.0:
        u = torch.rand(n0, generator=generator, device=dev, dtype=dt)
        z_kill = (-float(cfg.lambda_att)) * torch.log(u)
    else:
        z_kill = torch.full((n0,), math.inf, dtype=dt, device=dev)

    depths = [float(d) for d in record_depths]
    moments = {d: {"n": 0, "sx": 0.0, "sxx": 0.0, "sy": 0.0, "syy": 0.0,
                   "stx2": 0.0, "sty2": 0.0} for d in depths}
    pending = sorted(depths)

    n_exit = n_nuclear = n_ecut = 0
    steps = 0
    z = 0.0
    E = float(E0)
    R_rem = float(physics.csda_range(E))
    path_length = 0.0
    hit_max_steps = False
    # per-history kinetic energy, only used when straggling is enabled
    E_state = torch.full((n0,), float(E0), dtype=dt, device=dev) if straggling else None
    strag_n = 0
    strag_sum2 = 0.0
    strag_omega2 = 0.0
    # histogram of stopping depths: free diagnostic (all histories that stop in
    # a given step stop at the same z), and the direct measurement of the
    # emergent range-straggling distribution.
    stop_hist = np.zeros(nz, dtype=np.float64)
    if straggling and straggling_generator is None:
        straggling_generator = torch.Generator(device=dev)
        straggling_generator.manual_seed(int(cfg.straggling_seed_offset) + 12345)

    while True:
        if steps >= cfg.max_steps:
            hit_max_steps = True
            break
        alive = int(x.shape[0])
        if alive == 0:
            break

        # ---- 1. adaptive step, capped by the residual range -----------
        if straggling:
            # each history carries its own residual range; the shared step is
            # driven by the longest-lived proton so that nobody overshoots
            R_i = lookups.R_of_E(E_state)
            r_max = float(R_i.max())          # one host sync drives the schedule
            ds = min(cfg.ds_max, cfg.step_frac * r_max)
            ds = min(max(ds, cfg.min_ds), r_max)
            if ds <= 0.0:            # everyone below the range-table floor
                ds = 0.0
                z_new = z
                E_mid_t = E_state
                final_t = torch.ones_like(E_state, dtype=torch.bool)
                E_new_t = torch.full_like(E_state, float(cfg.e_cut))
                E_mid_ref = lookups.e_floor
            else:
                z_new = z + ds
                E_det_t = lookups.E_of_R((R_i - ds).clamp_min(0.0))
                # mid-step energy, linearly interpolated in energy: the error is
                # O(ds^2 dS/dE) and irrelevant for beta(E) (saves a range lookup)
                E_mid_t = 0.5 * (E_state + E_det_t)
                # Bohr: dOmega^2 = BOHR_COEF * ds / beta(E)^2, beta at the
                # current (mid-)step energy of each history.
                omega = (float(cfg.straggling_scale) * math.sqrt(bohr_coef * ds)
                         / _beta_of_E(E_mid_t, m_p_c2))
                delta = torch.randn(alive, generator=straggling_generator,
                                    device=dev, dtype=dt).mul_(omega)
                # The energy *loss* of a step cannot be negative.  Instead of
                # truncating the positive tail (which would fold the Gaussian and
                # remove variance), reflect it: delta -> 2*dE_det - delta for
                # delta > dE_det.  This preserves the mean/variance of the
                # symmetric Bohr distribution as far as positivity allows (the
                # residual correction is <1 % of sigma_R^2, see straggling_docs).
                dE_det_t = (E_state - E_det_t).clamp_min(0.0)
                delta = torch.where(delta > dE_det_t, 2.0 * dE_det_t - delta, delta)
                E_new_t = torch.clamp(E_det_t + delta, min=float(cfg.e_cut))
                # R(E) == 0 exactly at/below the range-table floor and R is
                # monotone, so this single comparison is "out of range"
                final_t = E_new_t <= max(lookups.e_floor, float(cfg.e_cut))
                # diagnostics: sampled vs theoretical fluctuation scale
                strag_n += alive
                strag_sum2 += float((delta * delta).sum())
                strag_omega2 += float((omega * omega).sum())
                E_mid_ref = float(lookups.E_of_R(
                    torch.as_tensor(max(r_max - 0.5 * ds, 0.0), dtype=dt, device=dev)))
            final = False
        else:
            ds = min(cfg.ds_max, cfg.step_frac * R_rem)
            ds = min(max(ds, cfg.min_ds), R_rem)
            if ds <= 0.0:  # no tabulated range left (E0 below the range-table floor)
                final = True
                ds = 0.0
                z_new = z
                E_new = float(physics.energy_from_range(0.0))
            elif cfg.integrator == "euler":
                z_new = z + ds
                E_new = max(E - float(physics.stopping_power_water(E)) * ds, 0.0)
                final = (E_new <= cfg.e_cut) or (float(physics.csda_range(E_new)) <= 0.0)
            else:
                z_new = z + ds
                E_new = float(physics.energy_from_range(R_rem - ds))
                final = (R_rem - ds <= 0.0) or (E_new <= cfg.e_cut)

        # ---- 2. multiple Coulomb scattering kick ----------------------
        # T is evaluated at the mid-step energy because the kick is applied at
        # the middle of the step; the left-endpoint value would bias the
        # accumulated angular variance low by ~(ds/2)*dT/dz (second order).
        # With straggling on, T is additionally rescaled per history by
        # (beta*pc_ref / beta*pc_i)^2, which is exact because the Highland
        # local form depends on the energy only through 1/(beta*pc)^2.
        if straggling:
            E_mid = E_mid_ref
        else:
            E_mid = (max(E - 0.5 * ds * float(physics.stopping_power_water(E)), 0.0)
                     if cfg.integrator == "euler"
                     else float(physics.energy_from_range(R_rem - 0.5 * ds)))
        T = float(physics.scattering_power(
            E_mid, z_over_X0=cfg.z_over_x0_reference, model=cfg.scattering_model))
        if straggling and ds > 0.0:
            bp_ref = _beta_pc(E_mid, m_p_c2)
            bp_i = _beta_pc(E_mid_t, m_p_c2)
            sigma_t = (T * ds) ** 0.5 * (bp_ref / bp_i)
            kx = _normal(alive).mul_(sigma_t)
            ky = _normal(alive).mul_(sigma_t)
        else:
            sigma = math.sqrt(max(T, 0.0) * ds)
            if sigma > 0.0:
                kx = _normal(alive).mul_(sigma)
                ky = _normal(alive).mul_(sigma)
            else:
                kx = torch.zeros(alive, dtype=dt, device=dev)
                ky = torch.zeros(alive, dtype=dt, device=dev)

        # ---- 3. phase-space moments at the requested depths -----------
        if pending:
            for d in tuple(pending):
                if z <= d < z_new:
                    frac = d - z
                    c = frac - 0.5 * ds
                    xq = x + thx * frac
                    yq = y + thy * frac
                    tqx, tqy = thx, thy
                    if c > 0.0:
                        xq = xq + kx * c
                        yq = yq + ky * c
                        tqx = thx + kx
                        tqy = thy + ky
                    m = moments[d]
                    m["n"] += alive
                    m["sx"] += float(xq.sum())
                    m["sxx"] += float((xq * xq).sum())
                    m["sy"] += float(yq.sum())
                    m["syy"] += float((yq * yq).sum())
                    m["stx2"] += float((tqx * tqx).sum())
                    m["sty2"] += float((tqy * tqy).sum())
                    pending.remove(d)

        # ---- 4. kick-drift, kick placed at mid-step -------------------
        half = 0.5 * ds
        sx = thx * ds + kx * half
        sy = thy * ds + ky * half
        xn = x + sx
        yn = y + sy
        thx = thx + kx
        thy = thy + ky

        # ---- 5. boundary crossing (lateral faces and distal face) -----
        outx = (xn < wx_lo) | (xn >= wx_hi)
        outy = (yn < wy_lo) | (yn >= wy_hi)
        t = torch.ones_like(xn)
        lateral_out_seen = bool(outx.any()) or bool(outy.any())
        if lateral_out_seen:
            den = torch.where(sx.abs() < 1e-12, torch.full_like(sx, 1e-12), sx)
            tx = torch.where(xn < wx_lo, (wx_lo - x) / den, (wx_hi - x) / den)
            t = torch.where(outx, tx.clamp_(0.0, 1.0), t)
            den = torch.where(sy.abs() < 1e-12, torch.full_like(sy, 1e-12), sy)
            ty = torch.where(yn < wy_lo, (wy_lo - y) / den, (wy_hi - y) / den)
            t = torch.where(outy, torch.minimum(ty.clamp_(0.0, 1.0), t), t)
        z_out = z_new >= wz_hi
        if z_out:
            t = torch.clamp(t, max=(wz_hi - z) / ds if ds > 0.0 else 0.0)
            out = torch.ones_like(outx)
            has_out = True
        else:
            out = outx | outy
            has_out = lateral_out_seen

        # ---- 6. deposit ------------------------------------------------
        xc = torch.where(out, x + t * sx, xn)
        yc = torch.where(out, y + t * sy, yn)
        zc = torch.where(out, z_new - (1.0 - t) * ds, z_new)
        if straggling:
            # per-history energy budget: the step loss, plus everything left for
            # a proton that stops here (range exhausted / at the energy cut)
            if cfg.deposit_escaped_energy:
                dep = torch.where(final_t | out, E_state, E_state - E_new_t)
            else:
                dep = torch.where(final_t, E_state,
                                  torch.where(out, (E_state - E_new_t) * t,
                                              E_state - E_new_t))
        else:
            # NOTE: 0-dim tensors (not bare floats) so that torch.where keeps the
            # transport dtype instead of falling back to the default float32.
            E_t = torch.as_tensor(E, dtype=dt, device=dev)
            dE_step = E - E_new
            dE_t = torch.as_tensor(dE_step, dtype=dt, device=dev)
            if final:
                dep = torch.full_like(xc, E)            # stop: bank everything left
            elif cfg.deposit_escaped_energy:
                dep = torch.where(out, E_t, dE_t)       # escape: bank all remaining
            else:
                dep = torch.where(out, dE_t * t, dE_t)

        ix = ((xc - gx0) * inv_dx).floor_()
        iy = ((yc - gy0) * inv_dy).floor_()
        izt = ((zc - gz0) * inv_dz).floor_().clamp_(0.0, nz - 1.0)
        if lateral_world:
            # a crossing point sits exactly on a face -> outermost voxel
            ix = torch.where(out, ix.clamp_(0.0, nx - 1.0), ix)
            iy = torch.where(out, iy.clamp_(0.0, ny - 1.0), iy)
        else:
            # laterally infinite phantom: everything is inside the grid in z
            # (iz is clamped) but a deposit may fall outside the scoring window
            inside = ((ix >= 0.0) & (ix < float(nx))
                      & (iy >= 0.0) & (iy < float(ny)))
        ix = ix.clamp_(0.0, nx - 1.0).to(torch.int64)
        iy = iy.clamp_(0.0, ny - 1.0).to(torch.int64)
        izt = izt.to(torch.int64)
        stride = ny * nx
        lin_xy = iy * nx + ix
        if not lateral_world:
            overflow_idx = torch.full_like(lin_xy, overflow)

        def _deposit(idx: torch.Tensor, vals: torch.Tensor) -> None:
            if not lateral_world:
                idx = torch.where(inside, idx, overflow_idx)
            if cfg.deterministic_deposit:
                dose.index_put_((idx,), vals, accumulate=True)
            else:
                dose.scatter_add_(0, idx, vals)

        # (a) protons that leave the grid: point deposit at their crossing point
        if has_out:
            dep_out = torch.where(out, dep, torch.zeros((), dtype=dt, device=dev))
            _deposit(lin_xy + izt * stride, dep_out)
            dep_main = dep - dep_out
        else:
            dep_main = dep
        # (b) everybody else: spread the step energy over the voxels it crosses
        #     (line integral along the track, removes the ~ds/dz endpoint alias)
        for iz_k, w_k in _z_overlap_weights(z, z_new, gz0, gspec.dz, nz):
            if w_k <= 0.0:
                continue
            idx = lin_xy + (iz_k * stride)
            _deposit(idx, dep_main if w_k == 1.0 else dep_main * w_k)

        # ---- 7. absorption and compaction ------------------------------
        steps += 1
        path_length += ds
        if z_out:
            n_exit += alive
            break
        n_out = int(out.sum())
        nuc = z_kill <= z_new
        if straggling:
            # histories that ran out of range stop individually; the four
            # counters are reduced in a single host sync (sync latency dominates
            # on Windows/WDDM)
            n_out, n_nuc_raw, n_nuc_ovl, n_fin = (
                int(v) for v in torch.stack((
                    out.sum(), nuc.sum(), (nuc & out).sum(),
                    (final_t & ~(out | nuc)).sum())).tolist())
            n_nuc = n_nuc_raw - n_nuc_ovl
            n_exit += n_out
            n_nuclear += n_nuc
            n_ecut += n_fin
            if n_fin:
                iz_stop = int((z_new - gz0) * inv_dz)
                stop_hist[min(max(iz_stop, 0), nz - 1)] += n_fin
            if n_out or n_nuc or n_fin:
                keep = ~(out | nuc | final_t)
                x = xn[keep]
                y = yn[keep]
                thx = thx[keep]
                thy = thy[keep]
                z_kill = z_kill[keep]
                E_state = E_new_t[keep]
            else:
                x = xn
                y = yn
                E_state = E_new_t
            z = z_new
            if int(x.shape[0]) == 0:
                break
        else:
            n_nuc = int(nuc.sum()) - int((nuc & out).sum())
            n_exit += n_out
            n_nuclear += n_nuc
            if n_out or n_nuc:
                keep = ~(out | nuc)
                x = xn[keep]
                y = yn[keep]
                thx = thx[keep]
                thy = thy[keep]
                z_kill = z_kill[keep]
            else:
                x = xn
                y = yn

            if final:
                n_ecut += int(x.shape[0])
                iz_stop = int((z_new - gz0) * inv_dz)
                stop_hist[min(max(iz_stop, 0), nz - 1)] += int(x.shape[0])
                break
            z = z_new
            E = E_new
            if cfg.integrator == "euler":
                R_rem = float(physics.csda_range(E_new))
            else:
                R_rem = R_rem - ds
        if cfg.verbose and steps % 100 == 0:
            print(f"    step {steps:4d}  z = {z:8.4f} cm  "
                  f"alive = {int(x.shape[0]):9d}")

    if straggling and E_state is not None and E_state.numel() > 0:
        E = float(E_state.max())

    info = {
        "n_history": n0,
        "n_exit": n_exit,
        "n_nuclear": n_nuclear,
        "n_ecut": n_ecut,
        "n_steps": steps,
        "n_steps_planned": int(mc_step_schedule(E0, cfg).size),
        "moments": moments,
        "hit_max_steps": hit_max_steps,
        "overflow_energy": float(dose[overflow]),
        "path_length_mean": path_length,
        "z_final": z,
        "E_final": E,
        "energy_straggling": bool(straggling),
        "straggling_n_draws": strag_n,
        "straggling_delta_rms": (math.sqrt(strag_sum2 / strag_n) if strag_n else 0.0),
        "straggling_omega_rms": (math.sqrt(strag_omega2 / strag_n) if strag_n else 0.0),
        "stop_hist": stop_hist,
    }
    return dose, info


# ---------------------------------------------------------------------------
# chunked driver for one spot
# ---------------------------------------------------------------------------
def _run_pencil(
    E0: float,
    n_histories: int,
    gspec: GridSpec,
    config: MCConfig,
    *,
    sigma_x0: float,
    sigma_theta0: float,
    rho0: float = 0.0,
    device: torch.device,
    seed: int,
    chunk_size: int,
    lateral_world: bool = True,
    record_depths: Sequence[float] = (),
) -> tuple[torch.Tensor, dict]:
    """Chunked driver around :func:`_transport_pencil`; returns an (nz,ny,nx) dose."""
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = max(int(n_histories), 1)
    chunk_size = int(chunk_size)
    n_left = int(n_histories)
    dose_flat = torch.zeros(gspec.n_voxels + 1, dtype=_DTYPES[config.dtype],
                            device=device)
    agg = {float(d): {"n": 0, "sx": 0.0, "sxx": 0.0, "sy": 0.0, "syy": 0.0,
                      "stx2": 0.0, "sty2": 0.0} for d in record_depths}
    tot = {"n_history": 0, "n_exit": 0, "n_nuclear": 0, "n_ecut": 0, "n_steps": 0,
           "n_steps_planned": 0, "overflow_energy": 0.0, "hit_max_steps": False,
           "n_chunks": 0, "path_length_mean": 0.0, "E_final": float(E0),
           "z_final": 0.0, "energy_straggling": False}
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    # independent stream for the energy-loss fluctuations (Bohr straggling):
    # the multiple-scattering draws then stay identical to a run with the
    # straggling switched off, and the two fluctuations are uncorrelated.
    gen_strg = torch.Generator(device=device)
    gen_strg.manual_seed(int(seed) + int(config.straggling_seed_offset))
    t0 = time.perf_counter()
    while n_left > 0:
        n_this = min(chunk_size, n_left)
        d, info = _transport_pencil(
            E0, n_this, gspec, config,
            sigma_x0=sigma_x0, sigma_theta0=sigma_theta0, rho0=rho0,
            device=device, generator=gen, lateral_world=lateral_world,
            record_depths=record_depths, straggling_generator=gen_strg)
        dose_flat += d
        n_left -= n_this
        tot["n_chunks"] += 1
        for k in ("n_history", "n_exit", "n_nuclear", "n_ecut"):
            tot[k] += info[k]
        tot["n_steps"] = max(tot["n_steps"], info["n_steps"])
        tot["n_steps_planned"] = info["n_steps_planned"]
        tot["overflow_energy"] += info["overflow_energy"]
        tot["hit_max_steps"] |= info["hit_max_steps"]
        tot["path_length_mean"] += info["path_length_mean"] * n_this
        tot["E_final"] = min(tot["E_final"], info["E_final"])
        tot["z_final"] = max(tot["z_final"], info.get("z_final", 0.0))
        tot["energy_straggling"] = bool(info.get("energy_straggling", False))
        tot.setdefault("straggling_n_draws", 0)
        tot.setdefault("straggling_sum2", 0.0)
        tot.setdefault("straggling_omega2", 0.0)
        tot.setdefault("stop_hist", None)
        if info.get("stop_hist") is not None:
            tot["stop_hist"] = (info["stop_hist"] if tot["stop_hist"] is None
                                else tot["stop_hist"] + info["stop_hist"])
        nd = int(info.get("straggling_n_draws", 0))
        if nd:
            tot["straggling_n_draws"] += nd
            tot["straggling_sum2"] += info["straggling_delta_rms"] ** 2 * nd
            tot["straggling_omega2"] += info["straggling_omega_rms"] ** 2 * nd
        for dpt, m in info["moments"].items():
            a = agg.setdefault(float(dpt), {"n": 0, "sx": 0.0, "sxx": 0.0, "sy": 0.0,
                                            "syy": 0.0, "stx2": 0.0, "sty2": 0.0})
            for k, v in m.items():
                a[k] += v
    if tot["n_history"] > 0:
        tot["path_length_mean"] /= tot["n_history"]
    tot["moments"] = agg
    tot["wall_time_s"] = time.perf_counter() - t0
    dose = dose_flat[:gspec.n_voxels].view(gspec.nz, gspec.ny, gspec.nx)
    return dose, tot


# ---------------------------------------------------------------------------
# depth-dose helpers
# ---------------------------------------------------------------------------
def depth_profile(dose: torch.Tensor) -> np.ndarray:
    """Depth-dose profile [MeV per z-slab] of an ``(nz, ny, nx)`` dose tensor."""
    d = dose.detach()
    if d.dim() == 3:
        d = d.sum(dim=(1, 2))
    return d.double().cpu().numpy()


def cumulative_depth_metric(dose: torch.Tensor, grid: Any, frac: float = 0.9) -> float:
    """Depth [cm] at which the *proximal* cumulative energy reaches ``frac``.

    This is the standard integral-depth-dose definition of R90/R80/...: for
    ``frac = 0.9`` it is the depth beyond which only 10 % of the deposited energy
    lies — equivalently, the depth where the cumulative **distal** energy
    reaches 90 % of the total when counted from the far side.

    The cumulative sum over voxels ``0..k`` equals the *continuous* cumulative
    at the **distal edge** of voxel ``k`` (all the energy of those voxels lies
    proximal to that edge), so the interpolation is done on the distal edges;
    using the voxel centres instead biases R90 proximally by ``dz/2``.
    """
    gspec = _as_grid_spec(grid)
    prof = depth_profile(dose)
    tot = float(prof.sum())
    if tot <= 0.0 or prof.size == 0:
        return float("nan")
    cum = np.cumsum(prof) / tot
    z_edge = gspec.centers("z") + 0.5 * gspec.dz
    i = int(np.searchsorted(cum, frac, side="left"))
    if i <= 0:
        return float(z_edge[0])
    if i >= cum.size:
        return float(z_edge[-1])
    c0, c1 = cum[i - 1], cum[i]
    z0, z1 = z_edge[i - 1], z_edge[i]
    if c1 <= c0:
        return float(z1)
    return float(z0 + (frac - c0) / (c1 - c0) * (z1 - z0))


def _lateral_rms_from_dose(dose: torch.Tensor, grid: GridSpec, depth: float) -> dict:
    """Voxelised lateral rms at a depth (biased by ``dx^2/12`` — see module docs)."""
    iz = int(np.clip((depth - grid.z0) / grid.dz, 0, grid.nz - 1))
    plane = dose[iz].detach().double().cpu().numpy()
    w = float(plane.sum())
    if not np.isfinite(w) or w <= 0.0:
        return {"sigma_x": float("nan"), "sigma_y": float("nan"), "weight": w}
    cx = grid.centers("x")[None, :]
    cy = grid.centers("y")[:, None]
    mx = float((plane * cx).sum() / w)
    my = float((plane * cy).sum() / w)
    vx = float((plane * (cx - mx) ** 2).sum() / w)
    vy = float((plane * (cy - my) ** 2).sum() / w)
    return {"sigma_x": math.sqrt(max(vx, 0.0)), "sigma_y": math.sqrt(max(vy, 0.0)),
            "weight": w}


def _moments_to_sigma(m: Mapping[str, float]) -> dict:
    n = int(m.get("n", 0))
    if n <= 1:
        return {"n": n, "sigma_x": float("nan"), "sigma_y": float("nan"),
                "sigma_theta": float("nan")}
    inv = 1.0 / n
    vx = max(m["sxx"] * inv - (m["sx"] * inv) ** 2, 0.0)
    vy = max(m["syy"] * inv - (m["sy"] * inv) ** 2, 0.0)
    vt = max(0.5 * (m["stx2"] + m["sty2"]) * inv, 0.0)
    return {"n": n, "sigma_x": math.sqrt(vx), "sigma_y": math.sqrt(vy),
            "sigma_theta": math.sqrt(vt)}


def _hist_sigma(hist, gspec: GridSpec) -> float:
    """Mean/std [cm] of a stopping-depth histogram (voxel-centre resolution)."""
    if hist is None:
        return float("nan")
    h = np.asarray(hist, dtype=np.float64)
    tot = h.sum()
    if tot <= 0.0:
        return float("nan")
    zc = gspec.centers("z")
    mean = float((h * zc).sum() / tot)
    var = float((h * (zc - mean) ** 2).sum() / tot)
    return math.sqrt(max(var, 0.0))


def _z_overlap_weights(z_a: float, z_b: float, gz0: float, dz: float, nz: int,
                       max_seg: int = 8):
    """Voxel-overlap weights of a uniform deposit along the step ``[z_a, z_b]``.

    Returns ``[(iz, w), ...]`` with ``sum(w) == 1``.  Scoring the step energy
    proportionally to the overlap of the step with each voxel (a line integral
    along the track, as real condensed-history codes do) removes the
    endpoint-aliasing artefact whose amplitude is ~ds/dz.
    """
    if not (z_b > z_a):
        iz = int(math.floor((z_a - gz0) / dz))
        return [(min(max(iz, 0), nz - 1), 1.0)]
    iz_a = int(math.floor((z_a - gz0) / dz))
    iz_b = int(math.floor((z_b - gz0) / dz))
    if iz_b - iz_a + 1 > max_seg:      # pathological step: fall back to the end
        return [(min(max(iz_b, 0), nz - 1), 1.0)]
    total = z_b - z_a
    out = []
    for iz in range(iz_a, iz_b + 1):
        lo = gz0 + iz * dz
        ov = min(z_b, lo + dz) - max(z_a, lo)
        if ov > 0.0:
            out.append((min(max(iz, 0), nz - 1), ov / total))
    return out or [(min(max(iz_b, 0), nz - 1), 1.0)]


def binned_gaussian_sigma(sigma: float, dx: float) -> float:
    """rms of a Gaussian of width ``sigma`` after binning into voxels of width ``dx``.

    Exact voxelisation bias of a scored lateral profile (no asymptotic
    ``dx^2/12`` approximation), assuming the beam axis lies on a voxel *edge* —
    true for a symmetric grid.  Use it to compare a ``voxelised_sigma_x_at_depth``
    entry against an analytic (continuous) width; the phase-space estimators
    ``sigma_x_at_depth`` need no such correction.
    """
    if sigma <= 0.0 or dx <= 0.0:
        return 0.0
    s2 = sigma * math.sqrt(2.0)
    half = int(math.ceil(40.0 * sigma / dx)) + 2
    num = den = 0.0
    for j in range(-half, half):
        lo, hi = j * dx, (j + 1) * dx
        w = 0.5 * (math.erf(hi / s2) - math.erf(lo / s2))
        if w <= 0.0:
            continue
        c = lo + 0.5 * dx
        num += w * c * c
        den += w
    return math.sqrt(num / den) if den > 0.0 else 0.0


# ---------------------------------------------------------------------------
# public API — single spot
# ---------------------------------------------------------------------------
def mc_spot_dose(
    E0: float,
    grid,
    n_histories: int = 2_000_000,
    *,
    sigma_x0: float = 0.0,
    sigma_theta0: float = 0.0,
    rho0: float = 0.0,
    config: "MCConfig" = DEFAULT_MC,
    device: str = "cuda",
    seed: int = 12345,
    chunk_size: int = 2_000_000,
) -> dict:
    """Transport a monoenergetic pencil beam and score its dose on a voxel grid.

    The beam enters at ``(x, y, z) = (0, 0, 0)`` travelling in ``+z`` with a
    Gaussian phase space ``x, y ~ N(0, sigma_x0^2)`` and
    ``theta_x, theta_y ~ N(0, sigma_theta0^2)`` (uncorrelated, i.e. ``rho0 = 0``
    in Fermi-Eyges notation), and is transported with the physics of
    :class:`MCConfig`: CSDA slowing down, Gaussian multiple Coulomb scattering
    with the same ``T(E)`` the analytic model integrates, and nuclear
    attenuation.  The grid faces are the boundary of the phantom.

    Parameters
    ----------
    E0 : float
        Nominal beam energy [MeV].
    grid : mapping or object
        Must expose ``nx, ny, nz, dx, dy, dz`` and optionally ``ox, oy, oz``
        (lower corner of voxel 0).  Voxel ``(k, j, i)`` spans
        ``[o + n*d, o + (n+1)*d)``; the proximal face must satisfy
        ``oz <= 0 < oz + nz*dz``.
    n_histories : int
        Number of proton histories (primary fluence).
    sigma_x0, sigma_theta0 : float
        Initial rms lateral size [cm] and divergence [rad] of the incident beam
        at ``z = 0``: ``x, y ~ N(0, sigma_x0^2)``,
        ``theta_x, theta_y ~ N(0, sigma_theta0^2)``, drawn once per history.
    rho0 : float
        Optional x-theta correlation coefficient (``-1 < rho0 < 1``), the same in
        both transverse planes, so that ``Cov(x, theta) = rho0 sigma_x0
        sigma_theta0`` — the ``rho0`` of :func:`pbdose.physics.fermi_eyges_moments`.
        Ignored when ``sigma_x0 == 0`` (no correlation is defined).
    config : MCConfig
        Transport tunables (see :class:`MCConfig`).
    device : str
        Torch device; ``"cuda"`` raises if CUDA is unavailable (never falls back
        silently).
    seed : int
        Seed of the global torch RNG and of the private ``torch.Generator``.
    chunk_size : int
        Maximum number of histories transported simultaneously (VRAM control).
        Results are reproducible for a fixed chunk size.

    Returns
    -------
    dict
        ``dose`` : (nz, ny, nx) tensor
            Deposited energy per voxel [MeV], summed over all histories, on the
            compute device (use ``dose.cpu().numpy()`` for numpy work).
            Convenience copies: ``dose_per_primary`` [MeV/voxel/history] and
            ``dose_per_cm3`` [MeV/cm^3] (i.e. MeV per cm^3 of water, since the
            phantom is unit-density water).
        ``n_primary``, ``n_absorbed``, ``n_history`` : int
            Started histories; histories removed without depositing their full
            energy (grid escapes + nuclear reactions); histories requested.
        ``stat`` : dict
            ``mean_energy_deposited`` [MeV/primary], ``r90_depth``,
            ``r80_depth``, ``r50_depth`` [cm], ``sigma_x_at_depth`` and
            ``sigma_theta_at_depth`` (phase-space estimators),
            ``voxelised_sigma_x_at_depth``, the absorption breakdown, the energy
            balance, device/dtype and timings.
        ``wall_time_s`` : float
    """
    t_start = time.perf_counter()
    if E0 <= 0.0:
        raise ValueError("E0 must be > 0")
    gspec = _as_grid_spec(grid)
    dev = _resolve_device(device)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    depths = tuple(float(d) for d in config.moment_depths if 0.0 < float(d) < gspec.z1)
    dose, info = _run_pencil(
        float(E0), int(n_histories), gspec, config,
        sigma_x0=float(sigma_x0), sigma_theta0=float(sigma_theta0),
        rho0=float(rho0), device=dev, seed=int(seed), chunk_size=int(chunk_size),
        lateral_world=True, record_depths=depths)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t_start

    n_hist = int(info["n_history"])
    total_energy = float(dose.sum())
    sig = {d: _moments_to_sigma(info["moments"][d]) for d in depths}
    stat = {
        "n_history": n_hist,
        "mean_energy_deposited": total_energy / n_hist if n_hist else float("nan"),
        "total_energy_deposited": total_energy,
        "r90_depth": cumulative_depth_metric(dose, gspec, 0.9),
        "r80_depth": cumulative_depth_metric(dose, gspec, 0.8),
        "r50_depth": cumulative_depth_metric(dose, gspec, 0.5),
        "n_exit": int(info["n_exit"]),
        "n_nuclear": int(info["n_nuclear"]),
        "n_ecut": int(info["n_ecut"]),
        "energy_outside_scoring_grid": float(info["overflow_energy"]),
        "energy_balance": total_energy / (n_hist * float(E0)) if n_hist else float("nan"),
        "n_steps": int(info["n_steps"]),
        "n_steps_planned": int(info["n_steps_planned"]),
        "path_length_mean": float(info["path_length_mean"]),
        "hit_max_steps": bool(info["hit_max_steps"]),
        "E_final_min": float(info["E_final"]),
        "wall_time_s": wall,
        "histories_per_s": n_hist / wall if wall > 0 else float("nan"),
        "n_chunks": int(info["n_chunks"]),
        "device": str(dev),
        "dtype": str(config.dtype),
        "chunk_size": int(chunk_size),
        "energy_straggling": bool(info["energy_straggling"]),
        "straggling_model": (str(config.straggling_model)
                             if info["energy_straggling"] else "none"),
        "straggling_delta_rms": (math.sqrt(info["straggling_sum2"]
                                           / info["straggling_n_draws"])
                                 if info.get("straggling_n_draws") else 0.0),
        "straggling_n_draws": int(info.get("straggling_n_draws", 0)),        "straggling_omega_rms": (math.sqrt(info["straggling_omega2"]
                                           / info["straggling_n_draws"])
                                 if info.get("straggling_n_draws") else 0.0),
        "straggling_sampled_over_theory": (
            math.sqrt(info["straggling_sum2"] / info["straggling_omega2"])
            if info.get("straggling_omega2") else float("nan")),
        "range_straggling_measured": _hist_sigma(info.get("stop_hist"), gspec),
        "sigma_x_at_depth": {d: sig[d]["sigma_x"] for d in depths},
        "sigma_y_at_depth": {d: sig[d]["sigma_y"] for d in depths},
        "sigma_theta_at_depth": {d: sig[d]["sigma_theta"] for d in depths},
        "n_at_depth": {d: sig[d]["n"] for d in depths},
        "voxelised_sigma_x_at_depth": {
            d: _lateral_rms_from_dose(dose, gspec, d)["sigma_x"] for d in depths},
        "voxel_variance_offset": gspec.dx * gspec.dy / 12.0,
    }
    if config.verbose:
        print(f"[mc_spot_dose] E0={E0:g} MeV  N={n_hist}  steps={stat['n_steps']}  "
              f"{stat['histories_per_s']:.4g} hist/s  wall={wall:.2f} s")

    return {
        "dose": dose,
        "n_primary": n_hist,
        "n_absorbed": int(info["n_exit"]) + int(info["n_nuclear"]),
        "n_history": n_hist,
        "stat": stat,
        "wall_time_s": wall,
        "dose_per_primary": dose / n_hist if n_hist else dose,
        "dose_per_cm3": dose / gspec.voxel_volume,
        "depth_profile": depth_profile(dose),
    }


# ---------------------------------------------------------------------------
# public API — IMPT field (translation-invariant pencil-beam reuse)
# ---------------------------------------------------------------------------
def _kernel_crop(kernel: torch.Tensor, rel_eps: float = 1e-9, pad: int = 4):
    """Lateral bounding box ``(j0, j1, i0, i1)`` of a pencil-beam kernel."""
    nz, ny, nx = kernel.shape
    px = kernel.sum(dim=(0, 1)).double().cpu().numpy()   # over z, y  -> x profile
    py = kernel.sum(dim=(0, 2)).double().cpu().numpy()   # over z, x  -> y profile
    bounds = []
    for prof, n in ((px, nx), (py, ny)):
        tot = float(prof.sum())
        if tot <= 0.0:
            bounds.append((0, n))
            continue
        cum = np.cumsum(prof) / tot
        lo = int(np.searchsorted(cum, rel_eps, side="left"))
        hi = int(np.searchsorted(cum, 1.0 - rel_eps, side="right"))
        lo, hi = max(0, lo - pad), min(n, hi + pad)
        if hi <= lo:
            lo, hi = 0, n
        bounds.append((lo, hi))
    (i0, i1), (j0, j1) = bounds
    return j0, j1, i0, i1


def _shift2d_add(dst: torch.Tensor, src: torch.Tensor, dj: int, di: int,
                 scale: float) -> None:
    """``dst[:, j + dj, i + di] += scale * src[:, j, i]`` with zero fill (in place)."""
    if scale == 0.0:
        return
    nz, ny, nx = src.shape
    NY, NX = dst.shape[1], dst.shape[2]
    sy0, sy1 = max(0, -dj), min(ny, NY - dj)
    sx0, sx1 = max(0, -di), min(nx, NX - di)
    if sy0 >= sy1 or sx0 >= sx1:
        return
    dst[:, sy0 + dj:sy1 + dj, sx0 + di:sx1 + di] += scale * src[:, sy0:sy1, sx0:sx1]


def _place_kernel(dose: torch.Tensor, kc: torch.Tensor, j0: int, i0: int,
                  fsx: float, fsy: float, weight: float) -> None:
    """Add a cropped kernel at a (possibly sub-voxel) lateral shift, bilinearly.

    ``kc`` holds full-kernel columns/rows ``i0..`` and the kernel axis sits at
    ``x = y = 0``; ``fsx/fsy`` are the spot offsets in voxel units.
    """
    di0 = int(math.floor(fsx))
    fx = fsx - di0
    dj0 = int(math.floor(fsy))
    fy = fsy - dj0
    for oi, wx in ((di0 + i0, 1.0 - fx), (di0 + i0 + 1, fx)):
        for oj, wy in ((dj0 + j0, 1.0 - fy), (dj0 + j0 + 1, fy)):
            if wx == 0.0 or wy == 0.0:
                continue
            _shift2d_add(dose, kc, dj=oj, di=oi, scale=weight * wx * wy)


def mc_impt_field(
    lattice,
    grid,
    histories_per_spot: int = 200_000,
    *,
    sigma_x0: float = 0.0,
    sigma_theta0: float = 0.0,
    rho0: float = 0.0,
    config=DEFAULT_MC,
    device="cuda",
    seed: int = 12345,
    chunk_size: int = 2_000_000,
) -> dict:
    """Weighted sum of pencil-beam MC doses over an IMPT spot lattice.

    ``lattice`` is a mapping (or object) with ``n_sx, n_sy, n_layers, dx_spot,
    dy_spot, x0, y0, energies`` (``n_layers``,) and ``weights``
    (``n_layers, n_sy, n_sx``).  Using the same convention as
    :class:`pbdose.model.SpotLattice`, spot ``(l, j, i)`` sits at
    ``x = x0 + (i - (n_sx-1)/2) * dx_spot`` and
    ``y = y0 + (j - (n_sy-1)/2) * dy_spot`` (i.e. ``x0, y0`` are the *centre* of
    the lattice, not the first spot), has energy ``energies[l]`` and weight
    ``weights[l, j, i]``; the weight is the physical fluence (number of protons)
    of that spot, so the returned dose is a directly usable weighted sum.

    Incident phase space
    --------------------
    ``sigma_x0`` [cm], ``sigma_theta0`` [rad] and ``rho0`` describe the finite
    source of the *pencil-beam kernel*: ``x, y ~ N(0, sigma_x0^2)``,
    ``theta_x, theta_y ~ N(0, sigma_theta0^2)`` with ``Cov(x, theta) =
    rho0 sigma_x0 sigma_theta0``, drawn once per history at ``z = 0`` (the same
    convention as :func:`mc_spot_dose`).  This is the nozzle spot size of the
    delivery system, e.g. ``sigma_x0 = 0.30`` cm for a 3 mm FWHM spot.  Because
    the kernel is *translated* laterally when it is placed at each spot, a
    finite source size does not break translation invariance: the phase space is
    a property of the kernel, not of its position.  Defaults are 0 (a point
    source), which reproduces earlier results exactly.

    Variance-reduction trick — exact for a homogeneous medium
    ---------------------------------------------------------
    In an infinite homogeneous water phantom the pencil-beam dose kernel is
    **translation invariant**: the dose of a spot at ``(X, Y)`` is
    ``K(x - X, y - Y, z; E)`` with a single kernel ``K`` per energy.  The engine
    therefore transports **one** pencil beam per energy layer
    (``histories_per_spot`` histories) and re-uses that same kernel for every
    spot of the layer, merely shifting it laterally and scaling it by the spot
    weight.  This is *exact in expectation* — a lateral translation cannot
    change the physics of a homogeneous phantom — and it removes the
    ``n_spots``-fold cost of a naive spot-by-spot simulation, which is what makes
    a 4x4x3 lattice instantaneous on a GPU.  Two consequences are worth stating:

    * the statistical uncertainty of the whole field equals that of a *single*
      pencil beam per layer (spots share one Monte-Carlo error, they are not
      independent), so this mode is a fast *mean-value* reference rather than a
      tool for field-level error bars;
    * the kernel is transported with no lateral boundary (the phantom is treated
      as laterally infinite; energy outside the scoring window is reported in
      ``stat['energy_outside_scoring_grid']``), cropped to where its lateral
      marginals are non-negligible, and placed with a bilinear shift-and-add
      that conserves the deposited energy exactly while blurring it by at most
      one voxel (a triangle of width ``dx``).  Make the grid larger than the
      field plus ~3 sigma so that no kernel tail is clipped.

    Returns the same dict layout as :func:`mc_spot_dose`, with ``dose`` the
    weighted sum over all layers and spots.
    """
    t_start = time.perf_counter()
    gspec = _as_grid_spec(grid)
    dev = _resolve_device(device)
    torch.manual_seed(int(seed))

    if isinstance(lattice, Mapping):
        get = lambda k, d=None: lattice.get(k, d)  # noqa: E731
    else:
        get = lambda k, d=None: getattr(lattice, k, d)  # noqa: E731
    n_sx = int(get("n_sx"))
    n_sy = int(get("n_sy"))
    n_layers = int(get("n_layers"))
    dx_spot = float(get("dx_spot"))
    dy_spot = float(get("dy_spot"))
    x0 = float(get("x0", 0.0) or 0.0)
    y0 = float(get("y0", 0.0) or 0.0)
    energies = np.atleast_1d(np.asarray(get("energies"), dtype=np.float64)).ravel()
    weights = np.asarray(get("weights"), dtype=np.float64)
    if energies.size != n_layers:
        raise ValueError(f"energies has {energies.size} entries, n_layers={n_layers}")
    if weights.shape != (n_layers, n_sy, n_sx):
        raise ValueError(
            f"weights has shape {weights.shape}, expected {(n_layers, n_sy, n_sx)}")

    dose = torch.zeros(gspec.shape, dtype=_DTYPES[config.dtype], device=dev)
    tot_primary = tot_exit = tot_nuc = tot_ecut = 0
    overflow_energy = 0.0
    clipped_energy = 0.0
    layer_stats = []
    n_spots_used = 0

    for l in range(n_layers):
        E_l = float(energies[l])
        w_l = weights[l]
        if not np.any(w_l):
            layer_stats.append({"E": E_l, "n_spots": 0, "r90_depth": float("nan"),
                                "wall_time_s": 0.0})
            continue
        ker, info = _run_pencil(
            E_l, int(histories_per_spot), gspec, config,
            sigma_x0=float(sigma_x0), sigma_theta0=float(sigma_theta0),
            rho0=float(rho0), device=dev,
            seed=int(seed) + 1000 * (l + 1), chunk_size=int(chunk_size),
            lateral_world=False, record_depths=())
        tot_primary += int(info["n_history"])
        tot_exit += int(info["n_exit"])
        tot_nuc += int(info["n_nuclear"])
        tot_ecut += int(info["n_ecut"])
        overflow_energy += float(info["overflow_energy"])

        full = float(ker.sum())
        j0, j1, i0, i1 = _kernel_crop(ker)
        kc = ker[:, j0:j1, i0:i1].contiguous()
        clipped_energy += max(full - float(kc.sum()), 0.0)
        for j in range(n_sy):
            for i in range(n_sx):
                w_ij = float(w_l[j, i])
                if w_ij == 0.0:
                    continue
                n_spots_used += 1
                # same lateral convention as pbdose.model.SpotLattice:
                # x0, y0 are the CENTRE of the n_sx x n_sy lattice
                _place_kernel(dose, kc, j0, i0,
                              fsx=(x0 + (i - 0.5 * (n_sx - 1)) * dx_spot) / gspec.dx,
                              fsy=(y0 + (j - 0.5 * (n_sy - 1)) * dy_spot) / gspec.dy,
                              weight=w_ij)
        layer_stats.append({
            "E": E_l, "n_spots": int(np.count_nonzero(w_l)),
            "r90_depth": cumulative_depth_metric(ker, gspec, 0.9),
            "kernel_energy_MeV": full, "wall_time_s": info["wall_time_s"],
            "n_steps": int(info["n_steps"]),
            "n_history": int(info["n_history"]),
        })
        if config.verbose:
            print(f"[mc_impt_field] layer {l + 1}/{n_layers}  E={E_l:g} MeV  "
                  f"kernel {tuple(ker.shape)} -> crop {tuple(kc.shape)}  "
                  f"{info['wall_time_s']:.2f} s")

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t_start
    total_energy = float(dose.sum())
    stat = {
        "n_history": tot_primary,
        "mean_energy_deposited": total_energy / tot_primary if tot_primary else float("nan"),
        "total_energy_deposited": total_energy,
        "n_exit": tot_exit, "n_nuclear": tot_nuc, "n_ecut": tot_ecut,
        "n_spots": n_spots_used, "n_layers": n_layers,
        "energy_outside_scoring_grid": overflow_energy,
        "kernel_crop_clipped_energy": clipped_energy,
        "finite": bool(torch.isfinite(dose).all()),
        "layers": layer_stats,
        "wall_time_s": wall,
        "histories_per_s": tot_primary / wall if wall > 0 else float("nan"),
        "device": str(dev), "dtype": str(config.dtype),
        "energy_straggling": bool(getattr(config, "energy_straggling", False)
                                  and str(config.straggling_model).lower() != "none"),
        "transport": ("one translation-invariant pencil-beam kernel per energy "
                      "layer, laterally shifted and scaled by the spot weight"),
    }
    return {
        "dose": dose,
        "n_primary": tot_primary,
        "n_absorbed": tot_exit + tot_nuc,
        "n_history": tot_primary,
        "stat": stat,
        "wall_time_s": wall,
    }


# ---------------------------------------------------------------------------
# __main__: 200 MeV single-spot self test
# ---------------------------------------------------------------------------
def _selftest_200MeV(n_histories: int = 2_000_000) -> dict:
    print("=" * 78)
    print("pbdose.montecarlo — 200 MeV single-spot self test (128^3 @ 0.2 cm)")
    print("=" * 78)
    print(mc_config_summary(DEFAULT_MC))
    gspec = GridSpec(nx=128, ny=128, nz=128, dx=0.2, dy=0.2, dz=0.2,
                     ox=-12.8, oy=-12.8, oz=0.0)
    print(f"grid   : {gspec.nx}x{gspec.ny}x{gspec.nz} @ {gspec.dx} cm   "
          f"x[{gspec.x0},{gspec.x1}] y[{gspec.y0},{gspec.y1}] z[{gspec.z0},{gspec.z1}]")
    print(f"device : {'cuda:' + str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'}"
          f"   torch {torch.__version__}")
    print(f"step schedule for 200 MeV: {mc_step_schedule(200.0).size} steps")

    res = mc_spot_dose(200.0, gspec, n_histories=n_histories, config=DEFAULT_MC,
                       device="cuda", seed=12345, chunk_size=2_000_000)
    st = res["stat"]
    R_csda = float(physics.csda_range(200.0))
    print("-" * 78)
    print(f"histories                 : {st['n_history']:,}")
    print(f"wall time                 : {res['wall_time_s']:.3f} s   "
          f"({'PASS' if res['wall_time_s'] < 60 else 'FAIL'} < 60 s)")
    print(f"histories / second        : {st['histories_per_s']:.4g}")
    print(f"transport steps           : {st['n_steps']} (planned {st['n_steps_planned']}), "
          f"chunks = {st['n_chunks']}, dtype {st['dtype']}")
    print(f"deposited energy / primary: {st['mean_energy_deposited']:.4f} MeV "
          f"(energy balance {st['energy_balance'] * 100:.3f} % of 200 MeV)")
    print(f"exits / nuclear / e-cut   : {st['n_exit']:,} / {st['n_nuclear']:,} / "
          f"{st['n_ecut']:,}    n_absorbed = {res['n_absorbed']:,}")
    if st["n_exit"]:
        print(f"  NOTE: the 128^3/0.2 cm grid ends at z = {gspec.z1:.2f} cm while "
              f"csda_range(200) = {R_csda:.3f} cm, so protons leave the distal face")
        print(f"        and bank their remaining energy there (energy conservation); "
              f"that energy is distal to R90, which keeps R90 unbiased.")
    print(f"energy outside grid       : {st['energy_outside_scoring_grid']:.3e} MeV")
    print("-" * 78)
    print(f"R90 (MC)                  : {st['r90_depth']:.4f} cm")
    print(f"csda_range(200)           : {R_csda:.4f} cm   [NIST PSTAR anchor 25.92 cm]")
    d90 = 100.0 * (st["r90_depth"] - R_csda) / R_csda
    print(f"deviation                 : {d90:+.3f} %  (tolerance 3 %)  "
          f"{'PASS' if abs(d90) < 3 else 'FAIL'}")
    print(f"R80 / R50                 : {st['r80_depth']:.4f} / {st['r50_depth']:.4f} cm")

    z_ref = np.linspace(0.0, R_csda * 1.0001, 1001)
    fe = physics.fermi_eyges_moments(z_ref, 200.0, sigma_x0=0.0, sigma_theta0=0.0)
    sig_fe = float(np.interp(10.0, z_ref, fe["sigma_x"]))
    sig_mc = st["sigma_x_at_depth"].get(10.0, float("nan"))
    sth_fe = float(np.sqrt(np.interp(10.0, z_ref, fe["A0"])))
    sth_mc = st["sigma_theta_at_depth"].get(10.0, float("nan"))
    dose_rms = st["voxelised_sigma_x_at_depth"].get(10.0, float("nan"))
    dlat = 100.0 * (sig_mc - sig_fe) / sig_fe
    # the scored slab [10.0, 10.2) receives the deposits of z = 10.0 and 10.1 cm,
    # so its voxelised width is compared with the binned FE width at 10.1 cm
    sig_slab = float(np.interp(10.1, z_ref, fe["sigma_x"]))
    exp_vox = binned_gaussian_sigma(sig_slab, gspec.dx)
    print("-" * 78)
    print("lateral width at z = 10 cm (phase-space estimator, no voxel bias)")
    print(f"  sigma_x  MC            : {sig_mc:.6f} cm   (n = {st['n_at_depth'].get(10.0, 0):,})")
    print(f"  sigma_x  Fermi-Eyges   : {sig_fe:.6f} cm")
    print(f"  deviation              : {dlat:+.3f} %  (tolerance 5 %)  "
          f"{'PASS' if abs(dlat) < 5 else 'FAIL'}")
    print(f"  sigma_th MC / FE       : {sth_mc:.6f} / {sth_fe:.6f} rad   "
          f"({100.0 * (sth_mc - sth_fe) / sth_fe:+.3f} %)")
    print(f"  voxelised dose plane   : {dose_rms:.6f} cm  vs binned-FE expectation "
          f"{exp_vox:.6f} cm (voxel bias, not a physics error)")
    print("-" * 78)
    ok = (res["wall_time_s"] < 60.0) and (abs(d90) < 3.0) and (abs(dlat) < 5.0)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return {"spot": res, "grid": gspec, "r_csda": R_csda, "sigma_fe": sig_fe,
            "sigma_mc": sig_mc, "ok": bool(ok)}


if __name__ == "__main__":
    _selftest_200MeV()
