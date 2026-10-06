# GPU-Accelerated Dose Reconstruction for Proton Therapy

A Fermi–Eyges pencil-beam dose engine for intensity-modulated proton therapy (IMPT),
with custom GPU kernels and an independent Monte Carlo code written to validate it.

**At clinical scale — 67,240 pencil beams over a 2.1-million-voxel grid — a full
3-D dose field is reconstructed in 7.41 ms, about 155,000× faster than the serial
CPU baseline.**


 axial slice, sagittal slice, depth dose, and lateral profile of a
10 × 10 cm IMPT field.*

---

## Why this matters

Proton therapy spares healthy tissue because protons stop at a well-defined depth —
the Bragg peak. That same sharpness makes it fragile: if the patient breathes or an
organ shifts, the beam lands in the wrong place. Online adaptive radiotherapy
therefore needs the whole dose field recomputed in **under a second**, while a
conventional CPU implementation takes minutes.

## The key idea

The standard algorithm has every voxel sum the contribution of every pencil beam —
O(M·N), or **1.41 × 10¹¹** multiply-adds at clinical scale. Because the lateral
Gaussian is separable and the beam spots sit on a regular lattice, each depth slice
is the convolution of a spot-amplitude map with a depth-dependent kernel:

**1.14 × 10⁸ multiply-adds — a 1,242× algorithmic gain, independent of hardware.**

One physics detail mattered more than the rest: the lateral width σ depends on each
layer's *local* energy E(R₀ − z), so different energy layers have genuinely different
σ at the same depth. Collapsing them into one averaged σ is not an approximation, it
is a modelling error — measured at **20.3% peak-dose error for only a 1.47× speed-up**.
The engine therefore computes every layer exactly.

## Results

| Metric | Result |
|---|---|
| Full-field dose reconstruction | **7.41 ms** (target < 100 ms) |
| Speed-up vs. serial CPU baseline | **155,544×** (1,242× algorithmic, 125× hardware) |
| Accuracy vs. float64 reference | 1.5 × 10⁻⁷ relative |
| Lateral beam width vs. independent Monte Carlo | **0.02%** |
| Range straggling σ_R vs. independent Monte Carlo | **2.4%** |
| Mid-depth (target-coverage) region, 3%/3mm gamma | **99.25 – 100%** |


*Acceleration chain and roofline analysis.*

### Acceleration chain

| Implementation | Device | Time | Speed-up |
|---|---|---:|---:|
| Naive per-beam accumulation (extrapolated) | CPU, 1 thread | 1,152,984 ms | 1× |
| NumPy separable | CPU, 1 thread | 6,708 ms | 172× |
| PyTorch batched matmul | CPU, 16 threads | 395 ms | 2,915× |
| PyTorch local-window gather | RTX 5070 Ti | 1,003 ms | 1,149× |
| Triton naive gather (custom kernel, extrapolated) | RTX 5070 Ti | 654 ms | 1,762× |
| Triton window gather + shared LUT | RTX 5070 Ti | 82.9 ms | 13,908× |
| Triton separable conv, fused Gaussian GEMM | RTX 5070 Ti | 19.6 ms | 58,859× |
| **PyTorch per-layer batched matmul** | **RTX 5070 Ti** | **7.41 ms** | **155,544×** |

## File directory
## 1. Documentation

Four typeset PDF reports. They are the technical record of the project: the derivation,
the measurements, the validation and the process.

| File | Size | Contents |
|---|---:|---|
| `01_fermi_eyges_derivation.pdf` | 300 KB | Full mathematical derivation — small-angle transport equation, Fermi–Eyges moment equations A₀/A₁/A₂, proof that σ_x²(z) = 2A₂(z), the Highland scattering power and its derivative, Bethe–Bloch stopping power, CSDA range, and the analytic integral depth dose construction. 30 numbered equations. |
| `02_performance_report.pdf` | 422 KB | Benchmarking methodology and the complete acceleration chain, from the serial CPU baseline to the GPU engine. Roofline and bottleneck analysis, VRAM constraints, the numerical-precision budget, and a ranked list of remaining optimisation opportunities. |
| `03_validation_report.pdf` | 712 KB | Five levels of validation — comparison against NIST PSTAR, CSDA range anchors, energy conservation, the Fermi–Eyges analytic identities, cross-comparison of five independent implementations, and the Monte Carlo reference with its full-field Gamma Index analysis. |
| `04_research_log.pdf` | 423 KB | Chronological log of the twelve problems encountered, the reasoning at each step and the fix — including the one that remains unfixed. |

## 2. Project documentation

| File | Size | Contents |
|---|---:|---|
| `README.md` | 5.7 KB | Project overview: the headline result, the algorithmic idea, the acceleration chain, what did not work, the limitations, and links to the four reports. |

## 3. Physics engine

The core library. Currently sitting in the repository root; imports assume the package
directory `pbdose/`.

| File | Size | Responsibility |
|---|---:|---|
| `physics.py` | 21.6 KB | Bethe–Bloch stopping power, numerically integrated CSDA range table, Highland scattering power and its derivative, Fermi–Eyges moments, and the analytic integral depth dose (IDD) with range straggling and nuclear attenuation. |
| `model.py` | 13.6 KB | Core data structures — `DoseGrid`, `SpotLattice`, `PBProblem` — and builders for Gaussian and flat IMPT fields plus the clinical layer energies. |
| `engines.py` | 17.6 KB | Five dose-calculation implementations: naive Python triple loop, naive NumPy per beam, separable NumPy, PyTorch batched matrix multiplication (per-layer exact), and PyTorch windowed gather. |
| `triton_kernels.py` | 12.8 KB | Three custom Triton kernels — naive gather, windowed gather with a shared-memory LUT, and a fused separable Gaussian convolution. JIT-compiled through PTX and `ptxas` to native `sm_120` code. |
| `montecarlo.py` | 78.0 KB | Independent class-II condensed-history Monte Carlo: Bohr energy straggling, nuclear attenuation and line-integral voxel scoring. Used as the independent reference for the Gamma analysis. |
| `gamma.py` | 10.4 KB | Gamma Index implementation following Low (1998) and AAPM TG-218 — global normalisation, dose threshold, and the distance-to-agreement search. |
| `benchmark.py` | 5.6 KB | Timing harness: warm-up, CUDA events, median of N runs, and operation counting. |
| `viz.py` | 16.4 KB | Figure generation for all seven result plots. |
| `__init__.py` | 1.1 KB | Package entry point. |

## 4. Executable scripts

Top-level entry points. Expect to be run from the repository root.

| File | Size | Responsibility |
|---|---:|---|
| `run_benchmark.py` | 13.5 KB | Runs the full benchmark sweep and writes `results/benchmark_full.json`. |
| `run_validation.py` | 9.3 KB | Runs the validation pipeline. Configurable via flags for histories, grid size, depth samples, voxel size, layer count, spot count, straggling model and beam spot size. |
| `make_figures.py` | 3.4 KB | Regenerates all seven figures from the stored result data. |

## 5. Verification and diagnostic tools

One-off checks and diagnostics used during development. They are kept because each one
corresponds to a specific question that had to be answered.

| File | Size | Responsibility |
|---|---:|---|
| `mc_selftest.py` | 33.4 KB | Monte Carlo internal self-tests — statistics, straggling behaviour and energy deposition. |
| `engine_check.py` | 3.1 KB | Cross-checks the five implementations against one another. |
| `triton_check2.py` | 2.9 KB | Correctness checks for the custom Triton kernels against the NumPy reference. |
| `mc_vs_analytic.py` | 2.6 KB | Compares the Monte Carlo and analytic lateral profiles and second moments. |
| `gamma_regions.py` | 2.4 KB | Decomposes Gamma failures by region — entrance, interior, lateral edge and distal falloff. |
| `gpu_microbench.py` | 2.3 KB | GPU hardware microbenchmarks: DRAM bandwidth, GEMM peak, machine balance. |
| `fetch_pstar.py` | 1.9 KB | Retrieves NIST PSTAR reference stopping-power data for the validation. |
| `grid_check.py` | 1.8 KB | Grid-convergence checks on the dose grid and depth sampling. |
| `crosscheck.py` | 1.7 KB | General cross-validation helpers shared by the other tools. |
| `profile_diff.py` | 1.7 KB | Depth-dose and lateral profile difference analysis. |
| `fluence_check.py` | 1.4 KB | Verifies the fluence-normalisation convention of the analytic IDD. |

## 6. Result data

Machine-readable output of the benchmark and validation runs. All numbers quoted in the
reports are traceable to these files.

| File | Size | Contents |
|---|---:|---|
| `benchmark_full.json` | 6.7 KB | Full benchmark output — per-implementation timings, speed-ups and operation counts. |
| `validation_bohr.json` | 4.1 KB | Validation results with the Bohr range-straggling model. |
| `gpu_microbench.json` | 0.4 KB | GPU microbenchmark results — DRAM bandwidth, GEMM peaks and machine balance. |

## 7. Figures

Seven result plots. Referenced by the reports; source data in `results/`.

| File | Size | Contents |
|---|---:|---|
| `fig4_accuracy_variants.png` | 330 KB | Accuracy comparison across implementation variants, including the single-σ failure mode. |
| `fig5_performance.png` | 254 KB | Acceleration chain and roofline analysis. |
| `fig1_physics_validation.png` | 242 KB | Triple validation of stopping power, CSDA range and the analytic Bragg peak. |
| `fig3_dose_maps.png` | 216 KB | Axial slice, sagittal slice, depth dose and lateral profile of a 10 × 10 cm IMPT field. |
| `fig6_gamma.png` | 206 KB | Gamma map, histogram and central-axis depth dose of the full-field validation. |
| `fig2_fermi_eyges.png` | 170 KB | Fermi–Eyges lateral broadening and the local scattering power. |
| `fig7_gamma_criteria.png` | 67 KB | Gamma pass rate as a function of the acceptance criteria. |

## 8. CUDA implementation

| File | Size | Contents |
|---|---:|---|
| `pencil_beam.cu` | 60.9 KB | Hand-written CUDA C++. Five `__global__` kernels — naive gather, shared-LUT gather, spot-amplitude scatter, separable Gaussian convolution and batched gather — plus the `extern "C"` host API and a `-DPB_STANDALONE_MAIN` self-test program. |

## What did not work

My full-field 3%/3mm gamma pass rate is **90.97%**, below my own 98% target. The
failures cluster at the surface and the lateral field edge; the mid-depth region that
determines target coverage passes at 99.25–100% with a distal falloff agreeing to
0.11 mm.

Getting there required finding **three implementation errors**, every one of which was
invisible to component-level tests:

1. The analytic model normalised each energy layer's depth-dose curve to the same
   peak, which silently turned **fluence-type** spot weights into **dose-type** weights
   (the 59 MeV layer's IDD peaks at 43.8 MeV/cm against 19.4 MeV/cm at 198 MeV).
2. The Monte Carlo always transported a **point** pencil beam while the analytic model
   used a 3 mm nozzle spot — the lateral scalloping of a 5 mm spot lattice differs
   substantially between the two.
3. A **spot-lattice centring convention mismatch** (edge-anchored versus centred)
   displaced the field by 5.0 cm.

A fourth cause remains: my range-straggling kernel convolves the whole depth-dose
curve with the *final* σ_R, but the range offset accumulates with depth, so the
physical kernel width at depth z is σ_ρ(z) = √∫₀ᶻ BOHR_COEF/(β²S²) dz′. At z = 0.2 cm
this over-smooths by **6.1×**. It is documented, not tuned away, and is the first item
on the list below.


*Gamma map, histogram and central-axis depth dose of the full-field validation.*


## Limitations — please read these before citing any number

- **Homogeneous water phantom only.** Heterogeneous media (bone, lung) need
  water-equivalent-depth ray tracing, which is not implemented.
- **Gaussian scattering only.** The large-angle Rutherford tail is neglected, so the
  penumbra is underestimated.
- **The Monte Carlo is not Geant4/TOPAS.** It is a home-built class-II
  condensed-history code. It shares the stopping power and scattering power with the
  analytic model, so it validates the *transport and implementation*, not the Gaussian
  assumption itself. No delta-ray transport, no nuclear secondaries.
- **The hand-written CUDA C++ file is delivered but not compiled.** This machine has
  no CUDA toolkit, so every measured number comes from functionally equivalent Triton
  kernels (which do compile to native sm_120 code).
- **Full-field 3%/3mm gamma is 90.97%, not the 98% target.** See above.
- **Run-to-run variance is about ±10%** on this laptop GPU; the three full benchmark
  runs gave 6.91 / 6.70 / 7.41 ms. All ratios in the tables come from a single run.

## Further reading

01_fermi_eyges_derivation.pdf，02_performance_report.pdf， 03_validation_report.pdf and 04_research_log.pdf the full
Fermi–Eyges derivation, a performance and profiling report, a physics and numerical
validation report, a research log of every problem encountered (with the reasoning and
the fix). 




