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

## Reproduce

```bash
pip install numpy torch matplotlib triton-windows pypdf reportlab

python -c "import sys;sys.path.insert(0,'.');from pbdose import physics as ph;ph.validate_stopping_power()"
python scripts/run_benchmark.py --repeat 15 --tag full   # ~10 min
python scripts/make_figures.py
python scripts/run_validation.py --histories 2000000 --tag bohr \
       --straggling bohr --sigma-x0 0.30                 # ~4 min
```

## Layout

```
pbdose/physics.py         Bethe-Bloch stopping power, CSDA range, Fermi-Eyges moments,
                          Highland scattering power, analytic Bragg peak, NIST PSTAR check
pbdose/model.py           dose grid, spot lattice, IMPT problem definition
pbdose/engines.py         five implementations (naive loop -> NumPy -> PyTorch -> GPU)
pbdose/triton_kernels.py  three custom GPU kernels (Triton -> native sm_120 code)
pbdose/montecarlo.py      independent condensed-history Monte Carlo transport (GPU)
pbdose/gamma.py           Gamma Index analysis (Low 1998 / AAPM TG-218)
cuda/pencil_beam.cu       hand-written CUDA C++ implementation (1,192 lines, 5 kernels)
docs/                     technical reports (Chinese, with an English summary)
apply/                    one-page English project summary
```

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

01_fermi_eyges_derivation.pdf，02_performance_report.pdf， 03_validation_report.md and 04_research_log.pdf the full
Fermi–Eyges derivation, a performance and profiling report, a physics and numerical
validation report, a research log of every problem encountered (with the reasoning and
the fix). 




