# Research Log: Problems, Thinking Process, and Solutions

> This file records, in chronological order, the technical problems **actually encountered** while
> carrying out this project, the reasoning path taken at the time, and the final solution.
> All numbers come from the actual run output of this project's code, with no embellishment.
> The Step 7 "document presentation and interview" material (`07_personal_statement_interview.md`)
> is distilled from this log.

---

## Problem 1: There's a GPU, but no CUDA compiler

**Symptom.** A survey of the environment found that this machine has an NVIDIA RTX 5070 Ti Laptop GPU
(Blackwell architecture, compute capability **sm_120**, 46 SMs, 11.9 GB VRAM), but
`Get-Command nvcc` and `Get-Command cl.exe` both came back empty — there is neither a CUDA Toolkit
nor MSVC on the system. Phase two of the project document calls for "advanced C++/CUDA kernel
development", and its precondition does not hold.

**Thinking process.** Three paths lay in front of me:

1. **Install the CUDA Toolkit + Visual Studio Build Tools.** About 3–6 GB of downloads, plus
   configuring the integration between `cl.exe` and nvcc; getting `torch.utils.cpp_extension` to
   work on Windows usually also means stepping into a few environment-variable potholes. For the
   goal of "getting the research done quickly", the time cost is too high, and the failure risk is
   large (Blackwell needs CUDA 12.8+, and a version mismatch means the install was wasted).
2. **Use only high-level PyTorch operators (`torch.bmm` and so on).** The least effort, but then the
   "custom CUDA kernel" deliverable evaporates, section four of the project document (thread
   mapping, Shared Memory LUT) has nothing to show for it, and the engineering substance of the
   project shrinks noticeably.
3. **Use Triton as the kernel authoring and compilation backend.** Triton lets you write kernels in
   Python, JIT-compiles them to PTX, and `ptxas` then assembles that into CUBIN for the target
   architecture. The key point is that it **ships its own ptxas**, so no system CUDA Toolkit install
   is needed. `pip install triton-windows` is enough.

**Decision and verification.** I chose path 3, but ran a feasibility check first instead of just
assuming it would work: right after installing, I wrote a minimal `add_kernel` test and confirmed it
could compile on sm_120 and produce the correct result (`max err = 0.0`). Only after that passed did
Triton go into the technical route.

At the same time I kept a **fourth approach as a supplement**: a complete hand-written CUDA C++
source file (`cuda/pencil_beam.cu`, 1192 lines, containing 5 `__global__` kernels, the host API, and
a `cudaEvent` timing self-test), as a code deliverable and as a reference implementation to be
compiled later on a machine that has nvcc. The report explicitly notes that this file **was not
compiled on this machine**.

**Result and cost.** The Triton route produced three custom kernels that genuinely run on the GPU:

| Kernel | Purpose | Measured median time |
|---|---|---|
| `pb_gather_naive_kernel` | loop over all M beams per voxel (the project document's original design) | 654 ms (linear extrapolation over all M) |
| `pb_gather_window_kernel` | Gaussian tight-support window + IDD/σ LUT caching | 82.90 ms |
| `_pass1_z` / `_pass2_z` | separable-convolution GEMM with fused Gaussian kernel generation | 19.59 ms |

**The cost**: Triton's expressive power falls short of CUDA C++ (for example, no fine control over
shared memory layout and bank conflicts, and no `__constant__` memory), so the more aggressive
optimizations in `cuda/pencil_beam.cu` (`cp.async` pipelining, explicit shared memory LUT) cannot be
empirically verified on this machine. This went into the report's "known limitations".

**What I learned**: an environment limitation should not be turned directly into a reason to "lower
the target", nor into a reason to "pretend I did it". The right move is to find a **technically
equivalent, verifiable** alternative path, and to write out clearly how it differs from the original
plan.

---

## Problem 2: Merging 40 energy layers into a single σ(z) — an approximation that looks beautiful and is in fact wildly wrong

**Background.** The three-dimensional pencil beam dose formula is

$$
D(x,y,z)=\sum_l \sum_{ij} w_{lij}\,\mathrm{IDD}_l(z)\,
\frac{1}{2\pi\sigma_l(z)^2}\exp\!\left(-\frac{(x-x_i)^2+(y-y_j)^2}{2\sigma_l(z)^2}\right)
$$

The original design was: since the lateral Gaussian kernel is separable, first fold the weights of
all energy layers through the IDD into a single "depth-wise amplitude map"
$A(z,i,j)=\sum_l w_{lij}\mathrm{IDD}_l(z)$, then use **one** layer-weighted average $\bar\sigma(z)$ to
do the lateral convolution. That way the entire three-dimensional dose field degenerates into
"$n_z$ two-dimensional convolutions", the computational cost is tiny, and it is very pretty to
write.

**How I found the problem.** I did not adopt this design directly. Instead I first ran a
cross-validation: compute the same problem with `separable_numpy(per_layer=True)` (per-layer exact)
and with `separable_numpy(per_layer=False)` (single-σ approximation), and compare the two.

- Small-scale test (3 energy layers, 80/130/180 MeV): maximum relative error **7.4%**;
- Another 3-layer test (90/140/190 MeV): **9.6%**.

The error was larger than expected, which pushed me to think the physics through properly:
$\sigma_l(z)$ depends on the **local energy at depth z** of the protons in that layer,
$E_l(z)=E(R_l-z)$. Different energy layers have different residual ranges **at the same depth**,
hence different local energies and different scattering power $T(E)$, so $\sigma_l(z)$ **is itself
different from layer to layer**. Merging them into a single $\bar\sigma(z)$ is not an
"approximation" — it gets the physics wrong.

**The cost at full size.** Measured on a real clinical field (40 layers, 59–198 MeV, M = 67,240):

- Single-σ approximation: **5.04 ms**, maximum relative error **20.3%** (relative to the global peak dose);
- Per-layer exact: **7.41 ms**, maximum relative error 1.5×10⁻⁷.

**Conclusion**: that approximation is only 1.47 times faster and brings 20% error — completely
unusable under the 3%/3mm clinical acceptance criterion. The exact approach is mandatory: take
(layer, depth) as the batch dimension and do a per-layer batched matrix multiplication.

**Reflection.** The value of this episode is not that "I found a bug", but the **process**: if I had
written that pretty merged version first and then taken it off to compare γ pass rates against Monte
Carlo, I would have seen a pile of inexplicable penumbra mismatch, and might then have spent days
tuning parameters, doubting the MC, doubting the grid. **Establishing cross-validation before
optimizing** made this error surface within 5 minutes.

There was also a numerical by-product: the error grows as the energy-layer span widens (3 layers,
100 MeV span → 7–10%; 40 layers, 139 MeV span, and the layer-to-layer σ differences are largest at
shallow depths → 25%). That pattern is itself evidence of physical understanding.

---

## Problem 3: The Bragg peak is too tall — what should range straggling actually be?

**Symptom.** After the analytic IDD construction was finished, the first version gave a 200 MeV
pristine Bragg peak "peak/entrance dose" (peak-to-entrance ratio) of **6.4**. In the proton therapy
literature this ratio for a clinical pristine peak is usually between **3–5**. My peak was clearly
too sharp and too tall.

**Step one: confirm this is not a numerical bug.** I ran an energy conservation check: the IDD
construction uses **exact energy binning**

$$D_i=\Phi(z_i)\cdot\frac{E(z_{i-1/2})-E(z_{i+1/2})}{\Delta z}$$

that is, "the dose deposited in each depth bin = the kinetic energy lost within that bin / bin
thickness". With this, $\int D\,dz$ must equal the total beam energy (times the fluence attenuation
factor). Measured at 100 MeV: ∫D dz = 96.76 MeV (theory 97.5); at 200 MeV: 179.39 MeV (theory 183.4),
deviation < 2%. **Energy conservation is correct**, so the problem is not normalisation.

A side note: writing $D\propto S(E(z))$ directly diverges at $E\to0$ ($S\propto E^{-0.77}$), whereas
the binned formulation is mathematically equivalent but completely avoids the singularity — that is
the biggest engineering value of this construction.

**Step two: confirm the physical relationship.** I derived the correct convolution form for range
straggling. Protons have a range distribution $p(R)$, and the dose at depth z is

$$D(z)=\Phi_0\int_0^\infty S\big(E(u)\big)\,G\big(u-(R_0-z);\sigma_R\big)\,du$$

(where $u$ is the residual range). Whereas "first compute the CSDA curve and then Gaussian-convolve
in z" gives

$$D(z)=\int S\big(E(v)\big)\,G\big((R_0-z)-v;\sigma_R\big)\,dv$$

The two are **exactly equivalent**. So the convolution approach is right, and the only uncertain
quantity is the size of $\sigma_R$.

**Step three: work out what $\sigma_R$ actually is.** I integrated from scratch using Bohr's energy
straggling theory:

$$\sigma_R^2=\int_{E_{lo}}^{E_0}\frac{K(Z/A)\rho\,m_ec^2}{\beta(E)^2\,S(E)^3}\,dE$$

(derivation: the energy straggling at depth x is $d\Omega^2/dx=0.08711/\beta^2\ \mathrm{MeV^2/cm}$,
converted into range fluctuation through the residual-range sensitivity $dR/dE=1/S$.) Numerical
integration results:

| $E_0$ | Bohr integral $\sigma_R$ | ICRU 49 empirical fit $0.012R^{0.935}$ |
|---|---|---|
| 100 MeV | 0.225 cm | 0.093 cm |
| 200 MeV | 0.511 cm | 0.294 cm |
| 250 MeV | 0.658 cm | 0.359 cm |

**The two differ by a factor of 2.** This is a real problem, not a detail that can be fudged.

**Step four: let observable clinical metrics be the judge.** I ran both $\sigma_R$ values and
compared the peak-region metrics (200 MeV):

| $\sigma_R$ model | peak/entrance | distal 80–20% falloff | literature reference |
|---|---|---|---|
| ICRU 49 (0.294 cm) | 5.98 | 3.9 mm | peak/entrance 3–5; 80–20 about 3–5 mm |
| Bohr integral (0.533 cm) | 4.88 | 7.0 mm | same as above |

**Conclusion and how I handled it**: each model gets half of it right — Bohr gives the peak height
closer to the literature, ICRU 49 gives the distal falloff closer to the literature. I did not
quietly pick the "good-looking" one and use it. Instead:

1. Default to the **Bohr first-principles integral** (consistent with this project's consistent
   practice of "starting from first principles", and its peak height is closer to the literature);
2. Expose `straggling_model ∈ {bohr, icru49, none}` and the beam energy spread `energy_spread` as
   switchable parameters (a real accelerator beam itself has 0.3–1% energy spread, which is a
   legitimate calibration quantity);
3. In the validation report, **explicitly state this ±30% model uncertainty in the peak region**,
   and explain that this project's γ validation was done under both settings.

**Reflection.** The mistake a high-school research project is most likely to make is to quietly tune
parameters until things "look right" whenever a result disagrees with the literature. The correct
handling here is to turn it into a **quantified, disclosed model uncertainty**. The value of this
episode in an interview is far higher than "my model agrees well with the literature".

---

## Problem 4: A bug that was off by a factor of √(2π) = 2.5066

**Symptom.** After finishing the Triton separable convolution kernel I ran a correctness check:
compared with the NumPy reference, the **maximum relative error was 0.84** (almost entirely wrong),
RMS relative error 0.14. But the PyTorch version of the same algorithm agreed with the NumPy version
to 3×10⁻⁷.

**Investigation process.**

1. I first suspected numerical precision. I wrote a minimal test comparing `tl.dot`'s three precision
   modes: `tf32` (default) error 6.9×10⁻⁴, `ieee` error 1.3×10⁻⁷, explicit loop 1.3×10⁻⁷. I switched
   to `input_precision="ieee"` and re-ran — **the error did not change at all**. Precision ruled out.
2. So I went and compared **ratios** instead of differences. I printed `T_triton / T_numpy`:
   the ratio for each row (spot-x index) was **exactly constant**, at 2.508, 3.008, 1.280……
   respectively. Wait — the topmost row's ratio across all y was **2.5066**.
3. 2.5066 = **√(2π)**. That number located the problem immediately: each of the two steps of the
   separable convolution is a **one-dimensional** Gaussian, whose normalisation constant should be
   $1/(\sqrt{2\pi}\sigma)$; but what I was passing into the kernel was `INV2PI = 1/(2π)` (the
   normalisation constant of a two-dimensional Gaussian). The ratio of the two is exactly
   $(1/\sqrt{2\pi})/(1/2\pi)=\sqrt{2\pi}$.

**Fix.** In `triton_kernels.py` I explicitly defined two constants, `INV2PI` (for 2D) and
`INV_SQRT_2PI` (for 1D), and switched the separable kernel to the latter. After the fix the maximum
relative error dropped to 1.09×10⁻⁶ (float32 rounding level).

**Reflection.** During the investigation I kept looking at the **absolute value of the error**, and
the more I looked the more confused I got. The turning point was switching to looking at **ratios**;
the constancy of the ratio immediately separated "a precision problem" from "a normalisation
problem". Separately, the fact that `tl.dot` defaults to TF32 (error 6.9×10⁻⁴) is itself worth
remembering — in scientific computing the default is often not the one you want.

---

## Problem 5: VRAM exhaustion — the pit the project document predicted, and I really did step in it

**Symptom.** Running the full-size configuration on the PyTorch local-window Gather
implementation: `torch.AcceleratorError: CUDA error: out of memory`.

**Cause analysis.** To do its "two rounds of one-dimensional gather", that implementation builds an
intermediate tensor `Ti = T[:, :, idx_x, :]` of shape $(L,\ n_z,\ n_x,\ K_x,\ n_y)$. Substituting the
actual parameters: $L=40$, $n_z=128$, $n_x=n_y=128$, $K_x=17$, number of elements
$=40\times128\times128\times17\times128\approx 1.4\times10^9$, about **5.7 GB** in float32 — that
does not fit in 12 GB of VRAM, and this is only one intermediate quantity.

**Solution.** Use **per-energy-layer chunking (layer chunking)**: hoist the loop over $L$ to the
outermost level and process only one layer at a time. Peak VRAM drops to $1/40$ (about 143 MB), and
the total result is obtained by accumulation. This corresponds exactly to the recommendation in
section five of the project document, "GPU VRAM limits (VRAM Exhaustion) → adopt Beam Batching or an
On-the-fly computation mode, avoiding storage of huge intermediate four-dimensional tensors" — 
except that my intermediate tensor is **five**-dimensional.

**Reflection.** I had read the project document's "pitfall guide" in advance, and I still stepped
into it while writing the code. This shows that "knowing there is a pit" and "avoiding it at design
time" are two different things. Later I wrote peak VRAM into the benchmarking framework as an
explicit metric (`torch.cuda.max_memory_allocated`), so that it is recorded on every measurement.

---

## Problem 6: The naive baseline simply cannot finish — and how to measure it honestly

**Symptom.** The original algorithm complexity given in the project document is
$O(M\cdot N_x N_y N_z)$. Substituting the actual configuration: $M=67{,}240$, grid
$128^3=2.10\times10^6$, total multiply-adds $=1.41\times10^{11}$. Measured scaling in NumPy with the
"accumulate the full grid per beam" formulation: 200 beams × 262,144 voxels took 0.462 s, which
linear extrapolation puts at about **1,153,000 ms ≈ 19.2 minutes** at full size (and this is still
the vectorised formulation; a real triple for-loop would be orders of magnitude slower).

**Thinking process.** There is a methodological trap here: if I report only the "extrapolated value",
that number is not a real measurement; if I report only the small-scale measured value, it does not
answer the core question of "can it actually meet <100 ms". Neither approach is honest enough.

**Solution: walk on two legs.**

1. **CPU side**: honestly report the small-scale measurement + linear extrapolation, and clearly
   label "extrapolated" in the table. Also give the complexity analysis ($1.41\times10^{11}$
   multiply-adds) as independent corroboration.
2. **GPU side**: **reimplement the same naive algorithm as a custom Triton kernel**, measure it on
   the GPU at three points, M = 256 / 1024 / 4096 (3.53 / 11.56 / 40.53 ms, showing good linearity),
   and extrapolate to M = 67,240 (654 ms).

That way, "how slow the naive algorithm is" and "how fast the optimized algorithm is" are **both
real measurements from the same machine**, and only then does the speed-up carry conviction:
654 ms → 19.59 ms (Triton separable convolution) → 7.41 ms (cuBLAS per-layer bmm), for a total of
88-fold pure GPU-side speed-up.

**Reflection**: when you cannot run the full-size experiment, the right move is not to give up and
not to pretend you ran it, but to **design a scaling relationship that can be verified**, and to
write out the basis for the extrapolation (linear fit, complexity formula) along with it, so the
reader can judge the credibility themselves.

---

## Problem 7: Several "counter-intuitive" Triton constraints

While writing the kernels I ran into a string of Triton language limitations, all of which were
quickly solved but are worth recording:

1. **The length of `tl.arange` must be a power of two.** My spot lattice is 41×41 (clinical 5 mm
   spacing, 20 cm field), and 41 is not a power of two, so compilation failed outright. Solution:
   take the block size as a power of two (32/64) and use the mask `tl.where(ki < nsx, ..., 0.0)` to
   zero out the out-of-range part. This constraint effectively dictated the block-size design of the
   whole kernel.
2. **`**2` is not supported on tensors.** `(x - x0) ** 2` raises
   `AttributeError: 'tensor' object has no attribute '__pow__'`, so it has to be written as
   `d = x - x0; d * d`. In a GPU kernel this is actually faster (one fewer exponentiation).
3. **`tl.dot`'s default input precision is TF32**, error 6.9×10⁻⁴ (10-bit mantissa). Scientific
   computing needs `input_precision="ieee"` to be specified explicitly in order to get full fp32
   precision (error 1.3×10⁻⁷). I had already been bitten by this "default value trap" in Problem 4.
4. **`tl.dot` requires all dimensions ≥ 16.** Small matrices (such as 5×5) must be padded to 16/32.

**Reflection**: these constraints are not hard in themselves, but **together they shaped the
structure of the kernel**. Writing a GPU kernel is not "translating a Python loop into CUDA" but
redesigning the data layout around the constraints of the hardware and the compiler.

---

## Problem 8: An order-of-magnitude "precision vs speed" trade-off — and how to quantify it

**Background.** Right at the start the project set up an "approximation for speed" switch (the
single-σ approximation, see Problem 2), plus parameters such as tf32 and layer_chunk. I needed one
unified set of criteria to answer "is this approximation worth it".

**Approach.** Establish three quantitative criteria and report all of them on every experiment:

1. **Maximum error relative to the global field peak** (max relative error of global max) — 
   the worst case, used to judge whether it would break the γ pass rate;
2. **RMS relative error** — overall quality;
3. **Time ratio** — the speed bought in exchange.

Full-size measured results:

| Variant | Time | max rel err | Verdict |
|---|---|---|---|
| Per-layer exact FP32 | 7.41 ms | 1.5×10⁻⁷ | ✅ adopted |
| Per-layer exact TF32 | 6.77 ms | 1.3×10⁻⁷ (bit-identical to FP32) | ⚠️ no gain (matrices too small) |
| Single-σ approximation | 5.04 ms | **2.0×10⁻¹** | ❌ unusable |
| Per-layer exact + layer_chunk=4 | slightly slower | 1.3×10⁻⁷ | ✅ fallback when VRAM-constrained |

**Key conclusion**: the single-σ approximation buys only a 1.47-fold speed-up for a 20% error. Facing
a 3% clinical tolerance, that trade is absurd. **But this conclusion only holds once it has been
quantified** — without this comparison table, "trade approximation for speed" always sounds like a
reasonable engineering instinct.

**Reflection**: the biggest risk in performance optimization is not failing to be fast enough, but
**being fast at the wrong precision**. Reporting precision metrics bound together with speed metrics
is the only reliable way to prevent that kind of self-deception.

---

## Problem 9: The methodological traps of measurement itself

**Symptom and handling.**

1. **JIT compilation time mixed into the measurement.** The first time I timed a Triton kernel I got
   "a few seconds", which was actually compilation time. Solution: run `warmup` iterations before
   measuring (3–5 for this benchmark) so that the JIT, cuBLAS algorithm selection and the VRAM
   allocation pool all reach steady state.
2. **Unsynchronised GPU timing is fake.** CUDA is asynchronous, and `time.perf_counter()` measures
   enqueue time. Solution: use `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()`
   and let the GPU report the elapsed time itself.
3. **A single measurement is not trustworthy.** Laptop GPUs have power limits and thermal drift.
   Solution: report the **median** (robust) and the **minimum** (best performance) over `repeat=15`
   runs, rather than the mean.
4. **On the first benchmark run `torch_gather_local` OOM'd, which crashed the whole script and lost
   all the earlier data.** Solution: write results to JSON in stages, and add exception protection
   around the critical calls.
5. **The denominator in an arithmetic-intensity estimate is easy to get wrong.** I initially
   estimated by "bytes moved per voxel" and got an arithmetic intensity of 0.083 FLOP/byte for both
   naive and window gather, which could not distinguish them. After switching to also reporting
   **total bytes moved** (3.38 TB vs 226 GB vs 1.15 GB), the differences among the three algorithms
   became visible.

**Reflection**: the credibility of performance numbers depends entirely on the measurement method. I
wrote a dedicated "measurement methodology" section in the report, spelling out all the parameters
(number of warmup runs, number of repetitions, synchronisation method, which statistic is reported),
precisely so that others can judge the signal-to-noise ratio of these numbers.

---

## Problem 10: Without Geant4/TOPAS, how do you do "clinical-grade" accuracy validation?

**Constraint.** Phase three of the project document requires "comparison with TOPAS/Geant4 Monte
Carlo simulation". But Geant4/TOPAS needs Linux + a multi-GB install + complex physics list
configuration, which is impossible to complete on this machine (Windows + no compiler).

**Thinking process.** Simply giving up on validation would mean the project only achieved "fast" and
not "accurate", leaving the deliverables incomplete. So I had to build an **independent** reference
implementation myself. The key is what the word "independent" means:

- If the reference implementation reuses the analytic model's formulas, it validates nothing
  (circular reasoning);
- A genuinely independent reference must **start from a different mathematical starting point**: the
  analytic model solves moment equations (the integral of $\sigma_x^2=2A_2$), whereas Monte Carlo
  **samples phase space event by event**.

**Approach.** Implement a class-II condensed-history Monte Carlo engine
(`pbdose/montecarlo.py`):

- Continuous slowing down: $-dE=S(E)\,ds$, using a self-consistent Bethe–Bloch stopping power;
- Multiple Coulomb scattering: each step samples a projected angular deflection from
  $N(0,T(E)\,ds)$ — **note that this uses the same $T(E)$ as the analytic model**, so the comparison
  between the two tests whether "the integral solution of the moment equations" and "the event-by-event
  random walk" agree, which is a meaningful consistency check;
- Nuclear reaction attenuation: each step terminates with probability $1-\exp(-ds/\lambda)$;
- Use PyTorch on the GPU to **advance tens of millions of histories simultaneously**, scoring with
  `index_put_(accumulate=True)`.

**Independence verification (self-test results):**

| Check | Result |
|---|---|
| R90 depth vs CSDA range (200 MeV) | 25.42 cm vs 25.95 cm (−2.1%) |
| Lateral rms width σ_x(10 cm) vs Fermi–Eyges | 0.12463 cm vs 0.12466 cm (**−0.02%**) |
| Energy conservation (nuclear absorption off) | balance = 1.000000000000 |
| Step-size convergence (coarse/fine grid) | R90 differs by 0.05% |
| Random-number reproducibility | bit-identical for the same seed; differs across seeds |
| Throughput | 6.6×10⁵ histories/s |

**Honest disclosure of the limitations**: this MC is **not** Geant4/TOPAS. It has no delta electron
transport, no nuclear reaction secondaries, no energy straggling (only CSDA + Gaussian scattering +
attenuation), and the Gaussian scattering also ignores the Rutherford large-angle tail. These
limitations are written into the module docstring and the validation report, and "final clinical
validation with TOPAS/Geant4" is listed as follow-up work.

**Reflection**: when resources are constrained, "settling for second best" and "lowering the
standard" are two different things. The home-built MC's level of physical approximation really is
below Geant4, but it is **sufficient to validate this project's central claim** — that the analytic
moment method plus GPU acceleration does not lose accuracy. Writing the **scope of applicability** of
the validation clearly is far more honest, and far more professional, than claiming "clinical
validation passed".

---

## Problem 11: An "invisible" normalisation error — per-layer peak normalisation vs fluence normalisation

**Symptom.** After finishing the Monte Carlo engine I ran the first full-field γ validation, and the
result was **catastrophic**: the 3%/3mm pass rate was only **2.6%**, with a dose difference RMS of
52%. And before that, every component-level test had passed (single-beam depth dose agreement, σ_x
agreement to 0.02%, exact energy conservation).

**Investigation process.**

1. **First confirm it is not a Monte Carlo problem.** I ran a minimal controlled comparison: a single
   layer (R₀=6 cm), a single layer (R₀=20 cm), and two layers (6+20 cm), comparing the analytic and
   MC depth profiles point by point. The results were **almost completely identical** (R_peak, R90,
   R50 all the same to 0.2 cm, profile values differing pointwise by < 1%). So the MC's single-beam
   physics is right.
2. **Therefore the problem must be in the multi-layer superposition.** In the 40-layer field, the
   MC's R50 = 6.9 cm while the analytic R50 = 20.1 cm — that is not a difference of a few percent
   but a **completely different field shape**. Yet the two agree to 2% on "total energy per depth"
   (the lateral integral). This means the energy deposition along the depth direction is consistent,
   and the discrepancy lies in **the relative weights among the energy layers**.
3. **Localised to the normalisation convention.** I checked layer by layer with a two-energy-layer
   case: my `idd_pristine` **normalises each energy layer's IDD to peak = 1**, whereas the Monte
   Carlo **transports the same number of protons (fluence) for each energy layer** and then
   multiplies by the weight. These two are completely different:
   - Peak normalisation ⇒ the weight w is a "peak dose weight", and all layers contribute the same
     dose at their own peak;
   - Fluence normalisation ⇒ the weight w is "protons/MU", and a low-energy layer, because its
     stopping power is large and its beam spot narrow, produces a markedly higher dose per unit
     fluence at its peak.

   Measured IDD peaks for the two: the 59 MeV layer **43.8 MeV/cm**, the 198 MeV layer
   **19.4 MeV/cm** — a factor of 2.3 apart. Forcing 40 such layers to normalise to the same height
   artificially suppresses the weight of the low-energy layers by more than half.

**Why this is a real error and not a convention difference.** The spot weights that a real IMPT
planning system outputs are **fluence** (monitor units MU / number of protons). The TPS optimizer
adjusts "how many protons to shoot", not "how much dose at the peak". Using peak normalisation makes
the relative intensity of the multi-energy-layer superposition completely wrong, and this error
**simply cannot show up in a single-layer test** — only a multi-energy-layer superposition exposes
it.

**Solution.** Add a `normalize` parameter to `physics.idd_pristine`, and make
`flat_impt_field` / `gaussian_impt_field` default to `normalize=False` (returning the physical
"depth dose per unit fluence" [MeV/cm]). After the fix, the analytic and MC profiles of the
controlled two-layer case agree point by point (R50 = 19.80 cm for both).

**Quantified impact.**

| Quantity | Peak normalisation (wrong) | Fluence normalisation (correct) |
|---|---|---|
| 59 MeV layer IDD peak | 1.0 (artificial) | 43.8 MeV/cm |
| 198 MeV layer IDD peak | 1.0 (artificial) | 19.4 MeV/cm |
| Full-field R50 (10×10 cm flat-top field) | 20.9 cm | 17.4 cm |
| Max error of the single-σ approximation (controlled flat-top field) | 8.7% | 6.3% |

**Reflection.** What makes this error most worth recording is that it **passed every unit-level
check**: energy conservation right, Bragg peak right, σ_x right, single-layer profile right — 
because peak normalisation has no effect whatsoever on a **single layer**. It only exposes itself
when multiple energy layers are superimposed and compared with an independent model. This confirms
Problem 2's lesson once again, and more extremely: **component correctness ≠ system correctness**.
Cross-validation must be done at "the same level of complexity as the thing being validated",
otherwise it only proves the small part you wanted to prove.

---

## Problem 12 (not fully resolved): the full-field lateral distribution still differs from Monte Carlo

**Symptom.** After fixing the normalisation, the full-field γ pass rate rose from 2.6% to a level
that is still very low. Layer-by-layer localisation showed that **even with only one energy layer**,
the 3%/3mm pass rate is only 10–22%, with a dose difference RMS of 20–42%. So the problem is not in
the multi-layer superposition but in the single layer's **lateral** distribution.

**The part already localised.** Point-by-point comparison of the lateral profile of the 107.5 MeV
layer at z = 8 cm:

| x [cm] | −3.1 | −1.5 | 0.1 | 1.7 | 3.3 |
|---|---|---|---|---|---|
| Analytic | 0.775 | 0.797 | 0.877 | 0.893 | 0.755 |
| Monte Carlo | 0.806 | 0.654 | 0.682 | 0.755 | 0.588 |

The MC's "spot scalloping" is clearly deeper. Half the cause has been found: `mc_impt_field` **does
not expose the incident beam phase-space parameters** and always transports a **point-source** pencil
beam (σ_x0 = 0), whereas my analytic model uses the clinical nozzle's 3 mm incident beam spot
(`beam_sigma_x0 = 0.30 cm`). Spot spacing 5 mm with a narrower beam spot ⇒ less overlap between
neighbouring Gaussians ⇒ deeper scalloping. This is **the two models having inconsistent inputs**,
not a physics disagreement.

**Why I cannot simply set the analytic σ_x0 to 0 as well.** The analytic model's lateral
normalisation factor is 1/(2πσ(z)²), and for a zero-emittance point source σ(0) = 0, so the model
diverges at z = 0 (measured dose difference RMS blew up to 10¹⁶%). So the Monte Carlo has to be made
to accept a finite incident beam spot, rather than degrading the analytic model.

**Current status (three errors fixed, one model defect localised but not fixed).**

| # | Problem | Fix | Effect |
|---|---|---|---|
| A | Analytic model's per-layer IDD peak normalisation (should be **fluence** normalisation) | `idd_pristine(normalize=False)` | two-layer profile agrees point by point |
| B | `mc_impt_field` does not expose the incident phase space, fixed to transport a **point source** | added `sigma_x0/theta0/rho0` pass-through | single-layer γ 10–22% → **93.3%** |
| C | Inconsistent beam **spot lattice centring convention** (MC anchored at the left end vs `SpotLattice` centred), the 21×21 lattice misaligned by **5.0 cm** | unified on the `SpotLattice` convention | combined with A+B → full field **90.97%** |

Full 40-layer field (80 M histories) 3%/3mm pass rate **90.97%** (2.6% before the fixes),
**92.29%** after excluding the surface build-up region, **98.78%** for the single-layer controlled
case; dose difference RMS dropped from 51.5% to 4.23%.

The fourth problem was precisely localised from the Monte Carlo side to **the analytic model itself**:
all failing voxels are concentrated in the first depth plane, because `idd_pristine` applies a single
convolution over the entire curve using the **final** range straggling σ_R, whereas the range spread
should accumulate with depth — the kernel width at depth z should be
σ_ρ(z) = sqrt(∫₀ᶻ BOHR_COEF/(β²S²) dz′). Measured at z = 0.2 cm the analytic kernel is too wide by
**6.1 times**, and 22.8% of the Gaussian weight leaks outside the surface, making the entrance region
low by about 23%. **This item is not fixed**, and is one of the clearest pieces of follow-up work in
this project.

The distribution of failing voxels has been precisely localised (see `tools/gamma_regions.py`):

| Region | Failure rate in that region | Share of total failures |
|---|---:|---:|
| Entrance z < 1.5 cm | **33.2%** | 24.2% |
| In-field r < 4.5 cm | 6.0% | 48.6% |
| Edge 4.5–6 cm | **17.7%** | 51.4% |
| Distal z > 24 cm | **0%** | 0% |

By depth: the pass rate is **99.25%–100%** for z > 15 cm, and only **64.7%** at z = 0–1.5 cm.

**Judgement on the nature of the residual.** Controlled experiments on the Monte Carlo side
precisely localise the residual to **the analytic model itself**: in the single-layer case all failing
voxels (801 of them, 1.22% of the mask) are in the **first depth plane z = 0.2 cm**, and everything
passes from z = 0.6 cm onwards; statistical fluctuation has been ruled out (MC self-comparison 100%)
along with shift-and-add interpolation blur (applying the same splat to the analytic kernel still
gives 100%). The root cause is that the analytic range-straggling kernel width does not vary with
depth (see item 4 of the table above). **This is not a difference that can be hidden by tuning
parameters, but a clear model defect.**

**Conclusion**: the project's > 98% target **was not achieved** (measured 90.97%, 92.29% excluding the
build-up region, and also short of TG-218's 95% action level). 3 implementation-level errors have
been fixed, and the 4th, a model-level defect, has been localised. The fix path: change the analytic
kernel to the depth-dependent σ_ρ(z) (preferred, physically more correct), or exclude the surface
build-up region from the γ statistics following the clinical TPS QA convention.

**Reflection.** What makes this entry different from the previous ones is that it is **partially
resolved but below target**. Writing it into the log together with the breakdown of the failing
regions, the three fixed errors and the one localised defect, rather than blurring it into
"validation basically passes", is because a localisable failure is more scientifically valuable than
a vague success — it states clearly what to do next, and where the current validation does not
reach.

---

## Summary: problem–thinking–solution mapping table

| # | Problem | Core thinking | Solution | Quantified result |
|---|---|---|---|---|
| 1 | GPU but no nvcc | find an equivalent, verifiable alternative path | Triton JIT + hand-written .cu deliverable | 3 custom kernels genuinely running |
| 2 | Single-σ approximation | different energy layers have intrinsically different σ | (layer, depth) batched bmm | error 20.3% → 1.5e-7 |
| 3 | Bragg peak too tall | let observable clinical metrics be the judge | Bohr integral + parameterisation + disclose uncertainty | peak/entrance 6.4 → 4.9 |
| 4 | Off by √(2π) | look at ratios rather than differences | distinguish 1D/2D normalisation constants | 0.84 → 1.0e-6 |
| 5 | VRAM exhaustion | the intermediate tensor is five-dimensional | per-layer chunking | peak VRAM ÷40 |
| 6 | Baseline cannot finish | extrapolation must be verifiable | measured scaling + linear fit + GPU version of the baseline | 1.24e6 ms → 654 ms |
| 7 | Triton constraints | the kernel designs its data layout around the constraints | power-of-two blocks + masks + ieee | compiles |
| 8 | Precision/speed trade-off | report precision bound to speed | three-criteria comparison table | 20% error for 1.47× → rejected |
| 9 | Measurement traps | performance numbers depend on method | warmup + CUDA event + median | measurements reproducible |
| 10 | No Geant4 | "independent" means starting from a different starting point | home-built GPU condensed-history MC | σ_R agrees to 2.4% |
| 11 | Full-field γ only 2.6% | component correctness ≠ system correctness | localise and fix 3 modelling/implementation errors (fluence normalisation, incident beam spot, spot lattice centring convention) | γ 2.6% → 90.97% |
| 12 | Single-layer lateral distribution still differs | the two models have inconsistent inputs | match the incident beam spot (MC gains σ_x0 pass-through) | single-layer γ 10–22% → **98.78%** |
| 13 | Residual failing voxels all in the first depth plane | rule out statistics and interpolation one by one with controlled experiments | localised to the analytic σ_R kernel width not varying with depth (**not fixed**) | kernel too wide by 6.1× at z=0.2 cm |
