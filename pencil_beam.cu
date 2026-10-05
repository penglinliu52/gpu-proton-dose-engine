// =============================================================================
//  pencil_beam.cu -- GPU-accelerated proton pencil-beam (PB) dose reconstruction / IMPT research
// -----------------------------------------------------------------------------
//  Dose model
//      D(x,y,z) = SUM_m N_m * IDD_m(z) * 1/(2*pi*sigma_m(z)^2)
//                 * exp( -[(x-x_m)^2 + (y-y_m)^2] / (2*sigma_m(z)^2) )
//      N_m = w[] weight of the m-th spot; IDD_m(z) = integral depth dose of the energy
//      layer that spot belongs to; sigma_m(z) = lateral spread of that layer
//      (standard deviation, cm); (x_m,y_m) = lateral centre of the spot.
//
//  Four __global__ kernels (matching the four stages of Section 4 of the proposal)
//      1) naive_gather_kernel        textbook gather baseline (expect it to be very slow, see its comment)
//      2) shared_lut_gather_kernel   LUT staged into shared memory + register blocking along z
//      3) separable_conv_kernel      separable Gaussian convolution (the optimised algorithm)
//         build_spot_amplitude_kernel scatter-adds the spot weights into an amplitude tensor
//      4) batched_gather_kernel      beam batching, caps the VRAM working set
//
//  Build
//      nvcc -arch=sm_120 -O3 -c pencil_beam.cu -o pencil_beam.o             # library
//      nvcc -arch=sm_120 -O3 -DPB_STANDALONE_MAIN pencil_beam.cu -o pb_demo # self-test
//      nvcc -arch=sm_120 -O3 -Xptxas -v -c pencil_beam.cu -o /dev/null      # register statistics
//
//  Conventions
//      * Dose is float; the host-side accumulation constants and the CPU reference use double.
//      * Output voxel layout d_out[((k*n_vox_y)+j)*n_vox_x + i] (z outer / y middle / x inner, x contiguous -> coalesced writes).
//      * Weight layout follows the proposal exactly: w[j*n_spot_x*n_layers + l*n_spot_x + i], i.e. [y][layer][x];
//        to switch to the more common [layer][y][x], PB_W_IDX is the only place to change.
//      * LUT layout idd[l*nz+k], sigma[l*nz+k]. No external dependencies, no Thrust.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

// ----------------------------- Compile-time tunables ----------------------------
#ifndef PB_CONV_TX
#define PB_CONV_TX 32 // separable convolution tile width (threads.x)
#endif
#ifndef PB_CONV_TY
#define PB_CONV_TY 32 // separable convolution tile height (threads.y)
#endif
#ifndef PB_CONV_KMAX
#define PB_CONV_KMAX 24 // max Gaussian kernel half-width, in voxels
#endif
#ifndef PB_Z_PER_THREAD
#define PB_Z_PER_THREAD 4 // number of z slices per thread in the gather-family kernels (register blocking)
#endif
#ifndef PB_BATCH_SPOTS
#define PB_BATCH_SPOTS 1024 // max number of spots staged per pass in the batched kernel
#endif
#ifndef PB_GAUSS_TRUNC
#define PB_GAUSS_TRUNC 3.0f // Gaussian truncation radius, in units of sigma
#endif
#ifndef PB_SIGMA_FLOOR
#define PB_SIGMA_FLOOR 1.0e-3f // lower bound on sigma (cm), guards against division by zero / NaN
#endif

// Exponential: high-precision expf by default; defining PB_FAST_MATH switches to the SFU version
// __expf (~2-4x faster, relative error ~1e-6, perfectly acceptable for dosimetry; -use_fast_math also maps expf to __expf).
#ifdef PB_FAST_MATH
#define PB_EXPF(x) __expf(x)
#else
#define PB_EXPF(x) expf(x)
#endif

#define PB_INV_2PI 0.15915494309189535f // 1/(2*pi)
#define PB_PI_D 3.14159265358979323846 // host-side double constant

static_assert(PB_CONV_TX * PB_CONV_TY <= 1024, "tile thread count exceeds the 1024-per-block limit");
static_assert(PB_CONV_TX >= 2 && PB_CONV_TY >= 2, "tile must be at least 2x2");
static_assert(PB_CONV_KMAX >= 1 && PB_CONV_KMAX <= 64, "KMAX must lie in the sensible range 1..64");
static_assert(PB_Z_PER_THREAD >= 1 && PB_Z_PER_THREAD <= 16, "Z_PER_THREAD must lie in the sensible range 1..16");

// =============================================================================
// 1. Problem descriptor — plain POD, can be constructed directly from ctypes / a torch extension
// =============================================================================
typedef struct PbProblem
{
    int n_spot_x, n_spot_y, n_layers; // spot lattice and number of energy layers
    float dx_spot, dy_spot; // spot spacing (cm)
    float x0_spot, y0_spot; // centre of spot 0 (cm)
    const float2* spot_xy; // optional explicit coordinate table (index m=j*n_spot_x+i),
                                                // NULL => computed analytically from the regular lattice (zero VRAM traffic)
    int nz; // number of depth LUT samples
    float z0, dz; // LUT depth origin / step (cm)
    const float* idd; // [n_layers*nz] integral depth dose
    const float* sigma; // [n_layers*nz] lateral sigma (cm)
    const float* w; // [n_spot_y][n_layers][n_spot_x], see PB_W_IDX
    int n_vox_x, n_vox_y, n_vox_z; // dose voxel grid
    float dvx, dvy, dvz; // voxel size (cm)
    float vox_x0, vox_y0, vox_z0; // centre coordinates of voxel (0,0,0) (cm)
    int deposit_bilinear; // 0=nearest-grid deposition, 1=bilinear deposition
} PbProblem;

// =============================================================================
// 2. Index helpers & device utilities

// All weight accesses go through PB_W_IDX — the layout assumption lives in this one place.
// =============================================================================
__host__ __device__ __forceinline__ size_t PB_W_IDX(const PbProblem& p, int l, int i, int j)
{
    return (size_t)j * p.n_spot_x * p.n_layers + (size_t)l * p.n_spot_x + (size_t)i;
}
// Dose voxel linear index = amplitude tensor index: z outer / y middle / x inner
__host__ __device__ __forceinline__ size_t PB_OU_IDX(int nx, int ny, int i, int j, int k)
{
    return ((size_t)k * ny + (size_t)j) * nx + (size_t)i;
}

// round-to-nearest (identical behaviour on host and device)
__host__ __device__ __forceinline__ int pb_iround(float v)
{
#if defined(__CUDA_ARCH__)
    return __float2int_rn(v);
#else
    return (int)((v >= 0.0f) ? (v + 0.5f) : (v - 0.5f));
#endif
}

// Linear interpolation on a uniform grid; row already points at the start of that layer's LUT row.
__host__ __device__ __forceinline__ float pb_lerp_z(const float* __restrict__ row,
                                                    int nz, float z0, float dz, float z)
{
    const float f = (z - z0) / dz;
    if (f <= 0.0f) return row[0];
    if (f >= (float)(nz - 1)) return row[nz - 1];
    const int k0 = (int)f; // f>0 and in range have been established => truncation is floor
    const float t = f - (float)k0;
    const float a = row[k0];
    return a + t * (row[k0 + 1] - a);
}

// Lateral spot coordinates: with an explicit coordinate table we read through the __ldg read-only cache,
// otherwise they are computed analytically from the lattice (warp-uniform branch, predicated load, no null deref).
__host__ __device__ __forceinline__ float2 pb_spot_position(const PbProblem& p, int m, int i, int j)
{
#if defined(__CUDA_ARCH__)
    if (p.spot_xy != nullptr) return __ldg(&p.spot_xy[m]);
#else
    if (p.spot_xy != nullptr) return p.spot_xy[m];
#endif
    return make_float2(p.x0_spot + (float)i * p.dx_spot,
                       p.y0_spot + (float)j * p.dy_spot);
}

// atomicAdd(double*) requires CC >= 6.0 (Pascal). Note that sm_120 gives __CUDA_ARCH__ == 1200, so the
// test must be ">="; an equality test such as "__CUDA_ARCH__ == 600" would take the wrong branch on Blackwell.
__device__ __forceinline__ double pb_atomic_add_double(double* address, double val)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(address, val); // native path
#else
    unsigned long long* a = reinterpret_cast<unsigned long long*>(address); // CAS fallback
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// =============================================================================
// 3. Kernel 1: naive_gather_kernel — the textbook baseline of Section 4 of the proposal
// -----------------------------------------------------------------------------
// grid/block: block=(16,16), grid=(ceil(nx/16), ceil(ny/16), nz); one thread = one voxel.
// Memory access: each thread walks the entire spot table along m => O(N_vox*M) logical loads.
// The weight address is uniform across the whole warp => L1 broadcast hits, so the physical
// DRAM traffic is far below the logical traffic, but the cost of "one LDG instruction per
// pair" is still paid. IDD/sigma depend only on (layer,z), not on the spot => hoisted out of the m loop
// (LICM); with spot_xy==NULL the coordinates are computed analytically, leaving just one 4B weight load per pair.
// Bottleneck: the arithmetic intensity is extremely low. Assuming nothing is cached at all, each
// (voxel,spot) pair needs 4B(w)+8B(coords)+2*4B(IDD/sigma interpolation) ~= 20 B and about
// 12 FLOP of work (including 2 expf) => ~=0.6 FLOP/B. The machine balance point of this box
// (46 SM, ~18.8 TFLOP/s FP32, ~448 GB/s GDDR7) is about 42 FLOP/B, a factor of ~70 away
// => a thoroughly memory-bandwidth-bound algorithm. More precisely: the spot table of this demo is only
// M*12B ~= 768 KB and fits entirely in L2, so the actual bottleneck degenerates into **instruction issue**:
// about 25~30 instructions/pair * 1.34e11 pairs ~= 4e12 instructions / (46SM*4sched*1.6GHz ~= 2.9e11 inst/s)
// ~= of order 13 s. Only when M*12B vastly exceeds L2 (M >> 2.5e6) does it truly become
// DRAM-bandwidth-bound. Neither regime is usable in production.
// =============================================================================
__global__ void naive_gather_kernel(const PbProblem p, float* __restrict__ d_out)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z;
    if (ix >= p.n_vox_x || iy >= p.n_vox_y || iz >= p.n_vox_z) return; // no barrier, safe to exit early

    const float x = p.vox_x0 + (float)ix * p.dvx;
    const float y = p.vox_y0 + (float)iy * p.dvy;
    const float z = p.vox_z0 + (float)iz * p.dvz;
    float acc = 0.0f;

    for (int l = 0; l < p.n_layers; ++l)
    {
        // Computed once per layer and per voxel: IDD, sigma, 1/(2 sigma^2), amplitude factor
        const float idd_z = pb_lerp_z(p.idd + (size_t)l * p.nz, p.nz, p.z0, p.dz, z);
        const float sig_z = fmaxf(pb_lerp_z(p.sigma + (size_t)l * p.nz, p.nz, p.z0, p.dz, z),
                                   PB_SIGMA_FLOOR);
        const float inv2s2 = 0.5f / (sig_z * sig_z); // 1/(2 sigma^2)
        const float amp = idd_z * inv2s2 * PB_INV_2PI; // IDD/(2 pi sigma^2)

        for (int j = 0; j < p.n_spot_y; ++j)
        {
            const float dy = y - (p.y0_spot + (float)j * p.dy_spot);
            const float dy2 = dy * dy;
            #pragma unroll 4 // register blocking: 4 independent expf chains
            for (int i = 0; i < p.n_spot_x; ++i)
            {
                const float w = __ldg(&p.w[PB_W_IDX(p, l, i, j)]); // warp-uniform address => broadcast
                if (w == 0.0f) continue; // zero-weight spots are common in IMPT
                const float2 pos = pb_spot_position(p, j * p.n_spot_x + i, i, j);
                const float dx = x - pos.x;
                acc += w * amp * PB_EXPF(-(dx * dx + dy2) * inv2s2);
            }
        }
    }
    d_out[PB_OU_IDX(p.n_vox_x, p.n_vox_y, ix, iy, iz)] = acc; // every voxel is written exactly once
}

// =============================================================================
// 4. Kernel 2: shared_lut_gather_kernel — LUT caching + z register blocking
// -----------------------------------------------------------------------------
// The gather structure is the same as Kernel 1, except that (a) the IDD/sigma row of each layer
// is cooperatively staged into shared memory, so every subsequent z interpolation read hits
// shared (~30 cycles, and no L1/L2 bandwidth consumed); (b) each thread owns PB_Z_PER_THREAD z
// slices (register blocking), so one weight/coordinate load feeds 4 voxels; (c) spot coordinates
// and weights go through __ldg (read-only data cache).
// grid/block: block=(16,16,1), grid=(ceil(nx/16), ceil(ny/16), ceil(nz/ZPT));
// thread (tx,ty) owns z ∈ [blockIdx.z*ZPT, +ZPT) of voxel (x,y).
// Shared memory: dynamic extern __shared__, 2*nz*4 B (only 1 KB at nz=128). If it exceeds the default 48 KB, the
// host requests the opt-in size via cudaFuncSetAttribute; if it still exceeds the device limit we
// return PB_ERR_UNSUPPORTED (see pb_dose_forward_shared_lut).
// ⚠ Synchronisation discipline: this kernel contains __syncthreads(), so it must **never return early**.
// Out-of-range threads mask their memory accesses with the active predicate, but they must still
// reach every barrier, otherwise the behaviour is undefined (on some architectures it simply hangs).
// Both barriers sit outside if (active).
// Bottleneck: still O(N_vox*M) pairs, but the instruction count per pair drops from ~28 to ~14 and L1 bandwidth
// is no longer consumed. Expect a 3~8x speedup over Kernel 1 in practice; the pair count is
// unchanged, so it still does not scale.
// =============================================================================
__global__ void shared_lut_gather_kernel(const PbProblem p, float* __restrict__ d_out)
{
    extern __shared__ float s_lut[]; // [0,nz)=IDD, [nz,2nz)=sigma
    float* __restrict__ s_idd = s_lut;
    float* __restrict__ s_sig = s_lut + p.nz;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int nthr = blockDim.x * blockDim.y, tid = ty * blockDim.x + tx;
    const int ix = blockIdx.x * blockDim.x + tx;
    const int iy = blockIdx.y * blockDim.y + ty;
    const int iz0 = blockIdx.z * PB_Z_PER_THREAD;
    const bool active = (ix < p.n_vox_x) && (iy < p.n_vox_y);
    const float x = p.vox_x0 + (float)ix * p.dvx;
    const float y = p.vox_y0 + (float)iy * p.dvy;

    float acc[PB_Z_PER_THREAD], zz[PB_Z_PER_THREAD];
    int zk[PB_Z_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < PB_Z_PER_THREAD; ++t) {
        acc[t] = 0.0f; zk[t] = iz0 + t; zz[t] = p.vox_z0 + (float)zk[t] * p.dvz;
    }

    for (int l = 0; l < p.n_layers; ++l)
    {
        // (a) cooperatively stage this layer's LUT — all threads take part, including out-of-range ones
        for (int k = tid; k < p.nz; k += nthr) {
            s_idd[k] = __ldg(&p.idd [(size_t)l * p.nz + k]);
            s_sig[k] = __ldg(&p.sigma[(size_t)l * p.nz + k]);
        }
        __syncthreads(); // ← must sit outside if (active)
        if (active)
        {
            // Per-layer coefficients of this thread's ZPT z slices, interpolated from shared
            float amp[PB_Z_PER_THREAD], inv2s2[PB_Z_PER_THREAD];
            #pragma unroll
            for (int t = 0; t < PB_Z_PER_THREAD; ++t) {
                if (zk[t] < p.n_vox_z) {
                    const float idd_z = pb_lerp_z(s_idd, p.nz, p.z0, p.dz, zz[t]);
                    const float sig_z = fmaxf(pb_lerp_z(s_sig, p.nz, p.z0, p.dz, zz[t]),
                                              PB_SIGMA_FLOOR);
                    inv2s2[t] = 0.5f / (sig_z * sig_z);
                    amp[t] = idd_z * inv2s2[t] * PB_INV_2PI;
                } else { inv2s2[t] = 0.0f; amp[t] = 0.0f; }
            }
            // (b) gather: one load feeds ZPT voxels
            for (int j = 0; j < p.n_spot_y; ++j) {
                const float dy = y - (p.y0_spot + (float)j * p.dy_spot);
                const float dy2 = dy * dy;
                #pragma unroll 4
                for (int i = 0; i < p.n_spot_x; ++i) {
                    const float w = __ldg(&p.w[PB_W_IDX(p, l, i, j)]);
                    if (w == 0.0f) continue;
                    const float2 pos = pb_spot_position(p, j * p.n_spot_x + i, i, j);
                    const float dx = x - pos.x;
                    const float r2 = dx * dx + dy2;
                    #pragma unroll
                    for (int t = 0; t < PB_Z_PER_THREAD; ++t)
                        acc[t] += w * amp[t] * PB_EXPF(-r2 * inv2s2[t]);
                }
            }
        }
        __syncthreads(); // ← also outside the if: protects shared memory from the next layer
    }

    if (active) { // Kernel 2 recomputes everything => assign, not add
        #pragma unroll
        for (int t = 0; t < PB_Z_PER_THREAD; ++t)
            if (zk[t] < p.n_vox_z)
                d_out[PB_OU_IDX(p.n_vox_x, p.n_vox_y, ix, iy, zk[t])] = acc[t];
    }
}
// =============================================================================
// 5. Kernel 3a: build_spot_amplitude_kernel — amplitude tensor construction (scatter-add)
// -----------------------------------------------------------------------------
// Goal: S_l(x,y,k) = SUM_{m in layer l} w_m * IDD_l(z_k), i.e. the rank-1 outer product of the
// lateral fluence map and the IDD. Note that sigma differs from layer to layer, so we **must
// build S per layer, convolve per layer and only then sum over layers**; hoisting Σ_l in front
// of the convolution is wrong (see the top of separable_conv_kernel).
// grid/block: block=(32,8)=256, one thread per (i,j) of the spot lattice,
// grid=(ceil(sx/32),ceil(sy/8)); each thread loops over z and writes the nz amplitude values of that (i,j) column.
// Memory access: for a fixed k, neighbouring threads in a warp write neighbouring x => coalesced
// writes. The IDD read is warp-uniform => broadcast.
// Two paths: (1) identity — the spot lattice is aligned 1:1 with the voxel grid, so there
// are no write conflicts and we assign directly with zero atomics; (2) the general path — 
// bilinear deposition, where several spots can land in the same voxel => atomicAdd is
// mandatory. A double* amplitude tensor (for high-precision accumulation studies) goes
// through atomicAdd(double*), guarded by __CUDA_ARCH__>=600.
// Bottleneck: atomic contention (general path) and write bandwidth. The work is O(sx*sy*nz),
// orders of magnitude smaller than the gather, hence negligible next to the convolution;
// the real cost sits in separable_conv_kernel.
// =============================================================================
__global__ void build_spot_amplitude_kernel(const PbProblem p,
                                            int layer, // >=0 single layer; <0 = sum over layers (literal implementation of the proposal's formula)
                                            float* __restrict__ s_lat, // [nz][ny][nx]
                                            double* __restrict__ s_lat_dp) // optional, NULL = use the float tensor
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.n_spot_x || j >= p.n_spot_y) return; // no barrier, safe to exit early

    const float sx = p.x0_spot + (float)i * p.dx_spot;
    const float sy = p.y0_spot + (float)j * p.dy_spot;

    // Weight of this spot (when layer<0, sum over layers first)
    float wsum = 0.0f;
    if (layer >= 0) {
        wsum = __ldg(&p.w[PB_W_IDX(p, layer, i, j)]);
    } else {
        for (int l = 0; l < p.n_layers; ++l) wsum += __ldg(&p.w[PB_W_IDX(p, l, i, j)]);
    }
    if (wsum == 0.0f) return;

    const int nx = p.n_vox_x, ny = p.n_vox_y;
    const float fx = (sx - p.vox_x0) / p.dvx; // fractional coordinate on the voxel grid
    const float fy = (sy - p.vox_y0) / p.dvy;
    const int offx = pb_iround((p.x0_spot - p.vox_x0) / p.dvx);
    const int offy = pb_iround((p.y0_spot - p.vox_y0) / p.dvy);
    const bool identity = (nx == p.n_spot_x) && (ny == p.n_spot_y)
        && (fabsf(p.dvx - p.dx_spot) <= 1.0e-5f * p.dvx)
        && (fabsf(p.dvy - p.dy_spot) <= 1.0e-5f * p.dvy)
        && (fabsf((p.x0_spot - p.vox_x0) - (float)offx * p.dvx) <= 1.0e-4f)
        && (fabsf((p.y0_spot - p.vox_y0) - (float)offy * p.dvy) <= 1.0e-4f);

    if (identity)
    {
        // ---------- 1:1 mapping: direct write, zero atomic ----------
        const int vx = i + offx, vy = j + offy;
        if (vx < 0 || vx >= nx || vy < 0 || vy >= ny) return;
        for (int k = 0; k < p.nz; ++k)
        {
            float a = 0.0f;
            if (layer >= 0) a = __ldg(&p.idd[(size_t)layer * p.nz + k]);
            else for (int l = 0; l < p.n_layers; ++l) a += __ldg(&p.idd[(size_t)l * p.nz + k]);
            const float v = wsum * a;
            const size_t idx = PB_OU_IDX(nx, ny, vx, vy, k);
            if (s_lat_dp != nullptr) s_lat_dp[idx] = (double)v;
            else s_lat[idx] = v;
        }
        return;
    }

    // ---------- general path: bilinear deposition + atomicAdd ----------
    int ix0, iy0; float tx, ty;
    if (p.deposit_bilinear) {
        ix0 = (int)floorf(fx); iy0 = (int)floorf(fy);
        tx = fx - (float)ix0; ty = fy - (float)iy0;
    } else {
        ix0 = pb_iround(fx); iy0 = pb_iround(fy);
        tx = 0.0f; ty = 0.0f;
    }
    const float wx[2] = { 1.0f - tx, tx };
    const float wy[2] = { 1.0f - ty, ty };

    for (int k = 0; k < p.nz; ++k)
    {
        float a = 0.0f;
        if (layer >= 0) a = __ldg(&p.idd[(size_t)layer * p.nz + k]);
        else for (int l = 0; l < p.n_layers; ++l) a += __ldg(&p.idd[(size_t)l * p.nz + k]);
        const float base = wsum * a;
        if (base == 0.0f) continue;

        #pragma unroll
        for (int q = 0; q < 2; ++q) { // 4 corners (bilinear stencil)
            const int yy = iy0 + q;
            if (wy[q] == 0.0f || yy < 0 || yy >= ny) continue;
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int xx = ix0 + r;
                if (wx[r] == 0.0f || xx < 0 || xx >= nx) continue;
                const float v = base * wx[r] * wy[q];
                const size_t idx = PB_OU_IDX(nx, ny, xx, yy, k);
                if (s_lat_dp != nullptr) pb_atomic_add_double(&s_lat_dp[idx], (double)v);
                else atomicAdd(&s_lat[idx], v);
            }
        }
    }
}

// =============================================================================
// 6. Kernel 3b: separable_conv_kernel — separable Gaussian convolution (horizontal + vertical pass)
// -----------------------------------------------------------------------------
// The 2D Gaussian is separable: G2D(dx,dy) = kx(dx)*ky(dy), kx(t)=exp(-t^2/(2 sigma_x^2))/Nx, so
// D_l(x,y,k) = IDD_l(z_k) * [ (A_l ⊛ kx) ⊛ ky ](x,y).
// ⚠ Why "sum over layers first, then convolve" is wrong: sigma depends on the energy layer (the
// multiple scattering differs with energy), and Σ_l [A_l ⊛ G_{sigma_l}] ≠ [Σ_l A_l] ⊛ G_sigma.
// This kernel therefore handles **one z slice of one layer** per launch; the host calls it from
// inside the layer loop and accumulates into the dose volume with accumulate=1. This is exactly
// why build_spot_amplitude_kernel takes a layer argument.
// grid/block: block=(PB_CONV_TX,PB_CONV_TY)=(32,32)=1024 (must match the compile-time tile; the
// host launches with dim3 block(PB_CONV_TX,PB_CONV_TY)); grid=(ceil(nx/32),
// ceil(ny/32), nz), with blockIdx.z being the z slice index.
// Shared memory (dynamic, sized by pb_separable_smem_bytes()): [0, HY*HX) halo tile (including the K=KMAX border)
// [.., +HY*TX) horizontal convolution result s_h (input of the vertical convolution)
// then the kx table, the ky table and 2 normalisation factors; HY=TY+2K, HX=TX+2K.
// Memory access: in the horizontal pass tx is contiguous within a warp => shared stride=1, no bank
// conflicts (HX=80 is not a multiple of 32, so the staggered row starts actually spread the
// banks out further); the vertical pass reads s_h with the same stride=1; the global dose
// write is contiguous in x => coalesced.
// Why the weight tables live in shared memory instead of __constant__: sigma varies per slice, so
// __constant__ would need one cudaMemcpyToSymbol per slice (host-device synchronisation,
// ~5-20 us each), i.e. 128 slices * 40 layers = 5120 synchronisations with the GPU waiting on
// the host the whole time. __constant__ only suits the "single sigma for the whole volume" case, where
// its broadcast really is faster than shared memory. The 64 KB constant-memory limit is not the binding constraint here.
// Bottleneck: 2*(2h+1) FMAs per output voxel (h=min(K, ceil(3 sigma/dv)), typically h≈19
// ~78 FLOP/voxel/slice) => compute-bound. The main inefficiency is the redundant halo read: a
// block reads (TX+2K)(TY+2K)=80*80 elements just to produce 32*32 outputs, an amplification of
// 6.25x (amplification factor =(TX+2K)(TY+2K)/(TX*TY)). Enlarging the tile, or splitting into
// "two independent kernels + a global intermediate buffer" (no redundant FLOP, at the price
// of 2 extra passes over global memory), would both improve this.
// =============================================================================
__global__ void separable_conv_kernel(const float* __restrict__ s_lat, // [nz][ny][nx] amplitude map
                                      const float* __restrict__ sigma_z, // [nz] per-slice sigma (cm) of this layer, may be NULL
                                      float sigma_const_cm, // constant sigma used when sigma_z==NULL
                                      float* __restrict__ dose, // [nz][ny][nx] dose volume
                                      int nx, int ny, int nz,
                                      float dvx, float dvy,
                                      int accumulate) // 1=accumulate, 0=overwrite
{
    // The tile sizes are compile-time constants and the host must match them exactly; no device
    // assert here (a device assert would require linking libcudadevrt). If blockDim does not
    const int K = PB_CONV_KMAX, HX = PB_CONV_TX + 2 * K, HY = PB_CONV_TY + 2 * K;

    extern __shared__ float smem[];
    float* __restrict__ s_halo = smem; // [HY][HX]
    float* __restrict__ s_h = s_halo + (size_t)HY * HX; // [HY][TX]
    float* __restrict__ s_kx = s_h + (size_t)HY * PB_CONV_TX; // [2K+1]
    float* __restrict__ s_ky = s_kx + (2 * K + 1); // [2K+1]
    float* __restrict__ s_nrm = s_ky + (2 * K + 1); // [2] = {1/Σkx, 1/Σky}

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * PB_CONV_TX + tx;
    const int k = blockIdx.z; // z slice handled by this block
    const int gx0 = blockIdx.x * PB_CONV_TX; // x/y origin of the output tile
    const int gy0 = blockIdx.y * PB_CONV_TY;

    // ---- 1) sigma of this slice and the two 1D Gaussian kernels (in voxels) ----
    const float sig_cm = (sigma_z != nullptr) ? __ldg(&sigma_z[k]) : sigma_const_cm;
    const float sigx = fmaxf(sig_cm, PB_SIGMA_FLOOR) / dvx;
    const float sigy = fmaxf(sig_cm, PB_SIGMA_FLOOR) / dvy;
    int hx = min((int)ceilf(PB_GAUSS_TRUNC * sigx), K); // run-time half-width <= compile-time cap K
    int hy = min((int)ceilf(PB_GAUSS_TRUNC * sigy), K);

    if (tid < 2 * K + 1) {
        const int t = tid - K; // tap offset ∈ [-K,K]
        const int at = (t < 0) ? -t : t; // |t| (avoids relying on abs() overload resolution)
        s_kx[tid] = (at <= hx) ? PB_EXPF(-0.5f * ((float)t / sigx) * ((float)t / sigx)) : 0.0f;
        s_ky[tid] = (at <= hy) ? PB_EXPF(-0.5f * ((float)t / sigy) * ((float)t / sigy)) : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        // Discrete normalisation: force Σkx = Σky = 1 (equivalent to midpoint quadrature of the
        // continuous integral), which conserves mass; more robust than dividing by sqrt(2pi)*sigma,
        float sx = 0.0f, sy = 0.0f;
        for (int t = 0; t < 2 * K + 1; ++t) { sx += s_kx[t]; sy += s_ky[t]; }
        s_nrm[0] = (sx > 1.0e-20f) ? 1.0f / sx : 0.0f;
        s_nrm[1] = (sy > 1.0e-20f) ? 1.0f / sy : 0.0f;
    }
    __syncthreads();

    // ---- 2) cooperatively load the halo tile (out of range -> 0 = zero padding) ----
    for (int r = ty; r < HY; r += PB_CONV_TY) {
        const int gy = gy0 + r - K;
        const bool row_ok = (gy >= 0) && (gy < ny);
        // The row pointer is only formed when the row index is in range — otherwise it would be an
        const float* __restrict__ row =
            row_ok ? (s_lat + ((size_t)k * ny + (size_t)gy) * nx) : s_lat;
        for (int c = tx; c < HX; c += PB_CONV_TX) {
            const int gx = gx0 + c - K;
            s_halo[(size_t)r * HX + c] = (row_ok && gx >= 0 && gx < nx) ? __ldg(&row[gx]) : 0.0f;
        }
    }
    __syncthreads();

    const float inv_x = s_nrm[0], inv_y = s_nrm[1];

    // ---- 3) horizontal pass: halo rows -> s_h, multiplying by 1/Σkx at the same time ----
    for (int r = ty; r < HY; r += PB_CONV_TY) {
        const float* __restrict__ hrow = s_halo + (size_t)r * HX + K; // points at c=0 of this row
        float a = 0.0f;
        for (int t = -hx; t <= hx; ++t) a += s_kx[t + K] * hrow[tx + t];
        s_h[(size_t)r * PB_CONV_TX + tx] = a * inv_x;
    }
    __syncthreads(); // s_h is ready

    // ---- 4) vertical pass: s_h -> global dose volume ----
    const int ox = gx0 + tx, oy = gy0 + ty;
    if ((ox < nx) && (oy < ny)) {
        float a = 0.0f;
        for (int t = -hy; t <= hy; ++t)
            a += s_ky[t + K] * s_h[(size_t)(ty + K + t) * PB_CONV_TX + tx];
        a *= inv_y;
        const size_t oidx = PB_OU_IDX(nx, ny, ox, oy, k);
        dose[oidx] = accumulate ? (dose[oidx] + a) : a;
    }
}

// Dynamic shared memory size of the separable convolution — host and device must call the same
__host__ __device__ __forceinline__ size_t pb_separable_smem_bytes()
{
    const size_t HY = PB_CONV_TY + 2 * PB_CONV_KMAX;
    const size_t HX = PB_CONV_TX + 2 * PB_CONV_KMAX;
    return (HY * HX + HY * PB_CONV_TX + 2 * (2 * PB_CONV_KMAX + 1) + 2) * sizeof(float);
}
// =============================================================================
// 7. Kernel 4: batched_gather_kernel — beam batching, caps the VRAM working set
// -----------------------------------------------------------------------------
// The current small batch of spots (a stretch of the lattice inside one energy layer) is staged
// into shared memory together with that layer's IDD/sigma, after which all threads gather
// repeatedly from shared memory. The number of reads of the spot table from global memory drops
// from O(number of threads) to O(number of blocks), and the VRAM working set is capped at BATCH spots.
// grid/block: block=(16,16,1)=256, each thread owns PB_Z_PER_THREAD z slices of voxel (x,y);
// grid=(ceil(nx/16),ceil(ny/16),ceil(nz/ZPT)); the host calls it repeatedly from a
// 2D (layer × batch) loop and accumulates with +=.
// Shared memory: float2 s_pos[count] | float s_w[count] | float s_idd[nz] | float s_sig[nz],
// bytes = 12*count + 8*nz, requested exactly for the actual count.
// Memory access: the reads of s_pos/s_w are warp-uniform => shared broadcast, zero bank
// conflicts; s_pos[t] spares the inner loop the %/ and the coordinate computation.
// The output uses += read-modify-write (each voxel belongs to exactly one thread
// within a launch => no atomics needed).
// On cudaMemcpyAsync / stream overlap: the spot table of this API already lives in VRAM (PbProblem
// holds device pointers), so there is no H2D copy to overlap; what genuinely needs
// double buffering is out-of-core streaming of the plan: 2 streams + 2 pinned staging
// buffers, cudaMemcpyAsync prefetching batch n+1 while batch n's kernel runs on the
// other stream, with cudaEventRecord / cudaStreamWaitEvent for dependency
// Bottleneck: same order as Kernel 2 (the pair count is unchanged), but the per-pair weight and
// coordinate access turns from "L1 broadcast + instructions" into "shared broadcast",
// and the L2 thrashing of a large plan disappears. The fundamental problem that the
// total number of pairs is unchanged still stands — this belongs to the gather
// family, it is not a separable algorithm.
// =============================================================================
__global__ void batched_gather_kernel(const PbProblem p, int layer,
                                      int cell_begin, int cell_count,
                                      float* __restrict__ d_out)
{
    extern __shared__ float2 s_pos[]; // [cell_count]
    float* __restrict__ s_w = reinterpret_cast<float*>(s_pos + cell_count);
    float* __restrict__ s_idd = s_w + cell_count;
    float* __restrict__ s_sig = s_idd + p.nz;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;
    const int vx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz0 = blockIdx.z * PB_Z_PER_THREAD;
    const bool active = (vx < p.n_vox_x) && (vy < p.n_vox_y);

    // ---- 1) cooperative staging: spot weights/coordinates of this batch + this layer's LUT ----
    for (int t = tid; t < cell_count; t += nthr) {
        const int cell = cell_begin + t; // cell == m == j*n_spot_x + i
        const int i = cell % p.n_spot_x, j = cell / p.n_spot_x;
        s_w[t] = __ldg(&p.w[PB_W_IDX(p, layer, i, j)]);
        s_pos[t] = pb_spot_position(p, cell, i, j);
    }
    for (int k = tid; k < p.nz; k += nthr) {
        s_idd[k] = __ldg(&p.idd [(size_t)layer * p.nz + k]);
        s_sig[k] = __ldg(&p.sigma[(size_t)layer * p.nz + k]);
    }
    __syncthreads(); // staging done

    float acc[PB_Z_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < PB_Z_PER_THREAD; ++t) acc[t] = 0.0f;

    if (active)
    {
        const float x = p.vox_x0 + (float)vx * p.dvx;
        const float y = p.vox_y0 + (float)vy * p.dvy;
        int zk[PB_Z_PER_THREAD];
        float amp[PB_Z_PER_THREAD], inv2s2[PB_Z_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < PB_Z_PER_THREAD; ++t) {
            zk[t] = iz0 + t;
            if (zk[t] < p.n_vox_z) {
                const float z = p.vox_z0 + (float)zk[t] * p.dvz;
                const float idd_z = pb_lerp_z(s_idd, p.nz, p.z0, p.dz, z);
                const float sig_z = fmaxf(pb_lerp_z(s_sig, p.nz, p.z0, p.dz, z), PB_SIGMA_FLOOR);
                inv2s2[t] = 0.5f / (sig_z * sig_z);
                amp[t] = idd_z * inv2s2[t] * PB_INV_2PI;
            } else { inv2s2[t] = 0.0f; amp[t] = 0.0f; }
        }
        // ---- 2) gather all spots of this batch from shared memory ----
        for (int t = 0; t < cell_count; ++t) {
            const float w = s_w[t];
            if (w == 0.0f) continue;
            const float2 pos = s_pos[t]; // shared broadcast
            const float dx = x - pos.x;
            const float dy = y - pos.y;
            const float r2 = dx * dx + dy * dy;
            #pragma unroll
            for (int q = 0; q < PB_Z_PER_THREAD; ++q)
                acc[q] += w * amp[q] * PB_EXPF(-r2 * inv2s2[q]);
        }
        // ---- 3) read-modify-write accumulation ----
        #pragma unroll
        for (int t = 0; t < PB_Z_PER_THREAD; ++t)
            if (zk[t] < p.n_vox_z)
                d_out[PB_OU_IDX(p.n_vox_x, p.n_vox_y, vx, vy, zk[t])] += acc[t];
    }
}

// =============================================================================
// 8. Host-side error handling & utilities
// =============================================================================
#define PB_OK 0
#define PB_ERR_NULL_ARG -1
#define PB_ERR_BAD_DIM -2
#define PB_ERR_ZGRID_MISMATCH -3
#define PB_ERR_CUDA -4
#define PB_ERR_UNSUPPORTED -5
#define PB_ERR_NO_SMEM -6

// Fatal error check (used by main / test code): print and exit on failure.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t pb_err_ = (call); \
        if (pb_err_ != cudaSuccess) { \
            std::fprintf(stderr, "[CUDA_CHECK] %s:%d %s -> %s\n", \
                         __FILE__, __LINE__, #call, cudaGetErrorString(pb_err_)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Inside library functions: translate a CUDA error code into an API error code (never exits the process)
static int pb_cuda_rc(cudaError_t e)
{
    if (e == cudaSuccess) return PB_OK;
    std::fprintf(stderr, "[pencil_beam] CUDA error: %s\n", cudaGetErrorString(e));
    return PB_ERR_CUDA;
}

// Always check cudaGetLastError after a launch — a failed kernel launch is silent!
static int pb_check_launch(const char* name)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[pencil_beam] launch '%s' failed: %s\n", name,
                     cudaGetErrorString(e));
        return PB_ERR_CUDA;
    }
    return PB_OK;
}

// Error code -> readable string; extern "C" keeps it unmangled so Python ctypes can look it up by name.
extern "C" const char* pb_strerror(int code)
{
    switch (code) {
        case PB_OK: return "ok";
        case PB_ERR_NULL_ARG: return "null argument";
        case PB_ERR_BAD_DIM: return "invalid dimension / spacing";
        case PB_ERR_ZGRID_MISMATCH: return "z grid mismatch (the separable path requires n_vox_z==nz, dvz==dz, vox_z0==z0)";
        case PB_ERR_CUDA: return "CUDA runtime error";
        case PB_ERR_UNSUPPORTED: return "unsupported configuration";
        case PB_ERR_NO_SMEM: return "insufficient shared memory per block";
        default: return "unknown error";
    }
}

static inline int pb_cdiv(int a, int b) { return (a + b - 1) / b; }

// Whether the spot lattice is aligned 1:1 with the voxel grid — the same test as the device-side predicate.
__host__ __device__ __forceinline__ bool pb_is_identity_lattice(const PbProblem& p)
{
    if (p.n_vox_x != p.n_spot_x || p.n_vox_y != p.n_spot_y) return false;
    if (fabsf(p.dvx - p.dx_spot) > 1.0e-5f * p.dvx) return false;
    if (fabsf(p.dvy - p.dy_spot) > 1.0e-5f * p.dvy) return false;
    const int offx = pb_iround((p.x0_spot - p.vox_x0) / p.dvx);
    const int offy = pb_iround((p.y0_spot - p.vox_y0) / p.dvy);
    if (fabsf((p.x0_spot - p.vox_x0) - (float)offx * p.dvx) > 1.0e-4f) return false;
    if (fabsf((p.y0_spot - p.vox_y0) - (float)offy * p.dvy) > 1.0e-4f) return false;
    return true;
}

// Generic parameter validation (called first by every public entry point); extern "C" for direct Python calls.
extern "C" int pb_problem_validate(const PbProblem* p)
{
    if (p == nullptr) return PB_ERR_NULL_ARG;
    if (p->n_spot_x <= 0 || p->n_spot_y <= 0 || p->n_layers <= 0) return PB_ERR_BAD_DIM;
    if (p->n_vox_x <= 0 || p->n_vox_y <= 0 || p->n_vox_z <= 0) return PB_ERR_BAD_DIM;
    if (p->nz <= 0) return PB_ERR_BAD_DIM;
    if (!(p->dx_spot > 0.f) || !(p->dy_spot > 0.f)) return PB_ERR_BAD_DIM;
    if (!(p->dvx > 0.f) || !(p->dvy > 0.f) || !(p->dvz > 0.f)) return PB_ERR_BAD_DIM;
    if (!(p->dz > 0.f)) return PB_ERR_BAD_DIM;
    if (p->idd == nullptr || p->sigma == nullptr || p->w == nullptr) return PB_ERR_NULL_ARG;
    if (p->deposit_bilinear != 0 && p->deposit_bilinear != 1) return PB_ERR_BAD_DIM;
    if (pb_cdiv(p->n_vox_z, PB_Z_PER_THREAD) > 65535) return PB_ERR_UNSUPPORTED; // gridDim.z limit
    return PB_OK;
}

// Dynamic shared memory request: at or below the default limit (usually 48 KB) do nothing; at or below the
// opt-in limit request it with cudaFuncSetAttribute (persists, set once); larger reports PB_ERR_NO_SMEM. The
// kernel goes through a function template rather than a const void* cast, selecting the typed runtime overload.
template <typename KernelT>
static int pb_ensure_dynamic_smem(KernelT kernel, size_t bytes, const char* name)
{
    int dev = 0, max_default = 0, max_optin = 0;
    int rc = pb_cuda_rc(cudaGetDevice(&dev));
    if (rc != PB_OK) return rc;
    cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    if ((int)bytes <= max_default) return PB_OK;
    if ((int)bytes > max_optin) {
        std::fprintf(stderr, "[pencil_beam] %s: needs %zu B of dynamic shared memory; opt-in limit is %d B\n",
                     name, bytes, max_optin);
        return PB_ERR_NO_SMEM;
    }
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)bytes);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[pencil_beam] cudaFuncSetAttribute(%s, %zu B) failed: %s\n",
                     name, bytes, cudaGetErrorString(e));
        return PB_ERR_NO_SMEM;
    }
    return PB_OK;
}


// =============================================================================
// 9. extern "C" host API — for Python ctypes / PyTorch C++ extensions. Returns 0 on success, a negative
// pb_strerror()-interpretable error code otherwise; no path ever exits the process.
// =============================================================================
extern "C" {

// Kernel 1: naive gather
int pb_dose_forward_naive(const PbProblem* p, float* d_out)
{
    int rc = pb_problem_validate(p);
    if (rc != PB_OK) return rc;
    if (d_out == nullptr) return PB_ERR_NULL_ARG;

    const dim3 block(16, 16, 1);
    const dim3 grid(pb_cdiv(p->n_vox_x, 16), pb_cdiv(p->n_vox_y, 16), p->n_vox_z);
    naive_gather_kernel<<<grid, block>>>(*p, d_out);
    return pb_check_launch("naive_gather_kernel");
}

// Kernel 2: shared LUT gather (with the 48 KB default allowance / opt-in guard)
int pb_dose_forward_shared_lut(const PbProblem* p, float* d_out)
{
    int rc = pb_problem_validate(p);
    if (rc != PB_OK) return rc;
    if (d_out == nullptr) return PB_ERR_NULL_ARG;

    const size_t smem = 2 * (size_t)p->nz * sizeof(float);
    rc = pb_ensure_dynamic_smem(shared_lut_gather_kernel, smem, "shared_lut_gather_kernel");
    if (rc != PB_OK) return rc;

    const dim3 block(16, 16, 1);
    const dim3 grid(pb_cdiv(p->n_vox_x, 16), pb_cdiv(p->n_vox_y, 16),
                    pb_cdiv(p->n_vox_z, PB_Z_PER_THREAD));
    shared_lut_gather_kernel<<<grid, block, smem>>>(*p, d_out);
    return pb_check_launch("shared_lut_gather_kernel");
}

// Kernel 3: separable convolution (per layer: build amplitude map -> per-slice convolution -> accumulate)
int pb_dose_forward_separable(const PbProblem* p, float* d_out)
{
    int rc = pb_problem_validate(p);
    if (rc != PB_OK) return rc;
    if (d_out == nullptr) return PB_ERR_NULL_ARG;

    // The fast path works on the LUT z grid (the kernel weights are computed per slice from sigma(z_k)),
    // so the dose z grid must coincide with it; if it does not, resample IDD/sigma onto the dose grid first.
    if (p->n_vox_z != p->nz) return PB_ERR_ZGRID_MISMATCH;
    if (fabsf(p->dvz - p->dz) > 1.0e-5f * p->dvz) return PB_ERR_ZGRID_MISMATCH;
    if (fabsf(p->vox_z0 - p->z0) > 1.0e-4f) return PB_ERR_ZGRID_MISMATCH;

    const size_t amp_bytes = (size_t)p->n_vox_x * p->n_vox_y * p->nz * sizeof(float);
    const size_t out_bytes = (size_t)p->n_vox_x * p->n_vox_y * p->n_vox_z * sizeof(float);

    float* d_amp = nullptr;
    rc = pb_cuda_rc(cudaMalloc((void**)&d_amp, amp_bytes));
    if (rc != PB_OK) return rc;

    const size_t smem = pb_separable_smem_bytes();
    rc = pb_ensure_dynamic_smem(separable_conv_kernel, smem, "separable_conv_kernel");
    if (rc != PB_OK) { cudaFree(d_amp); return rc; }

    const dim3 conv_block(PB_CONV_TX, PB_CONV_TY, 1);
    const dim3 conv_grid(pb_cdiv(p->n_vox_x, PB_CONV_TX), pb_cdiv(p->n_vox_y, PB_CONV_TY), p->nz);
    const dim3 amp_block(32, 8, 1);
    const dim3 amp_grid(pb_cdiv(p->n_spot_x, 32), pb_cdiv(p->n_spot_y, 8), 1);

    rc = pb_cuda_rc(cudaMemsetAsync(d_out, 0, out_bytes, 0)); // accumulated per layer, so zero first
    if (rc != PB_OK) { cudaFree(d_amp); return rc; }

    for (int l = 0; l < p->n_layers; ++l)
    {
        // The amplitude tensor must be cleared for every layer: the general path scatters with atomicAdd, and even
        // on the 1:1 identity fast path the zero-weight spots are skipped (their voxels are never written), so
        // leftovers from the previous layer would be stale data; costs 8 MB/layer ≈ 0.02 ms, negligible next to
        // the per-layer convolution (~10 ms).
        rc = pb_cuda_rc(cudaMemsetAsync(d_amp, 0, amp_bytes, 0));
        if (rc != PB_OK) { cudaFree(d_amp); return rc; }

        build_spot_amplitude_kernel<<<amp_grid, amp_block>>>(*p, l, d_amp, nullptr);
        rc = pb_check_launch("build_spot_amplitude_kernel");
        if (rc != PB_OK) { cudaFree(d_amp); return rc; }

        // sigma_z points at this layer's row; accumulate=1 adds this layer's contribution to the dose volume
        separable_conv_kernel<<<conv_grid, conv_block, smem>>>(
            d_amp, p->sigma + (size_t)l * p->nz, 0.0f, d_out,
            p->n_vox_x, p->n_vox_y, p->nz, p->dvx, p->dvy, /*accumulate=*/1);
        rc = pb_check_launch("separable_conv_kernel");
        if (rc != PB_OK) { cudaFree(d_amp); return rc; }
    }
    return pb_cuda_rc(cudaFree(d_amp));
}

// Kernel 4: batched gather (layer × batch double loop)
int pb_dose_forward_batched(const PbProblem* p, float* d_out)
{
    int rc = pb_problem_validate(p);
    if (rc != PB_OK) return rc;
    if (d_out == nullptr) return PB_ERR_NULL_ARG;

    const int S = p->n_spot_x * p->n_spot_y; // spots per layer
    const int batch = (PB_BATCH_SPOTS < S) ? PB_BATCH_SPOTS : S;
    const size_t smem = (size_t)batch * (sizeof(float2) + sizeof(float))
                      + 2 * (size_t)p->nz * sizeof(float); // 12 B/spot + 8 B/z
    rc = pb_ensure_dynamic_smem(batched_gather_kernel, smem, "batched_gather_kernel");
    if (rc != PB_OK) return rc;

    const dim3 block(16, 16, 1);
    const dim3 grid(pb_cdiv(p->n_vox_x, 16), pb_cdiv(p->n_vox_y, 16),
                    pb_cdiv(p->n_vox_z, PB_Z_PER_THREAD));
    const size_t out_bytes = (size_t)p->n_vox_x * p->n_vox_y * p->n_vox_z * sizeof(float);

    rc = pb_cuda_rc(cudaMemsetAsync(d_out, 0, out_bytes, 0)); // accumulation semantics => zero first
    if (rc != PB_OK) return rc;

    for (int l = 0; l < p->n_layers; ++l)
        for (int c0 = 0; c0 < S; c0 += batch) {
            const int count = (S - c0 < batch) ? (S - c0) : batch;
            batched_gather_kernel<<<grid, block, smem>>>(*p, l, c0, count, d_out);
            rc = pb_check_launch("batched_gather_kernel");
            if (rc != PB_OK) return rc;
        }
    return PB_OK;
}

// Print device information (shared memory limits, SM count, L2, ...) to check this machine's assumptions
void pb_device_info(void)
{
    int dev = 0, max_default = 0, max_optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    std::printf("---- device %d: %s ----\n", dev, prop.name);
    std::printf(" compute capability : %d.%d\n", prop.major, prop.minor);
    std::printf(" SMs : %d\n", prop.multiProcessorCount);
    std::printf(" global memory : %.1f MiB\n", (double)prop.totalGlobalMem / 1048576.0);
    std::printf(" shared/block : %d B default, %d B opt-in\n", max_default, max_optin);
    std::printf(" shared/SM : %d B\n", (int)prop.sharedMemPerMultiprocessor);
    std::printf(" regs/block : %d\n", prop.regsPerBlock);
    std::printf(" mem clock/bus : %.0f MHz / %d bit\n",
                (double)prop.memoryClockRate / 1000.0, prop.memoryBusWidth);
    std::printf(" L2 cache : %d B\n", prop.l2CacheSize);
}

// VRAM budget query (used by the caller to decide whether to batch)
int pb_problem_bytes(const PbProblem* p, size_t* out_bytes,
                     size_t* amp_bytes, size_t* spot_table_bytes)
{
    if (p == nullptr) return PB_ERR_NULL_ARG;
    if (out_bytes) *out_bytes =
        (size_t)p->n_vox_x * p->n_vox_y * p->n_vox_z * sizeof(float);
    if (amp_bytes) *amp_bytes =
        (size_t)p->n_vox_x * p->n_vox_y * p->nz * sizeof(float);
    if (spot_table_bytes) *spot_table_bytes =
        (size_t)p->n_spot_x * p->n_spot_y * p->n_layers * sizeof(float);
    return PB_OK;
}

} // extern "C"

// =============================================================================
// 10. Self-test / benchmark (standalone main)
//     nvcc -arch=sm_120 -O3 -DPB_STANDALONE_MAIN pencil_beam.cu -o pb_demo
// =============================================================================
#ifdef PB_STANDALONE_MAIN

#include <vector>

// Standalone self-test scaffolding: host-side data, synthetic problem builder and CPU reference.
typedef struct PbHostData {
    std::vector<float> idd, sigma, w; // host copy
    float *d_idd = nullptr, *d_sigma = nullptr, *d_w = nullptr, *d_out = nullptr;
    float* h_out = nullptr; // host mirror (for accuracy comparison)
    PbProblem p;
} PbHostData;

// Build a physically plausible IMPT problem: 40x40 spots / 40 layers / 128^3 by default
static void pb_build_synthetic(PbHostData& H, int n_spot, int n_layers, int n_vox,
                               float dx_spot, float dvx, int nz, float dz)
{
    PbProblem& p = H.p;
    std::memset(&p, 0, sizeof(p));
    p.n_spot_x = p.n_spot_y = n_spot;
    p.n_layers = n_layers;
    p.dx_spot = p.dy_spot = dx_spot;
    p.x0_spot = -0.5f * (float)(n_spot - 1) * dx_spot; // lattice centred on the origin
    p.y0_spot = p.x0_spot;
    p.spot_xy = nullptr; // regular lattice => analytic coordinates
    p.nz = nz; p.z0 = 0.0f; p.dz = dz;
    p.n_vox_x = p.n_vox_y = p.n_vox_z = n_vox;
    p.dvx = p.dvy = dvx;
    p.dvz = dz; // exactly coincident with the LUT z grid
    p.vox_x0 = -0.5f * (float)(n_vox - 1) * dvx;
    p.vox_y0 = p.vox_x0;
    p.vox_z0 = p.z0;
    p.deposit_bilinear = 1; // lattice spacing != voxel size => bilinear deposition

    // IDD: plateau + Bragg peak + steep distal falloff; sigma grows from 0.30 to ~1.0 cm with depth
    H.idd.assign((size_t)n_layers * nz, 0.0f);
    H.sigma.assign((size_t)n_layers * nz, 0.0f);
    for (int l = 0; l < n_layers; ++l) {
        const float z_peak = 4.0f + 0.6f * (float)l; // peak position sweeps from 4 cm to 27.4 cm
        const float w_peak = 0.9f;
        for (int k = 0; k < nz; ++k) {
            const float z = p.z0 + (float)k * dz;
            const float u = (z - z_peak) / w_peak;
            float idd = 0.35f + 6.0f * std::exp(-0.5f * u * u);
            if (z > z_peak + 1.6f * w_peak) idd = 0.02f; // distal falloff
            if (z < 1.0f) idd = 0.0f;
            H.idd[(size_t)l * nz + k] = idd;
            const float zc = (z > 30.0f) ? 30.0f : z;
            H.sigma[(size_t)l * nz + k] = 0.30f + 0.70f * (zc / 30.0f);
        }
    }
    // Weights: high in the middle, low at the edges (simulating a non-uniform field); layout [j][l][i] as proposed
    H.w.assign((size_t)n_spot * n_spot * n_layers, 0.0f);
    const float cx = 0.5f * (float)(n_spot - 1);
    for (int l = 0; l < n_layers; ++l)
        for (int j = 0; j < n_spot; ++j)
            for (int i = 0; i < n_spot; ++i) {
                const float rr = ((float)i - cx) * ((float)i - cx)
                               + ((float)j - cx) * ((float)j - cx);
                const float g = std::exp(-rr / (2.0f * 10.0f * 10.0f));
                const float lay = 1.0f - 0.5f * (float)l / (float)(n_layers - 1);
                H.w[(size_t)j * n_spot * n_layers + (size_t)l * n_spot + i] = g * lay;
            }

    CUDA_CHECK(cudaMalloc((void**)&H.d_idd, H.idd.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&H.d_sigma, H.sigma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&H.d_w, H.w.size() * sizeof(float)));
    const size_t out_n = (size_t)n_vox * n_vox * n_vox;
    CUDA_CHECK(cudaMalloc((void**)&H.d_out, out_n * sizeof(float)));
    H.h_out = (float*)std::malloc(out_n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(H.d_idd, H.idd.data(), H.idd.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(H.d_sigma, H.sigma.data(), H.sigma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(H.d_w, H.w.data(), H.w.size() * sizeof(float), cudaMemcpyHostToDevice));
    p.idd = H.d_idd; p.sigma = H.d_sigma; p.w = H.d_w;
}

// CPU reference (regular lattice only): full accumulation spot by spot, with a double accumulator.
// Computed on sampled voxels only — a full CPU reference is as slow as the GPU baseline, so pointless.
static void pb_cpu_reference(const PbProblem& p, const int* ids, int n_ids, float* out)
{
    for (int q = 0; q < n_ids; ++q) {
        const int id = ids[q];
        const int iv = id % p.n_vox_x;
        const int jv = (id / p.n_vox_x) % p.n_vox_y;
        const int kv = id / (p.n_vox_x * p.n_vox_y);
        const double x = (double)p.vox_x0 + (double)iv * p.dvx;
        const double y = (double)p.vox_y0 + (double)jv * p.dvy;
        const double z = (double)p.vox_z0 + (double)kv * p.dvz;

        double acc = 0.0;
        for (int l = 0; l < p.n_layers; ++l) {
            // Exactly the same "look up the table, then interpolate" path as the GPU
            const float* ri = p.idd + (size_t)l * p.nz;
            const float* rs = p.sigma + (size_t)l * p.nz;
            const double f = (z - (double)p.z0) / (double)p.dz;
            int k0 = (int)f;
            if (k0 < 0) k0 = 0;
            if (k0 > p.nz - 1) k0 = p.nz - 1;
            const int k1 = (k0 + 1 < p.nz) ? k0 + 1 : k0;
            const double t = (f > 0.0) ? (f - (double)k0) : 0.0;
            const double idd = (double)ri[k0] + t * ((double)ri[k1] - (double)ri[k0]);
            double sig = (double)rs[k0] + t * ((double)rs[k1] - (double)rs[k0]);
            if (sig < (double)PB_SIGMA_FLOOR) sig = (double)PB_SIGMA_FLOOR;
            const double inv2s2 = 1.0 / (2.0 * sig * sig);
            const double amp = idd * inv2s2 / (2.0 * PB_PI_D);

            for (int j = 0; j < p.n_spot_y; ++j) {
                const double ym = (double)p.y0_spot + (double)j * p.dy_spot;
                const double dy = y - ym;
                for (int i = 0; i < p.n_spot_x; ++i) {
                    const double w = (double)p.w[PB_W_IDX(p, l, i, j)];
                    if (w == 0.0) continue;
                    const double xm = (double)p.x0_spot + (double)i * p.dx_spot;
                    const double dx = x - xm;
                    acc += w * amp * std::exp(-(dx * dx + dy * dy) * inv2s2);
                }
            }
        }
        out[q] = (float)acc;
    }
}

// Comparison on sampled voxels: returns the relative L2 error; max|abs err| comes back through a pointer
static double pb_compare(const float* h_out, const int* ids, int n_ids,
                         const float* ref, double* max_abs_out)
{
    double num = 0.0, den = 0.0, max_abs = 0.0;
    for (int q = 0; q < n_ids; ++q) {
        const double d = (double)h_out[ids[q]] - (double)ref[q];
        num += d * d;
        den += (double)ref[q] * (double)ref[q];
        max_abs = std::fmax(max_abs, std::fabs(d));
    }
    if (max_abs_out) *max_abs_out = max_abs;
    return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// Timing: warmup warm-up runs + iters timed runs, with cudaDeviceSynchronize and event timing on both sides
template <typename F>
static float pb_bench(F&& launch, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ms / (float)iters;
}

// Uniform "run + verify + time" wrapper: rel_tol is the relative L2 tolerance
static void pb_run_and_check(const char* name, int (*fn)(const PbProblem*, float*),
                             PbHostData& H, const int* ids, int n_sample, const float* ref,
                             float rel_tol, int warmup, int iters, double work_pairs)
{
    const PbProblem& p = H.p;
    const size_t out_bytes = (size_t)p.n_vox_x * p.n_vox_y * p.n_vox_z * sizeof(float);
    const int rc = fn(&p, H.d_out);
    std::printf("\n[%s] rc=%d\n", name, rc);
    if (rc != PB_OK) { std::printf(" -> %s\n", pb_strerror(rc)); return; }

    CUDA_CHECK(cudaMemcpy(H.h_out, H.d_out, out_bytes, cudaMemcpyDeviceToHost));
    double mx = 0.0;
    const double rel = pb_compare(H.h_out, ids, n_sample, ref, &mx);
    std::printf(" rel-L2 vs CPU = %.3e max|abs err| = %.3e (tol %.0e)%s\n",
                rel, mx, rel_tol, (rel <= rel_tol) ? "" : " << outside tolerance");

    const float ms = pb_bench([&] { fn(&p, H.d_out); }, warmup, iters);
    std::printf(" time = %.3f ms", ms);
    if (work_pairs > 0.0)
        std::printf(" (%.3e spot-voxel pairs/s)", work_pairs / (ms * 1.0e-3));
    std::printf("\n");
}

int main(int argc, char** argv)
{
    int n_spot = 40, n_layers = 40, n_vox = 128, nz = 128;
    float dz = 0.3f, dx_spot = 0.5f, dvx = 0.15625f; // 38.4 cm deep / 20 cm field / 20 cm/128
    int run_naive = 1, naive_iters = 1;

    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "--help")) { std::printf(
                    "usage: %s [--spots N] [--layers N] [--vox N] [--skip-naive] [--naive-iters N]\n"
                    " --spots N spots per layer (default 40, i.e. a 40x40 lattice)\n"
                    " --layers N number of energy layers (default 40)\n"
                    " --vox N dose grid edge length (default 128; nz is set equal to N)\n"
                    " --skip-naive skip the O(N_vox*M) naive baseline (it can take tens of seconds)\n"
                    " --naive-iters N number of timed iterations of the naive kernel (default 1)\n", argv[0]); return 0; }
        else if (!std::strcmp(argv[a], "--spots") && a + 1 < argc) n_spot = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--layers") && a + 1 < argc) n_layers = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--vox") && a + 1 < argc) n_vox = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--skip-naive")) run_naive = 0;
        else if (!std::strcmp(argv[a], "--naive-iters") && a + 1 < argc) naive_iters = std::atoi(argv[++a]);
        else { std::fprintf(stderr, "unknown argument: %s (try --help)\n", argv[a]); return 2; }
    }
    if (n_spot < 1 || n_layers < 1 || n_vox < 2) {
        std::fprintf(stderr, "invalid arguments: spots/layers must be >= 1, vox must be >= 2\n");
        return 2;
    }
    nz = n_vox; // the separable path requires coincident z grids

    pb_device_info();
    PbHostData H;
    pb_build_synthetic(H, n_spot, n_layers, n_vox, dx_spot, dvx, nz, dz);
    const PbProblem& p = H.p;
    const double N_vox = (double)p.n_vox_x * p.n_vox_y * p.n_vox_z;
    const double M = (double)p.n_spot_x * p.n_spot_y * p.n_layers;

    size_t out_b = 0, amp_b = 0, spot_b = 0;
    pb_problem_bytes(&p, &out_b, &amp_b, &spot_b);
    std::printf("\n---- problem ----\n");
    std::printf(" spots : %dx%d x %d layers = %.0f\n", p.n_spot_x, p.n_spot_y,
                p.n_layers, M);
    std::printf(" voxels : %d^3 = %.3e ; voxel = %.5f cm\n", p.n_vox_x, N_vox, p.dvx);
    std::printf(" spot spacing : %.3f cm (lattice centered at 0)\n", p.dx_spot);
    std::printf(" pairs : N_vox*M = %.3e\n", N_vox * M);
    std::printf(" dose volume : %.2f MiB ; layer amplitude : %.2f MiB ; spot table : %.2f MiB\n",
                (double)out_b / 1048576.0, (double)amp_b / 1048576.0, (double)spot_b / 1048576.0);
    std::printf(" lattice 1:1 with voxel grid : %s\n",
                pb_is_identity_lattice(p) ? "yes" : "no (=> bilinear deposit + atomicAdd)");

    // Sample voxels for the CPU reference (a full CPU reference is pointless)
    const int n_sample = 512;
    const int n_vox_total = p.n_vox_x * p.n_vox_y * p.n_vox_z;
    std::vector<int> ids(n_sample);
    unsigned int seed = 12345u; // fixed seed => reproducible
    for (int q = 0; q < n_sample; ++q) {
        seed = seed * 1664525u + 1013904223u; // linear congruential
        ids[q] = (int)(seed % (unsigned int)n_vox_total);
    }
    std::vector<float> ref(n_sample, 0.0f);
    pb_cpu_reference(p, ids.data(), n_sample, ref.data());
    double ref_max = 0.0;
    for (int q = 0; q < n_sample; ++q) ref_max = std::fmax(ref_max, (double)ref[q]);
    std::printf("\n---- CPU reference: %d sampled voxels, double accumulation ----\n", n_sample);
    std::printf(" reference max dose = %.6g (arb. units)\n", ref_max);

    if (run_naive)
        pb_run_and_check("1 naive_gather_kernel", pb_dose_forward_naive, H, ids.data(),
                         n_sample, ref.data(), 1.0e-3, 1, naive_iters, N_vox * M);
    else
        std::printf("\n[1 naive_gather_kernel] skipped\n");

    pb_run_and_check("2 shared_lut_gather_kernel", pb_dose_forward_shared_lut, H, ids.data(),
                     n_sample, ref.data(), 1.0e-3, 1, 3, N_vox * M);

    std::printf("\n Note: the difference between kernel 3 and the analytic reference comes from\n"
                " (a) the discretely normalised kernel, (b) the %.0f-sigma truncation\n"
                " and (c) bilinear deposit -- expect a few %%, which is not a bug,\n"
                " but do not expect 1e-6 either.\n",
                (double)PB_GAUSS_TRUNC);
    pb_run_and_check("3 separable_conv_kernel", pb_dose_forward_separable, H, ids.data(),
                     n_sample, ref.data(), 0.15, 1, 3, 0.0);
    {
        int occ = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, separable_conv_kernel, PB_CONV_TX * PB_CONV_TY, pb_separable_smem_bytes()));
        std::printf(" occupancy = %d block/SM (%d threads/SM, shared %zu B/block)\n",
                    occ, occ * PB_CONV_TX * PB_CONV_TY, pb_separable_smem_bytes());
    }

    pb_run_and_check("4 batched_gather_kernel", pb_dose_forward_batched, H, ids.data(),
                     n_sample, ref.data(), 1.0e-3, 1, 3, N_vox * M);

    std::printf("\nNote: this file was written on a machine with no CUDA toolchain; it has never\n"
           " been compiled or measured.\n"
                " Re-measure the timings above on a machine with CUDA 12.8+.\n");

    CUDA_CHECK(cudaFree(H.d_idd));
    CUDA_CHECK(cudaFree(H.d_sigma));
    CUDA_CHECK(cudaFree(H.d_w));
    CUDA_CHECK(cudaFree(H.d_out));
    std::free(H.h_out);
    return 0;
}

#endif // PB_STANDALONE_MAIN
