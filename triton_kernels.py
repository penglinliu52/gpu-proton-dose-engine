"""
pbdose.triton_kernels — 自定义 GPU Kernel（Triton JIT → sm_120 PTX/CUBIN）
=========================================================================

对应课题文档第四节的交付物：自定义 CUDA Kernel 与显存/访存优化。
本机未安装 CUDA Toolkit（无 nvcc、无 MSVC），因此以 **Triton** 作为内核编写与
编译后端：Triton 直接生成 PTX，再由 ptxas 编译为 sm_120 的 CUBIN —— 这是真正
意义上的自定义 GPU kernel（不是 cuBLAS/库函数调用）。手写 CUDA C++ 版本另见
``cuda/pencil_beam.cu``（未编译，作为代码交付物与对照）。

四个内核构成本项目的"加速阶梯"
------------------------------
1. ``pb_gather_naive_kernel``      每体素循环 **全部 M 条** pencil beam（课题文档原始设计）
2. ``pb_gather_window_kernel``     高斯核紧支撑窗口 + IDD/sigma LUT 寄存器缓存（逐层精确）
3. ``_pass1_z`` / ``_pass2_z``     可分离卷积：高斯核在 kernel 内即时生成并融合进
                                   tensor-core GEMM，核矩阵不落显存
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False
    tl = None

INV2PI = float(1.0 / (2.0 * np.pi))                 # 二维高斯归一化 1/(2*pi*sigma^2)
INV_SQRT_2PI = float(1.0 / np.sqrt(2.0 * np.pi))    # 一维高斯归一化 1/(sqrt(2*pi)*sigma)


if HAS_TRITON:

    # =======================================================================
    # 内核 1：朴素 Gather —— 每体素循环全部 pencil beam
    # =======================================================================
    @triton.jit
    def pb_gather_naive_kernel(
        out_ptr,                              # (nz, ny, nx) float32
        sx_ptr, sy_ptr, sw_ptr, sl_ptr,       # (M,) spot x, y, weight, layer
        idd_ptr, sig_ptr,                     # (L, nz)
        nz, ny, nx, M,
        ox, oy, oz, dx, dy, dz, inv2pi,
        BLOCK_X: tl.constexpr,
    ):
        """
        网格: (ceil(nx/BLOCK_X), ny, nz)；每程序负责一行 BLOCK_X 个体素。
        瓶颈: 内层循环 M 次，每次都从 global memory 取 4 个标量 + 2 个 LUT 标量，
        算术强度 < 1 FLOP/Byte —— 典型的 memory-bound（详见性能报告）。
        """
        px = tl.program_id(0)
        py = tl.program_id(1)
        pz = tl.program_id(2)

        kx = px * BLOCK_X + tl.arange(0, BLOCK_X)
        mask = kx < nx
        x = ox + kx * dx
        y = oy + py * dy

        acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        for m in range(0, M):
            x0 = tl.load(sx_ptr + m)
            y0 = tl.load(sy_ptr + m)
            w = tl.load(sw_ptr + m)
            l = tl.load(sl_ptr + m).to(tl.int32)
            idd = tl.load(idd_ptr + l * nz + pz)
            s = tl.load(sig_ptr + l * nz + pz)
            dxv = x - x0
            dyv = y - y0
            e = tl.exp(-(dxv * dxv + dyv * dyv) / (2.0 * s * s))
            acc += (w * idd * inv2pi / (s * s)) * e

        tl.store(out_ptr + pz * ny * nx + py * nx + kx, acc, mask=mask)

    # =======================================================================
    # 内核 2：窗口 Gather + LUT 缓存（逐层物理精确）
    # =======================================================================
    @triton.jit
    def pb_gather_window_kernel(
        out_ptr,                              # (nz, ny, nx)
        w_ptr,                                # (L, nsy, nsx)
        idd_ptr, sig_ptr,                     # (L, nz)
        nz, ny, nx, L, nsy, nsx,
        ox, oy, oz, dx, dy, dz,
        xs0, ys0, dsx, dsy, inv2pi, kz,
        KX: tl.constexpr, KY: tl.constexpr, BLOCK_X: tl.constexpr,
    ):
        """
        网格: (ceil(nx/BLOCK_X), ny)；kz 由调用方逐个深度传入。
        优化点（对应课题文档第四节）:
          * 高斯核紧支撑: 每体素只累加 ±K 邻域，O(K^2) 而非 O(M)
          * 每层的 IDD[kz] / sigma[kz] 在层循环外一次性载入并复用（LUT 缓存）
          * 逐层累加，保持"不同能量层 sigma 不同"的物理精确性
        """
        px = tl.program_id(0)
        py = tl.program_id(1)

        kx = px * BLOCK_X + tl.arange(0, BLOCK_X)
        mask = kx < nx
        x = ox + kx * dx
        y = oy + py * dy

        i_c = tl.floor((x - xs0) / dsx + 0.5).to(tl.int32)
        j_c = tl.floor((y - ys0) / dsy + 0.5).to(tl.int32)

        acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        for l in range(0, L):
            idd_l = tl.load(idd_ptr + l * nz + kz)
            s = tl.load(sig_ptr + l * nz + kz)
            pre = idd_l * inv2pi / (s * s)
            inv2s2 = 1.0 / (2.0 * s * s)

            wsum = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for jj in range(-KY, KY + 1):
                j = j_c + jj
                jv = (j >= 0) & (j < nsy)
                jc = tl.minimum(tl.maximum(j, 0), nsy - 1)
                y0 = ys0 + jc * dsy
                dyv = y - y0
                gy = tl.where(jv, tl.exp(-(dyv * dyv) * inv2s2), 0.0)
                for ii in range(-KX, KX + 1):
                    i = i_c + ii
                    iv = (i >= 0) & (i < nsx)
                    ic = tl.minimum(tl.maximum(i, 0), nsx - 1)
                    x0 = xs0 + ic * dsx
                    dxv = x - x0
                    gx = tl.where(iv, tl.exp(-(dxv * dxv) * inv2s2), 0.0)
                    wv = tl.load(w_ptr + l * nsy * nsx + jc * nsx + ic,
                                 mask=iv & jv, other=0.0)
                    wsum += wv * gx * gy
            acc += pre * wsum

        tl.store(out_ptr + kz * ny * nx + py * nx + kx, acc, mask=mask)

    # =======================================================================
    # 内核 3：可分离卷积（高斯核即时生成 + 融合 GEMM）
    # =======================================================================
    @triton.jit
    def _pass1_z(a_ptr, t_ptr, sig_ptr, kz, ny, nsy, nsx,
                 oy, ys0, dyv, dsy, inv2pi,
                 BLOCK_Y: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
        """T[z,i,y] = sum_j A[z,j,i] * Gy[z,y,j] —— 沿 spot-j 方向收缩。"""
        py = tl.program_id(0)
        pi = tl.program_id(1)
        ky = py * BLOCK_Y + tl.arange(0, BLOCK_Y)
        ki = pi * BLOCK_I + tl.arange(0, BLOCK_I)
        mask_y = ky < ny
        mask_i = ki < nsx

        s = tl.load(sig_ptr + kz)
        inv2s2 = 1.0 / (2.0 * s * s)
        norm = inv2pi / s
        yv = oy + ky * dyv

        acc = tl.zeros((BLOCK_Y, BLOCK_I), dtype=tl.float32)
        for j0 in range(0, nsy, BLOCK_J):
            kj = j0 + tl.arange(0, BLOCK_J)
            mask_j = kj < nsy
            sj = ys0 + kj * dsy
            _dyv = yv[:, None] - sj[None, :]
            gy = norm * tl.exp(-(_dyv * _dyv) * inv2s2)   # (Y,J)
            gy = tl.where(mask_y[:, None] & mask_j[None, :], gy, 0.0)
            aptr = a_ptr + kz * nsy * nsx + kj[:, None] * nsx + ki[None, :]
            aa = tl.load(aptr, mask=mask_j[:, None] & mask_i[None, :], other=0.0)
            acc += tl.dot(gy, aa, input_precision="ieee")   # (Y,J)@(J,I) -> (Y,I)
        optr = t_ptr + kz * nsx * ny + ki[:, None] * ny + ky[None, :]
        tl.store(optr, tl.trans(acc), mask=mask_i[:, None] & mask_y[None, :])

    @triton.jit
    def _pass2_z(t_ptr, d_ptr, sig_ptr, kz, ny, nx, nsx,
                 ox, xs0, dxv, dsx, inv2pi,
                 BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_I: tl.constexpr):
        """D[z,y,x] = sum_i T[z,i,y] * Gx[z,x,i] —— 高斯核在 kernel 内即时生成。"""
        py = tl.program_id(0)
        px = tl.program_id(1)
        ky = py * BLOCK_Y + tl.arange(0, BLOCK_Y)
        kx = px * BLOCK_X + tl.arange(0, BLOCK_X)
        mask_y = ky < ny
        mask_x = kx < nx

        s = tl.load(sig_ptr + kz)
        inv2s2 = 1.0 / (2.0 * s * s)
        norm = inv2pi / s
        xv = ox + kx * dxv

        acc = tl.zeros((BLOCK_X, BLOCK_Y), dtype=tl.float32)
        for i0 in range(0, nsx, BLOCK_I):
            ki = i0 + tl.arange(0, BLOCK_I)
            mask_i = ki < nsx
            si = xs0 + ki * dsx
            _dxv = xv[:, None] - si[None, :]
            gx = norm * tl.exp(-(_dxv * _dxv) * inv2s2)   # (X,I)
            gx = tl.where(mask_x[:, None] & mask_i[None, :], gx, 0.0)
            tptr = t_ptr + kz * nsx * ny + ki[:, None] * ny + ky[None, :]
            tt = tl.load(tptr, mask=mask_i[:, None] & mask_y[None, :], other=0.0)
            acc += tl.dot(gx, tt, input_precision="ieee")
        optr = d_ptr + kz * ny * nx + ky[None, :] * nx + kx[:, None]
        tl.store(optr, acc, mask=mask_x[:, None] & mask_y[None, :])


# ===========================================================================
# Python 包装层
# ===========================================================================
def _require():
    if not HAS_TRITON:
        raise RuntimeError("Triton 不可用（pip install triton-windows）")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")


def _flat_spot_arrays(problem, m_limit=None):
    x, y, l, w = problem.lattice.flat_spot_table()
    if m_limit is not None and m_limit < len(w):
        x, y, l, w = x[:m_limit], y[:m_limit], l[:m_limit], w[:m_limit]
    dev = "cuda"
    return (torch.as_tensor(x.astype(np.float32), device=dev),
            torch.as_tensor(y.astype(np.float32), device=dev),
            torch.as_tensor(w.astype(np.float32), device=dev),
            torch.as_tensor(l.astype(np.int32), device=dev))


def triton_naive(problem, block_x: int = 32, m_limit=None, num_warps: int = 1):
    """内核 1：朴素 gather（每体素循环全部 beam）。"""
    _require()
    g, L = problem.grid, problem.lattice
    sx, sy, sw, sl = _flat_spot_arrays(problem, m_limit)
    M = sx.numel()
    dev = "cuda"
    out = torch.empty((g.nz, g.ny, g.nx), dtype=torch.float32, device=dev)
    idd = torch.as_tensor(problem.idd.astype(np.float32), device=dev)
    sig = torch.as_tensor(problem.sigma.astype(np.float32), device=dev)
    grid = (triton.cdiv(g.nx, block_x), g.ny, g.nz)
    pb_gather_naive_kernel[grid](
        out, sx, sy, sw, sl, idd, sig,
        g.nz, g.ny, g.nx, M,
        float(g.ox), float(g.oy), float(g.oz), g.dx, g.dy, g.dz, INV2PI,
        BLOCK_X=block_x, num_warps=num_warps,
    )
    return out


def triton_window(problem, kx: int = 6, ky: int = 6, block_x: int = 64,
                  num_warps: int = 4):
    """内核 2：窗口 gather + LUT 缓存（逐层精确）。"""
    _require()
    g, L = problem.grid, problem.lattice
    dev = "cuda"
    w = torch.as_tensor(np.ascontiguousarray(L.ensure_weights().astype(np.float32)),
                        device=dev)
    idd = torch.as_tensor(problem.idd.astype(np.float32), device=dev)
    sig = torch.as_tensor(problem.sigma.astype(np.float32), device=dev)
    out = torch.empty((g.nz, g.ny, g.nx), dtype=torch.float32, device=dev)
    xs0, ys0 = float(L.spot_x[0]), float(L.spot_y[0])
    grid = (triton.cdiv(g.nx, block_x), g.ny)
    for kz in range(g.nz):
        pb_gather_window_kernel[grid](
            out, w, idd, sig,
            g.nz, g.ny, g.nx, L.n_layers, L.n_sy, L.n_sx,
            float(g.ox), float(g.oy), float(g.oz), g.dx, g.dy, g.dz,
            xs0, ys0, L.dx_spot, L.dy_spot, INV2PI, kz,
            KX=kx, KY=ky, BLOCK_X=block_x, num_warps=num_warps,
        )
    return out


def triton_separable(problem, block_x: int = 32, block_y: int = 32,
                     block_i: int = 32, block_j: int = 32, num_warps: int = 4):
    """
    内核 3：完整可分离卷积（pass1 + pass2），单 sigma 快速路径。
    对应 `engines.torch_separable(per_layer=False)`。
    """
    _require()
    from .engines import layerwise_sigma_bar
    g, L = problem.grid, problem.lattice
    dev = "cuda"
    w = L.ensure_weights()
    A_np = (problem.idd.T @ w.reshape(L.n_layers, -1)).reshape(g.nz, L.n_sy, L.n_sx)
    A = torch.as_tensor(np.ascontiguousarray(A_np.astype(np.float32)), device=dev)
    sig = torch.as_tensor(np.ascontiguousarray(
        layerwise_sigma_bar(problem).astype(np.float32)), device=dev)
    T = torch.empty((g.nz, L.n_sx, g.ny), dtype=torch.float32, device=dev)
    D = torch.empty((g.nz, g.ny, g.nx), dtype=torch.float32, device=dev)

    gy_ = triton.cdiv(g.ny, block_y)
    gi = triton.cdiv(L.n_sx, block_i)
    gx_ = triton.cdiv(g.nx, block_x)
    for kz in range(g.nz):
        _pass1_z[(gy_, gi)](
            A, T, sig, kz, g.ny, L.n_sy, L.n_sx,
            float(g.oy), float(L.spot_y[0]), g.dy, L.dy_spot, INV_SQRT_2PI,
            BLOCK_Y=block_y, BLOCK_I=block_i, BLOCK_J=block_j, num_warps=num_warps,
        )
        _pass2_z[(gy_, gx_)](
            T, D, sig, kz, g.ny, g.nx, L.n_sx,
            float(g.ox), float(L.spot_x[0]), g.dx, L.dx_spot, INV_SQRT_2PI,
            BLOCK_X=block_x, BLOCK_Y=block_y, BLOCK_I=block_i, num_warps=num_warps,
        )
    return D
