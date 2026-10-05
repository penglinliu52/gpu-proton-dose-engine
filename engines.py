"""
pbdose.engines — Pencil Beam 剂量重构的多种实现（同一物理模型，不同算法/硬件）
=============================================================================

本模块提供课题 B 的"递进式加速链"，所有实现共享同一物理模型：

    D(z,y,x) = sum_l sum_ij  w[l,i,j] * IDD_l(z) * Gx_l(z, x, x_i) * Gy_l(z, y, y_j)
    Gx_l(z,x,x_i) = 1/(sqrt(2pi) sigma_l(z)) * exp( -(x-x_i)^2 / (2 sigma_l(z)^2) )

**关键物理细节（本项目的第一个真实技术坑）**：sigma_l(z) 依赖该能量层在深度 z 处的
局部能量 E_l(z) = E(R_l - z)，因此 **不同能量层在同一深度处的 sigma 不同**，横向核
不能跨层合并。正确做法是逐层（或按 (层, 深度) 批次）做横向卷积后按 IDD 权重求和。

实现清单
--------
  1. `naive_python_triple_loop`  教科书三重循环（每体素循环全部束流）—— 小规模正确性基线
  2. `naive_numpy_per_beam`      逐束流全网格累加（传统 TPS 写法, O(M*Nvox)）
  3. `separable_numpy`           可分离矩阵形式（NumPy float64 参考真值, 逐层精确）
  4. `torch_separable`           PyTorch 批矩阵乘法（CPU/GPU, 逐层精确 / 单 sigma 近似）
  5. `torch_gather_local`        PyTorch 局部窗口 Gather（GPU）

复杂度对比（M 束流, L 能量层, N = Nx*Ny*Nz 体素）
    naive      : O(M * N)
    separable  : O(L * nz * (nsx*nsy*ny + nx*nsx*ny))
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .model import PBProblem, DoseGrid, SpotLattice

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ===========================================================================
# 公共工具
# ===========================================================================
def lateral_kernel(vox_axis: np.ndarray, spot_axis: np.ndarray,
                   sigma: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    G[..., k_vox, k_spot] = (1/(sqrt(2pi) s)) exp(-(x_vox - x_spot)^2/(2 s^2))

    sigma 的 shape 为任意前置维度 (...,)；返回 shape (..., nv, ns)。
    """
    d = (vox_axis[:, None] - spot_axis[None, :]) ** 2          # (nv, ns)
    s = np.clip(np.asarray(sigma, dtype=np.float64), 1e-9, None)
    out = np.exp(-d / (2.0 * s[..., None, None] ** 2))
    if normalized:
        out = out * (INV_SQRT_2PI / s)[..., None, None]
    return out


def build_amplitude_map(problem: PBProblem) -> np.ndarray:
    """
    A[l, k, j, i] = w[l, j, i] * IDD[l, k]

    即"每条 pencil beam 在深度 k 处应沉积的幅度"（未做横向展宽）。
    """
    w = problem.lattice.ensure_weights()                        # (L, nsy, nsx)
    idd = problem.idd                                          # (L, nz)
    return idd[:, :, None, None] * w[:, None, :, :]


def layerwise_sigma_bar(problem: PBProblem, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """按层权重加权的平均 sigma(z)（仅供"单 sigma 近似"快速路径使用）。"""
    w = problem.lattice.ensure_weights()
    wl = w.reshape(w.shape[0], -1).sum(axis=1)
    wl = wl / max(wl.sum(), 1e-30)
    return (problem.sigma * wl[:, None]).sum(axis=0)


def operation_counts(problem: PBProblem) -> dict:
    """各算法的乘加次数（用于复杂度/加速比分析表）。"""
    g, L = problem.grid, problem.lattice
    per_layer = g.nz * (L.n_sx * L.n_sy * g.ny + g.nx * L.n_sx * g.ny)
    kx = int(np.ceil(4.0 * problem.sigma.max() / L.dx_spot)) + 1
    ky = int(np.ceil(4.0 * problem.sigma.max() / L.dy_spot)) + 1
    local = g.nz * (g.ny * (2 * ky + 1) * L.n_sx + g.nx * (2 * kx + 1) * g.ny)
    return {
        "naive": L.n_spots * g.n_vox,
        "separable_exact": per_layer,
        "gather_local": local,
        "amplitude_map": L.n_layers * g.nz * L.n_sx * L.n_sy,
        "M": L.n_spots, "N_vox": g.n_vox, "L": L.n_layers,
    }


# ===========================================================================
# 实现 1：教科书三重循环
# ===========================================================================
def naive_python_triple_loop(problem: PBProblem, max_spots: Optional[int] = None,
                             dtype=np.float64) -> np.ndarray:
    """
    纯 Python 三重循环：外层遍历体素，内层遍历全部 pencil beam。
    对应课题文档第二节的 O(M*Nx*Ny*Nz) 直接实现；仅用于小规模正确性验证与
    "为什么必须并行化"的教学对比（全尺寸问题下不可运行）。
    """
    g = problem.grid
    xs, ys, ls, ws = problem.lattice.flat_spot_table()
    m = len(ws)
    if max_spots is not None and m > max_spots:
        idx = np.linspace(0, m - 1, max_spots).astype(int)
        xs, ys, ls, ws = xs[idx], ys[idx], ls[idx], ws[idx]
        m = max_spots

    inv2pi = 1.0 / (2.0 * np.pi)
    dose = np.zeros(g.shape, dtype=dtype)
    for kz in range(g.nz):
        zc = g.oz + kz * g.dz
        idd_col = problem.idd[:, kz]
        sig_col = np.clip(problem.sigma[:, kz], 1e-9, None)
        for ky in range(g.ny):
            yc = g.oy + ky * g.dy
            for kx in range(g.nx):
                xc = g.ox + kx * g.dx
                acc = 0.0
                for mm in range(m):
                    l = ls[mm]
                    s = sig_col[l]
                    dx = xc - xs[mm]
                    dy = yc - ys[mm]
                    acc += (ws[mm] * idd_col[l] * inv2pi / (s * s)
                            * np.exp(-(dx * dx + dy * dy) / (2.0 * s * s)))
                dose[kz, ky, kx] = acc
    return dose


# ===========================================================================
# 实现 2：逐束流全网格累加
# ===========================================================================
def naive_numpy_per_beam(problem: PBProblem, max_spots: Optional[int] = None,
                         progress: bool = False) -> np.ndarray:
    """对每条 pencil beam 把其三维贡献加到整个网格（O(M*Nvox)，逐层 sigma 精确）。"""
    g = problem.grid
    xs, ys, ls, ws = problem.lattice.flat_spot_table()
    m = len(ws)
    if max_spots is not None and m > max_spots:
        idx = np.linspace(0, m - 1, max_spots).astype(int)
        xs, ys, ls, ws = xs[idx], ys[idx], ls[idx], ws[idx]
        m = max_spots

    X = g.x[None, None, :]
    Y = g.y[None, :, None]
    dose = np.zeros(g.shape, dtype=np.float64)
    t0 = time.perf_counter()
    for mm in range(m):
        l = ls[mm]
        sig = np.clip(problem.sigma[l], 1e-9, None)[:, None, None]
        gx = np.exp(-((X - xs[mm]) ** 2) / (2.0 * sig**2))
        gy = np.exp(-((Y - ys[mm]) ** 2) / (2.0 * sig**2))
        dose += ws[mm] * problem.idd[l][:, None, None] * gx * gy / (2.0 * np.pi * sig**2)
        if progress and (mm + 1) % max(1, m // 10) == 0:
            el = time.perf_counter() - t0
            print(f"    [{mm+1}/{m}] {el:7.2f} s  (预计总计 {el/(mm+1)*m:8.1f} s)")
    return dose


# ===========================================================================
# 实现 3：可分离矩阵形式（NumPy float64 参考真值）
# ===========================================================================
def separable_numpy(problem: PBProblem, per_layer: bool = True,
                    return_timing: bool = False):
    """
    NumPy 可分离实现（float64）。

    per_layer=True  : 逐能量层卷积后按 IDD 加权求和 —— **物理精确**
    per_layer=False : 用加权平均 sigma_bar(z) 的单核近似 —— 快速但近似
    """
    t0 = time.perf_counter()
    g, L = problem.grid, problem.lattice

    if not per_layer:
        A = (problem.idd.T @ L.ensure_weights().reshape(L.n_layers, -1)) \
            .reshape(g.nz, L.n_sy, L.n_sx)                    # (nz, nsy, nsx)
        sb = layerwise_sigma_bar(problem)
        Gx = lateral_kernel(g.x, L.spot_x, sb)                 # (nz, nx, nsx)
        Gy = lateral_kernel(g.y, L.spot_y, sb)                 # (nz, ny, nsy)
        T = np.einsum("zij,zvj->ziv", A, Gy)                   # (nz, nsx, ny)
        D = np.einsum("zai,ziy->zya", Gx, T).transpose(0, 2, 1)  # (nz, ny, nx)
    else:
        A = build_amplitude_map(problem)                       # (L, nz, nsy, nsx)
        Gx = lateral_kernel(g.x, L.spot_x, problem.sigma)      # (L, nz, nx, nsx)
        Gy = lateral_kernel(g.y, L.spot_y, problem.sigma)      # (L, nz, ny, nsy)
        T = np.einsum("zkji,zkvj->zkiv", A, Gy)                # (L, nz, nsx, ny)
        Dl = np.einsum("zkai,zkiy->zkay", Gx, T)               # (L, nz, nx, ny)
        D = Dl.sum(axis=0).transpose(0, 2, 1)                  # (nz, ny, nx)

    t1 = time.perf_counter()
    if return_timing:
        return D, t1 - t0
    return D


# ===========================================================================
# 实现 4：PyTorch 批矩阵乘法（CPU / GPU / TF32）
# ===========================================================================
def torch_separable(problem: PBProblem, device: str = "cuda", dtype=None,
                    allow_tf32: bool = False, per_layer: bool = True,
                    layer_chunk: Optional[int] = None, return_timing: bool = False,
                    return_intermediates: bool = False):
    """
    PyTorch 批矩阵乘法实现（把数万条 pencil beam 收缩为两次 bmm）。

    device        : 'cuda' / 'cpu'
    dtype         : torch.float32 / torch.float64
    allow_tf32    : 允许 TF32 张量核（速度 vs 精度，见性能报告）
    per_layer     : True 逐层精确；False 单 sigma 近似
    layer_chunk   : 每次处理的能量层数（控制显存峰值），None = 全部
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 不可用")
    if dtype is None:
        dtype = torch.float32 if str(device).startswith("cuda") else torch.float64

    t0 = time.perf_counter()
    g, L = problem.grid, problem.lattice
    nz, nsx, nsy, nx, ny = g.nz, L.n_sx, L.n_sy, g.nx, g.ny
    dev = torch.device(device)

    xv = torch.as_tensor(g.x, dtype=dtype, device=dev)
    yv = torch.as_tensor(g.y, dtype=dtype, device=dev)
    xs = torch.as_tensor(L.spot_x, dtype=dtype, device=dev)
    ys = torch.as_tensor(L.spot_y, dtype=dtype, device=dev)

    old_tf32 = None
    if str(device).startswith("cuda"):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)

    D = torch.zeros((nz, ny, nx), dtype=dtype, device=dev)
    try:
        if not per_layer:
            A = torch.as_tensor(
                (problem.idd.T @ L.ensure_weights().reshape(L.n_layers, -1))
                .reshape(nz, nsy, nsx), dtype=dtype, device=dev)
            sb = torch.as_tensor(layerwise_sigma_bar(problem), dtype=dtype, device=dev)
            s = sb.clamp_min(1e-9).view(nz, 1, 1)
            norm = INV_SQRT_2PI / s
            Gx = norm * torch.exp(-((xv.view(1, -1, 1) - xs.view(1, 1, -1)) ** 2)
                                  / (2.0 * s**2))
            Gy = norm * torch.exp(-((yv.view(1, -1, 1) - ys.view(1, 1, -1)) ** 2)
                                  / (2.0 * s**2))
            T = torch.bmm(A.transpose(1, 2), Gy.transpose(1, 2))     # (nz, nsx, ny)
            D = torch.bmm(Gx, T).transpose(1, 2).contiguous()        # (nz, ny, nx)
        else:
            w = torch.as_tensor(L.ensure_weights(), dtype=dtype, device=dev)   # (L,nsy,nsx)
            idd = torch.as_tensor(problem.idd, dtype=dtype, device=dev)        # (L,nz)
            sig = torch.as_tensor(problem.sigma, dtype=dtype, device=dev)      # (L,nz)
            step = layer_chunk or L.n_layers
            dx2 = (xv.view(1, 1, -1, 1) - xs.view(1, 1, 1, -1)) ** 2      # (1,1,nx,nsx)
            dy2 = (yv.view(1, 1, -1, 1) - ys.view(1, 1, 1, -1)) ** 2      # (1,1,ny,nsy)
            for l0 in range(0, L.n_layers, step):
                l1 = min(l0 + step, L.n_layers)
                ll = l1 - l0
                s4 = sig[l0:l1].clamp_min(1e-9).reshape(ll, nz, 1, 1)     # (ll,nz,1,1)
                norm = INV_SQRT_2PI / s4
                Gx = (norm * torch.exp(-dx2 / (2.0 * s4**2))).reshape(ll * nz, nx, nsx)
                Gy = (norm * torch.exp(-dy2 / (2.0 * s4**2))).reshape(ll * nz, ny, nsy)
                A = (idd[l0:l1, :, None, None] * w[l0:l1, None, :, :]).reshape(
                    ll * nz, nsy, nsx)
                T = torch.bmm(A.transpose(1, 2), Gy.transpose(1, 2))   # (ll*nz, nsx, ny)
                Dl = torch.bmm(Gx, T)                                  # (ll*nz, nx, ny)
                D += Dl.reshape(ll, nz, nx, ny).sum(0).transpose(1, 2)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    t1 = time.perf_counter()
    out = (D, t1 - t0) if return_timing else D
    return out


# ===========================================================================
# 实现 5：局部窗口 Gather
# ===========================================================================
def _local_window_indices(vox_axis, spot_axis, spacing, sigma_max, support_sigma,
                          max_k):
    k = int(np.ceil(support_sigma * sigma_max / spacing)) + 1
    k = max(1, min(k, max_k))
    off = np.arange(-k, k + 1)
    idx = np.round((vox_axis[:, None] - spot_axis[0]) / spacing).astype(np.int64) + off[None, :]
    valid = (idx >= 0) & (idx < len(spot_axis))
    return np.clip(idx, 0, len(spot_axis) - 1), valid


def torch_gather_local(problem: PBProblem, device: str = "cuda", dtype=None,
                       support_sigma: float = 4.0, return_timing: bool = False,
                       max_k: int = 64):
    """
    局部窗口 Gather：每个输出体素只累加其 ±support_sigma*sigma 邻域内的 beam。
    对应课题文档第四节"每个 Thread 负责计算单个体素内所有照射束的剂量叠加"，
    但利用高斯核紧支撑把每体素的 O(M) 降为 O(Kx*Ky)。
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 不可用")
    if dtype is None:
        dtype = torch.float32 if str(device).startswith("cuda") else torch.float64

    t0 = time.perf_counter()
    g, L = problem.grid, problem.lattice
    dev = torch.device(device)
    smax = float(problem.sigma.max())

    iy, vmy = _local_window_indices(g.y, L.spot_y, L.dy_spot, smax, support_sigma, max_k)
    ix, vmx = _local_window_indices(g.x, L.spot_x, L.dx_spot, smax, support_sigma, max_k)
    Ky, Kx = iy.shape[1], ix.shape[1]

    sig = torch.as_tensor(problem.sigma, dtype=dtype, device=dev).clamp_min(1e-9)  # (L,nz)
    w = torch.as_tensor(L.ensure_weights(), dtype=dtype, device=dev)               # (L,nsy,nsx)
    idd = torch.as_tensor(problem.idd, dtype=dtype, device=dev)                    # (L,nz)
    yv = torch.as_tensor(g.y, dtype=dtype, device=dev)
    xv = torch.as_tensor(g.x, dtype=dtype, device=dev)
    sy = torch.as_tensor(L.spot_y, dtype=dtype, device=dev)
    sx = torch.as_tensor(L.spot_x, dtype=dtype, device=dev)
    idx_y = torch.as_tensor(iy, device=dev)
    idx_x = torch.as_tensor(ix, device=dev)
    vmy_t = torch.as_tensor(vmy, dtype=dtype, device=dev)
    vmx_t = torch.as_tensor(vmx, dtype=dtype, device=dev)

    # 逐层累加（控制显存峰值：中间量 (nz, nx, Kx, ny) 约 40 MB/层）
    D = torch.zeros((g.nz, g.ny, g.nx), dtype=dtype, device=dev)
    sy_w = sy[idx_y].view(1, g.ny, Ky)                            # (1,ny,Ky)
    sx_w = sx[idx_x].view(1, g.nx, Kx)                            # (1,nx,Kx)
    vmy_v = vmy_t.view(1, g.ny, Ky)
    vmx_v = vmx_t.view(1, g.nx, Kx)
    for l in range(L.n_layers):
        s3 = sig[l].view(g.nz, 1, 1)
        norm = (INV_SQRT_2PI / sig[l]).view(g.nz, 1, 1)
        gy = norm * torch.exp(-((yv.view(1, -1, 1) - sy_w) ** 2)
                              / (2.0 * s3**2)) * vmy_v             # (nz, ny, Ky)
        gx = norm * torch.exp(-((xv.view(1, -1, 1) - sx_w) ** 2)
                              / (2.0 * s3**2)) * vmx_v             # (nz, nx, Kx)
        # pass 1: T[k,i,y] = sum_jj w[idx_y[y,jj], i] * gy[k,y,jj]
        wj = w[l][idx_y, :]                                       # (ny, Ky, nsx)
        T = torch.einsum("yji,kyj->kiy", wj, gy)                  # (nz, nsx, ny)
        T = T * idd[l].view(g.nz, 1, 1)
        # pass 2: Dl[k,y,x] = sum_ii T[k, idx_x[x,ii], y] * gx[k,x,ii]
        Ti = T[:, idx_x, :]                                       # (nz, nx, Kx, ny)
        D += torch.einsum("zxky,zxk->zyx", Ti, gx)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = (D.contiguous(), t1 - t0) if return_timing else D.contiguous()
    return out


# ===========================================================================
# 误差度量与归一化
# ===========================================================================
def relative_error(a, b, mask=None) -> dict:
    """计算 a 相对 b 的误差指标（a 待测, b 参考）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(b, dtype=bool)
    d = (a - b)[mask]
    ref_max = float(np.max(np.abs(b))) or 1.0
    return {
        "max_abs_err": float(np.abs(d).max()) if d.size else 0.0,
        "max_rel_err_global": float(np.abs(d).max() / ref_max) if d.size else 0.0,
        "rms_err": float(np.sqrt((d**2).mean())) if d.size else 0.0,
        "rms_rel_global": float(np.sqrt((d**2).mean()) / ref_max) if d.size else 0.0,
        "n_vox": int(mask.sum()),
    }


def normalize_dose(d, mode: str = "max", roi=None):
    """剂量归一化：'max' 全局最大值；'roi_max' ROI 内最大值。"""
    d = np.asarray(d, dtype=np.float64)
    if mode == "roi_max" and roi is not None:
        ref = float(d[roi].max())
    else:
        ref = float(d.max())
    return d / (ref if ref > 0 else 1.0), ref
