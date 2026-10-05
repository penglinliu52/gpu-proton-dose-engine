"""
pbdose.gamma — Gamma Index 分析（Low et al. 1998 / AAPM TG-218）
================================================================

临床剂量验证的黄金标准。对参考分布中的每个体素 r_ref：

    Gamma(r_ref) = min_{r_eval} sqrt( ( |r_eval - r_ref| / DTA )^2
                                      + ( (D_eval - D_ref) / D_crit )^2 )

判据：Gamma <= 1 视为通过。临床验收标准（TG-218 建议）
    * 3%/3mm 全局归一化 + 10% 剂量阈值：通过率 > 95%（本项目目标 > 98%）
    * 同时报告 2%/2mm 作为更严格的自检

实现要点
--------
* 全局归一化（global）：D_crit = dose_crit_pct * (参考分布的最大剂量或处方剂量)，
  这是 TG-218 推荐的临床做法；局域归一化（local）在高剂量梯度区会把低剂量点的
  容差压得极低，产生"假失败"，本模块默认使用 global。
* 10% 低剂量阈值：喉部/空气等区域的剂量统计噪声大且无临床意义，标准做法是排除。
* DTA 搜索球：仅遍历体素间距下欧氏距离 <= DTA 的偏移，保证 O(N * K) 复杂度。
* 支持 float64 numpy 与 torch(GPU) 两条路径，后者用于大网格的快速统计。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# DTA 搜索偏移
# ---------------------------------------------------------------------------
def dta_offsets(spacing, dta_cm: float, max_vox: int = 6):
    """
    返回所有满足 |offset| <= dta_cm 的整数体素偏移及其物理距离 [cm]。

    spacing: (dz, dy, dx) 体素尺寸 [cm]
    """
    dz, dy, dx = spacing
    nz = int(np.floor(dta_cm / dz + 1e-9))
    ny = int(np.floor(dta_cm / dy + 1e-9))
    nx = int(np.floor(dta_cm / dx + 1e-9))
    nz, ny, nx = (min(nz, max_vox), min(ny, max_vox), min(nx, max_vox))
    offs, dists = [], []
    for a in range(-nz, nz + 1):
        for b in range(-ny, ny + 1):
            for c in range(-nx, nx + 1):
                d = np.sqrt((a * dz) ** 2 + (b * dy) ** 2 + (c * dx) ** 2)
                if d <= dta_cm + 1e-12:
                    offs.append((a, b, c))
                    dists.append(d)
    return np.array(offs, dtype=np.int64), np.array(dists, dtype=np.float64)


def _shift(a: np.ndarray, off) -> np.ndarray:
    """把数组 a 平移 off（越界补 0），返回 same-shape 数组。"""
    a_ = a
    oz, oy, ox = off
    out = np.zeros_like(a_)
    sz = slice(max(0, oz), a_.shape[0] + min(0, oz))
    dz_ = slice(max(0, -oz), a_.shape[0] + min(0, -oz))
    sy = slice(max(0, oy), a_.shape[1] + min(0, oy))
    dy_ = slice(max(0, -oy), a_.shape[1] + min(0, -oy))
    sx = slice(max(0, ox), a_.shape[2] + min(0, ox))
    dx_ = slice(max(0, -ox), a_.shape[2] + min(0, -ox))
    out[dz_, dy_, dx_] = a_[sz, sy, sx]
    return out


# ---------------------------------------------------------------------------
# Gamma Index
# ---------------------------------------------------------------------------
def gamma_index(reference: np.ndarray, evaluation: np.ndarray,
                spacing=(0.2, 0.2, 0.2),
                dose_crit_pct: float = 3.0, dta_mm: float = 3.0,
                threshold_pct: float = 10.0,
                normalization: str = "global",
                ref_dose: Optional[float] = None,
                return_maps: bool = True) -> dict:
    """
    计算 Gamma Index。

    参数
    ----
    reference, evaluation : (nz, ny, nx) 剂量分布（同一网格）
    spacing               : (dz, dy, dx) [cm]
    dose_crit_pct         : 剂量判据百分比（3.0 表示 3%）
    dta_mm                : 距离一致性判据 [mm]
    threshold_pct         : 低剂量阈值（相对最大剂量）
    normalization         : 'global' 或 'local'
    ref_dose              : 归一化参考剂量（默认为 reference.max()）

    返回
    ----
    dict: gamma 图, pass_rate, 各项统计
    """
    ref = np.asarray(reference, dtype=np.float64)
    ev = np.asarray(evaluation, dtype=np.float64)
    if ref.shape != ev.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {ev.shape}")

    dta_cm = dta_mm / 10.0
    ref_max = float(ref.max()) if ref_dose is None else float(ref_dose)
    if ref_max <= 0:
        raise ValueError("reference dose max <= 0")

    # --- 剂量判据 ---
    if normalization == "global":
        dose_crit = dose_crit_pct / 100.0 * ref_max * np.ones_like(ref)
    else:  # local
        dose_crit = dose_crit_pct / 100.0 * np.maximum(ref, 1e-12)

    # --- 剂量阈值掩码（TG-218：通常取 10% 最大剂量） ---
    mask = ref >= (threshold_pct / 100.0) * ref_max
    if mask.sum() == 0:
        mask = ref > 0

    offs, dists = dta_offsets(spacing, dta_cm)
    gamma2 = np.full(ref.shape, np.inf, dtype=np.float64)
    for off, dist in zip(offs, dists):
        ev_shift = _shift(ev, tuple(off))
        dd = (ev_shift - ref) / dose_crit
        dr = dist / dta_cm
        g2 = dr * dr + dd * dd
        np.minimum(gamma2, g2, out=gamma2)

    gamma = np.sqrt(gamma2)
    g_masked = gamma[mask]
    passed = g_masked <= 1.0
    pass_rate = float(passed.mean() * 100.0) if g_masked.size else float("nan")

    res = {
        "pass_rate": pass_rate,
        "gamma_mean": float(g_masked.mean()),
        "gamma_max": float(g_masked.max()),
        "gamma_p95": float(np.percentile(g_masked, 95)),
        "gamma_p99": float(np.percentile(g_masked, 99)),
        "n_eval": int(g_masked.size),
        "n_pass": int(passed.sum()),
        "dose_crit_pct": dose_crit_pct,
        "dta_mm": dta_mm,
        "threshold_pct": threshold_pct,
        "normalization": normalization,
        "ref_max": ref_max,
        "n_dta_offsets": len(offs),
    }
    if return_maps:
        res["gamma"] = gamma
        res["mask"] = mask
        res["ref"] = ref
        res["eval"] = ev
    return res


# ---------------------------------------------------------------------------
# 剂量学对比指标
# ---------------------------------------------------------------------------
def dose_difference(reference, evaluation, threshold_pct: float = 10.0,
                    normalization: str = "global") -> dict:
    """剂量差异图与统计（相对最大剂量）。"""
    ref = np.asarray(reference, dtype=np.float64)
    ev = np.asarray(evaluation, dtype=np.float64)
    ref_max = float(ref.max())
    mask = ref >= threshold_pct / 100.0 * ref_max
    diff = ev - ref
    if normalization == "global":
        rel = diff / ref_max * 100.0
    else:
        rel = diff / np.maximum(ref, 1e-12) * 100.0
    return {
        "diff": diff, "rel_pct": rel, "mask": mask,
        "mean_rel_pct": float(rel[mask].mean()),
        "rms_rel_pct": float(np.sqrt((rel[mask] ** 2).mean())),
        "max_abs_rel_pct": float(np.abs(rel[mask]).max()),
        "p95_abs_rel_pct": float(np.percentile(np.abs(rel[mask]), 95)),
    }


def depth_profile(dose, z_axis, x_idx, y_idx, axis: str = "z"):
    """抽取一维剖面（用于 IDD / 横向剖面对比图）。"""
    d = np.asarray(dose, dtype=np.float64)
    if axis == "z":
        return z_axis.copy(), d[:, y_idx, x_idx]
    if axis == "y":
        return None, d[z_axis, :, x_idx]
    return None, d[z_axis, y_idx, :]


def gamma_pass_rate_vs_criteria(reference, evaluation, spacing=(0.2, 0.2, 0.2),
                                criteria=((3.0, 3.0), (2.0, 2.0), (1.0, 1.0),
                                          (3.0, 2.0), (2.0, 3.0)),
                                threshold_pct: float = 10.0) -> list:
    """扫描多组 (dose%, DTA mm) 判据，返回通过率列表（用于敏感性分析表）。"""
    out = []
    for dp, dt in criteria:
        r = gamma_index(reference, evaluation, spacing, dp, dt,
                        threshold_pct=threshold_pct, return_maps=False)
        out.append({"dose_pct": dp, "dta_mm": dt, **{k: v for k, v in r.items()}})
    return out


# ---------------------------------------------------------------------------
# GPU 版本（大网格加速）
# ---------------------------------------------------------------------------
def gamma_index_torch(reference, evaluation, spacing=(0.2, 0.2, 0.2),
                      dose_crit_pct: float = 3.0, dta_mm: float = 3.0,
                      threshold_pct: float = 10.0, ref_dose=None,
                      device: str = "cuda", threshold: float = 1e-6) -> dict:
    """
    GPU 版 Gamma Index。返回 numpy 结果字典（不含 gamma 图，除非请求）。
    与 numpy 版算法完全一致，用于大网格的快速统计。
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 不可用")
    dev = torch.device(device)
    ref = torch.as_tensor(np.asarray(reference, dtype=np.float64),
                          dtype=torch.float32, device=dev)
    ev = torch.as_tensor(np.asarray(evaluation, dtype=np.float64),
                         dtype=torch.float32, device=dev)
    ref_max = float(ref.max().item()) if ref_dose is None else float(ref_dose)
    dose_crit = dose_crit_pct / 100.0 * ref_max
    mask = ref >= (threshold_pct / 100.0) * ref_max

    offs, dists = dta_offsets(spacing, dta_mm / 10.0)
    g2 = torch.full(ref.shape, float("inf"), dtype=torch.float32, device=dev)
    for off, dist in zip(offs, dists):
        ev_s = torch.zeros_like(ev)
        oz, oy, ox = off
        sz = slice(max(0, oz), ev.shape[0] + min(0, oz))
        dz_ = slice(max(0, -oz), ev.shape[0] + min(0, -oz))
        sy = slice(max(0, oy), ev.shape[1] + min(0, oy))
        dy_ = slice(max(0, -oy), ev.shape[1] + min(0, -oy))
        sx = slice(max(0, ox), ev.shape[2] + min(0, ox))
        dx_ = slice(max(0, -ox), ev.shape[2] + min(0, -ox))
        ev_s[dz_, dy_, dx_] = ev[sz, sy, sx]
        dd = (ev_s - ref) / dose_crit
        dr = dist / (dta_mm / 10.0)
        torch.minimum(g2, dr * dr + dd * dd, out=g2)

    g = torch.sqrt(g2)[mask]
    pass_rate = float((g <= 1.0).float().mean().item() * 100.0)
    q = torch.tensor([0.95, 0.99], device=dev)
    pct = torch.quantile(g, q).cpu().numpy()
    return {
        "pass_rate": pass_rate,
        "gamma_mean": float(g.mean().item()),
        "gamma_max": float(g.max().item()),
        "gamma_p95": float(pct[0]), "gamma_p99": float(pct[1]),
        "n_eval": int(g.numel()), "n_pass": int((g <= 1.0).sum().item()),
        "dose_crit_pct": dose_crit_pct, "dta_mm": dta_mm,
        "threshold_pct": threshold_pct, "normalization": "global",
        "ref_max": ref_max, "n_dta_offsets": len(offs),
    }
