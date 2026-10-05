"""
pbdose.model — 计算域、束流点阵与 IMPT 问题定义
===============================================

坐标与索引约定
--------------
* 所有长度为 cm，能量 MeV。
* 剂量张量形状为 ``(nz, ny, nx)``，索引顺序 ``dose[k_z, k_y, k_x]``。
* 体素中心坐标:  x = ox + k_x * dx   (k_x = 0..nx-1)
* 束流点（spot）位于横向规则点阵上，允许与体素网格不对齐（任意偏移），
  以保证临床 5 mm 点间距 / 2 mm 体素 的真实配置可以被精确建模。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import physics as ph


# ---------------------------------------------------------------------------
# 剂量网格
# ---------------------------------------------------------------------------
@dataclass
class DoseGrid:
    """三维规则体素网格。"""
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
    def shape(self) -> tuple:
        return (self.nz, self.ny, self.nx)

    @property
    def n_vox(self) -> int:
        return self.nx * self.ny * self.nz

    def axis(self, name: str) -> np.ndarray:
        n, d, o = {"x": (self.nx, self.dx, self.ox),
                   "y": (self.ny, self.dy, self.oy),
                   "z": (self.nz, self.dz, self.oz)}[name]
        return o + np.arange(n) * d

    @property
    def x(self) -> np.ndarray:
        return self.axis("x")

    @property
    def y(self) -> np.ndarray:
        return self.axis("y")

    @property
    def z(self) -> np.ndarray:
        return self.axis("z")

    def voxel_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def as_dict(self) -> dict:
        return dict(nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy,
                    dz=self.dz, ox=self.ox, oy=self.oy, oz=self.oz)

    @staticmethod
    def centered(n: int, d: float, axis: str = "z") -> "DoseGrid":
        """生成以原点为中心的立方网格（z 从 0 开始，入射面在 z=0 平面）。"""
        half = (n - 1) * d / 2.0
        if axis == "z":
            return DoseGrid(n, n, n, d, d, d, -half, -half, 0.0)
        return DoseGrid(n, n, n, d, d, d, -half, -half, -half)


# ---------------------------------------------------------------------------
# 束流点阵（IMP T 能量层 + 横向点阵）
# ---------------------------------------------------------------------------
@dataclass
class SpotLattice:
    """
    IMPT 束流点阵。

    weights[l, j, i] 为第 l 能量层、横向索引 (i, j) 处 spot 的权重（MU / 监测单元）。
    """
    n_sx: int
    n_sy: int
    n_layers: int
    dx_spot: float
    dy_spot: float
    x0: float = 0.0
    y0: float = 0.0
    energies: Optional[np.ndarray] = None          # (n_layers,) MeV
    weights: Optional[np.ndarray] = None           # (n_layers, n_sy, n_sx)

    @property
    def n_spots(self) -> int:
        return self.n_sx * self.n_sy * self.n_layers

    @property
    def spot_x(self) -> np.ndarray:
        """(n_sx,) 横向 x 坐标。"""
        return self.x0 + (np.arange(self.n_sx) - (self.n_sx - 1) / 2.0) * self.dx_spot

    @property
    def spot_y(self) -> np.ndarray:
        return self.y0 + (np.arange(self.n_sy) - (self.n_sy - 1) / 2.0) * self.dy_spot

    def ensure_weights(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.weights is None:
            rng = rng or np.random.default_rng(0)
            self.weights = rng.random((self.n_layers, self.n_sy, self.n_sx))
        return self.weights

    def flat_spot_table(self) -> tuple:
        """
        展平为 (M, 4) 表: x[cm], y[cm], layer, weight。M = n_layers*n_sy*n_sx。
        """
        w = self.ensure_weights()
        sx, sy = self.spot_x, self.spot_y
        X, Y = np.meshgrid(sx, sy)                     # (n_sy, n_sx)
        X = np.broadcast_to(X, (self.n_layers, self.n_sy, self.n_sx))
        Y = np.broadcast_to(Y, (self.n_layers, self.n_sy, self.n_sx))
        L = np.broadcast_to(np.arange(self.n_layers)[:, None, None], w.shape)
        return (X.ravel().copy(), Y.ravel().copy(), L.ravel().copy(), w.ravel().copy())


# ---------------------------------------------------------------------------
# 完整问题：网格 + 点阵 + 预先算好的 IDD / sigma 查找表
# ---------------------------------------------------------------------------
@dataclass
class PBProblem:
    """
    Pencil Beam 剂量重构问题。

    idd[l, k]    : 第 l 层在深度 z[k] 处的 IDD。
                   **fluence_normalized=True 时为物理的"每单位注量深度剂量"
                   [MeV/cm]（推荐，与真实 IMPT 的注量型 spot 权重一致）；
                   否则为逐层归一化到峰值 1 的曲线。**
    sigma[l, k]  : 第 l 层在深度 z[k] 处的侧向 rms 宽度 [cm]
    """
    grid: DoseGrid
    lattice: SpotLattice
    idd: np.ndarray                     # (n_layers, nz)
    sigma: np.ndarray                   # (n_layers, nz)
    meta: dict = field(default_factory=dict)
    fluence_normalized: bool = True

    @property
    def n_layers(self) -> int:
        return self.lattice.n_layers

    @property
    def nz(self) -> int:
        return self.grid.nz

    def summary(self) -> str:
        m = self.meta
        lines = [
            "=" * 74,
            "Pencil Beam 剂量重构问题",
            "=" * 74,
            f"  体素网格      : {self.grid.nx} x {self.grid.ny} x {self.grid.nz}"
            f"  ({self.grid.n_vox/1e6:.2f} M voxels, {self.grid.dx*10:.1f} mm 体素)",
            f"  物理尺寸      : {self.grid.nx*self.grid.dx:.1f} x "
            f"{self.grid.ny*self.grid.dy:.1f} x {self.grid.nz*self.grid.dz:.1f} cm",
            f"  横向点阵      : {self.lattice.n_sx} x {self.lattice.n_sy}"
            f"  间距 {self.lattice.dx_spot*10:.1f} mm",
            f"  能量层数      : {self.lattice.n_layers}",
            f"  Pencil Beam 数: {self.lattice.n_spots:,} (M)",
            f"  束流能量范围  : {self.lattice.energies.min():.1f} - "
            f"{self.lattice.energies.max():.1f} MeV",
            f"  CSDA 射程范围 : {ph.csda_range(self.lattice.energies.min()):.2f} - "
            f"{ph.csda_range(self.lattice.energies.max()):.2f} cm",
            f"  Naive 复杂度  : M * Nx * Ny * Nz = "
            f"{self.lattice.n_spots * self.grid.n_vox/1e9:.1f} e9 次乘加",
        ]
        for k, v in m.items():
            lines.append(f"  {k:<14}: {v}")
        lines.append("=" * 74)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def gaussian_impt_field(
    energies: np.ndarray,
    grid: DoseGrid | None = None,
    *,
    n_spot: int = 41,
    spot_spacing: float = 0.5,          # cm  (5 mm 临床点间距)
    spot_sigma: float = 0.28,           # cm  横向权重包络
    seed: int = 0,
    idd_model: ph.IDDModel = ph.DEFAULT_IDD,
    beam_sigma_x0: float = 0.30,        # cm  入射束横向 rms 尺寸（喷嘴/准直）
    beam_sigma_theta0: float = 0.0,     # rad 入射束发散
    fluence_normalized: bool = True,
) -> PBProblem:
    """
    构造一个"临床风格"的 IMPT 场：
      * 横向束流权重为一个二维高斯包络（模拟一个适形野的横向展宽）
      * 各能量层权重略作随机扰动（模拟 IMPT 优化后的非均匀权重）
    """
    grid = grid or DoseGrid.centered(128, 0.2)
    n_layers = len(energies)
    rng = np.random.default_rng(seed)

    sx = np.arange(n_spot) - (n_spot - 1) / 2.0
    X, Y = np.meshgrid(sx * spot_spacing, sx * spot_spacing)
    env = np.exp(-(X**2 + Y**2) / (2.0 * spot_sigma**2))
    env /= env.max()
    w = env[None, :, :] * (1.0 + 0.25 * rng.standard_normal((n_layers, n_spot, n_spot)))
    w = np.clip(w, 0.0, None)

    lat = SpotLattice(n_spot, n_spot, n_layers, spot_spacing, spot_spacing,
                      energies=np.asarray(energies, dtype=np.float64), weights=w)

    z = grid.z
    idd = np.zeros((n_layers, grid.nz))
    sig = np.zeros((n_layers, grid.nz))
    for l, E in enumerate(lat.energies):
        idd[l], _ = ph.idd_pristine(z, E, idd_model,
                                    normalize=not fluence_normalized)
        mom = ph.fermi_eyges_moments(z, E, sigma_x0=beam_sigma_x0,
                                     sigma_theta0=beam_sigma_theta0)
        sig[l] = mom["sigma_x"]

    meta = {
        "IDD 模型": f"CSDA 能量守恒分箱 + 射程歧离 sigma_R=0.012R^0.935 + "
                    f"能量展宽 {idd_model.energy_spread*100:.2f}%",
        "注量衰减长度": f"{idd_model.attenuation_length:.0f} cm",
        "入射束 sigma_x0": f"{beam_sigma_x0*10:.1f} mm",
        "横向权重包络": f"Gaussian sigma={spot_sigma*10:.1f} mm",
        "权重类型": "注量型 (fluence-normalized IDD)" if fluence_normalized
                    else "峰值剂量型 (peak-normalized IDD)",
    }
    return PBProblem(grid=grid, lattice=lat, idd=idd, sigma=sig, meta=meta,
                     fluence_normalized=fluence_normalized)


def clinical_layer_energies(n_layers: int = 40,
                            r_min: float = 3.0, r_max: float = 25.5) -> np.ndarray:
    """
    生成等射程间隔的能量层（临床常用分层方式）。
    返回按射程升序排列的能量 [MeV]。
    """
    R = np.linspace(r_min, r_max, n_layers)
    return ph.energy_from_range(R)


def flat_field_weights(n_sx: int, n_sy: int, n_layers: int, *,
                       field_half_width: float = 5.0, spot_spacing: float = 0.5,
                       edge: float = 0.8, radial_falloff: float = 0.06,
                       layer_jitter: float = 0.15, seed: int = 0) -> np.ndarray:
    """
    构造"临床风格"的 IMPT 横向权重：**平顶野 + 平滑边缘**。

    真实 IMPT 计划中，束流权重在靶区横向上近似平坦（平顶），边缘由计划系统
    优化出的陡降构成，而不是像高斯包络那样从中心单调衰减。平顶野才能产生
    足够大的高剂量区，使 Gamma 统计有意义。

    参数
    ----
    field_half_width : 半高宽 [cm]（权重 = 1 的区域边界）
    edge             : 边缘过渡宽度 [cm]（tanh 过渡，模拟优化出的梯度）
    radial_falloff   : 边缘外侧的额外指数衰减尺度 [cm]
    layer_jitter     : 逐层随机扰动幅度（模拟 IMPT 逐层权重优化结果）

    返回 (n_layers, n_sy, n_sx)
    """
    rng = np.random.default_rng(seed)
    sx = (np.arange(n_sx) - (n_sx - 1) / 2.0) * spot_spacing
    X, Y = np.meshgrid(sx, sx)
    R = np.hypot(X, Y)
    w2 = 0.5 * (1.0 - np.tanh((R - field_half_width) / max(edge, 1e-6)))
    w2 = w2 * np.exp(-np.clip(R - field_half_width, 0, None) / max(radial_falloff, 1e-6))
    w = np.broadcast_to(w2, (n_layers, n_sy, n_sx)).copy()
    if layer_jitter > 0:
        w *= 1.0 + layer_jitter * rng.standard_normal(w.shape)
    return np.clip(w, 0.0, None)


def flat_impt_field(
    energies: np.ndarray,
    grid: DoseGrid | None = None,
    *,
    n_spot: int = 41,
    spot_spacing: float = 0.5,
    field_half_width: float = 5.0,
    seed: int = 0,
    idd_model: ph.IDDModel = ph.DEFAULT_IDD,
    beam_sigma_x0: float = 0.30,
    beam_sigma_theta0: float = 0.0,
    fluence_normalized: bool = True,
) -> PBProblem:
    """平顶横截面的 IMPT 场（用于精度验证，见 :func:`flat_field_weights`）。"""
    grid = grid or DoseGrid.centered(128, 0.2)
    energies = np.asarray(energies, dtype=np.float64)
    n_layers = len(energies)
    w = flat_field_weights(n_spot, n_spot, n_layers,
                           field_half_width=field_half_width,
                           spot_spacing=spot_spacing, seed=seed)
    lat = SpotLattice(n_spot, n_spot, n_layers, spot_spacing, spot_spacing,
                      energies=energies, weights=w)
    z = grid.z
    idd = np.zeros((n_layers, grid.nz))
    sig = np.zeros((n_layers, grid.nz))
    for l, E in enumerate(energies):
        idd[l], _ = ph.idd_pristine(z, E, idd_model,
                                    normalize=not fluence_normalized)
        mom = ph.fermi_eyges_moments(z, E, sigma_x0=beam_sigma_x0,
                                     sigma_theta0=beam_sigma_theta0)
        sig[l] = mom["sigma_x"]
    meta = {
        "横截面": f"平顶 {2*field_half_width:.1f} x {2*field_half_width:.1f} cm",
        "IDD 模型": f"{idd_model.straggling_model} 射程歧离, "
                    f"能量展宽 {idd_model.energy_spread*100:.2f}%",
        "入射束 sigma_x0": f"{beam_sigma_x0*10:.1f} mm",
        "权重类型": "注量型 (fluence-normalized IDD)" if fluence_normalized
                    else "峰值剂量型 (peak-normalized IDD)",
    }
    return PBProblem(grid=grid, lattice=lat, idd=idd, sigma=sig, meta=meta,
                     fluence_normalized=fluence_normalized)


def single_spot_problem(E0: float = 200.0, grid: DoseGrid | None = None,
                        n_spot: int = 1, spot_spacing: float = 0.5,
                        beam_sigma_x0: float = 0.30) -> PBProblem:
    """单点（或小点阵）问题——用于横向剖面对比验证。"""
    grid = grid or DoseGrid(256, 256, 300, 0.1, 0.1, 0.1, -12.8, -12.8, 0.0)
    lat = SpotLattice(n_spot, n_spot, 1, spot_spacing, spot_spacing,
                      energies=np.array([E0]), weights=np.ones((1, n_spot, n_spot)))
    z = grid.z
    idd, _ = ph.idd_pristine(z, E0)
    mom = ph.fermi_eyges_moments(z, E0, sigma_x0=beam_sigma_x0)
    return PBProblem(grid, lat, idd[None, :], mom["sigma_x"][None, :],
                     meta={"E0": f"{E0} MeV"})
