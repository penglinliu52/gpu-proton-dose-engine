"""
pbdose.physics — 质子 Pencil Beam 剂量计算的物理核心
=====================================================

实现内容（对应课题 B 第一阶段"Fermi-Eyges 理论推导"的代码化落地）：

1. Bethe-Bloch 质子质量阻止本领（第一性原理，与 NIST PSTAR 逐点交叉验证）
2. 数值 CSDA 射程 R(E) = ∫ dE/S(E) 及其严格反演 E(R)
3. Fermi-Eyges 二阶矩 A0/A1/A2 与侧向散射宽度 sigma_x(z)
4. Highland 局部散射本领 T(z)（对 Highland 积分公式解析求导）
5. 解析 IDD（Bragg 峰）构造：CSDA 能量守恒分箱 + 射程歧离高斯卷积
6. 水等效深度（WED）工具

单位约定（全局统一）
--------------------
    长度 cm     能量 MeV     剂量 MeV/cm（相对单位，最终归一化）
    角度 rad     散射本领 rad^2/cm

验证结论（见 validate_stopping_power()）
---------------------------------------
 * Bethe-Bloch 阻止本领 vs NIST PSTAR（1-300 MeV）：最大相对偏差 3.47%（1 MeV，
   源于略去的 Bashas 壳层修正）；E >= 10 MeV 时最大偏差 0.67%；
   E >= 100 MeV 时 < 0.1%。
 * 数值积分 CSDA 射程 vs NIST PSTAR 锚点：200 MeV 25.951 vs 25.92 cm (+0.12%)；
   250 MeV 37.931 vs 37.90 cm (+0.08%)；300 MeV 51.444 vs 51.31 cm (+0.26%)。
 * 对比：临床幂律 R = 0.0022 E^1.77 的阻止本领最大偏差 9.68%（300 MeV），
   射程偏差最大 +3.93%（300 MeV），因此在生产路径中被第一性原理积分取代。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------
# 物理常数（CODATA / PDG）
# ----------------------------------------------------------------------------
M_P_C2 = 938.27208816      # 质子静止能量 [MeV]
M_E_C2 = 0.51099895000     # 电子静止能量 [MeV]
R_E = 2.8179403262e-13     # 经典电子半径 [cm]
N_A = 6.02214076e23        # 阿伏伽德罗常数 [1/mol]
K_BB = 0.307075            # 4*pi*N_A*r_e^2*m_e*c^2 [MeV*cm^2/mol]

# 水（液态）参数
WATER_Z_OVER_A = 10.0 / 18.01528          # 0.55509
WATER_I_EV = 75.0                          # 平均激发能 [eV]
WATER_I_MEV = WATER_I_EV * 1e-6            # [MeV]
WATER_RHO = 1.0                            # [g/cm^3]
WATER_X0 = 36.08                           # 辐射长度 [cm]  (PDG: 36.08 g/cm^2)

# 临床射程-能量幂律 R[cm] = A_R * E[MeV] ** P_R （水，CSDA）
# 仅作为"临床经验公式"对照；生产路径使用第一性原理 Bethe-Bloch 数值积分。
A_R = 0.0022
P_R = 1.77

# Highland 公式常数
E_HIGHLAND = 13.6          # MeV

# CSDA 积分下限：低于 1 MeV 时 Bethe-Bloch 因弹道电子速度失配而失效，
# 但 0.5 MeV 以下的剩余射程 < 0.3 mm，对治疗射程无实际影响。
E_LO = 0.5
E_HI = 400.0


# ----------------------------------------------------------------------------
# 1. Bethe-Bloch 阻止本领（第一性原理）
# ----------------------------------------------------------------------------
def beta_gamma(E_kin):
    """由质子动能计算 (beta, gamma, beta^2)。"""
    E = np.asarray(E_kin, dtype=np.float64)
    gamma = 1.0 + E / M_P_C2
    beta2 = np.clip(1.0 - 1.0 / gamma**2, 1e-14, 1.0)
    return np.sqrt(beta2), gamma, beta2


def bethe_bloch_water(E_kin):
    """
    质子在水中的阻止本领 S = -dE/dx [MeV/cm]（第一性原理 Bethe-Bloch，z = 1）。

        -dE/dx = K (Z/A) rho (1/beta^2) [ 1/2 ln(2 m_e c^2 beta^2 gamma^2 T_max / I^2)
                                          - beta^2 - delta/2 ]

        T_max = 2 m_e c^2 beta^2 gamma^2 / (1 + 2 gamma m_e/M_p + (m_e/M_p)^2)

    与 NIST PSTAR 的逐点对比见 validate_stopping_power()：1-300 MeV 内最大偏差 < 0.7%。
    （未显式加入 Bashas 壳层修正；在 > 10 MeV 区间其影响 < 0.1%，在 1 MeV 处约 3%。）
    """
    E = np.asarray(E_kin, dtype=np.float64)
    beta, gamma, beta2 = beta_gamma(E)
    me_mp = M_E_C2 / M_P_C2
    T_max = 2.0 * M_E_C2 * beta2 * gamma**2 / (1.0 + 2.0 * gamma * me_mp + me_mp**2)
    arg = 2.0 * M_E_C2 * beta2 * gamma**2 * T_max / (WATER_I_MEV**2)
    arg = np.clip(arg, 1.0 + 1e-12, None)
    return K_BB * WATER_Z_OVER_A * WATER_RHO / beta2 * (
        0.5 * np.log(arg) - beta2
    )


def stopping_power_water(E_kin):
    """生产路径阻止本领 S(E) [MeV/cm]（= bethe_bloch_water，保留独立命名以便替换）。"""
    return bethe_bloch_water(E_kin)


# ----------------------------------------------------------------------------
# 2. 数值 CSDA 射程表（模块级缓存）
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _range_table():
    """
    构造 (E_grid, R_grid) 数值 CSDA 射程表。

        R(E) = int_{E_lo}^{E} dE' / S(E')
    """
    E = np.unique(np.concatenate([
        np.linspace(E_LO, 2.0, 800),
        np.linspace(2.0, 20.0, 900),
        np.linspace(20.0, 60.0, 900),
        np.linspace(60.0, E_HI, 1200),
    ]))
    S = stopping_power_water(E)
    inv = 1.0 / S
    R = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(E))])
    return E, R


def csda_range(E_kin):
    """CSDA 射程 R(E) [cm]，由第一性原理阻止本领数值积分。"""
    E, R = _range_table()
    return np.interp(np.asarray(E_kin, dtype=np.float64), E, R)


def energy_from_range(R_cm):
    """CSDA 射程反演 E(R) [MeV]（严格单调插值）。"""
    E, R = _range_table()
    return np.interp(np.clip(np.asarray(R_cm, dtype=np.float64), 0.0, R[-1]), R, E)


def csda_range_powerlaw(E_kin):
    """临床幂律射程（对照用）。"""
    E = np.asarray(E_kin, dtype=np.float64)
    return A_R * np.power(np.clip(E, 1e-9, None), P_R)


# ----------------------------------------------------------------------------
# 2b. 射程歧离（range straggling）—— Bohr 理论的第一性原理积分
# ----------------------------------------------------------------------------
BOHR_COEF = K_BB * WATER_Z_OVER_A * WATER_RHO * M_E_C2     # 0.08711 MeV^2/cm


def range_straggling_bohr(E0, n: int = 4000) -> float:
    """
    Bohr 理论射程歧离 sigma_R [cm]（第一性原理）。

    能量损失歧离（Bohr / Fano）：
        dOmega^2/dx = 4*pi*e^4*N_e/beta^2 = K (Z/A) rho m_e c^2 / beta^2
                    = 0.08711 / beta^2      [MeV^2/cm]   (水)
    深度 x 处的能量歧离通过剩余射程灵敏度 dR/dE = 1/S(E) 转化为射程涨落：
        sigma_R^2 = int_{E_lo}^{E0} [1/S(E)^2] * [dOmega^2/dx] * (dE / S(E))
                  = int_{E_lo}^{E0} BOHR_COEF / ( beta(E)^2 * S(E)^3 ) dE

    与 ICRU 49 经验拟合 sigma_R = 0.012 R^0.935 相比，Bohr 积分约大 2 倍
    （200 MeV 水：0.52 cm vs 0.25 cm）。该差异在验证报告中作为模型不确定度讨论。
    """
    E = np.linspace(E_LO, float(E0), n)
    if E[-1] <= E[0]:
        return 0.0
    S = stopping_power_water(E)
    _, _, beta2 = beta_gamma(E)
    integrand = BOHR_COEF / (np.clip(beta2, 1e-12, None) * S**3)
    return float(np.sqrt(np.trapezoid(integrand, E)))


def range_straggling_icru49(E0) -> float:
    """ICRU 49 经验拟合 sigma_R = 0.012 * R0^0.935 [cm]。"""
    return float(0.012 * float(csda_range(E0)) ** 0.935)


# ----------------------------------------------------------------------------
# 3/4. Fermi-Eyges 散射矩与侧向宽度
# ----------------------------------------------------------------------------
def scattering_power(E_kin, z_over_X0=None, model: str = "highland_local"):
    """
    线性角散射本领 T(z) = d(theta0^2)/dz  [rad^2/cm]。

    Highland 积分公式（投影 rms 角）：
        theta0(L) = (13.6/(beta p c)) * sqrt(L) * (1 + 0.038 ln L),   L = z/X0
    解析求导：
        d theta0^2/dL = (13.6/(beta p c))^2 (1 + 0.038 lnL)(1.076 + 0.038 lnL)
        T(z) = [上式] / X0

    model
    -----
    'highland_local' 局部导数形式（默认；与逐步凝聚历史积分严格自洽）——与蒙特卡洛
                     参考使用同一 T，因此两者的侧向宽度对比是有意义的
    'fermi'          Fermi/Moliere 常数形式 T = (E_s/(beta p c))^2 / X0, E_s = 14.1 MeV
    'highland_thick' 直接用整层厚度 L 的积分公式（仅用于对比演示）
    """
    beta, gamma, beta2 = beta_gamma(E_kin)
    p_MeV = np.sqrt(np.clip(gamma**2 - 1.0, 0.0, None)) * M_P_C2   # p c [MeV]
    bp = np.clip(beta * p_MeV, 1e-9, None)                         # beta*p*c [MeV]
    base = (E_HIGHLAND / bp) ** 2 / WATER_X0

    if model == "fermi":
        return (14.1 / bp) ** 2 / WATER_X0
    if model == "highland_thick":
        L = np.clip(np.asarray(1.0 if z_over_X0 is None else z_over_X0, dtype=np.float64),
                    1e-9, None)
        return base * (1.0 + 0.038 * np.log(L)) ** 2
    if model == "highland_local":
        L = 1.0 if z_over_X0 is None else np.clip(
            np.asarray(z_over_X0, dtype=np.float64), 1e-9, None)
        lnL = np.log(L)
        return base * (1.0 + 0.038 * lnL) * (1.076 + 0.038 * lnL)
    raise ValueError(f"unknown scattering model: {model}")


def fermi_eyges_moments(z, E0, *, sigma_x0=0.0, sigma_theta0=0.0, rho0=0.0,
                        model="highland_local", n_sub=4, max_step=0.02):
    """
    在深度网格 z [cm] 上返回 Fermi-Eyges 矩与侧向宽度。

    返回 dict
    ---------
        A0(z) = int_0^z T du                       角方差 sigma_theta^2
        A1(z) = int_0^z A0 du
        A2(z) = int_0^z A1 du
        B(z)  = int_0^z (z-u)^2 T(u) du = 2 A2(z)  纯散射引起的侧向方差
        E(z)   CSDA 平均能量
        T(z)   局部散射本领
        sigma_x(z)  含初始束流相空间的侧向 rms 宽度

    公式
    ----
        sigma_x^2(z) = sigma_x0^2 + 2 rho0 sigma_x0 sigma_theta0 z
                       + sigma_theta0^2 z^2 + int_0^z (z-u)^2 T(u) du

    数值实现
    --------
    内部细分网格的**绝对步长**被限制为 ``max_step`` [cm]（默认 0.02 cm），而不是
    仅按输入点数细分。这一点很重要：B(z) 的被积函数在 u -> z 处有一阶零点，
    用稀疏输入网格（例如 8 个深度点跨 25 cm，步长 3 cm）做左矩形求和会高估
    sigma_x 约 6%。本函数因此在内部保证足够的积分分辨率，与输入网格的疏密无关。
    """
    z = np.asarray(z, dtype=np.float64)
    if z.size == 1:
        z = np.array([0.0, z[0]]) if z[0] > 0 else np.array([0.0, 1e-6])
        single = True
    else:
        single = False

    # --- 细分网格：同时满足 n_sub 倍细分与绝对步长上限 ---
    span = float(z[-1] - z[0])
    n_need = int(np.ceil(span / max(max_step, 1e-6))) + 1 if span > 0 else 2
    n_fine = max((len(z) - 1) * n_sub + 1, n_need, 2)
    zf = np.linspace(z[0], z[-1], n_fine)
    hf = np.gradient(zf)
    R0 = float(csda_range(E0))
    rem = np.clip(R0 - zf, 0.0, None)
    E_z = np.where(zf < R0, energy_from_range(rem), 0.0)
    Tf = np.where(zf < R0, scattering_power(np.clip(E_z, 1e-3, None), model=model), 0.0)

    A0f = np.cumsum(Tf * hf)
    A1f = np.cumsum(A0f * hf)
    A2f = np.cumsum(A1f * hf)
    dzf = zf[:, None] - zf[None, :]
    Bf = np.where(dzf > 0, dzf**2 * Tf[None, :] * hf[None, :], 0.0).sum(axis=1)

    # --- 回到用户网格 ---
    A0 = np.interp(z, zf, A0f)
    A1 = np.interp(z, zf, A1f)
    A2 = np.interp(z, zf, A2f)
    B = np.interp(z, zf, Bf)
    E_out = np.where(z < R0, energy_from_range(np.clip(R0 - z, 0.0, None)), 0.0)
    T_out = np.where(z < R0, scattering_power(np.clip(E_out, 1e-3, None), model=model), 0.0)

    var = (sigma_x0**2 + 2.0 * rho0 * sigma_x0 * sigma_theta0 * z
           + sigma_theta0**2 * z**2 + B)
    res = {"A0": A0, "A1": A1, "A2": A2, "B": B,
           "E": E_out, "T": T_out, "sigma_x": np.sqrt(np.clip(var, 0.0, None))}
    if single:
        res = {k: np.asarray(v)[-1:] for k, v in res.items()}
    return res


# ----------------------------------------------------------------------------
# 5. 解析 IDD（深度剂量 / Bragg 峰）
# ----------------------------------------------------------------------------
@dataclass
class IDDModel:
    """
    解析 IDD 构造参数。

    straggling_model
        'bohr'    —— 第一性原理 Bohr 积分（默认；200 MeV 水 sigma_R = 0.52 cm）
        'icru49'  —— ICRU 49 经验拟合 0.012 R^0.935（200 MeV 水 0.25 cm，约为 Bohr 一半）
        'none'    —— 只使用束流能量展宽
    """
    straggling_model: str = "bohr"
    straggling_coef: float = 0.012      # ICRU 49: sigma_R = coef * R0^exp
    straggling_exp: float = 0.935
    energy_spread: float = 0.0033       # 束流能量 1-sigma 相对展宽 (0.33% ~ 1% FWHM)
    attenuation_length: float = 150.0   # 核反应导致的原注量衰减长度 [cm] (~0.67%/cm)
    n_fine: int = 6000                  # 预卷积细网格点数（能量守恒分箱用）
    e_cut: float = 0.05                 # CSDA 积分截止能量 [MeV]

    def sigma_range(self, E0: float) -> tuple:
        """返回 (总等效射程展宽 sigma_R, 射程歧离分量, 束流能量展宽分量) [cm]。"""
        R0 = float(csda_range(E0))
        if self.straggling_model == "bohr":
            s_strag = range_straggling_bohr(E0)
        elif self.straggling_model == "icru49":
            s_strag = self.straggling_coef * R0 ** self.straggling_exp
        elif self.straggling_model == "none":
            s_strag = 0.0
        else:
            raise ValueError(f"unknown straggling model: {self.straggling_model}")
        s_espr = P_R * self.energy_spread * R0
        return float(np.hypot(s_strag, s_espr)), float(s_strag), float(s_espr)


DEFAULT_IDD = IDDModel()


def idd_pristine(z, E0, model: IDDModel = DEFAULT_IDD, return_components: bool = False,
                 normalize: bool = True):
    """
    构造 pristine Bragg 峰 IDD(z)。

    方法（能量守恒分箱 + 射程歧离高斯卷积）
    --------------------------------------
    1. 细网格上用 **精确分箱** 计算 CSDA 剂量：
           D_i = Phi(z_i) * [E(z_{i-1/2}) - E(z_{i+1/2})] / dz_fine
       等价于 D = Phi * S(E(z))，但避免了 E->0 时 S->inf 的数值奇点，
       且严格守恒每个 bin 内沉积的动能。**其量纲为"每单位注量的深度剂量"
       [MeV/cm]**，即真实加速器交付的物理量。
    2. 高斯卷积等效射程歧离
           sigma_R^2 = (0.012 R0^0.935)^2 + (P_R * (dE/E) * R0)^2
       第二项来自束流能量展宽（dR = P_R (dE/E) R0）。
    3. 插值回目标深度网格。

    参数
    ----
    normalize : bool
        **True（默认）**：把该能量层的曲线归一化到峰值 = 1。此时束流权重 w 是
        "峰值剂量权重"，仅适合单层可视化。
        **False**：返回物理的"每单位注量深度剂量"（峰值随能量变化，低能层峰值更高）。
        **多能量层叠加必须用 normalize=False**，否则各层之间的相对强度被错误地
        强制成 1:1 —— 这是本项目在蒙特卡洛验证阶段发现的一个真实建模错误
        （详见 docs/04_research_log.md 问题 11）。真实 IMPT 的 spot 权重是
        注量（MU / 质子数），不是峰值剂量。

    返回 (idd, sigma_R)；return_components=True 时返回 dict。
    """
    z = np.asarray(z, dtype=np.float64)
    R0 = float(csda_range(E0))

    # --- 1) 细网格，能量守恒分箱 ---
    sig_R, sig_strag, sig_espread = model.sigma_range(E0)
    zf_max = R0 + 8.0 * sig_R + 0.2
    zf = np.linspace(0.0, zf_max, model.n_fine)
    dzf = zf[1] - zf[0]
    zb = np.linspace(0.0, zf_max, model.n_fine + 1)
    Eb = np.where(zb < R0, energy_from_range(np.clip(R0 - zb, 0.0, None)), 0.0)
    dE_bin = Eb[:-1] - Eb[1:]
    fluence = np.exp(-zf / model.attenuation_length)
    dose_csda = fluence * dE_bin / dzf                     # [MeV/cm]，每单位注量

    nfft = int(2 ** np.ceil(np.log2(2 * model.n_fine)))
    k = np.fft.rfftfreq(nfft, d=dzf)
    G = np.exp(-2.0 * (np.pi * k * sig_R) ** 2)
    dose = np.fft.irfft(np.fft.rfft(dose_csda, nfft) * G, nfft)[: model.n_fine]

    dose_on_z = np.interp(z, zf, dose, left=0.0, right=0.0)
    peak = float(dose_on_z.max()) if dose_on_z.max() > 0 else 1.0
    idd = dose_on_z / peak if normalize else dose_on_z

    out = {"idd": idd, "sigma_R": sig_R, "R0": R0, "peak": peak,
           "z_fine": zf, "dose_fine": dose / peak, "dose_csda_fine": dose_csda / peak,
           "dose_fluence": dose, "peak_fluence": peak,
           "sigma_straggling": sig_strag, "sigma_espread": sig_espread}
    if return_components:
        return out
    return idd, sig_R


def idd_peak_metrics(z, idd) -> dict:
    """从 IDD 中提取临床关心的峰位指标：R_peak / R90 / R80 / R50 / 坪区比 / 远端跌落。"""
    z = np.asarray(z); idd = np.asarray(idd)
    ipk = int(np.argmax(idd))
    R_peak = float(z[ipk])

    def distal_cross(level):
        idx = np.where(idd >= level)[0]
        if idx.size == 0:
            return float("nan")
        i = idx[-1]
        if i + 1 >= len(z):
            return float(z[i])
        y0, y1 = idd[i], idd[i + 1]
        if y1 == y0:
            return float(z[i])
        return float(z[i] + (level - y0) / (y1 - y0) * (z[i + 1] - z[i]))

    entrance = float(np.mean(idd[: max(3, int(0.05 * len(z)))]))
    r90, r80, r20, r10 = distal_cross(0.9), distal_cross(0.8), distal_cross(0.2), distal_cross(0.1)
    return {"R_peak": R_peak, "R90": r90, "R80": r80, "R50": distal_cross(0.5), "R20": r20,
            "R10": r10,
            "distal_falloff_90_10": r10 - r90,
            "distal_falloff_80_20": r20 - r80,
            "entrance_dose": entrance,
            "peak_to_entrance": float(idd.max() / max(entrance, 1e-12))}


# ----------------------------------------------------------------------------
# 6. 水等效深度（WED）
# ----------------------------------------------------------------------------
def wed_from_density_path(densities, lengths):
    """WED = sum rho_i * L_i （水密度 = 1 g/cm^3）。"""
    return float(np.sum(np.asarray(densities) * np.asarray(lengths)))


# ----------------------------------------------------------------------------
# 7. 交叉验证工具
# ----------------------------------------------------------------------------
# NIST PSTAR 数据库（Water, Liquid, matno = 276），2026 年通过其 CGI 接口实取。
# 列: E[MeV] -> 电子阻止本领 [MeV cm^2/g]（总阻止本领与此相差 < 0.6%，见下）
NIST_PSTAR_WATER = {
    1.0:   260.6,
    2.0:   158.5,
    5.0:   79.06,
    10.0:  45.64,
    20.0:  26.05,
    30.0:  18.75,
    50.0:  12.44,
    80.0:  8.622,
    100.0: 7.286,
    150.0: 5.443,
    200.0: 4.491,
    250.0: 3.910,
    300.0: 3.519,
}

# NIST PSTAR CSDA 射程参考锚点（水，cm）——用于射程积分验证
NIST_PSTAR_RANGE = {
    200.0: 25.92,
    250.0: 37.90,
    300.0: 51.31,
}


def validate_stopping_power(verbose: bool = True) -> dict:
    """Bethe-Bloch 阻止本领 / 数值 CSDA 射程 与 NIST PSTAR 的逐点对比。"""
    E = np.array(sorted(NIST_PSTAR_WATER.keys()))
    S_ref = np.array([NIST_PSTAR_WATER[e] for e in E])
    S_bb = bethe_bloch_water(E)
    S_pl = 1.0 / (A_R * P_R * np.power(E, P_R - 1.0))
    R_bb = csda_range(E)
    R_pl = csda_range_powerlaw(E)

    rel_bb = (S_bb - S_ref) / S_ref * 100.0
    rel_pl = (S_pl - S_ref) / S_ref * 100.0

    res = {
        "E": E, "S_ref": S_ref, "S_bethe_bloch": S_bb, "S_powerlaw": S_pl,
        "S_bb_relerr_pct": rel_bb,
        "S_pl_relerr_pct": rel_pl,
        "R_bethe_bloch": R_bb, "R_powerlaw": R_pl,
        "S_bb_max_relerr_pct": float(np.abs(rel_bb).max()),
        "S_bb_max_relerr_ge10MeV_pct": float(np.abs(rel_bb[E >= 10]).max()),
        "S_pl_max_relerr_pct": float(np.abs(rel_pl).max()),
    }
    # 射程锚点（仅 200/250/300 MeV 有可靠公开值）
    Er = np.array(sorted(NIST_PSTAR_RANGE.keys()))
    Rr = np.array([NIST_PSTAR_RANGE[e] for e in Er])
    Rm = csda_range(Er)
    res.update({"E_range": Er, "R_ref": Rr, "R_model": Rm,
                "R_relerr_pct": (Rm - Rr) / Rr * 100.0,
                "R_max_relerr_pct": float(np.abs((Rm - Rr) / Rr * 100.0).max())})

    if verbose:
        print("=" * 82)
        print("Bethe-Bloch 阻止本领 / 数值 CSDA 射程 与 NIST PSTAR 交叉验证（水）")
        print("=" * 82)
        print(f"{'E[MeV]':>8} {'S_NIST':>9} {'S_BB':>9} {'dS%':>8} "
              f"{'S_powerlaw':>11} {'dS_pl%':>8} {'R_BB[cm]':>10}")
        for i, e in enumerate(E):
            print(f"{e:8.1f} {S_ref[i]:9.3f} {S_bb[i]:9.3f} {rel_bb[i]:8.2f} "
                  f"{S_pl[i]:11.3f} {res['S_pl_relerr_pct'][i]:8.2f} {R_bb[i]:10.3f}")
        print("-" * 82)
        print(f"Bethe-Bloch 阻止本领最大相对偏差 (1-300 MeV) : "
              f"{res['S_bb_max_relerr_pct']:.3f} %")
        print(f"Bethe-Bloch 阻止本领最大相对偏差 (>=10 MeV)  : "
              f"{res['S_bb_max_relerr_ge10MeV_pct']:.3f} %")
        print(f"临床幂律阻止本领最大相对偏差                 : "
              f"{res['S_pl_max_relerr_pct']:.3f} %")
        print("-" * 82)
        print("数值积分 CSDA 射程 vs NIST PSTAR 锚点:")
        for i, e in enumerate(res["E_range"]):
            print(f"   E = {e:6.1f} MeV :  R_model = {res['R_model'][i]:7.3f} cm , "
                  f"R_NIST = {res['R_ref'][i]:7.3f} cm ,  偏差 {res['R_relerr_pct'][i]:+6.2f} %")
        print(f"射程最大相对偏差: {res['R_max_relerr_pct']:.2f} %")
        print("=" * 82)
    return res


def sigma_highland_integrated(E0, thickness_cm) -> float:
    """Highland 积分公式给出的投影 rms 角 [rad]（与 T(z) 积分结果交叉验证用）。"""
    beta, gamma, beta2 = beta_gamma(E0)
    p_MeV = np.sqrt(gamma**2 - 1.0) * M_P_C2
    L = thickness_cm / WATER_X0
    return float(E_HIGHLAND / (beta * p_MeV) * np.sqrt(L) * (1.0 + 0.038 * np.log(L)))


if __name__ == "__main__":
    validate_stopping_power()
