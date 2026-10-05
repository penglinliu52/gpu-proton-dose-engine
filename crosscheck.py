# -*- coding: utf-8 -*-
"""交叉验证：Fermi-Eyges sigma_x、B=2A2 恒等式、Bragg 峰指标（两种射程歧离模型）。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pbdose import physics as ph

z = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0])
m = ph.fermi_eyges_moments(z, 200.0)
print("sigma_x (sigma_x0=0) [cm]:", np.array2string(m["sigma_x"], precision=5))
print("B/(2*A2) =", np.array2string(m["B"] / np.maximum(2 * m["A2"], 1e-30), precision=5))
print("T [rad^2/cm] =", np.array2string(m["T"], precision=3))
print()
for E0 in (100.0, 150.0, 200.0):
    mm = ph.fermi_eyges_moments(z, E0, sigma_x0=0.3)
    print("E0=%5.0f  sigma_x[cm] =" % E0, np.array2string(mm["sigma_x"], precision=4))
print()
zz = np.arange(0.0, 34.0, 0.02)
for model in ("bohr", "icru49"):
    print("=== straggling_model =", model, "===")
    for E0 in (100.0, 150.0, 200.0, 230.0):
        mo = ph.IDDModel(straggling_model=model)
        o = ph.idd_pristine(zz, E0, mo, return_components=True)
        mt = ph.idd_peak_metrics(zz, o["idd"])
        print("  %5.0f MeV  R0=%.2f R_peak=%.2f R80=%.2f R50=%.2f  80-20=%.2f mm  "
              "90-10=%.2f mm  pk/ent=%.2f  sigma_R=%.3f cm"
              % (E0, o["R0"], mt["R_peak"], mt["R80"], mt["R50"],
                 mt["distal_falloff_80_20"] * 10, mt["distal_falloff_90_10"] * 10,
                 mt["peak_to_entrance"], o["sigma_R"]))
print()
print("Highland 积分角宽交叉检查 (200 MeV, 10 cm):",
      "theta0 = %.4f mrad" % (ph.sigma_highland_integrated(200.0, 10.0) * 1e3))
print("Fermi-Eyges A0(10cm) = %.4f mrad^2 -> sqrt = %.4f mrad"
      % (m["A0"][4] * 1e6, np.sqrt(m["A0"][4]) * 1e3))
