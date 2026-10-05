"""
pbdose — GPU 强化的二维/三维 Pencil Beam 剂量重构引擎
=====================================================

模块
----
physics          质子物理核心：Bethe-Bloch 阻止本领、CSDA 射程、Fermi-Eyges 散射矩、
                 Highland 散射本领、解析 IDD（Bragg 峰）、NIST PSTAR 交叉验证
model            计算域（DoseGrid）、束流点阵（SpotLattice）、IMPT 问题定义
engines          5 类剂量重构实现：朴素三重循环 / NumPy 逐束流 /
                 NumPy 可分离 / PyTorch 批矩阵乘法 / 局部窗口 Gather
triton_kernels   自定义 GPU Kernel（Triton → sm_120 PTX/CUBIN）：
                 朴素 gather、窗口 gather + LUT、融合高斯核的可分离卷积 GEMM
montecarlo       独立蒙特卡洛参考引擎（凝聚历史法，GPU 加速）
gamma            Gamma Index 分析（Low 1998 / AAPM TG-218）
benchmark        计时框架、roofline 估算、结果导出
viz              科研图表生成
"""

__version__ = "1.0.0"
__all__ = ["physics", "model", "engines", "triton_kernels", "gamma",
           "montecarlo", "benchmark", "viz"]
