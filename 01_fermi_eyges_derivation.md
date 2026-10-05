# 第一部分：Fermi–Eyges 散射理论与 Pencil Beam 剂量模型的数学推导

> 本文是 GPU 加速质子调强治疗（IMPT）剂量引擎的数学基础：从小角输运方程出发推导 Fermi–Eyges 横向展宽，给出散射本领（scattering power）与阻止本领（stopping power）的局域形式，构造解析积分深度剂量（IDD），并把三维 pencil beam 剂量写成逐层二维卷积，给出复杂度与加速比。所有积分恒等式均给出推导步骤。

---

## 1. 物理问题背景

质子束进入介质后，三种同时发生的过程决定剂量分布：

1. **电子阻止（electronic stopping）**：质子与介质原子中的电子发生大量非弹性库仑碰撞，按 Bethe–Bloch 公式连续损失能量。这是质子射程有限、并在射程末端形成 Bragg 峰的根本原因。
2. **多重库仑散射（multiple Coulomb scattering, MCS）**：质子与原子核的库仑场发生大量微小偏转，单次偏转角典型量级远小于 1 mrad，但次数极多，累积成毫弧度量级的角分布，并通过"角度搬移横向位置"转化为束斑展宽。
3. **核反应导致的注量衰减（nuclear attenuation）**：弹性/非弹性核散射与靶碎裂（target fragmentation）使原初质子注量随深度近似指数衰减，水中约每厘米 1%。

因为偏转次数极多、每次偏转近似独立且方差有限，中心极限定理（central limit theorem）使横向注量分布趋于高斯分布。这就是 **Fermi–Eyges / Gaussian pencil beam 近似**的核心：整个束流被视为许多互不干涉的高斯型"笔形束"（pencil beam）的线性叠加。

本文显式采用两条假设：

- **(A1) 小角近似**：只保留散射角的一、二阶矩，忽略大角度单次散射（single large-angle scattering）的 Rutherford 尾部。因此横向分布必然是高斯的，模型会低估半影（penumbra）与窄束的剂量尾部。
- **(A2) 连续慢化近似（CSDA, continuous slowing down approximation）**：能量损失按 $-\mathrm dE/\mathrm dz=S(E)$ 连续处理，不追踪单次大能量转移；射程歧离（range straggling）在 §5 中以高斯卷积作为后处理引入。同时采用**局域近似**：散射本领与阻止本领只依赖该点的局部能量。

---

## 2. Fermi–Eyges 输运方程与二阶矩

### 2.1 小角输运方程

设 $\Phi(x,y,\theta_x,\theta_y,z)$ 为深度 $z$ 处、横向位置 $(x,y)$、方向余弦 $(\theta_x,\theta_y)$ 的角注量（angular fluence），并归一化 $\iint\Phi\,\mathrm dx\,\mathrm d\theta=1$。在小角近似下（Fermi 1940；Eyges 1948）：

$$
\frac{\partial\Phi}{\partial z}+\boldsymbol\theta\cdot\nabla_\perp\Phi=\frac{T(z)}{4}\nabla_\theta^2\Phi ,
\qquad
\boldsymbol\theta=(\theta_x,\theta_y),\quad
\nabla_\perp=(\partial_x,\partial_y),\quad
\nabla_\theta^2=\partial_{\theta_x}^2+\partial_{\theta_y}^2
\tag{1}
$$

其中 $T(z)$ 为线性角散射本领（linear angular scattering power），单位 $\mathrm{rad^2/cm}$。式 (1) 左端第二项是漂移（角度把横向位置搬移），右端是多次散射引起的角空间扩散。

> **因子约定说明。** 若 $\boldsymbol\theta\in\mathbb R^2$，扩散项给每个分量 $\mathrm d\langle\theta_x^2\rangle/\mathrm dz=2\cdot(T/4)=T/2$，而空间角满足 $\mathrm d\langle\theta^2\rangle/\mathrm dz=T$；此时式 (1) 中的 $T$ 是**空间角散射本领**。质子治疗文献惯用**投影角散射本领** $T_{\mathrm{proj}}\equiv \mathrm d\langle\theta_x^2\rangle/\mathrm dz$（Highland 公式给出的正是投影角宽度，见 §3），此时 $T=2T_{\mathrm{proj}}$，一维投影形式为

$$
\frac{\partial\Phi}{\partial z}+\theta_x\frac{\partial\Phi}{\partial x}=\frac{T_{\mathrm{proj}}}{2}\frac{\partial^2\Phi}{\partial\theta_x^2},
\qquad
\frac{\mathrm d\langle\theta_x^2\rangle}{\mathrm dz}=T_{\mathrm{proj}} .
\tag{2}
$$

**本文以下统一采用投影角约定**，并把符号简写为 $T\equiv T_{\mathrm{proj}}$。这样 §3 中对 Highland 公式微分得到的 $T$ 与本节 $A_0=\int_0^z T\,\mathrm du$ 的含义完全一致；两套约定物理结果相同，差别只是同一字母 $T$ 的归一化。

### 2.2 矩方程

由式 (2) 出发，对 $x$ 与 $\theta_x$ 积分，定义矩 $\langle f\rangle=\iint f\,\Phi\,\mathrm dx\,\mathrm d\theta_x$，并利用 $\Phi\to0\ (|x|,|\theta|\to\infty)$ 使一切边界项为零。核反应衰减为乘性因子，不影响归一化矩，故暂不写出。

**(i) 横向均方宽度。** 用式 (2) 替换 $\partial_z\Phi$：

$$
\frac{\mathrm d\langle x^2\rangle}{\mathrm dz}
=\iint x^2\frac{\partial\Phi}{\partial z}\mathrm dx\,\mathrm d\theta
=-\underbrace{\iint x^2\theta\,\partial_x\Phi}_{I_1}
+\frac{T}{2}\underbrace{\iint x^2\partial_\theta^2\Phi}_{I_2}.
$$

逐项计算。$I_2=\int x^2\Big([\partial_\theta\Phi]_{-\infty}^{\infty}\Big)\mathrm dx=0$。对 $I_1$ 先对 $x$ 分部积分：

$$
\int x^2\partial_x\Phi\,\mathrm dx=\big[x^2\Phi\big]_{-\infty}^{\infty}-2\int x\Phi\,\mathrm dx=-2\int x\Phi\,\mathrm dx
\;\Longrightarrow\;
I_1=2\iint x\theta\,\Phi=2\langle x\theta\rangle ,
$$

$$
\boxed{\ \frac{\mathrm d\langle x^2\rangle}{\mathrm dz}=2\langle x\theta\rangle\ }
\tag{3}
$$

**(ii) 混合矩。** 同理

$$
\frac{\mathrm d\langle x\theta\rangle}{\mathrm dz}
=-\iint x\theta^2\partial_x\Phi+\frac{T}{2}\iint x\theta\,\partial_\theta^2\Phi .
$$

第一项：$\int x\partial_x\Phi\,\mathrm dx=[x\Phi]-\int\Phi\,\mathrm dx=-\int\Phi\,\mathrm dx$，故该项为 $\iint\theta^2\Phi=\langle\theta^2\rangle$。第二项：$\int\theta\,\partial_\theta^2\Phi\,\mathrm d\theta=[\theta\partial_\theta\Phi]-\int\partial_\theta\Phi\,\mathrm d\theta=0$，故为零。于是

$$
\boxed{\ \frac{\mathrm d\langle x\theta\rangle}{\mathrm dz}=\langle\theta^2\rangle\ }
\tag{4}
$$

**(iii) 角均方。** 由 $\iint\theta^3\partial_x\Phi=0$（对 $x$ 的全微分）与

$$
\int\theta^2\partial_\theta^2\Phi\,\mathrm d\theta
=\big[\theta^2\partial_\theta\Phi\big]-\int 2\theta\,\partial_\theta\Phi\,\mathrm d\theta
=-2\Big(\big[\theta\Phi\big]-\int\Phi\,\mathrm d\theta\Big)=2\int\Phi\,\mathrm d\theta ,
$$

得

$$
\boxed{\ \frac{\mathrm d\langle\theta^2\rangle}{\mathrm dz}=T(z)\ }
\tag{5}
$$

对以 $z$ 轴正入射的 pencil beam，$\langle x\rangle=\langle\theta\rangle=0$，角分布方位对称，三个二阶矩即完全决定横向分布 $x\sim\mathcal N(0,\sigma_x^2)$。

### 2.3 Fermi–Eyges 矩 $A_0,A_1,A_2$

依次积分式 (5)(4)(3)，并取入射条件 $\langle\theta^2\rangle(0)=\langle x\theta\rangle(0)=0$：

$$
A_0(z)\equiv\langle\theta^2\rangle(z)=\int_0^{z}T(u)\,\mathrm du
\tag{6}
$$

$$
A_1(z)\equiv\int_0^{z}A_0(u)\,\mathrm du=\langle x\theta\rangle(z)
\tag{7}
$$

$$
A_2(z)\equiv\int_0^{z}A_1(u)\,\mathrm du
\tag{8}
$$

即：$A_0$ 是投影角均方展宽 $\sigma_\theta^2$，$A_1$ 是位置–角度关联 $\langle x\theta\rangle$，$A_2$ 是横向均方展宽的一半（见下）。

### 2.4 $\sigma_x^2(z)=\int_0^z(z-u)^2T(u)\,\mathrm du=2A_2(z)$ 的严格证明

在小角近似下轨迹为 $x(z)=\int_0^z\theta(u)\,\mathrm du$。对以 $z$ 轴入射的 pencil beam，$\theta(u)=\theta_s(u)$ 是纯散射角，初值为零，且在本模型中为**独立增量过程**（Markov 扩散），$\mathrm{Var}\,\theta_s(u)=A_0(u)$。

**引理（两点关联函数）。** 对零均值独立增量过程，$\langle\theta_s(u)\theta_s(v)\rangle=A_0(\min(u,v))$。

*证明*：设 $v\ge u$，写 $\theta_s(v)=\theta_s(u)+[\theta_s(v)-\theta_s(u)]$。增量与 $\theta_s(u)$ 独立、且 $\langle\theta_s(v)-\theta_s(u)\rangle=0$，故

$$
\langle\theta_s(u)\theta_s(v)\rangle
=\langle\theta_s(u)^2\rangle+\langle\theta_s(u)\rangle\big\langle\theta_s(v)-\theta_s(u)\big\rangle
=A_0(u)+0=A_0(\min(u,v)).\qquad\blacksquare
$$

**第一步：双重积分。** 由于 $\langle x(z)\rangle=0$，

$$
\sigma_x^2(z)=\big\langle x(z)^2\big\rangle
=\int_0^{z}\!\!\int_0^{z}\langle\theta_s(u)\theta_s(v)\rangle\,\mathrm du\,\mathrm dv
=\int_0^{z}\!\!\int_0^{z}A_0(\min(u,v))\,\mathrm du\,\mathrm dv .
$$

按 $u<v$ 与 $u>v$ 对称分片（被积函数关于 $u\leftrightarrow v$ 对称）：

$$
\sigma_x^2(z)=2\int_0^{z}\mathrm dv\int_0^{v}\mathrm du\,A_0(u)
=2\int_0^{z}A_0(u)\,(z-u)\,\mathrm du .
\tag{9}
$$

**第二步：分部积分给出 $2A_2$。** 取 $\mathrm dA_1/\mathrm du=A_0(u)$，

$$
\int_0^{z}A_0(u)(z-u)\,\mathrm du
=\Big[A_1(u)(z-u)\Big]_0^{z}+\int_0^{z}A_1(u)\,\mathrm du
=\underbrace{0}_{u=z}-\underbrace{0}_{A_1(0)=0}+A_2(z)=A_2(z),
$$

$$
\boxed{\ \sigma_x^2(z)=2A_2(z)\ }
\tag{10}
$$

**第三步：化为 $T$ 的显式积分。** 利用 $(z-u)^2=\int_u^{z}\!\!\int_u^{z}\mathrm dv\,\mathrm dw=2\int_u^{z}(v-u)\,\mathrm dv$，并用 Fubini 定理交换次序：

$$
\int_0^{z}(z-u)^2T(u)\,\mathrm du
=2\int_0^{z}T(u)\left[\int_u^{z}(v-u)\,\mathrm dv\right]\mathrm du
=2\int_0^{z}\mathrm dv\int_0^{v}\mathrm du\,T(u)\,(v-u).
$$

内层再分部积分（用 $A_0'(u)=T(u)$）：

$$
\int_0^{v}T(u)(v-u)\,\mathrm du
=\Big[(v-u)A_0(u)\Big]_0^{v}+\int_0^{v}A_0(u)\,\mathrm du=0+A_1(v),
$$

$$
\int_0^{z}(z-u)^2T(u)\,\mathrm du=2\int_0^{z}A_1(v)\,\mathrm dv=2A_2(z).
\tag{11}
$$

比较 (10)、(11) 即得所需恒等式：

$$
\boxed{\ \sigma_x^2(z)=\int_0^{z}(z-u)^2T(u)\,\mathrm du=2A_2(z)\ }
\tag{12}
$$

**常数 $T$ 检验。** 若 $T=\mathrm{const}$：$A_0=Tz,\ A_1=Tz^2/2,\ A_2=Tz^3/6$，于是 $\sigma_x^2=2A_2=Tz^3/3$，即经典结果 $\sigma_x=\sqrt{Tz^3/3}$。数值示例：200 MeV 质子在水表面（$\beta pc\approx364.9\ \mathrm{MeV}$，$L=1/36.08=0.0277$）的局部 $T\approx3.12\times10^{-5}\ \mathrm{rad^2/cm}$，故 1 cm 深处 $\sigma_x=\sqrt{Tz^3/3}\approx3.2\times10^{-3}\ \mathrm{cm}=32\ \mu\mathrm m$，与入射坪区笔形束展宽仅几十微米的预期一致。

### 2.5 含初值协方差的一般情形

若表面 $z=0$ 处束流已有均方宽度 $\sigma_{x0}^2$、均方发散 $\sigma_{\theta0}^2$ 与相关系数 $\rho_0$（$|\rho_0|\le1$，$\langle x_0\theta_0\rangle=\rho_0\sigma_{x0}\sigma_{\theta0}$），则轨迹分解为

$$
x(z)=x_0+\theta_0 z+\int_0^{z}\theta_s(u)\,\mathrm du ,
$$

其中散射增量与初始条件统计独立。平方取期望：

$$
\langle x^2\rangle=\langle x_0^2\rangle+z^2\langle\theta_0^2\rangle+2z\langle x_0\theta_0\rangle
+2\Big\langle x_0\!\!\int_0^{z}\!\theta_s\Big\rangle
+2z\Big\langle\theta_0\!\!\int_0^{z}\!\theta_s\Big\rangle
+\Big\langle\Big(\int_0^{z}\!\theta_s\Big)^2\Big\rangle .
$$

因 $\langle\theta_s\rangle=0$ 且独立，两个交叉项为零；最后一项已由 §2.4 算出等于 $2A_2(z)$。于是

$$
\boxed{\ \sigma_x^2(z)=\sigma_{x0}^2+2\rho_0\sigma_{x0}\sigma_{\theta0}z+\sigma_{\theta0}^2z^2+\int_0^{z}(z-u)^2T(u)\,\mathrm du\ }
\tag{13}
$$

四项物理含义：初始束斑尺寸；初始"腰–发散"耦合（相空间倾斜）；几何发散；介质多次散射的累积贡献。写成协方差矩阵形式（$\Sigma=\begin{pmatrix}\sigma_x^2 & \langle x\theta\rangle\\ \langle x\theta\rangle & \langle\theta^2\rangle\end{pmatrix}$，$R(z)=\begin{pmatrix}1& z\\ 0& 1\end{pmatrix}$）：

$$
\Sigma(z)=R(z)\,\Sigma(0)\,R(z)^{\mathsf T}
+\begin{pmatrix}2A_2(z) & A_1(z)\\ A_1(z) & A_0(z)\end{pmatrix}
\tag{14}
$$

$y$ 方向完全同理，只需换成 $\sigma_{y0},\rho_{0y},T_y$；在方位对称介质中 $T_x=T_y$，故 $\sigma_x=\sigma_y$。

---

## 3. 散射本领 $T(z)$ 的取值

### 3.1 Highland 公式（积分量）

穿越几何厚度 $z$ 后的**投影角** rms 由 Highland（1975）经验公式给出（PDG 常用形式）：

$$
\theta_0(z)=\frac{13.6\ \mathrm{MeV}}{\beta pc}\sqrt{\frac{z}{X_0}}\left[1+0.038\ln\!\frac{z}{X_0}\right]
\tag{15}
$$

水中辐射长度（radiation length）$X_0=36.08\ \mathrm{cm}$；混合物用 Bragg 加性 $1/X_0=\sum_i w_i/X_{0,i}$。

### 3.2 微分为局域散射本领 $T(z)$

令 $L=z/X_0$，则 $\theta_0^2=\left(\frac{13.6}{\beta pc}\right)^2 L\,[1+0.038\ln L]^2$。对 $z$ 求导，把 $\beta pc$ 视为该点的局部常数（能量依赖由 CSDA 在每一步单独更新），并注意 $\mathrm dL/\mathrm dz=1/X_0$：

$$
\frac{\mathrm d\theta_0^2}{\mathrm dz}
=\left(\frac{13.6}{\beta pc}\right)^2
\left\{
\frac{\mathrm dL}{\mathrm dz}\big[1+0.038\ln L\big]^2
+L\cdot 2\big[1+0.038\ln L\big]\cdot 0.038\cdot\frac{1}{L}\cdot\frac{\mathrm dL}{\mathrm dz}
\right\}
$$

$$
=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}
\Big\{[1+0.038\ln L]^2+0.076\,[1+0.038\ln L]\Big\}
$$

$$
=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\big[1+0.038\ln L\big]\Big(\big[1+0.038\ln L\big]+0.076\Big)
$$

$$
\boxed{\ T(z)=\frac{\mathrm d\theta_0^2}{\mathrm dz}
=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\big(1+0.038\ln L\big)\big(1.076+0.038\ln L\big),
\qquad L=\frac{z}{X_0}\ }
\tag{16}
$$

### 3.3 与 Fermi 常数形式的交叉检验

另一常用形式为

$$
T\simeq\left(\frac{E_s}{\beta pc}\right)^2\frac{1}{X_0},\qquad E_s=14.1\ \mathrm{MeV}
\tag{17}
$$

在 $L=1$（$z=X_0=36.08\ \mathrm{cm}$）处，式 (16) 给出 $T=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\times1.076$，而 $(14.1/13.6)^2=1.0754$，两者相差 **0.05%**：即 Highland 微分式与 $E_s=14.1$ MeV 的 Fermi 常数式在 $z\approx X_0$ 处几乎重合，可作为实现时的量纲与系数自检。

### 3.4 为什么必须用"局域" $T$ 而非整层积分值

逐步积分时，每一步的角方差增量是 $\mathrm d\langle\theta^2\rangle=T(z)\,\mathrm dz$，只由**该步的局部能量与局部厚度**决定；$T$ 是 $\theta_0^2$ 的导数，而不是 $\theta_0^2$ 与厚度的比值。若把整层积分值平摊为常数，即用

$$
T_{\mathrm{avg}}(z)=\frac{\theta_0^2(z)}{z}=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\big[1+0.038\ln L\big]^2 ,
$$

则 $\dfrac{T}{T_{\mathrm{avg}}}=\dfrac{1.076+0.038\ln L}{1+0.038\ln L}=1+\dfrac{0.076}{1+0.038\ln L}$：在 $L\lesssim1$ 范围内该比值恒为 $+7.6\%\sim+9\%$，即平摊法系统性**低估**局部散射本领约 8%。更粗略地把路径末端的 $\theta_0^2(R_0)$ 当作局部 $T$，则在入射坪区会严重高估（该处能量高、散射弱），误差可达数倍。

### 3.5 对数修正因子的影响

**表 1** 列出对数因子在若干 $L$ 处的数值（最后一列为局域 $T$ 相对无对数形式 $\left(\frac{13.6}{\beta pc}\right)^2/X_0$ 的倍数）：

| $L=z/X_0$ | $\ln L$ | 积分角宽因子 $[1+0.038\ln L]^2$ | 局域 $T$ 因子 $(1+0.038\ln L)(1.076+0.038\ln L)$ | 相对无对数形式 |
|---|---|---|---|---|
| 0.1 | −2.303 | 0.833 | 0.902 | −10% |
| 0.5 | −0.693 | 0.948 | 1.022 | +2% |
| 1.0 | 0.000 | 1.000 | 1.076 | +7.6% |
| 3.0 | 1.099 | 1.085 | 1.164 | +16% |
| 5.0 | 1.609 | 1.126 | 1.207 | +21% |
| 10.0 | 2.303 | 1.183 | 1.265 | +27% |

可见对数修正并非单向：它使**薄层**的积分角宽明显变小（$L=0.1$ 时 $\theta_0^2$ 比无对数形式小约 17%），却使**局域 $T$** 在 $L\gtrsim1$ 时比无对数形式大 8%–25%。在治疗质子关心的区间（200 MeV 质子水中 $R_0\approx26\ \mathrm{cm}$，$L=R_0/X_0\approx0.72$；全路径 $L\lesssim0.8$）内，对数项对 $T$ 的净影响量级为 **10%–15%**（$L=1$ 处由求导产生的因子本身即给出 $+7.6\%$）。忽略它会给 $\sigma_x^2$ 带来同量级误差（深部偏差 10% 以上），因此不能省略。

---

## 4. 阻止本领与 CSDA 射程

### 4.1 Bethe–Bloch 公式（质子）

对非相对论性入射粒子（质子）配相对论性电子运动学：

$$
-\frac{\mathrm dE}{\mathrm dx}=Kz^2\frac{Z}{A}\frac{1}{\beta^2}
\left[\frac12\ln\!\frac{2m_ec^2\beta^2\gamma^2T_{\max}}{I^2}-\beta^2-\frac{\delta}{2}\right],
\qquad K=0.307075\ \mathrm{MeV\,mol^{-1}\,cm^2}
\tag{18}
$$

$$
T_{\max}=\frac{2m_ec^2\beta^2\gamma^2}{1+2\gamma m_e/M_p+(m_e/M_p)^2},
\qquad M_pc^2=938.272\ \mathrm{MeV},\quad m_ec^2=0.510999\ \mathrm{MeV}
\tag{19}
$$

其中 $z$ 为入射粒子电荷数（质子 $z=1$），$Z,A$ 为介质原子序数与原子量，$I$ 为平均激发能，$\delta$ 为密度效应修正（治疗能区可近似 $\delta\approx0$）。水中取 $Z/A=0.55509$、$I=75.0\ \mathrm{eV}$、$\rho=1.0\ \mathrm{g/cm^3}$；质量阻止本领 $S/\rho$ 乘 $\rho$ 得线性阻止本领 $S$。

**数值锚点**（由式 (18)(19) 直接算得，与 ICRU 49 / PSTAR 表一致）：200 MeV 时 $S\approx4.5\ \mathrm{MeV/cm}$；10 MeV 时 $\approx46\ \mathrm{MeV/cm}$；1 MeV 时 $\approx2.7\times10^{2}\ \mathrm{MeV/cm}$。末端两个数量级的增长正是 Bragg 峰的来源；主导因子是 $1/\beta^2$，对数项只是缓变修正。

### 4.2 CSDA 射程与能量–深度映射

$$
R(E_0)=\int_0^{E_0}\frac{\mathrm dE}{S(E)}
\tag{20}
$$

对 $E\gtrsim1$ MeV，$S(E)$ 单调递减，故 $R(E)$ 单调递增、可逆，得到逆映射 $E(R)$（实现上用样条或分段对数插值）。能量–深度常微分方程为

$$
\frac{\mathrm dE}{\mathrm dz}=-S(E),\quad E(0)=E_0
\;\Longrightarrow\;
z(E)=\int_E^{E_0}\frac{\mathrm dE'}{S(E')}=R(E_0)-R(E),
\quad\text{即}\quad
E(z)=R^{-1}\!\big(R(E_0)-z\big)
\tag{21}
$$

水中常用拟合 $R\approx0.0022\,E^{1.77}$（$R$ 单位 cm，$E$ 单位 MeV），在 1–200 MeV 内精度约百分之几。

### 4.3 水等效深度 WED

对非均匀介质，沿射线把各种材料折算为水：

$$
\mathrm{WED}(z)=\int_0^{z}\frac{\rho_e}{\rho_{e,w}}(z')\,\mathrm dz'
\tag{22}
$$

更严格地应按阻止本领比（stopping power ratio, SPR）加权：$\mathrm{WED}(z)=\int_0^{z}\big(S_m/S_w\big)\mathrm dz'$；电子密度比 (22) 是它的常用近似（100–200 MeV 质子在软组织/水中两者相差在百分量级）。射线追踪用 Siddon（1985）的精确体素穿越算法或 Joseph 的增量步进算法。得到 WED 后用水的 CSDA 关系读出局部能量 $E(\mathrm{WED})$ 与阻止本领 $S$。注意：WED 只修正**沿射线方向**的能量损失；横向散射本领的换算需另用 $T_m/T_w$（同时含 $X_0$ 与 $\beta pc$ 的比值）。

---

## 5. 深度剂量 IDD 的解析构造

### 5.1 宽束近似下的中心轴剂量

对横向足够宽的平行束，横向注量损失可忽略，中心轴剂量正比于"局部注量 × 局部阻止本领"：

$$
D(z)\propto\Phi(z)\,S\big(E(z)\big),
\qquad
\Phi(z)=\Phi_0\exp\!\left(-\int_0^{z}n\sigma_{\mathrm{nuc}}\,\mathrm du\right)
\tag{23}
$$

$n$ 为靶核数密度，$\sigma_{\mathrm{nuc}}$ 为质子–核反应截面。水中 $n\sigma_{\mathrm{nuc}}\approx0.01\ \mathrm{cm^{-1}}$（约每厘米 1%），对 26 cm 射程意味着约 23% 的原初注量被核反应移除——这部分能量并未消失，而是以次级粒子形式在别处沉积（见 §7）。

### 5.2 形式发散与三步实用构造

$S(E)\propto1/\beta^2\sim1/E$（$E\to0$）故 $S\to\infty$，而 $R$ 有限，于是 $D_0(z)\propto S(E(z))$ 在 $z\to R_0$ 时发散。由 $R\propto E^{1.77}\Rightarrow S\propto E^{-0.77}$ 得

$$
S\big(E(z)\big)\propto (R_0-z)^{-\alpha},\qquad \alpha=\frac{0.77}{1.77}=0.435
\tag{24}
$$

实用构造为：

**(i) 积分到截止能量。** 用式 (21) 的 ODE 从 $E_0$ 积到很小的 $E_{\mathrm{cut}}$（例如 0.5–1 MeV），得 $z_{\max}=R(E_0)-R(E_{\mathrm{cut}})$。对 200 MeV 质子，$E_{\mathrm{cut}}=1$ MeV 只削掉末端约 0.2 mm——**截断只去掉形式上发散的最后一丝，真正把发散正则化的是射程歧离卷积**。

**(ii) 截断。** $D_0(z)=\Phi(z)S(E(z))$ 用于 $z\le z_{\max}$，$z>z_{\max}$ 置零。

**(iii) 射程歧离卷积并归一化。**

$$
D(z)=\int_{-\infty}^{\infty}D_0(z')\,G(z-z';\sigma_R)\,\mathrm dz',
\qquad
G(\zeta;\sigma_R)=\frac{1}{\sigma_R\sqrt{2\pi}}\exp\!\left(-\frac{\zeta^2}{2\sigma_R^2}\right)
\tag{25}
$$

$$
\sigma_R=0.012\,R_0^{\,0.935}
\qquad(R_0\ \text{单位 cm},\ \sigma_R\ \text{单位 cm；ICRU 49 经验拟合，亦见 Bortfeld 1997})
\tag{26}
$$

对 200 MeV（$R_0\approx26$ cm）：$\sigma_R\approx0.25\ \mathrm{cm}=2.5$ mm。最后把 $D(z)$ 归一化到最大值 1。

### 5.3 卷积为何既压低又展宽 Bragg 峰，以及坪峰比为何约 0.3

- **压低**：无卷积时 $D_0$ 在末端发散；能量守恒要求末端沉积的能量摊在宽度 $\sim\sigma_R$ 的区间上，峰值因而被限制为"末端 $\pm\sigma_R$ 内沉积的能量 / $(\sqrt{2\pi}\,\sigma_R)$"。同时峰位略向近端移动。
- **展宽与远端跌落**：卷积把阶跃截断变成 Gauss 误差函数型边缘，80%–20% 跌落宽度 $=2\times0.8416\,\sigma_R\approx1.68\sigma_R\approx4.2$ mm（200 MeV），与临床实测的远端跌落 4–5 mm 相符。
- **坪峰比**：把 $D_0\propto(R_0-z)^{-\alpha}$ 与高斯核卷积，峰值处的积分给出（$\alpha=0.435$）

$$
D_{\text{peak}}\simeq 0.77\,\Big(\frac{R_0}{\sigma_R}\Big)^{\alpha}D_{\text{坪}},
\qquad \Big(\frac{R_0}{\sigma_R}\Big)^{0.435}\approx 104^{0.435}\approx7.5 ,
$$

其中常数 $0.77=2^{-(\alpha+1)/2}\Gamma\!\big(\tfrac{1-\alpha}{2}\big)/\sqrt{2\pi}$ 来自 $\int_0^\infty t^{-\alpha}e^{-t^2/2}\mathrm dt$。故未修正的峰/坪 $\approx5.8$；再乘该深度注量衰减因子 $e^{-0.01\times26}\approx0.77$，并对坪区取平均（坪区 $D_0$ 随深度缓慢上升，平均值约为入坪值的 1.3 倍），得峰/坪 $\approx5.8\times0.77/1.3\approx3.4$，即 **坪/峰 $\approx0.29\approx0.3$**。可见"坪/峰 ≈ 0.3"是"$1/E$ 型阻止本领增长"与"有限射程歧离宽度"两个因素竞争的结果，而非任何单一常数决定；能量更高时 $\sigma_R/R_0$ 变小，坪/峰比进一步下降。

---

## 6. 三维 Pencil Beam 剂量公式

### 6.1 主公式

$$
\boxed{\
D(x,y,z)=\sum_m N_m\,\mathrm{IDD}\big(\mathrm{WED}_m(z)\big)\,
\frac{1}{2\pi\sigma_x(z)\sigma_y(z)}
\exp\!\left(-\frac{(x-x_m)^2}{2\sigma_x^2(z)}-\frac{(y-y_m)^2}{2\sigma_y^2(z)}\right)\
}
\tag{27}
$$

各符号含义：$m$ 为斑点（spot）编号，共 $M$ 个；$N_m$ 为该斑点的权重（MU/粒子数）；$x_m,y_m$ 为其横向标称位置；(x,y,z) 为剂量网格坐标；$\mathrm{WED}_m(z)$ 为沿斑点 $m$ 射线通过 Siddon/Joseph 追踪得到的水等效深度；$\mathrm{IDD}(\cdot)$ 为 §5 构造的、按斑点能量归一化的积分深度剂量；$\sigma_x(z),\sigma_y(z)$ 为核宽，由式 (13) 沿该斑点射线取局部 $T$、$S$ 积分得到。

- **均匀介质中 $\sigma_x=\sigma_y$**（方位对称）；在非均匀介质中，不同斑点的射线穿越不同材料，$\mathrm{WED}_m$ 与局部 $T$ 均不同，核宽逐斑点（甚至逐层）不同；椭圆型初始相空间（$\sigma_{x0}\ne\sigma_{y0}$ 或 $\rho_{0x}\ne\rho_{0y}$）也会使 $\sigma_x\ne\sigma_y$。扫描磁铁带来的是纯几何位置偏移，不改变核宽。
- **核在 $x,y$ 上可分离**：

$$
\frac{1}{2\pi\sigma_x\sigma_y}e^{-\frac{(x-x_m)^2}{2\sigma_x^2}}e^{-\frac{(y-y_m)^2}{2\sigma_y^2}}
=\Big[\tfrac{1}{\sqrt{2\pi}\sigma_x}e^{-\frac{(x-x_m)^2}{2\sigma_x^2}}\Big]
\Big[\tfrac{1}{\sqrt{2\pi}\sigma_y}e^{-\frac{(y-y_m)^2}{2\sigma_y^2}}\Big],
$$

即二维核是两个一维核的直积（separable kernel），可先沿 $x$ 再沿 $y$ 做两次一维卷积，每体素代价从 $O(K^2)$ 降到 $O(2K)$。

- **求和与横向卷积可交换**：式 (27) 对 $m$ 是线性的，横向卷积对 $D$ 也是线性的，故 $\sum_m(\text{卷积})=\text{卷积}(\sum_m)$。

### 6.2 逐层二维卷积与复杂度

定义**逐层斑点幅度图**（把落在同一横向格点 $(i,j)$ 上的所有斑点的深度剂量先累加）：

$$
S_z(i,j)=\sum_{m\in(i,j)}N_m\,\mathrm{IDD}_m(z)
\tag{28}
$$

若该深度层内所有斑点共用同一横向核 $G_z$（均匀介质中最自然的假设；非均匀时见下文讨论），则

$$
D(i,j,z)=\sum_{p,q}S_z(i-p,\,j-q)\,G_z(p,q)=\big(S_z\circledast G_z\big)(i,j)
\tag{29}
$$

即**每个深度 $z$ 上只需做一次二维离散卷积**，$z$ 方向循环即可。

**复杂度计数**（$M=64\,000$ 个斑点，$128^3$ 网格，核支撑 $K=35$ 抽头/体素）：

| 方法 | 复杂度 | 运算量 |
|---|---|---|
| 逐斑点直接求和 | $O(M\,N_xN_yN_z)$ | $64\,000\times128^3=6.4\times10^4\times2.097\times10^6\approx1.34\times10^{11}$ |
| 逐层二维卷积 | $O(N_zN_xN_yK)$ | $128\times128\times128\times35=7.34\times10^{7}$ |
| 幅度图构建（附加） | $O(MN_z)$ | $64\,000\times128=8.2\times10^6$（$\ll$ 卷积代价） |

**加速比** $=1.34\times10^{11}/7.34\times10^{7}\approx1.83\times10^{3}=M/K=64\,000/35\approx1829$，即约三个数量级；加速比即 $M/K$，斑点越多、核越紧凑，收益越大。可分离核使每体素只需 $2K$ 抽头而非 $K_xK_y\approx K^2/4$，是 $K=35$ 这一小值可实现的前提。$K$ 按高斯截断选取，例如 $K=2\lceil3.5\sigma_x/h\rceil+1$（$h$ 为体素边长），保证截断处核值 $<10^{-6}$。

**严格性条件与工程折中**：式 (29) 只有在核与斑点无关时才严格成立。非均匀介质中 $\sigma_x(z)$ 逐斑点不同，工程上通常 (a) 按 $\mathrm{WED}$/能量分组（每组共用一枚核，分别卷积后叠加），或 (b) 退回"剂量影响矩阵"逐斑点累加（失去上述加速）。

### 6.3 二维轴对称约化

对单斑点二维剂量图（径向对称，用于 commissioning 的积分深度剂量与横向剖面比对）：

$$
D(r,z)=N\,\mathrm{IDD}(z)\,\frac{1}{2\pi\sigma^2(z)}\exp\!\left(-\frac{r^2}{2\sigma^2(z)}\right),
\qquad r=\sqrt{x^2+y^2}
\tag{30}
$$

其中 $r$ 在垂直于束轴的平面内测量，$\sigma(z)$ 由式 (13) 用该射线的 WED 与局部 $T$ 计算；该式即式 (27) 在 $M=1$、$\sigma_x=\sigma_y=\sigma$ 时的特例。

---

## 7. 模型适用边界与已知偏差

1. **核反应产物（靶碎裂）**：次级质子、氘核、$\alpha$ 粒子等射程短但会形成"核晕（nuclear halo）"，把剂量沉积到原初射程**之外**。纯原初质子模型因此远端跌落过陡、远端与尾区剂量被低估（远端 mm 量级的位置偏差、尾区几个百分点的剂量偏差）。修正需 Monte Carlo 或人为叠加 halo 项。
2. **大角度单次散射**：纯高斯核低估半影与窄束的剂量尾部。真实 Molière 分布在 $\theta$ 较大处近似 $\theta^{-4}$ 衰减，比高斯重得多；改进办法是"Fermi–Eyges 核 $\circledast$ 单次散射核"（双高斯/核–尾分解）或直接使用 Molière 参数化。
3. **非均匀介质**：需要 WED 射线追踪（Siddon 1985；Joseph）；在组织界面（骨–软组织、肺–胸壁）附近精度最差。物理原因是界面两侧散射本领突变造成侧向失平衡，单一 $\sigma(z)$（"密度标度"）无法描述界面后的再分布，局部误差可达 5%–10% 以上，肺内尤甚。
4. **原初注量损失的处理过粗**：本文用每厘米约 1% 的指数衰减近似式 (23)，而完整 MC 会同时追踪被移除质子的次级产物及其能量沉积位置；此外 $\sigma_{\mathrm{nuc}}$ 随能量变化，把注量损失简单当作能量损失只是近似。
5. **其它**：忽略体内磁场偏转（mrad 量级以下）、忽略散射各向异性与射程歧离对 $\sigma_x$ 的二阶影响、忽略 $E_{\mathrm{cut}}$ 与截断位置的人为依赖。

**结论**：本模型适合作为快速解析引擎、GPU 加速原型与教学演示；任何临床级使用都必须以 Monte Carlo 参考（TOPAS/Geant4，或 FLUKA/MCNP）与实测数据（ICRU 35/49 提供阻止本领与射程歧离基准；Paganetti 2012 综述了质子治疗物理与射程不确定度）进行验证。

---

## 8. 符号表

**表 2：输运、散射与能量损失**

| 符号 | 含义 | 单位 |
|---|---|---|
| $\Phi$ | 角注量（angular fluence） | 粒子/(cm²·rad²) |
| $\theta_x,\theta_y$ | 投影/横向角度（小角近似下 $\theta\approx\tan\theta$） | rad |
| $\theta_s$ | 纯散射角（零均值独立增量过程） | rad |
| $T$ | 局部线性角散射本领（本文为投影角约定） | rad²/cm |
| $T_{\mathrm{proj}}$ | 投影角散射本领，$\mathrm d\langle\theta_x^2\rangle/\mathrm dz$ | rad²/cm |
| $A_0,A_1,A_2$ | Fermi–Eyges 零/一/二重积分矩，式 (6)–(8) | rad², rad²·cm, rad²·cm² |
| $\sigma_x,\sigma_y$ | 横向 rms 束斑尺寸（核宽） | cm |
| $\sigma_{x0},\sigma_{\theta0},\rho_0$ | 表面初始 rms 尺寸、rms 发散与相关系数 | cm, rad, 无量纲 |
| $X_0$ | 辐射长度（水 36.08 cm） | cm |
| $L=z/X_0$ | 以辐射长度为单位的归一化厚度 | 无量纲 |
| $\beta,\gamma,p$ | 相对速度、Lorentz 因子、动量 | 无量纲, 无量纲, MeV/c |
| $E_s$ | Fermi 常数（14.1 MeV） | MeV |
| $S(E)=-{\mathrm dE}/{\mathrm dx}$ | 线性阻止本领（Bethe–Bloch） | MeV/cm |
| $I,\delta$ | 平均激发能、密度效应修正 | eV, 无量纲 |
| $M_pc^2,\ m_ec^2$ | 质子、电子静止能量 | MeV |
| $T_{\max}$ | 单次碰撞最大能量转移 | MeV |
| $R(E),R_0,E_{\mathrm{cut}}$ | CSDA 射程、初始能量射程、截止能量 | cm, cm, MeV |
| $\sigma_R$ | 射程歧离（range straggling）rms | cm |
| $n,\sigma_{\mathrm{nuc}}$ | 靶核数密度、质子–核反应截面 | cm⁻³, cm² |
| $\rho_e/\rho_{e,w}$ | 电子密度比 | 无量纲 |
| WED | 水等效深度（water-equivalent depth） | cm |

**表 3：剂量引擎与数值实现**

| 符号 | 含义 | 单位 |
|---|---|---|
| $D(x,y,z)$ | 吸收剂量 | Gy（或相对剂量） |
| IDD | 积分深度剂量（归一化到峰值 1） | 无量纲 |
| $N_m$ | 第 $m$ 个斑点权重 | MU 或粒子数 |
| $M$ | 斑点总数（示例 64 000） | 无量纲 |
| $x_m,y_m$ | 斑点标称横向位置 | cm |
| $S_z(i,j)$ | 深度 $z$ 的逐层斑点幅度图，式 (28) | 无量纲 |
| $G_z$ | 深度 $z$ 的横向高斯核（可分离） | cm⁻² |
| $K$ | 每体素核抽头数（示例 35） | 无量纲 |
| $N_x,N_y,N_z$ | 网格点数（示例 128³） | 无量纲 |
| $h$ | 体素边长 | cm |
| $r=\sqrt{x^2+y^2}$ | 轴对称径向坐标 | cm |
| $m,(i,j)$ | 斑点索引、横向格点索引 | 无量纲 |

---

### 参考文献（按姓名引用，不列页码）

Fermi (1940)；Eyges (1948)；Highland (1975)；Bortfeld (1997)；Low et al. (1998)；Siddon (1985)；ICRU Report 35；ICRU Report 49；Paganetti (ed.), *Proton Therapy Physics* (2012)；PDG *Review of Particle Physics*（粒子通过物质的章节）。
