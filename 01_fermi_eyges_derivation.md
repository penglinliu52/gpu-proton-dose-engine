# Part I: Mathematical Derivation of Fermi–Eyges Scattering Theory and the Pencil Beam Dose Model

> This document is the mathematical foundation of a GPU-accelerated intensity-modulated proton therapy (IMPT) dose engine: starting from the small-angle transport equation, we derive the Fermi–Eyges lateral spread, give the local forms of the scattering power and the stopping power, construct an analytical integral depth dose (IDD), and write the three-dimensional pencil beam dose as a layer-by-layer two-dimensional convolution, together with its complexity and speed-up. Every integral identity is derived step by step.

---

## 1. Physical Background

Once a proton beam enters a medium, three processes that occur simultaneously determine the dose distribution:

1. **Electronic stopping**: the proton undergoes a great many inelastic Coulomb collisions with the electrons of the atoms of the medium, losing energy continuously according to the Bethe–Bloch formula. This is the root cause of the proton's finite range and of the Bragg peak that forms at the end of that range.
2. **Multiple Coulomb scattering (MCS)**: the proton is deflected through a great many tiny angles by the Coulomb fields of atomic nuclei. A typical single deflection is far below 1 mrad, but the number of deflections is enormous, and they accumulate into an angular distribution of order a milliradian, which is then converted into beam-spot broadening because angle displaces lateral position.
3. **Nuclear attenuation**: elastic and inelastic nuclear scattering together with target fragmentation make the primary proton fluence decay approximately exponentially with depth — about 1% per centimetre in water.

Because the number of deflections is very large, each deflection approximately independent and of finite variance, the central limit theorem drives the lateral fluence distribution towards a Gaussian. This is the essence of the **Fermi–Eyges / Gaussian pencil beam approximation**: the whole beam is treated as the linear superposition of many mutually non-interfering Gaussian pencil beams.

Two assumptions are adopted explicitly in this document:

- **(A1) Small-angle approximation**: only the first and second moments of the scattering angle are retained; the Rutherford tail of single large-angle scattering is ignored. The lateral distribution is therefore necessarily Gaussian, and the model underestimates the penumbra and the dose tail of narrow beams.
- **(A2) Continuous slowing down approximation (CSDA)**: energy loss is treated continuously as $-\mathrm dE/\mathrm dz=S(E)$, without tracking individual large energy transfers; range straggling is introduced in §5 as a post-processing Gaussian convolution. We simultaneously adopt a **local approximation**: the scattering power and the stopping power depend only on the local energy at that point.

---

## 2. The Fermi–Eyges Transport Equation and Second Moments

### 2.1 Small-Angle Transport Equation

Let $\Phi(x,y,\theta_x,\theta_y,z)$ be the angular fluence at depth $z$, lateral position $(x,y)$ and direction cosine $(\theta_x,\theta_y)$, normalised so that $\iint\Phi\,\mathrm dx\,\mathrm d\theta=1$. Under the small-angle approximation (Fermi 1940; Eyges 1948):

$$
\frac{\partial\Phi}{\partial z}+\boldsymbol\theta\cdot\nabla_\perp\Phi=\frac{T(z)}{4}\nabla_\theta^2\Phi ,
\qquad
\boldsymbol\theta=(\theta_x,\theta_y),\quad
\nabla_\perp=(\partial_x,\partial_y),\quad
\nabla_\theta^2=\partial_{\theta_x}^2+\partial_{\theta_y}^2
\tag{1}
$$

Here $T(z)$ is the linear angular scattering power, with units $\mathrm{rad^2/cm}$. The second term on the left-hand side of Eq. (1) is the drift term (angle displacing lateral position), while the right-hand side is the angular-space diffusion caused by multiple scattering.

> **A note on the factor convention.** If $\boldsymbol\theta\in\mathbb R^2$, the diffusion term gives each component $\mathrm d\langle\theta_x^2\rangle/\mathrm dz=2\cdot(T/4)=T/2$, whereas the space angle satisfies $\mathrm d\langle\theta^2\rangle/\mathrm dz=T$; in that case the $T$ in Eq. (1) is the **space-angle scattering power**. The proton therapy literature conventionally uses the **projected-angle scattering power** $T_{\mathrm{proj}}\equiv \mathrm d\langle\theta_x^2\rangle/\mathrm dz$ (what the Highland formula supplies is precisely the projected angular width — see §3), for which $T=2T_{\mathrm{proj}}$, and the one-dimensional projected form reads

$$
\frac{\partial\Phi}{\partial z}+\theta_x\frac{\partial\Phi}{\partial x}=\frac{T_{\mathrm{proj}}}{2}\frac{\partial^2\Phi}{\partial\theta_x^2},
\qquad
\frac{\mathrm d\langle\theta_x^2\rangle}{\mathrm dz}=T_{\mathrm{proj}} .
\tag{2}
$$

**Throughout the remainder of this document we adopt the projected-angle convention** and abbreviate the symbol to $T\equiv T_{\mathrm{proj}}$. The $T$ obtained in §3 by differentiating the Highland formula then has exactly the same meaning as $A_0=\int_0^z T\,\mathrm du$ in this section; the two conventions give identical physics, the only difference being the normalisation attached to the same letter $T$.

### 2.2 Moment Equations

Starting from Eq. (2), we integrate over $x$ and $\theta_x$, define the moments $\langle f\rangle=\iint f\,\Phi\,\mathrm dx\,\mathrm d\theta_x$, and use $\Phi\to0\ (|x|,|\theta|\to\infty)$ so that every boundary term vanishes. Nuclear attenuation enters as a multiplicative factor and does not affect the normalised moments, so we leave it unwritten for now.

**(i) Mean-square lateral width.** Substituting Eq. (2) for $\partial_z\Phi$:

$$
\frac{\mathrm d\langle x^2\rangle}{\mathrm dz}
=\iint x^2\frac{\partial\Phi}{\partial z}\mathrm dx\,\mathrm d\theta
=-\underbrace{\iint x^2\theta\,\partial_x\Phi}_{I_1}
+\frac{T}{2}\underbrace{\iint x^2\partial_\theta^2\Phi}_{I_2}.
$$

Take the terms one at a time. $I_2=\int x^2\Big([\partial_\theta\Phi]_{-\infty}^{\infty}\Big)\mathrm dx=0$. For $I_1$ we integrate by parts in $x$ first:

$$
\int x^2\partial_x\Phi\,\mathrm dx=\big[x^2\Phi\big]_{-\infty}^{\infty}-2\int x\Phi\,\mathrm dx=-2\int x\Phi\,\mathrm dx
\;\Longrightarrow\;
I_1=2\iint x\theta\,\Phi=2\langle x\theta\rangle ,
$$

$$
\boxed{\ \frac{\mathrm d\langle x^2\rangle}{\mathrm dz}=2\langle x\theta\rangle\ }
\tag{3}
$$

**(ii) Mixed moment.** Likewise

$$
\frac{\mathrm d\langle x\theta\rangle}{\mathrm dz}
=-\iint x\theta^2\partial_x\Phi+\frac{T}{2}\iint x\theta\,\partial_\theta^2\Phi .
$$

First term: $\int x\partial_x\Phi\,\mathrm dx=[x\Phi]-\int\Phi\,\mathrm dx=-\int\Phi\,\mathrm dx$, so this term equals $\iint\theta^2\Phi=\langle\theta^2\rangle$. Second term: $\int\theta\,\partial_\theta^2\Phi\,\mathrm d\theta=[\theta\partial_\theta\Phi]-\int\partial_\theta\Phi\,\mathrm d\theta=0$, so it vanishes. Hence

$$
\boxed{\ \frac{\mathrm d\langle x\theta\rangle}{\mathrm dz}=\langle\theta^2\rangle\ }
\tag{4}
$$

**(iii) Mean-square angle.** From $\iint\theta^3\partial_x\Phi=0$ (a total derivative in $x$) and

$$
\int\theta^2\partial_\theta^2\Phi\,\mathrm d\theta
=\big[\theta^2\partial_\theta\Phi\big]-\int 2\theta\,\partial_\theta\Phi\,\mathrm d\theta
=-2\Big(\big[\theta\Phi\big]-\int\Phi\,\mathrm d\theta\Big)=2\int\Phi\,\mathrm d\theta ,
$$

we obtain

$$
\boxed{\ \frac{\mathrm d\langle\theta^2\rangle}{\mathrm dz}=T(z)\ }
\tag{5}
$$

For a pencil beam incident along the $z$ axis, $\langle x\rangle=\langle\theta\rangle=0$ and the angular distribution is azimuthally symmetric, so these three second moments completely determine the lateral distribution $x\sim\mathcal N(0,\sigma_x^2)$.

### 2.3 The Fermi–Eyges Moments $A_0,A_1,A_2$

Integrating Eqs. (5), (4) and (3) in turn, and taking the entrance conditions $\langle\theta^2\rangle(0)=\langle x\theta\rangle(0)=0$:

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

That is: $A_0$ is the mean-square projected angular spread $\sigma_\theta^2$, $A_1$ is the position–angle correlation $\langle x\theta\rangle$, and $A_2$ is half the mean-square lateral spread (see below).

### 2.4 Rigorous Proof that $\sigma_x^2(z)=\int_0^z(z-u)^2T(u)\,\mathrm du=2A_2(z)$

Under the small-angle approximation the trajectory is $x(z)=\int_0^z\theta(u)\,\mathrm du$. For a pencil beam incident along the $z$ axis, $\theta(u)=\theta_s(u)$ is the purely scattered angle with zero initial value, and in this model it is a process with **independent increments** (a Markov diffusion), with $\mathrm{Var}\,\theta_s(u)=A_0(u)$.

**Lemma (two-point correlation function).** For a zero-mean independent-increment process, $\langle\theta_s(u)\theta_s(v)\rangle=A_0(\min(u,v))$.

*Proof*: take $v\ge u$ and write $\theta_s(v)=\theta_s(u)+[\theta_s(v)-\theta_s(u)]$. The increment is independent of $\theta_s(u)$, and $\langle\theta_s(v)-\theta_s(u)\rangle=0$, so

$$
\langle\theta_s(u)\theta_s(v)\rangle
=\langle\theta_s(u)^2\rangle+\langle\theta_s(u)\rangle\big\langle\theta_s(v)-\theta_s(u)\big\rangle
=A_0(u)+0=A_0(\min(u,v)).\qquad\blacksquare
$$

**Step 1: the double integral.** Since $\langle x(z)\rangle=0$,

$$
\sigma_x^2(z)=\big\langle x(z)^2\big\rangle
=\int_0^{z}\!\!\int_0^{z}\langle\theta_s(u)\theta_s(v)\rangle\,\mathrm du\,\mathrm dv
=\int_0^{z}\!\!\int_0^{z}A_0(\min(u,v))\,\mathrm du\,\mathrm dv .
$$

Splitting symmetrically into $u<v$ and $u>v$ (the integrand is symmetric under $u\leftrightarrow v$):

$$
\sigma_x^2(z)=2\int_0^{z}\mathrm dv\int_0^{v}\mathrm du\,A_0(u)
=2\int_0^{z}A_0(u)\,(z-u)\,\mathrm du .
\tag{9}
$$

**Step 2: integration by parts gives $2A_2$.** Taking $\mathrm dA_1/\mathrm du=A_0(u)$,

$$
\int_0^{z}A_0(u)(z-u)\,\mathrm du
=\Big[A_1(u)(z-u)\Big]_0^{z}+\int_0^{z}A_1(u)\,\mathrm du
=\underbrace{0}_{u=z}-\underbrace{0}_{A_1(0)=0}+A_2(z)=A_2(z),
$$

$$
\boxed{\ \sigma_x^2(z)=2A_2(z)\ }
\tag{10}
$$

**Step 3: reduction to an explicit integral over $T$.** Using $(z-u)^2=\int_u^{z}\!\!\int_u^{z}\mathrm dv\,\mathrm dw=2\int_u^{z}(v-u)\,\mathrm dv$ and swapping the order of integration by Fubini's theorem:

$$
\int_0^{z}(z-u)^2T(u)\,\mathrm du
=2\int_0^{z}T(u)\left[\int_u^{z}(v-u)\,\mathrm dv\right]\mathrm du
=2\int_0^{z}\mathrm dv\int_0^{v}\mathrm du\,T(u)\,(v-u).
$$

Integrating the inner integral by parts once more (using $A_0'(u)=T(u)$):

$$
\int_0^{v}T(u)(v-u)\,\mathrm du
=\Big[(v-u)A_0(u)\Big]_0^{v}+\int_0^{v}A_0(u)\,\mathrm du=0+A_1(v),
$$

$$
\int_0^{z}(z-u)^2T(u)\,\mathrm du=2\int_0^{z}A_1(v)\,\mathrm dv=2A_2(z).
\tag{11}
$$

Comparing (10) and (11) yields the required identity:

$$
\boxed{\ \sigma_x^2(z)=\int_0^{z}(z-u)^2T(u)\,\mathrm du=2A_2(z)\ }
\tag{12}
$$

**Check with constant $T$.** If $T=\mathrm{const}$: $A_0=Tz,\ A_1=Tz^2/2,\ A_2=Tz^3/6$, hence $\sigma_x^2=2A_2=Tz^3/3$, the classical result $\sigma_x=\sqrt{Tz^3/3}$. Numerical example: for a 200 MeV proton at the water surface ($\beta pc\approx364.9\ \mathrm{MeV}$, $L=1/36.08=0.0277$) the local $T\approx3.12\times10^{-5}\ \mathrm{rad^2/cm}$, so at 1 cm depth $\sigma_x=\sqrt{Tz^3/3}\approx3.2\times10^{-3}\ \mathrm{cm}=32\ \mu\mathrm m$, consistent with the expectation that a pencil beam spreads by only a few tens of micrometres in the entrance plateau.

### 2.5 General Case with an Initial Covariance

If at the surface $z=0$ the beam already has a mean-square width $\sigma_{x0}^2$, a mean-square divergence $\sigma_{\theta0}^2$ and a correlation coefficient $\rho_0$ ($|\rho_0|\le1$, $\langle x_0\theta_0\rangle=\rho_0\sigma_{x0}\sigma_{\theta0}$), the trajectory decomposes as

$$
x(z)=x_0+\theta_0 z+\int_0^{z}\theta_s(u)\,\mathrm du ,
$$

where the scattering increment is statistically independent of the initial conditions. Squaring and taking expectations:

$$
\langle x^2\rangle=\langle x_0^2\rangle+z^2\langle\theta_0^2\rangle+2z\langle x_0\theta_0\rangle
+2\Big\langle x_0\!\!\int_0^{z}\!\theta_s\Big\rangle
+2z\Big\langle\theta_0\!\!\int_0^{z}\!\theta_s\Big\rangle
+\Big\langle\Big(\int_0^{z}\!\theta_s\Big)^2\Big\rangle .
$$

Because $\langle\theta_s\rangle=0$ and the two are independent, the two cross terms vanish; the last term was already shown in §2.4 to equal $2A_2(z)$. Hence

$$
\boxed{\ \sigma_x^2(z)=\sigma_{x0}^2+2\rho_0\sigma_{x0}\sigma_{\theta0}z+\sigma_{\theta0}^2z^2+\int_0^{z}(z-u)^2T(u)\,\mathrm du\ }
\tag{13}
$$

The four terms carry the following physical meanings: the initial spot size; the initial "waist–divergence" coupling (phase-space tilt); geometric divergence; and the accumulated contribution of multiple scattering in the medium. Written in covariance-matrix form ($\Sigma=\begin{pmatrix}\sigma_x^2 & \langle x\theta\rangle\\ \langle x\theta\rangle & \langle\theta^2\rangle\end{pmatrix}$, $R(z)=\begin{pmatrix}1& z\\ 0& 1\end{pmatrix}$):

$$
\Sigma(z)=R(z)\,\Sigma(0)\,R(z)^{\mathsf T}
+\begin{pmatrix}2A_2(z) & A_1(z)\\ A_1(z) & A_0(z)\end{pmatrix}
\tag{14}
$$

The $y$ direction follows in exactly the same way, with $\sigma_{y0},\rho_{0y},T_y$ substituted throughout; in an azimuthally symmetric medium $T_x=T_y$, hence $\sigma_x=\sigma_y$.

---

## 3. Values of the Scattering Power $T(z)$

### 3.1 The Highland Formula (Integrated Quantity)

The rms **projected angle** after traversing a geometric thickness $z$ is given by the empirical Highland (1975) formula (the form commonly used by the PDG):

$$
\theta_0(z)=\frac{13.6\ \mathrm{MeV}}{\beta pc}\sqrt{\frac{z}{X_0}}\left[1+0.038\ln\!\frac{z}{X_0}\right]
\tag{15}
$$

The radiation length in water is $X_0=36.08\ \mathrm{cm}$; for mixtures, Bragg additivity gives $1/X_0=\sum_i w_i/X_{0,i}$.

### 3.2 Differentiation to the Local Scattering Power $T(z)$

Setting $L=z/X_0$ gives $\theta_0^2=\left(\frac{13.6}{\beta pc}\right)^2 L\,[1+0.038\ln L]^2$. Differentiating with respect to $z$, treating $\beta pc$ as the local constant at that point (the energy dependence is updated separately at each CSDA step), and noting that $\mathrm dL/\mathrm dz=1/X_0$:

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

### 3.3 Cross-Check against the Fermi-Constant Form

Another frequently used form is

$$
T\simeq\left(\frac{E_s}{\beta pc}\right)^2\frac{1}{X_0},\qquad E_s=14.1\ \mathrm{MeV}
\tag{17}
$$

At $L=1$ ($z=X_0=36.08\ \mathrm{cm}$), Eq. (16) gives $T=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\times1.076$, whereas $(14.1/13.6)^2=1.0754$, a difference of **0.05%**: that is, the differentiated Highland expression and the Fermi-constant expression with $E_s=14.1$ MeV nearly coincide at $z\approx X_0$, which serves as a dimensional and coefficient self-check during implementation.

### 3.4 Why One Must Use the "Local" $T$ Rather Than a Layer-Integrated Value

When integrating step by step, the angular-variance increment of each step is $\mathrm d\langle\theta^2\rangle=T(z)\,\mathrm dz$, determined solely by **the local energy and the local thickness of that step**; $T$ is the derivative of $\theta_0^2$, not the ratio of $\theta_0^2$ to thickness. If instead the layer-integrated value is spread out as a constant, that is, if one uses

$$
T_{\mathrm{avg}}(z)=\frac{\theta_0^2(z)}{z}=\left(\frac{13.6}{\beta pc}\right)^2\frac{1}{X_0}\big[1+0.038\ln L\big]^2 ,
$$

then $\dfrac{T}{T_{\mathrm{avg}}}=\dfrac{1.076+0.038\ln L}{1+0.038\ln L}=1+\dfrac{0.076}{1+0.038\ln L}$: over the range $L\lesssim1$ this ratio is consistently $+7.6\%\sim+9\%$, i.e. the averaging approach systematically **underestimates** the local scattering power by about 8%. A cruder approach — taking $\theta_0^2(R_0)$ at the end of the path as the local $T$ — badly overestimates it in the entrance plateau (where the energy is high and the scattering weak), with errors reaching a factor of several.

### 3.5 Effect of the Logarithmic Correction Factor

**Table 1** lists the numerical value of the logarithmic factor at several $L$ (the last column gives the local $T$ as a multiple of the no-logarithm form $\left(\frac{13.6}{\beta pc}\right)^2/X_0$):

| $L=z/X_0$ | $\ln L$ | Integral angular-width factor $[1+0.038\ln L]^2$ | Local $T$ factor $(1+0.038\ln L)(1.076+0.038\ln L)$ | Relative to the no-logarithm form |
|---|---|---|---|---|
| 0.1 | −2.303 | 0.833 | 0.902 | −10% |
| 0.5 | −0.693 | 0.948 | 1.022 | +2% |
| 1.0 | 0.000 | 1.000 | 1.076 | +7.6% |
| 3.0 | 1.099 | 1.085 | 1.164 | +16% |
| 5.0 | 1.609 | 1.126 | 1.207 | +21% |
| 10.0 | 2.303 | 1.183 | 1.265 | +27% |

The logarithmic correction is thus not one-directional: it makes the integral angular width of **thin layers** distinctly smaller ($\theta_0^2$ at $L=0.1$ is about 17% below the no-logarithm form), yet it makes the **local $T$** 8%–25% larger than the no-logarithm form once $L\gtrsim1$. Over the interval of interest for therapeutic protons (a 200 MeV proton in water has $R_0\approx26\ \mathrm{cm}$, $L=R_0/X_0\approx0.72$; over the full path $L\lesssim0.8$), the net effect of the logarithmic term on $T$ is of order **10%–15%** (at $L=1$ the factor produced by the differentiation alone already gives $+7.6\%$). Neglecting it introduces an error of the same order into $\sigma_x^2$ (deviations of more than 10% at depth), so it cannot be omitted.

---

## 4. Stopping Power and CSDA Range

### 4.1 The Bethe–Bloch Formula (Protons)

For a non-relativistic incident particle (a proton) with relativistic electron kinematics:

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

Here $z$ is the charge number of the incident particle ($z=1$ for a proton), $Z,A$ are the atomic number and atomic weight of the medium, $I$ is the mean excitation energy, and $\delta$ is the density-effect correction (in the therapeutic energy range $\delta\approx0$ is a good approximation). For water, $Z/A=0.55509$, $I=75.0\ \mathrm{eV}$, $\rho=1.0\ \mathrm{g/cm^3}$; multiplying the mass stopping power $S/\rho$ by $\rho$ gives the linear stopping power $S$.

**Numerical anchors** (computed directly from Eqs. (18) and (19), and consistent with the ICRU 49 / PSTAR tables): at 200 MeV, $S\approx4.5\ \mathrm{MeV/cm}$; at 10 MeV, $\approx46\ \mathrm{MeV/cm}$; at 1 MeV, $\approx2.7\times10^{2}\ \mathrm{MeV/cm}$. This two-orders-of-magnitude growth at the end is precisely the origin of the Bragg peak; the dominant factor is $1/\beta^2$, the logarithmic term being only a slowly varying correction.

### 4.2 CSDA Range and the Energy–Depth Mapping

$$
R(E_0)=\int_0^{E_0}\frac{\mathrm dE}{S(E)}
\tag{20}
$$

For $E\gtrsim1$ MeV, $S(E)$ decreases monotonically, so $R(E)$ increases monotonically and is invertible, giving the inverse mapping $E(R)$ (implemented in practice with splines or piecewise logarithmic interpolation). The energy–depth ordinary differential equation is

$$
\frac{\mathrm dE}{\mathrm dz}=-S(E),\quad E(0)=E_0
\;\Longrightarrow\;
z(E)=\int_E^{E_0}\frac{\mathrm dE'}{S(E')}=R(E_0)-R(E),
\quad\text{i.e.}\quad
E(z)=R^{-1}\!\big(R(E_0)-z\big)
\tag{21}
$$

A fit commonly used for water is $R\approx0.0022\,E^{1.77}$ ($R$ in cm, $E$ in MeV), accurate to about a few per cent over 1–200 MeV.

### 4.3 Water-Equivalent Depth (WED)

For a heterogeneous medium, the various materials along the ray are converted to water:

$$
\mathrm{WED}(z)=\int_0^{z}\frac{\rho_e}{\rho_{e,w}}(z')\,\mathrm dz'
\tag{22}
$$

More rigorously one should weight by the stopping power ratio (SPR): $\mathrm{WED}(z)=\int_0^{z}\big(S_m/S_w\big)\mathrm dz'$; the electron-density ratio of Eq. (22) is its usual approximation (for 100–200 MeV protons in soft tissue/water the two differ by of order a per cent). Ray tracing uses Siddon's (1985) exact voxel-traversal algorithm or Joseph's incremental stepping algorithm. Once the WED is known, the local energy $E(\mathrm{WED})$ and the stopping power $S$ are read off from the water CSDA relation. Note that the WED corrects only the energy loss **along the ray direction**; converting the lateral scattering power requires $T_m/T_w$ instead (which contains the ratios of both $X_0$ and $\beta pc$).

---

## 5. Analytical Construction of the Integral Depth Dose (IDD)

### 5.1 Central-Axis Dose under the Broad-Beam Approximation

For a parallel beam that is wide enough laterally, the lateral fluence loss can be neglected and the central-axis dose is proportional to "the local fluence × the local stopping power":

$$
D(z)\propto\Phi(z)\,S\big(E(z)\big),
\qquad
\Phi(z)=\Phi_0\exp\!\left(-\int_0^{z}n\sigma_{\mathrm{nuc}}\,\mathrm du\right)
\tag{23}
$$

Here $n$ is the target-nucleus number density and $\sigma_{\mathrm{nuc}}$ the proton–nucleus reaction cross-section. In water $n\sigma_{\mathrm{nuc}}\approx0.01\ \mathrm{cm^{-1}}$ (about 1% per centimetre), which for a 26 cm range means that roughly 23% of the primary fluence is removed by nuclear reactions — that energy does not disappear, but is deposited elsewhere in the form of secondary particles (see §7).

### 5.2 Formal Divergence and a Three-Step Practical Construction

Since $S(E)\propto1/\beta^2\sim1/E$ as $E\to0$, $S\to\infty$ while $R$ remains finite, and hence $D_0(z)\propto S(E(z))$ diverges as $z\to R_0$. From $R\propto E^{1.77}\Rightarrow S\propto E^{-0.77}$ we obtain

$$
S\big(E(z)\big)\propto (R_0-z)^{-\alpha},\qquad \alpha=\frac{0.77}{1.77}=0.435
\tag{24}
$$

The practical construction is:

**(i) Integrate down to a cutoff energy.** Integrate the ODE of Eq. (21) from $E_0$ down to a very small $E_{\mathrm{cut}}$ (for example 0.5–1 MeV), giving $z_{\max}=R(E_0)-R(E_{\mathrm{cut}})$. For a 200 MeV proton, $E_{\mathrm{cut}}=1$ MeV shaves off only about the last 0.2 mm — **the truncation only removes the last sliver of the formal divergence; what actually regularises the divergence is the range-straggling convolution**.

**(ii) Truncation.** $D_0(z)=\Phi(z)S(E(z))$ is used for $z\le z_{\max}$, and set to zero for $z>z_{\max}$.

**(iii) Range-straggling convolution and normalisation.**

$$
D(z)=\int_{-\infty}^{\infty}D_0(z')\,G(z-z';\sigma_R)\,\mathrm dz',
\qquad
G(\zeta;\sigma_R)=\frac{1}{\sigma_R\sqrt{2\pi}}\exp\!\left(-\frac{\zeta^2}{2\sigma_R^2}\right)
\tag{25}
$$

$$
\sigma_R=0.012\,R_0^{\,0.935}
\qquad(R_0\ \text{in cm},\ \sigma_R\ \text{in cm; ICRU 49 empirical fit, see also Bortfeld 1997})
\tag{26}
$$

For 200 MeV ($R_0\approx26$ cm): $\sigma_R\approx0.25\ \mathrm{cm}=2.5$ mm. Finally $D(z)$ is normalised to a maximum value of 1.

### 5.3 Why the Convolution Both Lowers and Broadens the Bragg Peak, and Why the Plateau-to-Peak Ratio Is About 0.3

- **Lowering**: without the convolution $D_0$ diverges at the end; energy conservation demands that the energy deposited at the end be spread over an interval of width $\sim\sigma_R$, so the peak value is constrained to be "the energy deposited within $\pm\sigma_R$ of the end / $(\sqrt{2\pi}\,\sigma_R)$". The peak position also shifts slightly towards the proximal side.
- **Broadening and the distal falloff**: the convolution turns the step truncation into a Gaussian error-function edge, the 80%–20% falloff width being $=2\times0.8416\,\sigma_R\approx1.68\sigma_R\approx4.2$ mm (200 MeV), in agreement with the 4–5 mm distal falloff measured clinically.
- **Plateau-to-peak ratio**: convolving $D_0\propto(R_0-z)^{-\alpha}$ with the Gaussian kernel, the integral at the peak gives ($\alpha=0.435$)

$$
D_{\text{peak}}\simeq 0.77\,\Big(\frac{R_0}{\sigma_R}\Big)^{\alpha}D_{\text{plateau}},
\qquad \Big(\frac{R_0}{\sigma_R}\Big)^{0.435}\approx 104^{0.435}\approx7.5 ,
$$

where the constant $0.77=2^{-(\alpha+1)/2}\Gamma\!\big(\tfrac{1-\alpha}{2}\big)/\sqrt{2\pi}$ comes from $\int_0^\infty t^{-\alpha}e^{-t^2/2}\mathrm dt$. Hence the uncorrected peak/plateau $\approx5.8$; multiplying further by the fluence-attenuation factor at that depth, $e^{-0.01\times26}\approx0.77$, and averaging over the plateau (where $D_0$ rises slowly with depth, the mean value being about 1.3 times the entrance-plateau value), gives peak/plateau $\approx5.8\times0.77/1.3\approx3.4$, that is, **plateau/peak $\approx0.29\approx0.3$**. Clearly "plateau/peak ≈ 0.3" is the outcome of competition between two factors — the "$1/E$-type growth of the stopping power" and the "finite range-straggling width" — and is not determined by any single constant; at higher energies $\sigma_R/R_0$ becomes smaller and the plateau/peak ratio falls further.

---

## 6. Three-Dimensional Pencil Beam Dose Formula

### 6.1 Master Formula

$$
\boxed{\
D(x,y,z)=\sum_m N_m\,\mathrm{IDD}\big(\mathrm{WED}_m(z)\big)\,
\frac{1}{2\pi\sigma_x(z)\sigma_y(z)}
\exp\!\left(-\frac{(x-x_m)^2}{2\sigma_x^2(z)}-\frac{(y-y_m)^2}{2\sigma_y^2(z)}\right)\
}
\tag{27}
$$

The symbols are as follows: $m$ is the spot index, with $M$ spots in total; $N_m$ is the weight of that spot (MU/number of particles); $x_m,y_m$ is its nominal lateral position; (x,y,z) are the dose-grid coordinates; $\mathrm{WED}_m(z)$ is the water-equivalent depth obtained by tracing along the ray of spot $m$ with Siddon/Joseph; $\mathrm{IDD}(\cdot)$ is the integral depth dose constructed in §5 and normalised to the spot energy; $\sigma_x(z),\sigma_y(z)$ are the kernel widths, obtained from Eq. (13) by integrating the local $T$ and $S$ along that spot's ray.

- **In a homogeneous medium $\sigma_x=\sigma_y$** (azimuthal symmetry); in a heterogeneous medium the rays of different spots traverse different materials, so that $\mathrm{WED}_m$ and the local $T$ both differ and the kernel width varies from spot to spot (indeed from layer to layer); an elliptical initial phase space ($\sigma_{x0}\ne\sigma_{y0}$ or $\rho_{0x}\ne\rho_{0y}$) also makes $\sigma_x\ne\sigma_y$. What the scanning magnets produce is a purely geometric positional offset, which does not change the kernel width.
- **The kernel is separable in $x,y$**:

$$
\frac{1}{2\pi\sigma_x\sigma_y}e^{-\frac{(x-x_m)^2}{2\sigma_x^2}}e^{-\frac{(y-y_m)^2}{2\sigma_y^2}}
=\Big[\tfrac{1}{\sqrt{2\pi}\sigma_x}e^{-\frac{(x-x_m)^2}{2\sigma_x^2}}\Big]
\Big[\tfrac{1}{\sqrt{2\pi}\sigma_y}e^{-\frac{(y-y_m)^2}{2\sigma_y^2}}\Big],
$$

that is, the two-dimensional kernel is the direct product of two one-dimensional kernels (a separable kernel), so two one-dimensional convolutions can be performed, first along $x$ and then along $y$, reducing the cost per voxel from $O(K^2)$ to $O(2K)$.

- **The sum and the lateral convolution commute**: Eq. (27) is linear in $m$, and the lateral convolution is linear in $D$, so $\sum_m(\text{convolution})=\text{convolution}(\sum_m)$.

### 6.2 Layer-by-Layer Two-Dimensional Convolution and Complexity

Define the **per-layer spot amplitude map** (first accumulating the depth dose of all spots that fall on the same lateral grid point $(i,j)$):

$$
S_z(i,j)=\sum_{m\in(i,j)}N_m\,\mathrm{IDD}_m(z)
\tag{28}
$$

If all spots within that depth layer share the same lateral kernel $G_z$ (the most natural assumption in a homogeneous medium; for the heterogeneous case see the discussion below), then

$$
D(i,j,z)=\sum_{p,q}S_z(i-p,\,j-q)\,G_z(p,q)=\big(S_z\circledast G_z\big)(i,j)
\tag{29}
$$

that is, **only one two-dimensional discrete convolution is needed at each depth $z$**, looping over $z$.

**Complexity accounting** ($M=64\,000$ spots, a $128^3$ grid, kernel support $K=35$ taps/voxel):

| Method | Complexity | Operation count |
|---|---|---|
| Direct per-spot summation | $O(M\,N_xN_yN_z)$ | $64\,000\times128^3=6.4\times10^4\times2.097\times10^6\approx1.34\times10^{11}$ |
| Layer-by-layer 2D convolution | $O(N_zN_xN_yK)$ | $128\times128\times128\times35=7.34\times10^{7}$ |
| Amplitude-map construction (additional) | $O(MN_z)$ | $64\,000\times128=8.2\times10^6$ ($\ll$ convolution cost) |

**Speed-up** $=1.34\times10^{11}/7.34\times10^{7}\approx1.83\times10^{3}=M/K=64\,000/35\approx1829$, i.e. about three orders of magnitude; the speed-up is simply $M/K$, so the more spots and the more compact the kernel, the greater the gain. The separable kernel, requiring only $2K$ taps per voxel instead of $K_xK_y\approx K^2/4$, is the precondition that makes such a small value as $K=35$ achievable. $K$ is chosen by truncating the Gaussian, for example $K=2\lceil3.5\sigma_x/h\rceil+1$ ($h$ being the voxel edge length), guaranteeing a kernel value $<10^{-6}$ at the truncation point.

**Conditions for exactness and engineering compromises**: Eq. (29) holds rigorously only when the kernel is independent of the spot. In a heterogeneous medium $\sigma_x(z)$ differs from spot to spot, and in practice one either (a) groups by $\mathrm{WED}$/energy (each group sharing one kernel, convolved separately and then summed), or (b) falls back to accumulating spot by spot through a "dose influence matrix" (forfeiting the speed-up above).

### 6.3 Two-Dimensional Axisymmetric Reduction

For a single-spot two-dimensional dose map (radially symmetric, used for commissioning comparisons of the integral depth dose and the lateral profile):

$$
D(r,z)=N\,\mathrm{IDD}(z)\,\frac{1}{2\pi\sigma^2(z)}\exp\!\left(-\frac{r^2}{2\sigma^2(z)}\right),
\qquad r=\sqrt{x^2+y^2}
\tag{30}
$$

Here $r$ is measured in the plane perpendicular to the beam axis, and $\sigma(z)$ is computed from Eq. (13) using the WED and local $T$ of that ray; this expression is the special case of Eq. (27) with $M=1$ and $\sigma_x=\sigma_y=\sigma$.

---

## 7. Limits of Validity and Known Deviations of the Model

1. **Nuclear reaction products (target fragmentation)**: secondary protons, deuterons, $\alpha$ particles and so on have short ranges but form a "nuclear halo" that deposits dose **beyond** the primary range. A purely primary-proton model therefore has too steep a distal falloff and underestimates the dose distally and in the tail region (positional deviations of order mm distally, dose deviations of a few per cent in the tail region). Correcting this requires Monte Carlo or the artificial addition of a halo term.
2. **Large-angle single scattering**: a purely Gaussian kernel underestimates the penumbra and the dose tail of narrow beams. The true Molière distribution decays approximately as $\theta^{-4}$ at larger $\theta$, far heavier than a Gaussian; remedies are a "Fermi–Eyges kernel $\circledast$ single-scattering kernel" (a double-Gaussian/core–tail decomposition) or direct use of a Molière parameterisation.
3. **Heterogeneous media**: WED ray tracing is required (Siddon 1985; Joseph); accuracy is worst near tissue interfaces (bone–soft tissue, lung–chest wall). The physical reason is that the abrupt change in scattering power across the interface creates a lateral disequilibrium, and a single $\sigma(z)$ (a "density scaling") cannot describe the redistribution behind the interface, with local errors reaching 5%–10% or more, especially in lung.
4. **Overly crude treatment of primary fluence loss**: here we approximate Eq. (23) by an exponential decay of about 1% per centimetre, whereas a complete MC would track both the secondary products of the removed protons and the location of their energy deposition; moreover $\sigma_{\mathrm{nuc}}$ varies with energy, so treating fluence loss simply as energy loss is only an approximation.
5. **Other**: deflection by magnetic fields in the body is neglected (below the mrad level), as are the anisotropy of scattering and the second-order influence of range straggling on $\sigma_x$, and any artificial dependence on $E_{\mathrm{cut}}$ and the truncation position.

**Conclusion**: this model is suitable as a fast analytical engine, a GPU-acceleration prototype and a teaching demonstration; any clinical-grade use must be validated against Monte Carlo references (TOPAS/Geant4, or FLUKA/MCNP) and measured data (ICRU 35/49 provide stopping-power and range-straggling benchmarks; Paganetti 2012 reviews proton therapy physics and range uncertainty).

---

## 8. Table of Symbols

**Table 2: Transport, scattering and energy loss**

| Symbol | Meaning | Unit |
|---|---|---|
| $\Phi$ | Angular fluence | particles/(cm²·rad²) |
| $\theta_x,\theta_y$ | Projected/lateral angle ($\theta\approx\tan\theta$ under the small-angle approximation) | rad |
| $\theta_s$ | Purely scattered angle (zero-mean independent-increment process) | rad |
| $T$ | Local linear angular scattering power (projected-angle convention in this document) | rad²/cm |
| $T_{\mathrm{proj}}$ | Projected-angle scattering power, $\mathrm d\langle\theta_x^2\rangle/\mathrm dz$ | rad²/cm |
| $A_0,A_1,A_2$ | Fermi–Eyges zeroth/first/second integral moments, Eqs. (6)–(8) | rad², rad²·cm, rad²·cm² |
| $\sigma_x,\sigma_y$ | Lateral rms beam-spot size (kernel width) | cm |
| $\sigma_{x0},\sigma_{\theta0},\rho_0$ | Initial surface rms size, rms divergence and correlation coefficient | cm, rad, dimensionless |
| $X_0$ | Radiation length (36.08 cm in water) | cm |
| $L=z/X_0$ | Normalised thickness in units of the radiation length | dimensionless |
| $\beta,\gamma,p$ | Relative velocity, Lorentz factor, momentum | dimensionless, dimensionless, MeV/c |
| $E_s$ | Fermi constant (14.1 MeV) | MeV |
| $S(E)=-{\mathrm dE}/{\mathrm dx}$ | Linear stopping power (Bethe–Bloch) | MeV/cm |
| $I,\delta$ | Mean excitation energy, density-effect correction | eV, dimensionless |
| $M_pc^2,\ m_ec^2$ | Proton and electron rest energy | MeV |
| $T_{\max}$ | Maximum energy transfer in a single collision | MeV |
| $R(E),R_0,E_{\mathrm{cut}}$ | CSDA range, range at the initial energy, cutoff energy | cm, cm, MeV |
| $\sigma_R$ | Range straggling rms | cm |
| $n,\sigma_{\mathrm{nuc}}$ | Target-nucleus number density, proton–nucleus reaction cross-section | cm⁻³, cm² |
| $\rho_e/\rho_{e,w}$ | Electron-density ratio | dimensionless |
| WED | Water-equivalent depth | cm |

**Table 3: Dose engine and numerical implementation**

| Symbol | Meaning | Unit |
|---|---|---|
| $D(x,y,z)$ | Absorbed dose | Gy (or relative dose) |
| IDD | Integral depth dose (peak-normalised to 1) | dimensionless |
| $N_m$ | Weight of the $m$-th spot | MU or number of particles |
| $M$ | Total number of spots (64 000 in the example) | dimensionless |
| $x_m,y_m$ | Nominal lateral position of the spot | cm |
| $S_z(i,j)$ | Per-layer spot amplitude map at depth $z$, Eq. (28) | dimensionless |
| $G_z$ | Lateral Gaussian kernel at depth $z$ (separable) | cm⁻² |
| $K$ | Number of kernel taps per voxel (35 in the example) | dimensionless |
| $N_x,N_y,N_z$ | Number of grid points ($128^3$ in the example) | dimensionless |
| $h$ | Voxel edge length | cm |
| $r=\sqrt{x^2+y^2}$ | Axisymmetric radial coordinate | cm |
| $m,(i,j)$ | Spot index, lateral grid index | dimensionless |

---

### References (cited by name, without page numbers)

Fermi (1940); Eyges (1948); Highland (1975); Bortfeld (1997); Low et al. (1998); Siddon (1985); ICRU Report 35; ICRU Report 49; Paganetti (ed.), *Proton Therapy Physics* (2012); PDG *Review of Particle Physics* (the chapter on passage of particles through matter).
