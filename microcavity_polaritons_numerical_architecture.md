# Расчет дисперсии exciton–cavity polaritons: формулы, параметры и численная архитектура

## 1. Цель проекта

Рассчитать дисперсию exciton–cavity polaritons в planar DBR microcavity и последовательно сравнить:

1. аналитическую модель связанных photon/exciton modes;
2. finite-DBR transfer-matrix model (TMM);
3. TE и TM поляризации;
4. комплексные собственные моды, linewidth и lifetime;
5. влияние числа DBR-пар и положения QW;
6. в конце — несколько QW как дополнительное усложнение.

Motional narrowing и inhomogeneous broadening из §4.4.5 **не включаются**.

Основные разделы книги, на которые опирается модель:

- Kavokin et al., *Microcavities*, §§2.5–2.7 — propagation in multilayer planar structures, photonic eigenmodes, planar microcavities;
- §4.3.3 — quantum wells;
- §4.4.2 — exciton–photon coupling in a QW, QW reflection/transmission;
- §4.4.4 — polaritons in planar microcavities, QW transfer matrix, oblique incidence, TE/TM;
- §§5.8.1–5.8.2 — polariton dispersion, Hamiltonian and Hopfield transformation.

Простые формулы для Q-factor, FWHM и lifetime отдельно к книге не привязываются.

---

# 2. Архитектура проекта

```text
microcavity_polaritons/
├── microcavity_polaritons.ipynb
├── microcavity.py
├── requirements.txt
├── figures/
└── report.pdf
```

## `microcavity_polaritons.ipynb`

Содержит:

- imports;
- обычный `dict` с конфигурацией;
- запуск расчетов;
- формулы и короткие пояснения;
- графики;
- проверки;
- выводы.

Конфиг находится **только в notebook**, без JSON и без `@dataclass`.

## `microcavity.py`

Один файл со всеми основными функциями. Внутри логически разделить комментариями:

1. constants and units;
2. analytical dispersion;
3. layer construction;
4. transfer matrices;
5. QW resonant response;
6. reflectivity and field reconstruction;
7. branch extraction;
8. complex eigenmodes;
9. small plotting helpers, если они действительно нужны.

В `microcavity.py` не хранить конкретные параметры образца — они приходят из `cfg` notebook.

---

# 3. Базовый экспериментальный ориентир

За основу геометрии берем структуру из:

**P. Cilibrizzi et al., “Polariton condensation in a strain-compensated planar microcavity with InGaAs quantum wells”, Applied Physics Letters 105, 191118 (2014), DOI: 10.1063/1.4901814.**

Из статьи используем как экспериментальные ориентиры:

- planar `2λ` GaAs-based microcavity;
- верхний DBR: 23 периода;
- нижний DBR: 26 периодов;
- DBR основаны на GaAs и strain-compensated AlAs/AlP/AlAs layers;
- три doublets из `6 nm` In\(_{0.08}\)Ga\(_{0.92}\)As QWs расположены в antinodes cavity field;
- еще две QW используются как carrier-collection wells около минимумов поля и не входят в основной strong-coupling contribution;
- суммарный vacuum Rabi splitting активных QWs: примерно `8 meV`;
- исследуемый detuning: примерно от `-8 meV` до `+1 meV`;
- reported lower estimate of Q-factor: около `5500`;
- рабочая спектральная область polariton branches находится около `1.45 eV`.

## Упрощения нашей модели относительно статьи

1. Слева и справа — **air**, как было выбрано для проекта; реальный образец статьи не является air/air structure.
2. Strain-compensating AlP insert в DBR в основной версии не моделируется; low-index DBR layer заменяется на effective AlAs layer.
3. В основной части сначала одна QW; multiple-QW structure добавляется в конце.
4. Material dispersion `n(E)` в первой версии не вводится: refractive indices считаются постоянными в рассматриваемом диапазоне.

---

# 4. Единицы и основные физические константы

Во всем проекте использовать только:

\[
E\ [\mathrm{eV}],\qquad
z,d\ [\mu\mathrm m],\qquad
k_\parallel\ [\mu\mathrm m^{-1}].
\]

Основные константы:

\[
\hbar c = 0.1973269804\ \mathrm{eV\,\mu m},
\]

\[
hc = 1.2398419843\ \mathrm{eV\,\mu m},
\]

\[
\hbar = 6.582119569\times10^{-16}\ \mathrm{eV\,s},
\]

\[
\frac{\hbar^2}{2m_0}
=3.80998212\times10^{-8}\ \mathrm{eV\,\mu m^2}.
\]

Числа хранить как `float64`, комплексные матрицы — `complex128`.

---

# 5. Базовые параметры модели

## 5.1. Материалы

Рабочие постоянные значения показателя преломления:

| область | материал | значение |
|---|---|---:|
| external | air | \(n=1\) |
| high-index DBR | GaAs | \(n_H=3.60\) |
| low-index DBR | effective AlAs | \(n_L=2.95\) |
| cavity spacer | GaAs | \(n_C=3.60\) |
| QW background | In\(_{0.08}\)Ga\(_{0.92}\)As | \(n_W\approx3.65\) |

Эти `n` — **рабочие приближенные константы модели**, а не числа, выписанные из Cilibrizzi et al. Если потребуется высокоточное совпадение с экспериментом, следующим уровнем улучшения будет material dispersion `n(E)`.

## 5.2. Центральная энергия

Для baseline:

\[
E_0=E_X(0)=1.450\ \mathrm{eV}.
\]

Это соответствует спектральной области экспериментального образца Cilibrizzi et al.; точное значение затем остается настраиваемым.

Соответствующая длина волны в вакууме:

\[
\lambda_0=\frac{hc}{E_0}
\approx0.85506\ \mu\mathrm m.
\]

## 5.3. Quarter-wave DBR

Начальные толщины:

\[
d_H=\frac{\lambda_0}{4n_H}
\approx 0.05938\ \mu\mathrm m
=59.38\ \mathrm{nm},
\]

\[
d_L=\frac{\lambda_0}{4n_L}
\approx0.07246\ \mu\mathrm m
=72.46\ \mathrm{nm}.
\]

Baseline:

\[
N_{top}=23,\qquad N_{bottom}=26.
\]

## 5.4. Cavity and exact DBR order

Baseline structure is fixed as

\[
\boxed{
\mathrm{air}\,|\,(HL)^{23}\,|\,C_1\,|\,W\,|\,C_2\,|\,(LH)^{26}\,|\,\mathrm{air}
}
\]

with

\[
H=\mathrm{GaAs},\qquad L=\mathrm{AlAs}.
\]

Definitions:

- **top DBR** is the left mirror with 23 pairs;
- **bottom DBR** is the right mirror with 26 pairs;
- light is incident from the **left**, i.e. through the top DBR;
- the layer adjacent to the cavity is \(L\) on both sides.

Thus the full left-to-right sequence is

\[
\mathrm{air}\,|\,
H\,L\,H\,L\,\ldots\,H\,L
\,|\,C_1\,|\,W\,|\,C_2\,|\,
L\,H\,L\,H\,\ldots\,L\,H
\,|\,\mathrm{air}.
\]

The exact termination is part of the model and is not changed between calculations.

The article uses a `2λ` cavity. The initial homogeneous-GaAs estimate is

\[
L_C^{(0)}=\frac{2\lambda_0}{n_C}
\approx0.47504\ \mu\mathrm m.
\]

This is only an initial value. Final \(L_C\) is tuned numerically because the finite DBRs have nonzero reflection phase and penetration depth.

## 5.5. Meaning of \(L_C\) after inserting the QW

The QW **replaces** an equal-thickness part of the GaAs spacer. It is not added on top of the cavity.

Therefore

\[
\boxed{
L_C=d_{C_1}+d_{QW}+d_{C_2}
}
\]

and for any QW position

\[
d_{C_1}+d_{C_2}=L_C-d_{QW}.
\]

If the QW center is at \(z_{QW}\), measured from the left cavity boundary,

\[
d_{C_1}=z_{QW}-\frac{d_{QW}}{2},
\]

\[
d_{C_2}=L_C-z_{QW}-\frac{d_{QW}}{2}.
\]

Thus moving the QW changes only \(d_{C_1}\) and \(d_{C_2}\); the total spacer thickness remains \(L_C\).

## 5.6. Quantum well

Baseline QW:

\[
d_{QW}=6\ \mathrm{nm}=0.006\ \mu\mathrm m.
\]

Exciton in-plane mass:

\[
M_X\approx0.5m_0.
\]

Homogeneous non-radiative exciton HWHM for the starting calculation:

\[
\gamma_X=0.25\ \mathrm{meV}.
\]

The corresponding non-radiative FWHM is

\[
\boxed{
\Gamma_X^{nr}=2\gamma_X
}
\]

and this relation is used throughout the project.

Radiative width \(\Gamma_0\) is not chosen independently; it is calibrated from the single-QW polariton splitting.

# 6. Analytical scalar baseline

См. *Microcavities*, §§5.8.1–5.8.2.

## 6.1. Cavity photon

Для идеальной planar cavity с фиксированным \(k_z\):

\[
E_C(k_\parallel)
=
\frac{\hbar c}{n_C}
\sqrt{k_z^2+k_\parallel^2}.
\]

Если задано \(E_C(0)\), удобнее писать:

\[
E_C(k_\parallel)
=
\sqrt{
E_C(0)^2+
\left(\frac{\hbar c}{n_C}k_\parallel\right)^2
}.
\]

При малом \(k_\parallel\):

\[
E_C(k_\parallel)
\approx
E_C(0)+\frac{\hbar^2k_\parallel^2}{2m_{ph}},
\]

\[
m_{ph}=\frac{E_C(0)n_C^2}{c^2}.
\]

Для \(E_C(0)=1.45\) eV и \(n_C=3.60\):

\[
m_{ph}\approx3.68\times10^{-5}m_0.
\]

В финальном сравнении effective photon mass лучше дополнительно получить фитированием bare-cavity TMM branch около \(k_\parallel=0\), поскольку finite DBR modifies curvature.

## 6.2. QW exciton

\[
E_X(k_\parallel)
=
E_X(0)+
\frac{\hbar^2k_\parallel^2}{2M_X}.
\]

## 6.3. Detuning

\[
\Delta(k_\parallel)=E_C(k_\parallel)-E_X(k_\parallel),
\]

\[
\delta=E_C(0)-E_X(0).
\]

## 6.4. Polariton Hamiltonian

\[
H(k_\parallel)=
\begin{pmatrix}
E_C(k_\parallel) & \Omega_R/2\\
\Omega_R/2 & E_X(k_\parallel)
\end{pmatrix}.
\]

Собственные энергии:

\[
E_{UP,LP}(k_\parallel)
=
\frac{E_C+E_X}{2}
\pm
\frac12
\sqrt{\Delta^2+\Omega_R^2}.
\]

При resonance \(E_C=E_X\):

\[
E_{UP}-E_{LP}=\Omega_R.
\]

## 6.5. Hopfield coefficients

Для lower polariton:

\[
|C_{LP}|^2
=
\frac12
\left(
1-
\frac{\Delta}
{\sqrt{\Delta^2+\Omega_R^2}}
\right),
\]

\[
|X_{LP}|^2=1-|C_{LP}|^2.
\]

Для upper polariton:

\[
|C_{UP}|^2=|X_{LP}|^2,
\qquad
|X_{UP}|^2=|C_{LP}|^2.
\]

## 6.6. Photon energy used in the TMM versus \(2\times2\) comparison

For the final comparison with the full polariton TMM calculation, the photon
diagonal of the \(2\times2\) Hamiltonian is not the ideal planar-cavity
formula and not a reflectivity minimum. It is the real part of the outgoing
eigenmode of the same finite background microcavity with the exciton response
switched off:

\[
\boxed{
E_C^{bg}(k_\parallel)
=
\operatorname{Re}\mathcal E_C^{bg}(k_\parallel)
},
\]

\[
M_{22}^{bg}(\mathcal E_C^{bg},k_\parallel)=0,
\qquad \Gamma_0=0.
\]

Thus the scalar comparison uses

\[
H_{2\times2}(k_\parallel)=
\begin{pmatrix}
E_C^{bg}(k_\parallel) & \Omega_R/2\\
\Omega_R/2 & E_X(k_\parallel)
\end{pmatrix},
\]

\[
E_{LP,UP}^{(2\times2)}
=
\frac{E_C^{bg}+E_X}{2}
\mp
\frac12\sqrt{(E_C^{bg}-E_X)^2+\Omega_R^2}.
\]

The TE comparison uses \(\mathcal E_C^{bg,TE}\). If a separate TM
comparison is made, its photon diagonal is independently obtained from
\(\mathcal E_C^{bg,TM}\). At \(k_\parallel=0\), the tuned background cavity
satisfies

\[
E_C^{bg}(0)=E_X(0).
\]

The ideal photon dispersion remains a separate diagnostic in the bare-cavity
comparison. This separation makes deviations in the polariton comparison a
test of the two-mode approximation rather than a mixture of that approximation
and an idealized photon dispersion.

---

# 7. Transfer-matrix method: единая convention

См. *Microcavities*, §§2.5–2.7 и §4.4.4.

## 7.1. State vector and TE/TM amplitude normalization

Use the forward/backward wave-amplitude vector

\[
\mathbf v(z)=
\begin{pmatrix}
A^+(z)\\
A^-(z)
\end{pmatrix},
\]

with \(A^+\) propagating toward \(+z\) and \(A^-\) toward \(-z\).

The full matrix propagates amplitudes from left to right:

\[
\boxed{
\mathbf v_R=M\mathbf v_L
}.
\]

To stay consistent with the substitutions used in §2.5 of the book:

- in **TE**, \(A^\pm\) are tangential electric-field amplitudes \(E_\tau^\pm\);
- in **TM**, \(A^\pm\) are tangential magnetic-field amplitudes \(cB_\tau^\pm\).

Therefore

\[
Y_j^{TE}=n_j\cos\phi_j,
\]

\[
Y_j^{TM}=\frac{\cos\phi_j}{n_j}.
\]

This is exactly why the same interface formula can be used for both polarizations.

For the final field-profile plot and antinode search we use **TE at \(k_\parallel=0\)** only. No TM electric-field reconstruction is needed for the main project.

## 7.2. Longitudinal wavevector

For layer \(j\),

\[
q_j(E,k_\parallel)
=
\sqrt{
\left(\frac{n_jE}{\hbar c}\right)^2-k_\parallel^2
}.
\]

Also

\[
\cos\phi_j=
\frac{q_j}{n_jE/(\hbar c)}.
\]

### Deterministic square-root branch

For real positive \(E\), choose the root with

\[
\operatorname{Re}q_j>0.
\]

For complex poles, start from the same branch and continue it analytically as \(E\) moves into the lower half-plane. Algorithmically:

1. compute the principal complex square root;
2. if \(\operatorname{Re}q_j<0\), multiply by \(-1\);
3. if \(|\operatorname{Re}q_j|\) is numerically zero, choose the sign for which \(\operatorname{Im}q_j\) has the same sign as \(\operatorname{Im}E\).

In the selected \(E,k_\parallel\) range the solution stays well away from the light-line branch point, so branch switching should not occur.

## 7.3. Propagation matrix

\[
P_j=
\begin{pmatrix}
e^{iq_jd_j} & 0\\
0 & e^{-iq_jd_j}
\end{pmatrix}.
\]

## 7.4. Interface matrix

For interface \(i\to j\),

\[
I_{i\to j}
=
\frac{1}{2Y_j}
\begin{pmatrix}
Y_i+Y_j & Y_j-Y_i\\
Y_j-Y_i & Y_i+Y_j
\end{pmatrix}.
\]

## 7.5. Total matrix

If physical elements are encountered left-to-right as \(M_1,M_2,\ldots,M_N\),

\[
M=M_NM_{N-1}\cdots M_1.
\]

## 7.6. Reflection and transmission

For incidence from the left,

\[
\mathbf v_L=
\begin{pmatrix}
1\\r
\end{pmatrix},
\qquad
\mathbf v_R=
\begin{pmatrix}
t\\0
\end{pmatrix}.
\]

Hence

\[
\boxed{
r=-\frac{M_{21}}{M_{22}}
},
\]

\[
\boxed{
t=\frac{\det M}{M_{22}}
}.
\]

Reflectivity:

\[
R=|r|^2.
\]

For a lossless structure,

\[
T=
\frac{\operatorname{Re}Y_R}
{\operatorname{Re}Y_L}
|t|^2,
\]

\[
R+T\approx1.
\]

The TE/TM basis choice above is also used when interpreting \(t\) and checking energy conservation.

### Note on the book convention

The QW transfer matrix in §4.4.4 is consistent with the boundary condition

\[
M(1,r)^T=(t,0)^T.
\]

In the project all reflection/transmission formulas are derived directly from this fixed convention; matrix-element formulas from different conventions are not mixed.

# 8. Quantum-well resonant response

См. *Microcavities*, §4.4.2 и §4.4.4.

## 8.1. Exciton resonance

\[
r_X(E,k_\parallel)
=
\frac{i\Gamma_0^{pol}}
{E_X(k_\parallel)-E-i\left(\Gamma_0^{pol}+\gamma_X\right)}.
\]

Here both \(\Gamma_0\) and \(\gamma_X\) are **HWHM energy parameters** in the QW response.

\[
\Gamma_X^{nr}=2\gamma_X
\]

is the non-radiative exciton FWHM used in the complex coupled-oscillator comparison.

The radiative parameter \(2\Gamma_0\) is **not added** to the diagonal exciton linewidth of that comparison model. In the TMM, \(\Gamma_0\) already controls the optical QW coupling through the resonant scattering response; adding it again to the diagonal linewidth would double-count the same channel.

For electric-field scattering amplitudes:

\[
t_X^E=1+r_X^E.
\]

## 8.2. TE/TM dependence

\[
\Gamma_0^{TE}
=
\frac{\Gamma_0}{\cos\phi_W},
\]

\[
\Gamma_0^{TM}
=
\Gamma_0\cos\phi_W.
\]

At \(k_\parallel=0\),

\[
\Gamma_0^{TE}=\Gamma_0^{TM}=\Gamma_0.
\]

For TE, the TMM amplitudes are electric-field amplitudes:

\[
r_X^{TE,A}=r_X^E,
\qquad
t_X^{TE,A}=t_X^E.
\]

For TM, the TMM amplitudes are \(cB_\tau^\pm\). Reversing the propagation
direction reverses the sign relating electric and magnetic amplitudes, hence

\[
r_X^{TM,A}=-r_X^E,
\qquad
t_X^{TM,A}=t_X^E.
\]

Thus \(t_X^{TM,A}=1-r_X^{TM,A}\), not \(1+r_X^{TM,A}\). This basis
conversion is required for an active QW to satisfy
\(R_{TE}(E,0)=R_{TM}(E,0)\).

## 8.3. Resonant-sheet matrix

In the TE electric-amplitude basis,

\[
T_X^{TE}
=
\frac1{t_X^E}
\begin{pmatrix}
(t_X^E)^2-(r_X^E)^2 & r_X^E\\
-r_X^E & 1
\end{pmatrix}.
\]

With \(S=\operatorname{diag}(1,-1)\), the TM magnetic-amplitude matrix is

\[
\boxed{T_X^{TM}=S T_X^{TE} S}.
\]

Equivalently, defining

\[
\beta=
\frac{i\Gamma_0^{pol}}
{E_X-E-i\gamma_X},
\]

gives the numerically stable forms

\[
T_X^{TE}=
\begin{pmatrix}
1+\beta & \beta\\
-\beta & 1-\beta
\end{pmatrix},
\qquad
T_X^{TM}=
\begin{pmatrix}
1+\beta & -\beta\\
\beta & 1-\beta
\end{pmatrix}.
\]

## 8.4. Finite QW background layer

The 6 nm InGaAs QW replaces 6 nm of GaAs spacer.

For a QW centered in its finite background layer,

\[
M_{QW}
=
I_{W\to C}\,
P_W(d_{QW}/2)\,
T_X\,
P_W(d_{QW}/2)\,
I_{C\to W}.
\]

With the exciton resonance switched off,

\[
T_X=I
\]

and the block remains an ordinary finite-index InGaAs layer. This **background-QW structure**, not a homogeneous GaAs spacer, defines the bare photon reference used for zero detuning.

# 9. Photon-reference cavity and zero-detuning tuning

The term **photon-reference cavity** means the full intended geometry including the finite-index QW background, but with the exciton resonance switched off:

\[
\Gamma_0=0
\quad\Longleftrightarrow\quad
T_X=I.
\]

This avoids a hidden photon-energy shift when \(n_W\neq n_C\).

## Algorithm

1. Start from
   \[
   L_C^{(0)}=\frac{2\lambda_0}{n_C}.
   \]
2. Build the full background structure
   \[
   \mathrm{air}|(HL)^{23}|C_1|W|C_2|(LH)^{26}|\mathrm{air}
   \]
   with \(T_X=I\).
3. At \(k_\parallel=0\), find its cavity resonance \(E_C^{bg}(0)\).
4. Solve
   \[
   f(L_C)=E_C^{bg}(0;L_C)-E_{target}=0
   \]
   using `scipy.optimize.brentq`.

For zero detuning,

\[
E_{target}=E_X(0).
\]

For a detuning sweep,

\[
E_{target}=E_X(0)+\delta.
\]

Acceptance:

\[
|E_C^{bg}(0)-E_{target}|<10^{-6}\ \mathrm{eV}.
\]

Whenever QW position or the number of QW background blocks changes and the goal is to compare coupling at fixed detuning, this background tuning is repeated before turning the exciton resonance on.

---

# 10. Field reconstruction and baseline QW position

The field is reconstructed only for the tuned **background** cavity, with exciton resonance off, at

\[
E=E_C^{bg}(0),\qquad k_\parallel=0,\qquad \text{TE}.
\]

## 10.1. Partial-matrix reconstruction

For incidence from the left,

\[
\mathbf v_L=
\begin{pmatrix}
1\\r
\end{pmatrix}.
\]

Accumulate transfer matrices from the left boundary to every layer boundary. If the TE amplitudes at the left edge of a homogeneous layer are

\[
\begin{pmatrix}
A_j^+\\A_j^-
\end{pmatrix},
\]

then at a distance \(\xi\) inside that layer,

\[
E_j(\xi)
=
A_j^+e^{iq_j\xi}
+
A_j^-e^{-iq_j\xi}.
\]

Use the common field grid

\[
\Delta z=0.5\ \mathrm{nm}.
\]

For plotting and antinode finding, normalize only after reconstruction:

\[
I(z)=\frac{|E(z)|^2}{\max_{\mathrm{cavity}}|E(z)|^2}.
\]

## 10.2. Antinode finding

Find local maxima of \(I(z)\) inside the spacer. To suppress accidental duplicate peaks, require a minimum peak separation of

\[
\frac{\lambda_0}{4n_C}.
\]

The baseline single-QW position is the detected antinode closest to the geometric cavity center.

Let its offset from the cavity center be

\[
x_0=z_{antinode}-\frac{L_C}{2}.
\]

For a QW-position sweep define a physical displacement \(\Delta z_{QW}\) and keep

\[
z_{QW}
=
\frac{L_C}{2}+x_0+\Delta z_{QW}.
\]

When \(L_C\) is retuned to keep detuning fixed, the same physical offset \(x_0+\Delta z_{QW}\) is preserved.

---

# 11. Calibration of the single-QW coupling

Cilibrizzi et al. report approximately

\[
\Omega_R^{(6QW)}\approx8\ \mathrm{meV}
\]

for six strongly coupled QWs. Assuming approximately equivalent coherent contributions,

\[
\Omega_R^{(1)}
\approx
\frac{8}{\sqrt6}\ \mathrm{meV}
=
3.266\ \mathrm{meV}.
\]

This \(3.266\) meV value is an inferred single-QW target, not a directly reported experimental number.

## Calibration observable

Calibration is performed at **exact zero detuning** using the **real parts of the two complex poles at \(k_\parallel=0\)**:

\[
\boxed{
\operatorname{Re}\mathcal E_{UP}(0)
-
\operatorname{Re}\mathcal E_{LP}(0)
=
3.266\ \mathrm{meV}
}.
\]

Do not calibrate to reflectivity minima and do not minimize the splitting over \(k_\parallel\) inside the calibration loop.

The scalar calibration function is

\[
F(\Gamma_0)
=
\operatorname{Re}\mathcal E_{UP}(0;\Gamma_0)
-
\operatorname{Re}\mathcal E_{LP}(0;\Gamma_0)
-
3.266\ \mathrm{meV}.
\]

Solve

\[
F(\Gamma_0)=0
\]

with `scipy.optimize.brentq`.

The two poles are always sorted by real energy so that

\[
\operatorname{Re}\mathcal E_{LP}
<
\operatorname{Re}\mathcal E_{UP}.
\]

# 12. Complex eigenmodes

For the fixed convention

\[
\mathbf v_R=M\mathbf v_L,
\]

an outgoing eigenmode has no incoming waves:

\[
A_L^+=0,\qquad A_R^-=0.
\]

Therefore

\[
\boxed{
M_{22}(\mathcal E,k_\parallel)=0
}.
\]

## 12.1. Complex-energy convention

Use

\[
\boxed{
\mathcal E=E-i\frac{\Gamma}{2}
}
\]

where \(\Gamma\) is the energy FWHM.

Thus

\[
E=\operatorname{Re}\mathcal E,
\]

\[
\Gamma=-2\operatorname{Im}\mathcal E,
\]

\[
\tau=\frac{\hbar}{\Gamma},
\]

\[
\boxed{
Q=\frac{\operatorname{Re}\mathcal E}{\Gamma}
}.
\]

## 12.2. Complex coupled-oscillator comparison

Use

\[
H=
\begin{pmatrix}
E_C-i\Gamma_C/2 & \Omega_R/2\\
\Omega_R/2 & E_X-i\Gamma_X^{nr}/2
\end{pmatrix},
\]

with

\[
\Gamma_X^{nr}=2\gamma_X.
\]

The cavity FWHM \(\Gamma_C\) is taken from the bare/background cavity pole.

The radiative sheet parameter \(2\Gamma_0\) is not added again to \(\Gamma_X^{nr}\).

# 13. Численная сетка

Используется **один основной набор сеток**, без fast/accurate profiles.

## 13.1. Energy grid

\[
E\in[1.35,1.55]\ \mathrm{eV}.
\]

\[
N_E=4001.
\]

Шаг:

\[
\Delta E=0.05\ \mathrm{meV}.
\]

Эта сетка достаточно широкая для stop-band и одновременно достаточно плотная для визуализации narrow resonances.

Точные resonance positions и linewidths **не определяются только дискретной сеткой** — после grid search используется локальное refinement.

## 13.2. In-plane momentum grid

\[
k_\parallel\in[0,4.0]\ \mu\mathrm m^{-1}.
\]

\[
N_k=241.
\]

\[
\Delta k_\parallel\approx0.0167\ \mu\mathrm m^{-1}.
\]

При \(E=1.35\) eV внешний light cone air имеет

\[
E/(\hbar c)\approx6.84\ \mu\mathrm m^{-1},
\]

поэтому весь выбранный диапазон соответствует propagating external waves.

## 13.3. Field-profile grid

Нужна только внутри cavity:

\[
\Delta z=0.5\ \mathrm{nm}
=5\times10^{-4}\ \mu\mathrm m.
\]

Она используется для:

- \(|E(z)|^2\);
- поиска antinodes;
- размещения QWs.

TMM не дискретизирует слои по \(z\): каждый homogeneous layer считается аналитически одной matrix.

---

# 14. Как считать reflectivity maps

Основная стратегия:

\[
\boxed{
\text{Python loop по }k_\parallel
+\text{ vectorized NumPy calculation по всей }E\text{-grid}
}
\]

Не использовать двойной Python loop по \((E,k_\parallel)\).

Для одного \(k_\parallel\):

- \(E\) — NumPy array длины 4001;
- \(q_j(E)\), phases и все четыре matrix elements считаются сразу arrays;
- loop остается только по физическим слоям.

Это сохраняет простой код и дает достаточную скорость без Numba/JAX/CUDA.

The saved heatmaps use a much denser raw energy grid than can be represented
by the final raster image.  For display, divide the raw energy grid into bins
and plot

\[
R_{TE}^{bin}(\bar E,k_\parallel)
=
\min_{E\in\mathrm{bin}(\bar E)}R_{TE}(E,k_\parallel).
\]

Thus figures 4 and 6 are explicitly **energy-binned reflectivity-minimum
maps**, not point samples of the original \(R_{TE}(E,k_\parallel)\) array.
Their titles, energy-axis labels, and colorbars must use this same definition.
Raw-grid \(\Delta E\) and \(\Delta k\) values stay in notebook diagnostics and
are not printed inside the figures.

---

# 15. Branch extraction from reflectivity

Reflectivity minima are used for visualization and comparison with complex poles. Complex poles remain the authoritative eigenmode result.

## 15.1. Predictions

For the first \(k_\parallel\), use analytical LP/UP estimates.

For subsequent points, use continuation from the previous \(k_\parallel\).

Let predicted energies satisfy

\[
g_{LP}<g_{UP}.
\]

Define

\[
m=\frac{g_{LP}+g_{UP}}{2}.
\]

## 15.2. Non-overlapping adaptive brackets

Use a nominal half-window

\[
w=2.5\ \mathrm{meV}.
\]

The LP and UP minimizations must use non-overlapping intervals:

\[
B_{LP}
=
[g_{LP}-w,\ \min(g_{LP}+w,m-\varepsilon)],
\]

\[
B_{UP}
=
[\max(g_{UP}-w,m+\varepsilon),\ g_{UP}+w],
\]

with

\[
\varepsilon=0.05\ \mathrm{meV}.
\]

Within each bracket use

`scipy.optimize.minimize_scalar(method="bounded")`.

After minimization require:

1. the minimum lies strictly inside its bracket;
2. the minimum is lower than both bracket endpoints;
3. \(E_{LP}<E_{UP}\).

If the brackets collapse or two distinct dips are not resolved, the reflectivity branches are marked unresolved at that \(k_\parallel\); they are **not forced** into two artificial minima. Complex-pole continuation is still used.

Target energy refinement:

\[
10^{-6}\ \mathrm{eV}.
\]

`scipy.signal.find_peaks` is not the main branch tracker.

# 16. Linewidth from real-energy reflectivity

The final linewidth is obtained from the complex pole. Reflectivity FWHM is only a diagnostic and an initial guess for the complex root.

For a resolved reflectivity dip at \(E_{min}\):

1. use a local window
   \[
   E_{min}\pm5\ \mathrm{meV};
   \]
2. find the nearest local maximum on the left and on the right;
3. denote their values by \(R_L^{max}\) and \(R_R^{max}\);
4. define the local background
   \[
   \boxed{
   R_{bg}=\frac{R_L^{max}+R_R^{max}}{2}
   };
   \]
5. define
   \[
   R_{1/2}
   =
   \frac{R_{bg}+R_{min}}{2};
   \]
6. solve
   \[
   R(E)=R_{1/2}
   \]
   on the left and right sides with `scipy.optimize.brentq`;
7. obtain
   \[
   \Gamma_{FWHM}^{(R)}=E_R-E_L.
   \]

If the dip is not bracketed by local maxima on both sides, no reflectivity FWHM is reported for that point. The pole linewidth remains available.

No separate ultra-fine energy grid is introduced.

# 17. Numerical solution of complex eigenmodes

Solve

\[
M_{22}(\mathcal E,k_\parallel)=0
\]

with `scipy.optimize.root` applied to

\[
\operatorname{Re}M_{22}=0,
\qquad
\operatorname{Im}M_{22}=0.
\]

The deterministic square-root branch from §7.2 must be used in every layer.

## Initial guess

For the first \(k_\parallel\):

\[
\operatorname{Re}\mathcal E_0
=
E_{reflectivity\ minimum},
\]

\[
\operatorname{Im}\mathcal E_0
=
-\Gamma_{FWHM}^{(R)}/2.
\]

If reflectivity FWHM is unavailable, use the corresponding complex coupled-oscillator estimate as the initial imaginary part.

For subsequent \(k_\parallel\):

\[
\mathcal E_{start}(k_i)
=
\mathcal E(k_{i-1}).
\]

If continuation fails, retry once from the reflectivity/analytical estimate at the same \(k_\parallel\). No separate calculation profiles are introduced.

## Acceptance

Require stable energies to

\[
|\Delta\operatorname{Re}\mathcal E|<10^{-6}\ \mathrm{eV},
\]

\[
|\Delta\operatorname{Im}\mathcal E|<10^{-6}\ \mathrm{eV}.
\]

Also check

\[
\frac{|M_{22}|}
{\sqrt{|M_{11}|^2+|M_{12}|^2+|M_{21}|^2+|M_{22}|^2}}
<10^{-9}.
\]

This normalized residual is the primary target. For an exciton-like pole,
complex128 rounding can occasionally leave a residual slightly above this
threshold even though the pole energy cannot be improved by one floating-point
step. In that case a root may be accepted as **machine-limited** only if

\[
\frac{|M_{22}|}{\|M\|_F}<5\times10^{-8}
\]

and the local energy backward error

\[
\left|\frac{M_{22}}{\partial M_{22}/\partial\mathcal E}\right|
<10^{-10}\ \mathrm{eV}.
\]

The derivative is evaluated by a centered finite difference after complex
Newton polishing. Machine-limited roots are flagged in diagnostics rather than
silently reported as strict-residual solutions.

At every \(k_\parallel\), sort the two roots by real energy:

\[
\operatorname{Re}\mathcal E_{LP}
<
\operatorname{Re}\mathcal E_{UP}.
\]

# 18. Numerical integration

В основной модели **численное интегрирование не требуется**.

| часть | метод |
|---|---|
| photon/exciton dispersion | аналитически |
| Hopfield LP/UP | аналитически |
| multilayer optics | matrix multiplication |
| QW resonant response | аналитически |
| field profile | аналитическая propagation внутри каждого layer |
| cavity tuning | scalar root finding |
| branch refinement | scalar minimization |
| linewidth crossings | scalar root finding |
| complex modes | 2D root finding |

Не вводить overlap-integrals вручную для QW-position sweep. Положение resonant sheet уже входит в TMM, поэтому изменение coupling должно возникать из полной optical calculation.

---

# 19. TE/TM calculation

Отдельная angle grid не нужна.

Независимая переменная остается \(k_\parallel\), а для каждого layer автоматически вычисляются:

\[
q_j(E,k_\parallel),
\qquad
\cos\phi_j.
\]

TE/TM отличаются только:

1. admittance \(Y_j\);
2. QW radiative width \(\Gamma_0^{TE/TM}\).

Весь остальной TMM code общий.

Обязательная проверка:

\[
R_{TE}(E,0)=R_{TM}(E,0).
\]

Численно требовать примерно:

\[
\max_E|R_{TE}-R_{TM}|<10^{-10}
\]

для структуры без численного шума и при одинаковых material parameters.

---

# 20. DBR-number sweep

Использовать те же routines и те же grids.

Пример:

\[
(N_t,N_b)=
(8,8),(12,12),(16,16),(23,26).
\]

Для каждого случая:

1. tune bare cavity к тому же \(E_{target}\);
2. найти bare complex pole;
3. получить cavity linewidth;
4. при необходимости повторить polariton calculation.

Главная зависимость:

\[
\Gamma_C(N_{DBR}).
\]

Не использовать reported \(Q\approx5500\) как жесткий input: из-за air/air boundaries и упрощенного DBR наша модель не обязана дать точно тот же Q. Значение статьи используется как внешний ориентир.

---

# 21. QW-position sweep

The purpose of this sweep is to isolate the influence of QW position on coupling.

Use

\[
z_{QW}
=
\frac{L_C}{2}+x_0+\Delta z_{QW},
\]

where \(x_0\) is the baseline antinode offset found in §10.

For every \(\Delta z_{QW}\):

1. replace an equal-thickness part of the GaAs spacer by the 6 nm QW background layer;
2. keep the total spacer thickness equal to \(L_C\);
3. with \(T_X=I\), retune \(L_C\) so that the same bare detuning is preserved;
4. turn the exciton resonance back on;
5. solve LP/UP complex poles.

Thus the sweep changes coupling, not accidental bare-photon detuning.

The directly calculated spectral separation is taken from the complex poles:

\[
\Delta E_{pole}(\Delta z)
=
\operatorname{Re}\mathcal E_{UP}(0;\Delta z)
-
\operatorname{Re}\mathcal E_{LP}(0;\Delta z).
\]

No manual \(|E(z_{QW})|^2\) factor is inserted into the coupling.

## 21.1. Field-scaling comparison

The local-field law is used only as a post-processing comparison, not as an
input to the TMM pole calculation.  From the normalized bare-cavity field

\[
I(z)=\frac{|E(z)|^2}{\max |E(z)|^2}
\]

construct a separate local-field coupling estimate

\[
2g_{field}(\Delta z)
=
\Omega_1
\sqrt{
\frac{I(z_0+\Delta z)}{I(z_0)}
}.
\]

Here \(z_0\) is the selected antinode and \(\Omega_1\) is the calibrated
single-QW coupling scale.  Calibration gives
\(2g_{field}(0)=\Delta E_{pole}(0)=\Omega_1\), so the two curves coincide at
the central strong-coupling point.  They are not defined as the same quantity:
near field nodes, losses and the non-Hermitian pole structure can make
\(\Delta E_{pole}\) differ from the local coupling scale \(2g_{field}\).
The two quantities must therefore use different line/marker encodings and
explicit legend labels in the combined two-panel figure.

---

# 22. Multiple-QW extension

The same TMM and pole solver are reused.

The selected article contains three active QW doublets, but the exact intra-doublet barrier thickness is not specified in the chosen source. Therefore the final extension is explicitly an **effective-doublet model**, not an exact reconstruction of every epitaxial layer.

## 22.1. Selecting three antinodes

Use the tuned TE background field at \(k_\parallel=0\).

1. detect all cavity-field antinodes as in §10;
2. rank them by normalized intensity \(I(z)\), highest first;
3. if intensities are numerically equal, prefer the antinode closer to the cavity center;
4. select the first three.

This is deterministic even if more than three antinodes are found.

## 22.2. Effective doublet blocks

At each selected antinode insert one effective 6 nm background block that replaces 6 nm of GaAs spacer, so the total cavity thickness is still

\[
L_C.
\]

The effective block carries twice the single-QW oscillator strength:

\[
\Gamma_{0,doublet}
=
2\Gamma_{0,single}.
\]

This represents two equivalent QWs coherently coupled at nearly the same field antinode without inventing an unknown intra-doublet barrier thickness.

After placing all three background blocks:

1. set all exciton resonances off;
2. retune \(L_C\) once to the target bare detuning;
3. turn the three effective doublet resonances on;
4. calculate complex LP/UP poles.

The validation target is approximately

\[
\operatorname{Re}\mathcal E_{UP}(0)
-
\operatorname{Re}\mathcal E_{LP}(0)
\sim8\ \mathrm{meV}.
\]

The spacer thickness is never increased by adding QWs; QW/effective-doublet regions always replace equal-thickness GaAs regions.

# 23. Основные numerical checks

## TMM

1. Zero-thickness layer дает identity matrix.
2. Interface \(i\to j\) и обратный \(j\to i\) должны давать identity с numerical precision.
3. Для lossless stack:
   \[
   R+T\approx1.
   \]
4. При \(k_\parallel=0\):
   \[
   R_{TE}=R_{TM}.
   \]

## Bare cavity

5. После tuning:
   \[
   |E_C(0)-E_{target}|<0.001\ \mathrm{meV}.
   \]
6. При росте числа DBR periods complex-pole linewidth должна уменьшаться.
7. TMM photon branch около \(k=0\) должна быть approximately parabolic.

## QW/polariton

8. При \(\Gamma_0\to0\) исчезает anticrossing.
9. При zero detuning single-QW splitting после calibration соответствует target \(3.266\) meV.
10. При смещении QW от antinode splitting уменьшается.
11. LP/UP из reflectivity minima должны быть близки к real parts complex poles.

## Multiple QWs

12. Для equivalent QWs coupling должен приблизительно следовать:

\[
\Omega_R(N)\propto\sqrt N.
\]

13. Article-matching six-active-QW extension должна давать splitting порядка `8 meV` после single-QW calibration.

---

# 24. Библиотеки и где их использовать

| библиотека / инструмент | применение |
|---|---|
| `numpy` | grids, complex arrays, vectorized transfer matrices, analytical dispersions |
| `scipy.optimize.brentq` | cavity-length tuning, \(\Gamma_0\) calibration, FWHM crossings |
| `scipy.optimize.minimize_scalar` | точное положение reflectivity minima |
| `scipy.optimize.root` | complex eigenmodes: Re/Im \(M_{22}=0\) |
| `scipy.signal.find_peaks` | только optional diagnostic при отладке, не основной branch tracker |
| `matplotlib` | все графики и heatmaps |
| `tqdm` | progress bar для loop по \(k_\parallel\) и parameter sweeps |
| `jupyter` | основной notebook |
| `nbconvert` | экспорт notebook/report при необходимости |

Не использовать без необходимости:

- pandas;
- numba;
- jax;
- cupy;
- sympy;
- отдельные optimization profiles;
- отдельные config files.

---

# 25. Последовательность расчетов в notebook

## 0. Imports and configuration

Один `cfg` dictionary.

## 1. Materials and geometry

- получить \(\lambda_0,d_H,d_L,L_C^{(0)}\);
- построить layer list;
- вывести \(n(z)\).

## 2. Analytical scalar model

- ideal photon dispersion;
- exciton dispersion;
- LP/UP;
- Hopfield coefficients;
- detuning examples.

## 3. Background finite-DBR cavity

- build the exact DBR termination and finite-index QW background;
- switch the exciton resonance off;
- tune \(L_C\) to the target photon energy;
- calculate TE/TM reflectivity maps;
- continue the bare complex photon pole and define
  \(E_C^{bg}(k)=\operatorname{Re}\mathcal E_C^{bg}(k)\);
- extract reflectivity photon branches only as a separate observable/diagnostic;
- reconstruct the TE \(k_\parallel=0\) field and find antinodes.

## 4. Single-QW polariton TMM

- place the QW at the selected antinode while preserving total \(L_C\);
- retune the background cavity at zero detuning;
- calibrate \(\Gamma_0\) from the \(k_\parallel=0\) complex-pole splitting;
- calculate TE/TM polariton maps;
- extract LP/UP;
- compare reflectivity branches with complex poles and the scalar model.

## 5. DBR-number sweep

- linewidth vs number of periods.

## 6. QW-position sweep

- splitting vs displacement from antinode.

## 7. TE/TM splitting

- TE/TM LP/UP branches;
- \(\Delta_{TE-TM}^{LP,UP}(k)\).

## 8. Complex eigenmodes

- solve \(M_{22}=0\);
- real parts;
- FWHM;
- lifetimes;
- compare poles with reflectivity minima.

## 9. Multiple-QW extension

- effective doublets at antinodes;
- splitting vs number of active QWs/effective doublets;
- compare final splitting with experimental ~8 meV.

## 10. Numerical checks and conclusions

Коротко вывести diagnostics и основные выводы.

## 11. Final figure set

The notebook saves exactly 13 figures:

1. `01_refractive_index_profile.png`;
2. `02_analytical_polariton_dispersion.png`;
3. `03_hopfield_coefficients.png`;
4. `04_bare_cavity_reflectivity_map.png`;
5. `05_bare_photon_dispersion_comparison.png`;
6. `06_polariton_reflectivity_map_te.png`;
7. `07_polariton_tmm_vs_2x2.png`;
8. `08_qw_position_coupling.png` — bare-cavity field and QW-position sweep;
9. `09_dbr_linewidth_sweep.png`;
10. `10_te_tm_polariton_splitting.png` — TE/TM branches and their difference;
11. `11_polariton_linewidths.png`;
12. `12_reflectivity_dips_vs_poles.png`;
13. `13_multiple_qw_scaling.png`.

---

# 26. Итоговые решения, которые считаются зафиксированными

1. Конфиг — обычный `dict` внутри `.ipynb`.
2. Один основной `.py` файл: `microcavity.py`.
3. Свет падает слева через top DBR:
   \[
   \mathrm{air}|(HL)^{23}|C_1|W|C_2|(LH)^{26}|\mathrm{air}.
   \]
4. QW заменяет равную толщину GaAs:
   \[
   L_C=d_{C_1}+d_{QW}+d_{C_2}.
   \]
5. Zero detuning определяется по **background-QW cavity** с \(T_X=I\), а не по cavity без QW layer.
6. TE amplitudes are \(E_\tau^\pm\); TM amplitudes are \(cB_\tau^\pm\).
7. Forward TMM convention:
   \[
   \mathbf v_R=M\mathbf v_L.
   \]
8. Reflection:
   \[
   r=-M_{21}/M_{22}.
   \]
9. Complex eigenmodes:
   \[
   M_{22}=0.
   \]
10. Complex-energy convention:
   \[
   \mathcal E=E-i\Gamma/2.
   \]
11. Linewidth relations:
   \[
   \Gamma_X^{nr}=2\gamma_X,
   \qquad
   Q=\operatorname{Re}\mathcal E/\Gamma.
   \]
12. \(2\Gamma_0\) is not added again to the exciton diagonal linewidth of the complex \(2\times2\) comparison.
13. Single-QW calibration observable:
   \[
   \operatorname{Re}\mathcal E_{UP}(0)
   -
   \operatorname{Re}\mathcal E_{LP}(0)
   =
   3.266\ \mathrm{meV}.
   \]
14. Calibration uses complex poles at \(k_\parallel=0\), not reflectivity minima and not \(\min_k\).
15. Основная сетка:
   \[
   E=1.35\ldots1.55\ \mathrm{eV},\quad N_E=4001,
   \]
   \[
   k_\parallel=0\ldots4\ \mu\mathrm m^{-1},\quad N_k=241.
   \]
16. Field grid:
   \[
   \Delta z=0.5\ \mathrm{nm}.
   \]
17. Complex square roots use a deterministic branch continuous from real positive energy.
18. Reflectivity LP/UP use non-overlapping adaptive brackets; unresolved double dips are not forced.
19. Reflectivity FWHM uses nearest bracketing local maxima to define \(R_{bg}\).
20. Complex poles are traced by continuation in \(k_\parallel\).
21. Field reconstruction uses partial matrices; antinodes are found from the normalized TE \(k_\parallel=0\) field.
22. QW-position sweep retunes the background cavity so bare detuning stays fixed.
23. Multiple-QW extension selects the three highest-field antinodes deterministically.
24. In the TMM-versus-\(2\times2\) polariton comparison, the photon diagonal is
    \(E_C^{bg}(k)=\operatorname{Re}\mathcal E_C^{bg}(k)\) from the same finite
    background cavity with \(\Gamma_0=0\), not the ideal photon formula and not
    a reflectivity minimum.
24. Effective doublet blocks replace GaAs rather than increasing total spacer thickness.
25. Numerical integration is not required in the main model.
26. §4.4.5 motional narrowing / inhomogeneous broadening is not included.

# Источники

1. A. V. Kavokin, J. J. Baumberg, G. Malpuech, F. P. Laussy, *Microcavities*, Oxford University Press, 2007. Основные используемые места: §§2.5–2.7, 4.3.3, 4.4.2, 4.4.4, 5.8.1–5.8.2.
2. P. Cilibrizzi et al., “Polariton condensation in a strain-compensated planar microcavity with InGaAs quantum wells”, *Applied Physics Letters* **105**, 191118 (2014), DOI: 10.1063/1.4901814.
