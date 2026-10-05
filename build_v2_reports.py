"""Build revised Russian and English TeX reports from the preserved masters.

Run after executing microcavity_polaritons_v2.ipynb.  The numerical table is
calculated from the saved Figure 15 data, so the prose cannot drift from it.
"""

from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent
LATEX = ROOT / "latex"
DATA = ROOT / "data_v2" / "fig15_three_methods_comparison.csv"


def once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one anchor, found {count}: {old[:95]!r}")
    return text.replace(old, new)


def between(text, start, end, replacement):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Non-unique interval: {start[:70]!r} / {end[:70]!r}")
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement + text[j:]


def replace_figure(text, old_name, new_name, caption, label=None):
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines)
               if line.startswith(f"\\reportfigure{{{old_name}}}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one figure {old_name}, found {len(matches)}")
    i = matches[0]
    if label is None:
        label = lines[i].rsplit("}{", 1)[-1].split("}")[0]
    lines[i] = f"\\reportfigure{{{new_name}}}{{{caption}}}{{{label}}}\n"
    return "".join(lines)


def after_figure(text, name, addition):
    lines = [line for line in text.splitlines()
             if line.startswith(f"\\reportfigure{{{name}}}")]
    if len(lines) != 1:
        raise ValueError(f"Non-unique figure insertion: {name}")
    return once(text, lines[0] + "\n", lines[0] + "\n\n" + addition + "\n")


def stats_table(russian):
    data = np.genfromtxt(DATA, delimiter=",", names=True)
    rows = []
    for branch in ("lp", "up"):
        valid = data[f"{branch}_R_min_valid"].astype(bool)
        k = data["k_um_inverse"][valid]
        for method, description in (("2x2", "$2\\times2$"),
                                    ("R_min", "$R_{\\min}$")):
            residual = data[f"{branch}_{method}_minus_pole_ueV"][valid]
            rows.append(
                f"{branch.upper()} & {description} & {len(residual)} & "
                f"{k[0]:.1f}--{k[-1]:.1f} & "
                f"{np.sqrt(np.mean(residual**2)):.2f} & "
                f"{np.max(np.abs(residual)):.2f} & "
                f"{residual[0]:+.2f} & {residual[-1]:+.2f} \\\\"
            )
    intro = ("Статистика разностей (в $\\mu$эВ) для точек с разрешёнными минимумами отражения:" if russian
             else "Energy-residual statistics (in $\\mu$eV) for points with resolved reflectivity minima:")
    method = "Способ" if russian else "Method"
    region = "$k$-диапазон" if russian else "$k$ range"
    return rf"""
{intro}
\begin{{center}}
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Branch & {method} & $N$ & {region} & RMS & Max $|\delta E|$ & $k=0$ & $k=k_{{\max}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}
"""



def revise_qw_position(s, russian):
    title = r"\section{Поле в полости и положение QW}" if russian else r"\section{Cavity field and QW position}"
    next_title = r"\section{Влияние числа пар DBR}" if russian else r"\section{Effect of the number of DBR pairs}"
    begin = s.index(title)
    end = s.index("\\clearpage\n" + next_title, begin)
    block = s[begin:end]
    block = block.replace("I(", r"\mathrm{I}(").replace(r"\sqrt{I}", r"\sqrt{\mathrm{I}}")
    block = once(
        block,
        r"2g_{\mathrm{field}}(\Delta z)=\Omega_1\sqrt{\frac{\mathrm{I}(z_0+\Delta z)}{\mathrm{I}(z_0)}}.",
        r"2g_{\mathrm{field}}(\Delta z_{QW})=\Omega_1\sqrt{\frac{\mathrm{I}(z_0+\Delta z_{QW})}{\mathrm{I}(z_0)}}.",
    )
    if russian:
        profile = r"Профиль поля восстанавливается численно с шагом $\Delta z=0.5$ нм. Одиночная QW помещается в пучность, ближайшую к центру полости, а затем смещается относительно нее."
        description = r"""Профиль поля восстанавливается численно с шагом $0.5$ нм. При выбранной длине $L_C=474.87$ нм, близкой к $2\lambda_0/n_C$, период интенсивности равен $\lambda_0/(2n_C)\simeq118.76$ нм. Поэтому видны четыре периода интенсивности (два периода электрического поля): три внутренние пучности и максимумы у обеих границ полости. Пучности задаются фазами отражения DBR. Со стороны полости оба зеркала начинаются со слоя $L$, и на центральной частоте $r\simeq+1$: падающая и отражённая волны складываются у границы. Поле продолжается внутрь зеркал.

Центр одиночной QW помещается в ближайшую к середине полости пучность: $z_0\simeq237.42$ нм. Затем яму смещают относительно этой точки. Толщина QW равна $6$ нм; при таком расположении амплитуда на обеих границах ямы составляет $|E|/|E|_{\max}\simeq0.997$. Размещение в пучности усиливает связь, но максимум поля на границах полости не является обязательным условием работы QW."""
        scale_intro = r"Поскольку $\mathrm{I}(z)\propto|E(z)|^2$, оценку величины $2g$ можно записать как"
        scale_new = r"Поскольку $\mathrm{I}(z)\propto|E(z)|^2$, нормированную оценку расщепления по амплитуде поля запишем как"
        old_definition = r"где $\Omega_1$ --- расщепление одной QW в выбранной пучности, а $z_0$ --- положение этой пучности. Эта формула не подставляется в численный расчет: она строится отдельно и сравнивается с $\Delta E_{\mathrm{pole}}$."
        definition = r"""Здесь $\Omega_1=3.266$ мэВ --- расщепление действительных частей полюсов для одной QW в выбранном положении $z_0$, заданное при калибровке. Корень из отношения интенсивностей равен отношению амплитуд поля. Обозначение $2g_{\mathrm{field}}$ используется для этой оценки: при конечных потерях расщепление полюсов не строго равно удвоенной энергии связи $2g$. Формула использует исходный профиль поля и строится отдельно от полного расчёта TMM."""
        caption = r"Положение QW и сила экситон--фотонной связи. Сверху --- нормированная интенсивность резонансного поля фоновой полости с выключенным экситонным откликом; точки отмечают внутренние пучности, красная линия --- исходное положение QW. Снизу --- оценка расщепления по амплитуде поля и расстояние между действительными частями поляритонных полюсов полного TMM."
        old_end = r"На рис.~\ref{fig:qw-position} обе зависимости почти совпадают в большей части диапазона. Это прямо показывает, что сила связи определяется локальной амплитудой поля. Только совсем близко к узлам поля реальное расстояние между полюсами может уменьшаться быстрее простой оценки, потому что при очень слабой связи заметнее становится влияние потерь."
        discussion = r"""На верхней панели рис.~\ref{fig:qw-position} координата $z$ показана в мкм для всей полости. Нижняя панель использует смещение $\Delta z_{QW}$ в нм около центральной пучности, между ближайшими узлами на расстоянии примерно $59.38$ нм от неё. Для каждого смещения при настройке резонанса сохраняется положение QW относительно середины полости; изменение $L_C$ не превышает $0.17$ нм.

В большей части диапазона оценка согласуется с полным расчётом. Это со\-ответствует зависимости связи от локальной амплитуды поля. У узлов влияние потерь усиливается: реальная часть расщепления уменьшается быстрее оценки по полю. Полюса при этом могут сохранять разные мнимые части."""
    else:
        profile = r"The field profile is reconstructed numerically with a step $\Delta z=0.5$~nm. A single QW is first placed at the field antinode closest to the center of the cavity and is then displaced from this position."
        description = r"""The field profile is reconstructed numerically with a step of $0.5$~nm. For the chosen length $L_C=474.87$~nm, approximately $2\lambda_0/n_C$, the intensity period is $\lambda_0/(2n_C)\simeq118.76$~nm. Thus, the profile contains four intensity periods (two electric-field periods): three interior antinodes and maxima at both cavity boundaries. The antinode positions are set by the DBR reflection phases. Viewed from the cavity, both mirrors start with an $L$ layer and have $r\simeq+1$ at the central frequency, so incident and reflected waves add at the boundary. The field extends into the mirrors.

A single QW is placed at the antinode closest to the cavity center, with coordinate $z_0\simeq237.42$~nm, and is then displaced from this position. The QW thickness is $6$~nm; at this position, the amplitude at both QW boundaries is $|E|/|E|_{\max}\simeq0.997$. Placement at an antinode enhances coupling, but maximum field intensity at the cavity boundaries is not a requirement for QW operation."""
        scale_intro = r"Since $\mathrm{I}(z)\propto|E(z)|^2$, the estimate for $2g$ can be written as"
        scale_new = r"Since $\mathrm{I}(z)\propto|E(z)|^2$, a normalized field-amplitude estimate of the splitting is written as"
        old_definition = r"Here $\Omega_1$ is the splitting for one QW at the selected antinode, and $z_0$ is the position of this antinode. This expression is not inserted into the numerical model. It is calculated separately and compared with $\Delta E_{\mathrm{pole}}$."
        definition = r"""Here $\Omega_1=3.266$~meV is the separation between the real parts of the poles for one QW at the selected position $z_0$, imposed during calibration. The square root of the intensity ratio equals the field-amplitude ratio. The notation $2g_{\mathrm{field}}$ denotes this estimate: with finite losses, the pole splitting is not strictly equal to twice the coupling energy $2g$. The expression uses the original field profile and is evaluated separately from full TMM."""
        caption = r"QW position and exciton--photon coupling strength. The upper panel shows the normalized resonant-field intensity of the background cavity with the exciton response switched off; points mark interior antinodes and the red line marks the initial QW position. The lower panel compares the field-amplitude estimate of the splitting with the separation between the real parts of the full-TMM polariton poles."
        old_end = r"In Figure~\ref{fig:qw-position}, the two dependences almost coincide over most of the range. This directly shows that the coupling strength is controlled by the local field amplitude. Very close to field nodes, the real separation of the poles can decrease faster than the simple estimate because losses become more important when the coupling is very weak."
        discussion = r"""The upper panel of Figure~\ref{fig:qw-position} shows the coordinate $z$ in $\mu$m over the entire cavity. The lower panel uses the displacement $\Delta z_{QW}$ in nm around the central antinode, between the nearest nodes about $59.38$~nm away. For each displacement, the QW position relative to the cavity center is held fixed during resonance tuning; the change in $L_C$ does not exceed $0.17$~nm.

The estimate agrees with full TMM over most of the range, consistent with coupling controlled by the local field amplitude. Near nodes, losses become more important and the real part of the splitting decreases faster than the field estimate. The poles can still retain different imaginary parts."""
    block = once(block, profile, description)
    block = once(block, scale_intro, scale_new)
    block = once(block, old_definition, definition)
    block = replace_figure(block, "09_qw_position_coupling.png", "09_qw_position_coupling.png", caption)
    block = once(block, old_end, discussion)
    return s[:begin] + block + s[end:]

def build(russian):
    old = LATEX / ("report.tex" if russian else "report_en.tex")
    new = LATEX / ("report_v2.tex" if russian else "report_en_v2.tex")
    s = old.read_text()
    if russian:
        s = once(s, r"\graphicspath{{figures/}}", r"\graphicspath{{figures_v2/}}")
    else:
        s = once(s, r"\graphicspath{{figures/}{./}}", r"\graphicspath{{figures_v2/}{./}}")

    if russian:
        captions = {
            "02_analytical_polariton_dispersion.png": "Аналитические дисперсии фотона, экситона и поляритонов при нулевой расстройке. Слева --- исходные ветви по $k_\\parallel$, справа --- их угловое представление при $n_{\\rm ext}=1$. Энергия отсчитывается от $E_X(0)$.",
            "04_bare_cavity_reflectivity_map.png": "TE-отражение фоновой конечной DBR-микрополости без экситонного отклика. Слева карта $R(E,k_\\parallel)$, справа напрямую рассчитанная карта $R(E,\\theta_{\\rm ext})$ с пересчётом $k_\\parallel(E,\\theta)$ для каждой энергии. На рисунке показан минимум в энергетическом интервале $0.050$ мэВ только для визуализации.",
            "05_bare_photon_dispersion_comparison.png": "Фотонная дисперсия идеальной плоской полости и конечной DBR-структуры, полученная из $\\operatorname{Re}\\mathcal E_C^{bg}$. Слева исходные ветви по $k_\\parallel$, справа их визуальное угловое представление. Обе ветви имеют $E_C(0)=E_X(0)$.",
            "06_2x2_photon_model_comparison.png": "Влияние выбора фотонной дисперсии на энергии поляритонов в модели $2\\times2$. Одна и та же связь рассчитана с фотонной ветвью идеальной полости и с фотонной ветвью конечной DBR-структуры; на нижней панели показаны разности энергий LP и UP при одинаковом $k_\\parallel$.",
            "07_polariton_reflectivity_map_te.png": "TE-отражение микрополости с экситонным откликом QW. Слева карта по $k_\\parallel$, справа карта при фиксированном внешнем угле; в обоих случаях минимум в энергетическом интервале $0.025$ мэВ используется только для отображения двух поляритонных особенностей.",
            "11_te_tm_polariton_splitting.png": "TE--TM-расщепление по действительным частям комплексных полюсов. Сверху ветви LP/UP показаны по $k_\\parallel$: TE --- широкая полупрозрачная сплошная линия, TM --- тонкая штриховая. Их различие почти не видно в масштабе мэВ, поэтому снизу отдельно показано $E^{TE}-E^{TM}$ в $\\mu$эВ при том же $k_\\parallel$.",
            "12_fixed_x_transmission_peaks.png": r"Максимумы прохождения при фиксированном тангенциальном $E_x$ и сборе обеих выходных поляризаций. Сверху --- $E_{\alpha,m}-E_X(0)$ для всех найденных LP/UP-максимумов; цвет показывает $T_x$ в максимуме. Чёрный срез $k_y=0$ даёт чистую TM, зелёный $k_x=0$ --- TE. Снизу --- $\delta E_{\alpha,x}$ наиболее сильного максимума относительно среднего чистых TE/TM-максимумов. Разрывы нижних панелей отмечают переключение между пиками. Масштабы одинаковы внутри каждого ряда.",
            "13_polariton_linewidths.png": "Параметры ширины комплексных полюсов LP и UP, $\\Gamma_{LP,UP}=-2\\operatorname{Im}\\mathcal E_{LP,UP}$, как функции $k_\\parallel$ и внешнего угла. Угловые кривые получены визуальным пересчётом уже найденных fixed-$k_\\parallel$ мод.",
            "15_three_methods_comparison.png": "Сравнение трёх способов определения энергии TE-поляритонного резонанса для одной конечной DBR-структуры и одной калибровки: собственных значений эффективной модели $2\\times2$, минимумов отражения и действительных частей комплексных полюсов TMM. Разности рассчитаны при одинаковом реальном $k_\\parallel$ относительно энергий полюсов; штриховые линии соответствуют эффективной модели, кружки --- минимумам отражения.",
        }
    else:
        captions = {
            "02_analytical_polariton_dispersion.png": "Analytical photon, exciton and polariton dispersions at zero detuning. The original branches versus $k_\\parallel$ are shown on the left and their angular representation for $n_{\\rm ext}=1$ on the right. Energy is relative to $E_X(0)$.",
            "04_bare_cavity_reflectivity_map.png": "TE reflectivity of the finite background DBR cavity without the exciton response. The left map is $R(E,k_\\parallel)$; the right map is calculated directly as $R(E,\\theta_{\\rm ext})$, evaluating $k_\\parallel(E,\\theta)$ at every energy. The minimum within a $0.050$-meV energy bin is displayed only for visualization.",
            "05_bare_photon_dispersion_comparison.png": "Photon dispersions of the ideal planar cavity and finite DBR structure, the latter obtained from $\\operatorname{Re}\\mathcal E_C^{bg}$. The original $k_\\parallel$ branches and their visual angular representations are shown. Both satisfy $E_C(0)=E_X(0)$.",
            "06_2x2_photon_model_comparison.png": "Influence of the photon-dispersion model on the $2\\times2$ polariton energies. The same coupling is evaluated with either the ideal-cavity photon dispersion or the finite-DBR photon branch. The lower panel shows the LP and UP differences at equal $k_\\parallel$.",
            "07_polariton_reflectivity_map_te.png": "TE reflectivity of the microcavity with the QW exciton response. The left map is versus $k_\\parallel$ and the right map is calculated at fixed external angle. In both, the minimum within a $0.025$-meV energy bin serves only to display the two polariton features.",
            "11_te_tm_polariton_splitting.png": "TE--TM splitting from the real parts of complex poles. The upper LP/UP curves are plotted versus $k_\\parallel$: TE is a broad translucent solid line and TM a thin dashed line. Their difference is barely visible on the meV scale, so $E^{TE}-E^{TM}$ is plotted separately below in $\\mu$eV at the same $k_\\parallel$.",
            "12_fixed_x_transmission_peaks.png": r"Transmission maxima for fixed tangential $E_x$, collecting both output polarizations. The upper panels show $E_{\alpha,m}-E_X(0)$ for all detected LP/UP maxima; colour gives $T_x$ at each maximum. The black $k_y=0$ cut is pure TM and the green $k_x=0$ cut is TE. The lower panels show $\delta E_{\alpha,x}$ of the strongest maximum relative to the mean pure TE/TM peak energy. Gaps in the lower panels mark peak switches. Axis limits match within each row.",
            "13_polariton_linewidths.png": "Complex-pole linewidth parameters $\\Gamma_{LP,UP}=-2\\operatorname{Im}\\mathcal E_{LP,UP}$ versus $k_\\parallel$ and external angle. The angular curves are visual transformations of the already calculated fixed-$k_\\parallel$ modes.",
            "15_three_methods_comparison.png": "Comparison of three TE polariton resonance-energy definitions for the same finite DBR structure and calibration: effective $2\\times2$ eigenvalues, reflectivity minima, and real parts of complex TMM poles. Residuals are evaluated at equal real $k_\\parallel$ relative to pole energies; dashed lines denote the effective model and circles the reflectivity minima.",
        }

    renames = {
        "06_polariton_reflectivity_map_te.png": "07_polariton_reflectivity_map_te.png",
        "08_qw_position_coupling.png": "09_qw_position_coupling.png",
        "09_dbr_linewidth_sweep.png": "10_dbr_linewidth_sweep.png",
        "10_te_tm_polariton_splitting.png": "11_te_tm_polariton_splitting.png",
        "11_polariton_linewidths.png": "13_polariton_linewidths.png",
        "13_multiple_qw_scaling.png": "14_multiple_qw_scaling.png",
    }
    for old_name in ("02_analytical_polariton_dispersion.png",
                     "04_bare_cavity_reflectivity_map.png",
                     "05_bare_photon_dispersion_comparison.png"):
        s = replace_figure(s, old_name, old_name, captions[old_name])
    for old_name, new_name in renames.items():
        s = replace_figure(s, old_name, new_name,
                           captions.get(new_name) or next(
                               line.split("}{", 1)[1].rsplit("}{", 1)[0]
                               for line in s.splitlines()
                               if line.startswith(f"\\reportfigure{{{old_name}}}")))

    # Original reports and PDFs are control versions; only the v2 outputs change.
    s = revise_ru(s, captions) if russian else revise_en(s, captions)
    s = revise_qw_position(s, russian)
    new.write_text(s)
    print(new.relative_to(ROOT), len(s.splitlines()), "lines")


def revise_ru(s, captions):
    s = once(s, r"\section{TE--TM расщепление}",
             r"\section{TE--TM-расщепление и прохождение при фик\-си\-ро\-ван\-ной тангенциальной поляризации}")
    angle = r"""
Помимо поперечного волнового вектора дисперсия показана как функция внешнего угла
$\theta_{\rm ext}$, отсчитываемого от нормали. Сохранение компоненты волнового
вектора вдоль слоёв даёт
\[
k_\parallel(E,\theta_{\rm ext})
=\frac{n_{\rm ext}E}{\hbar c}\sin\theta_{\rm ext}.
\]
Для воздуха $n_{\rm ext}=1$. Уже найденную ветвь с действительной энергией
переводим в угловое представление по формуле
\[
\theta_{\rm ext}=\arcsin\left(\frac{\hbar c k_\parallel}{n_{\rm ext}E}\right).
\]
Для ветви комплексных полюсов здесь берётся $E=\operatorname{Re}\mathcal E$,
действительная часть комплексной энергии собственной моды. Это визуальное
преобразование fixed-$k_\parallel$ ветви. Спектр при фиксированном угле
$R(E,\theta)$ рассчитывается отдельно: $k_\parallel(E,\theta)$ меняется при
сканировании энергии.
"""
    s = once(s, r"\reportfigure{02_analytical_polariton_dispersion.png}",
             angle + "\n" + r"\reportfigure{02_analytical_polariton_dispersion.png}")

    t_block = r"""
Та же матрица переноса даёт прошедшую амплитуду
\[
t=M_{11}+M_{12}r=\frac{\det M}{M_{22}},
\qquad \det M=\frac{Y_L}{Y_R}.
\]
В общем случае коэффициент прохождения по мощности равен
\[
T=\frac{\operatorname{Re}Y_R}{\operatorname{Re}Y_L}|t|^2.
\]
В рассматриваемой структуре с обеих сторон находится воздух, поэтому $T=|t|^2$.
Величина $A=1-R-T$ показывает долю падающей мощности, выведенную из
когерентных отражённого и прошедшего каналов. При принятой интерпретации
$\gamma_X$ как нерадиационного однородного уширения её можно называть
поглощением.

Для смешанной поляризации $a_j$ --- входные амплитуды TE/TM,
нормированные по потоку, $\tau_j$ --- соответствующие комплексные
амплитуды передачи, $T_j=|\tau_j|^2$, а $u$ --- нормированный в том же
базисе вектор анализатора. Тогда
\[
T_{\rm all}=\sum_{j=TE,TM}|a_j|^2T_j,\qquad
T_u=\left|\sum_{j=TE,TM}u_j^*\tau_j a_j\right|^2.
\]
Поворот TE/TM-базиса и поляризационное расщепление микрополости
рассмотрены в \href{https://doi.org/10.1364/AO.33.005080}{McGuire, Chipman,
\textit{Appl. Opt.} \textbf{33}, 5080 (1994)} и
\href{https://doi.org/10.1103/PhysRevB.59.5082}{Panzarini et al.,
\textit{Phys. Rev. B} \textbf{59}, 5082 (1999)}.
"""
    s = once(s, "R=|r|^2.\n\\]\n", "R=|r|^2.\n\\]\n" + t_block)

    early = r"""
Перед введением полного экситонного отклика сравним два варианта эффективной
модели $2\times2$. Первый использует фотонную дисперсию идеальной полости,
второй --- действительную часть фотонного полюса фоновой конечной структуры:
\[
E_C^{\rm ideal}(k_\parallel)
=\sqrt{E_X(0)^2+\left(\frac{\hbar c k_\parallel}{n_C}\right)^2},
\qquad
E_C^{\rm finite}(k_\parallel)
=\operatorname{Re}\mathcal E_C^{bg}(k_\parallel).
\]
Обе ветви настроены на $E_C^{\rm ideal}(0)=E_C^{\rm finite}(0)=E_X(0)$.
Для каждой диагонализируется одна и та же матрица
\[
H(k_\parallel)=
\begin{pmatrix}E_C(k_\parallel)&g\\g&E_X(k_\parallel)\end{pmatrix},
\qquad g=\Omega_{\rm target}/2,\quad
\Omega_{\rm target}=8/\sqrt6\ {\rm meV}\approx3.27\ {\rm meV}.
\]
В TMM радиационный параметр $\Gamma_0$ калибруется по расщеплению
действительных частей комплексных полюсов. Из-за потерь эта процедура и
подстановка $g=\Omega_{\rm target}/2$ в эрмитову матрицу не обязаны быть
строго эквивалентны на уровне $\mu$эВ.

\reportfigure{06_2x2_photon_model_comparison.png}{CAPTION}{fig:2x2-photon-model}

Разность $\delta E_\alpha^{\rm photon}
=E_\alpha^{2\times2,\rm ideal}-E_\alpha^{2\times2,\rm finite}$ на рис.~\ref{fig:2x2-photon-model}
выделяет влияние выбора фотонной дисперсии. Точность этих вариантов
относительно полного TMM оценивается ниже по отдельным разностям энергий.
""".replace("CAPTION", captions["06_2x2_photon_model_comparison.png"])
    s = once(s, "\\clearpage\n\\section{Квантовая яма в микрополости}",
             early + "\n\\clearpage\n\\section{Квантовая яма в микрополости}")

    qw = r"""
Здесь $r_X$ --- комплексная амплитуда резонансного отражения, а
$\Gamma_0^{pol}$ и $\gamma_X$ задают радиационное и дополнительное однородное
уширения. Знаменатель описывает один однородно уширенный экситонный переход:
экспоненциальное затухание когерентной поляризации во времени даёт в
частотной области полюс $(E_X-E-i\gamma)^{-1}$. В узкой спектральной
окрестности резонанса, где зависимостью $\Gamma_0^{pol}(E)$ можно пренебречь,
квадрат модуля изолированного отклика имеет лоренцеву форму. Гауссова форма
обычно связана с неоднородным распределением резонансных энергий; такое
уширение здесь не введено. Полный спектр отражения многослойной структуры
не обязан быть лоренцевым из-за интерференционного фона.

Параметры $\Gamma_0^{pol}$ и $\gamma_X$ входят в знаменатель как полуширины.
Для изолированного лоренцева профиля
\[
\gamma_{QW}^{pol}=\Gamma_0^{pol}+\gamma_X\quad(\mathrm{HWHM}),
\qquad
\Gamma_{QW}^{pol}=2\gamma_{QW}^{pol}\quad(\mathrm{FWHM}).
\]
Для собственной моды открытой полости
$\mathcal E=E_{\rm mode}-i\Gamma_{\rm pole}/2$,
где $E_{\rm mode}=\operatorname{Re}\mathcal E$ --- действительная часть
комплексной энергии моды, а
$\Gamma_{\rm pole}=-2\operatorname{Im}\mathcal E$ --- параметр ширины
комплексного полюса. Для изолированного лоренцева резонанса он совпадает со
спектральной FWHM. Положение $E_{R_{\min}}$ и ширина полного провала
отражения в общем случае могут отличаться от
$\operatorname{Re}\mathcal E$ и $\Gamma_{\rm pole}$ вследствие интерференции с
нерезонансным фоном.

"""
    s = between(s, "Здесь $r_X$ --- амплитуда отраженной волны",
                "Амплитуда прохождения связана", qw)

    map_text = r"""
Здесь карта отражения показывает две спектральные особенности полной
структуры. Количественное сравнение модели $2\times2$, минимумов отражения и
комплексных полюсов приведено в заключительном разделе. Положения минимумов
для этого сравнения определяются из небинированных спектров.

"""
    s = between(s, "Далее численный результат сравнивается",
                "\\clearpage\n\\section{Поле в полости и положение QW}", map_text)

    s = once(s, "На верхней панели удобно показать сами четыре ветви, а на нижней --- их малую разность, которая на общей энергетической шкале почти не видна.",
             "Рис.~\\ref{fig:te-tm} показывает четыре ветви по $k_\\parallel$ и отдельно их малые разности в $\\mu$эВ.")
    pole_note = ("Комплексные полюса TE и TM ищутся на той же сетке $k_\\parallel=0\\ldots4\\ \\mu\\mathrm m^{-1}$, "
                 "что и остальные дисперсионные ветви. По энергии отдельная сетка для полюса не нужна: "
                 "комплексный корень уточняется непосредственно для каждого значения $k_\\parallel$.")
    s = once(s, pole_note + "\n\n", "")
    s = after_figure(s, "11_te_tm_polariton_splitting.png", pole_note)
    s = once(s, "На общей шкале верхней панели рис.~\\ref{fig:te-tm} TE- и TM-ветви почти совпадают. Нижняя панель показывает, что расщепление все же есть и растет до десятков $\\mu$эВ.",
             "На общей шкале рис.~\\ref{fig:te-tm} TE- и TM-ветви почти совпадают. Панель разностей показывает, что расщепление всё же есть и растёт до десятков $\\mu$эВ.")

    surface = r"""
На рис.~\ref{fig:te-tm-3d} верхние поверхности показывают положения
максимумов спектра прохождения при фиксированном входном тангенциальном
поле вдоль $x$: $E_x=E_0$, $E_y=0$. Детектор собирает обе выходные поляризации без анализатора.
Пусть $k_\parallel=\sqrt{k_x^2+k_y^2}$,
$\varphi=\operatorname{atan2}(k_y,k_x)$ и
$\sin\theta_{\rm ext}=\hbar c k_\parallel/(n_{\rm ext}E)$, где $\theta_{\rm ext}$ --- угол
падения во входной среде ($n_{\rm ext}=1$). Для каждой энергии и
$k_\parallel$ рассчитываются матрицы $M_j$, $j=TE,TM$.
При воздухе с обеих сторон
\[
t_j=\frac{\det M_j}{(M_j)_{22}},\qquad T_j=|t_j|^2.
\]
В изотропной планарной структуре TE- и TM-каналы независимы.
При $k_\parallel\ne0$ они образуют локальный базис собственных поляризаций
относительно плоскости падения. При $k_\parallel=0$ каналы вырождены,
и поперечный базис выбирается произвольно. Фиксированное входное поле
задаёт вклады этих каналов в измеряемый спектр.

Условие поперечности $\mathbf{k}\cdot\mathbf{E}=0$ задаёт компоненту
$E_z$ при фиксированном $E_x=E_0$. Тангенциальная проекция TM-поля
равна $E_{TM}^{in}\cos\theta_{\rm ext}=E_0\cos\varphi$, поэтому
амплитуды по единичным векторам поляризации равны
\[
E_{TM}^{in}=\frac{E_0\cos\varphi}{\cos\theta_{\rm ext}},\qquad
E_{TE}^{in}=-E_0\sin\varphi.
\]
При регистрации полной прошедшей мощности без анализатора вклады
ортогональных TE- и TM-компонент складываются без интерференционного
слагаемого; это верно и для когерентного входного поля.
Нормировка по полной входной мощности даёт
\[
T_x(E,k_\parallel,\varphi)=
\frac{\cos^2\varphi\,T_{TM}(E,k_\parallel)
+\cos^2\theta_{\rm ext}\sin^2\varphi\,T_{TE}(E,k_\parallel)}
{\cos^2\varphi+\cos^2\theta_{\rm ext}\sin^2\varphi}.
\]
В изотропной структуре $M_j$ от азимута не зависят; азимут меняет
веса поляризаций. Поэтому чёрный срез $k_y=0$ соответствует
$T_x=T_{TM}$, а зелёный $k_x=0$ --- $T_x=T_{TE}$.

В каждой области $\alpha=LP,UP$ положения локальных максимумов
$E_{\alpha,m}(k_\parallel,\varphi)$ определяются условиями
\[
\left.\frac{\partial T_x}{\partial E}\right|_{E_{\alpha,m}}=0,
\qquad
\left.\frac{\partial^2T_x}{\partial E^2}\right|_{E_{\alpha,m}}<0.
\]
Сверху показаны $E_{\alpha,m}-E_X(0)$ в мэВ для всех найденных
максимумов; цвет задаётся $T_x(E_{\alpha,m},k_\parallel,\varphi)$.
Для нижних панелей выбирается наиболее сильный максимум:
\[
\begin{aligned}
m_*&=\operatorname*{arg\,max}_{m}
T_x(E_{\alpha,m},k_\parallel,\varphi),\\
\delta E_{\alpha,x}
&=E_{\alpha,m_*}
-\frac{E_{\alpha,\rm peak}^{TM}+E_{\alpha,\rm peak}^{TE}}{2}.
\end{aligned}
\]
Здесь $E_{\alpha,\rm peak}^{j}$ --- положение максимума чистого
$j$-спектра при том же $k_\parallel$; $\delta E_{\alpha,x}$ показано
в $\mu$эВ. Разрывы нижних поверхностей отмечают смену наиболее
сильного пика. Эти величины определяются по спектру прохождения;
энергии собственных мод показаны отдельно на рис.~\ref{fig:te-tm}.
\reportfigure{12_fixed_x_transmission_peaks.png}{CAPTION}{fig:te-tm-3d}

""".replace("CAPTION", captions["12_fixed_x_transmission_peaks.png"])
    s = once(s, "\\clearpage\n\\section{Открытая микрополость: энергия и затухание мод}",
             surface + "\\clearpage\n\\section{Открытая микрополость: энергия и затухание мод}")

    pole = r"""Здесь $E_{\rm mode}=\operatorname{Re}\mathcal E$ --- действительная часть
комплексной энергии собственной моды, а
$\Gamma_{\rm pole}=-2\operatorname{Im}\mathcal E$ --- параметр её ширины и
затухания. Для изолированного лоренцева резонанса он равен спектральной
FWHM; ширина и центр провала отражения всей многослойной структуры могут
отличаться из-за интерференции с нерезонансным фоном. Если
$|\psi(t)|^2\propto\exp(-\Gamma_{\rm pole}t/\hbar)$, то характерное время
жизни равно
\[
\tau(k_\parallel)=\frac{\hbar}{\Gamma_{\rm pole}(k_\parallel)}.
\]

"""
    s = between(s, "Здесь $E=\\operatorname{Re}\\mathcal E$",
                "Чтобы найти такую моду", pole)
    s = once(s, "Отдельный рисунок с действительными частями комплексных энергий здесь не нужен: те же дисперсионные ветви уже показаны выше.",
             "\n\nДействительные части полюсов сравниваются с другими определениями энергии в итоговом разделе.\n\n")
    s = between(s, "Минимумы отражения и комплексные полюса относятся",
                "\\clearpage\n\\section{Коллективная связь нескольких квантовых ям}",
                "Сопоставление положений минимумов отражения с действительными частями полюсов выполнено в итоговом разделе.\n\n")

    final = r"""
\clearpage
\section{Сравнение способов определения энергии\\ поляритонного резонанса}

Для TE-поляризации одной и той же структуры с 23 и 26 парами DBR и одной QW
рассмотрим три определения резонансной энергии при одинаковом действительном
$k_\parallel$. Эффективная модель даёт собственные значения
\[
H(k_\parallel)=
\begin{pmatrix}E_C(k_\parallel)&g\\g&E_X(k_\parallel)\end{pmatrix},
\qquad
E_C(k_\parallel)=\operatorname{Re}\mathcal E_C^{bg}(k_\parallel).
\]
Во втором способе из небинированного действительного TMM-спектра
$R(E,k_\parallel)$ локально уточняются минимумы
$E_{LP,UP}^{R_{\min}}$. Если минимум не разрешён, точка помечается как
недействительная и исключается из статистики; модельная энергия вместо
него не подставляется. В третьем способе решается
\[
M_{22}(\mathcal E,k_\parallel)=0
\]
для двух комплексных полюсов $\mathcal E_{LP,UP}^{pole}$ с отдельным
отслеживанием ветвей. Их действительные части служат опорными энергиями,
а мнимые части используются для ширин.

Это не три независимые физические модели: минимум отражения и полюс
извлекаются из одной полной TMM-структуры, но представляют разные определения
резонанса; матрица $2\times2$ есть эффективное двухмодовое приближение.
Разности энергий определены при одном и том же действительном $k_\parallel$:
\[
\delta E_\alpha^{2\times2}
=E_\alpha^{2\times2}-\operatorname{Re}\mathcal E_\alpha^{pole},
\qquad
\delta E_\alpha^{R}
=E_\alpha^{R_{\min}}-\operatorname{Re}\mathcal E_\alpha^{pole},
\quad \alpha=LP,UP.
\]
\reportfigure{15_three_methods_comparison.png}{CAPTION}{fig:three-methods}

""".replace("CAPTION", captions["15_three_methods_comparison.png"])
    final += stats_table(True)
    final += r"""
Все 241 точки каждой ветви в диапазоне
$k_\parallel=0\ldots4\ \mu\mathrm m^{-1}$ имеют разрешённые минимумы.
Для LP модель $2\times2$ даёт меньший RMS относительно полюса, чем минимум
отражения; для UP минимум отражения имеет немного меньший RMS. Поэтому
согласие методов количественно зависит от ветви. Разность между минимумом отражения и действительной частью полюса
характеризует различие определений резонанса.

"""
    s = once(s, "\\clearpage\n\\section{Заключение}", final + "\n\\clearpage\n\\section{Заключение}")
    conclusion = r"""\section{Заключение}

Аналитическая модель $2\times2$ объясняет антипересечение и фотонные и
экситонные доли LP/UP. Полный TMM-расчёт учитывает конечные DBR, положение
QW, TE/TM-поляризацию и открытость микрополости. Исходные графики по
$k_\parallel$ дополнены угловыми представлениями и картами отражения при фиксированном
внешнем угле.

Сравнение идеальной и конечной фотонных ветвей в модели $2\times2$ показывает,
как приближение фотонной дисперсии влияет на энергии поляритонов. Итоговый
расчёт разностей относительно комплексных полюсов даёт RMS около
$1.93$ и $2.72\ \mu$эВ для LP и UP модели $2\times2$ и около
$6.85$ и $2.01\ \mu$эВ для минимумов отражения. Эти числа относятся к
выбранной структуре, калибровке и диапазону $k_\parallel$; они не делают
разные определения резонанса тождественными.

Изменение ширин комплексных полюсов согласуется со сменой фотонной и
экситонной долей. Для нескольких QW модель воспроизводит коллективный закон
$\Omega_R\propto\sqrt{N_{QW}}$.

"""
    s = between(s, "\\section{Заключение}", "\\end{document}", conclusion)
    return s


def revise_en(s, captions):
    s = once(s, r"\section{TE--TM splitting}",
             r"\section{TE--TM splitting and transmission for a fixed tangential polarization}")
    angle = r"""
In addition to the in-plane wave vector, dispersion is displayed against the
external angle $\theta_{\rm ext}$ measured from the normal. Conservation of
in-plane momentum gives
\[
k_\parallel(E,\theta_{\rm ext})
=\frac{n_{\rm ext}E}{\hbar c}\sin\theta_{\rm ext}.
\]
Here the external medium is air, so $n_{\rm ext}=1$. An already calculated
real-energy branch is displayed against
\[
\theta_{\rm ext}=\arcsin\left(\frac{\hbar c k_\parallel}{n_{\rm ext}E}\right).
\]
For a complex pole we use $E=\operatorname{Re}\mathcal E$, the real part of
the complex eigenmode energy, in this visual transformation. A spectrum at
fixed angle, $R(E,\theta)$, is calculated separately with
$k_\parallel(E,\theta)$ evaluated for every energy.
"""
    s = once(s, r"\reportfigure{02_analytical_polariton_dispersion.png}",
             angle + "\n" + r"\reportfigure{02_analytical_polariton_dispersion.png}")

    t_block = r"""
The same transfer matrix gives the transmitted amplitude
\[
t=M_{11}+M_{12}r=\frac{\det M}{M_{22}},
\qquad \det M=\frac{Y_L}{Y_R}.
\]
For arbitrary external media, power transmittance is
\[
T=\frac{\operatorname{Re}Y_R}{\operatorname{Re}Y_L}|t|^2.
\]
Air is present on both sides here, so $T=|t|^2$. The quantity $A=1-R-T$
is the incident-power fraction removed from the coherent reflected and
transmitted channels. With $\gamma_X$ interpreted as non-radiative
homogeneous loss, it may be called absorptance.

For mixed polarization, let $a_j$ be incident flux-normalized TE/TM
amplitudes, $\tau_j$ the corresponding complex transmission amplitudes,
$T_j=|\tau_j|^2$, and $u$ an analyzer state normalized in the same basis.
Then
\[
T_{\rm all}=\sum_{j=TE,TM}|a_j|^2T_j,\qquad
T_u=\left|\sum_{j=TE,TM}u_j^*\tau_j a_j\right|^2.
\]
TE/TM-basis rotation and polarization splitting in microcavities are
treated by \href{https://doi.org/10.1364/AO.33.005080}{McGuire and Chipman,
\textit{Appl. Opt.} \textbf{33}, 5080 (1994)} and
\href{https://doi.org/10.1103/PhysRevB.59.5082}{Panzarini et al.,
\textit{Phys. Rev. B} \textbf{59}, 5082 (1999)}.
"""
    s = once(s, "R=|r|^2.\n\\]\n", "R=|r|^2.\n\\]\n" + t_block)

    early = r"""
Before introducing the full exciton response, we compare two effective
$2\times2$ models. One uses the ideal planar-cavity photon dispersion and
the other uses the real part of the background finite-DBR photon pole:
\[
E_C^{\rm ideal}(k_\parallel)
=\sqrt{E_X(0)^2+\left(\frac{\hbar c k_\parallel}{n_C}\right)^2},
\qquad
E_C^{\rm finite}(k_\parallel)
=\operatorname{Re}\mathcal E_C^{bg}(k_\parallel).
\]
Both satisfy $E_C^{\rm ideal}(0)=E_C^{\rm finite}(0)=E_X(0)$.
The same matrix is diagonalized for each photon branch,
\[
H(k_\parallel)=
\begin{pmatrix}E_C(k_\parallel)&g\\g&E_X(k_\parallel)\end{pmatrix},
\qquad g=\Omega_{\rm target}/2,\quad
\Omega_{\rm target}=8/\sqrt6\ {\rm meV}\approx3.27\ {\rm meV}.
\]
In TMM, the radiative parameter $\Gamma_0$ is calibrated from the splitting
of the real parts of the complex poles. In the presence of loss, that
procedure and $g=\Omega_{\rm target}/2$ in a Hermitian matrix need not be
equivalent at the $\mu$eV level.

\reportfigure{06_2x2_photon_model_comparison.png}{CAPTION}{fig:2x2-photon-model}

The difference $\delta E_\alpha^{\rm photon}
=E_\alpha^{2\times2,\rm ideal}-E_\alpha^{2\times2,\rm finite}$
in Figure~\ref{fig:2x2-photon-model} isolates the effect of the photon
dispersion. Accuracy relative to the full TMM calculation is quantified
later using separate energy residuals.
""".replace("CAPTION", captions["06_2x2_photon_model_comparison.png"])
    s = once(s, "\\clearpage\n\\section{Quantum well in the microcavity}",
             early + "\n\\clearpage\n\\section{Quantum well in the microcavity}")

    qw = r"""
Here $r_X$ is a complex resonant reflection amplitude, while
$\Gamma_0^{pol}$ and $\gamma_X$ are radiative and additional homogeneous
broadening parameters. The denominator describes a single homogeneously
broadened exciton transition: exponential decay of coherent polarization
in time gives a frequency-domain pole $(E_X-E-i\gamma)^{-1}$. In a narrow
interval around resonance, where the energy dependence of
$\Gamma_0^{pol}(E)$ is negligible, the squared magnitude of the isolated
response has a Lorentzian line shape. Gaussian broadening is associated
primarily with an inhomogeneous distribution of transition energies;
that effect is not included. The complete multilayer reflectivity need
not have a Lorentzian shape because of its interference background.

The parameters $\Gamma_0^{pol}$ and $\gamma_X$ enter the denominator as
half widths. For an isolated Lorentzian profile,
\[
\gamma_{QW}^{pol}=\Gamma_0^{pol}+\gamma_X\quad(\mathrm{HWHM}),
\qquad
\Gamma_{QW}^{pol}=2\gamma_{QW}^{pol}\quad(\mathrm{FWHM}).
\]
For an open-cavity eigenmode,
$\mathcal E=E_{\rm mode}-i\Gamma_{\rm pole}/2$,
where $E_{\rm mode}=\operatorname{Re}\mathcal E$ is the real part of the
complex eigenmode energy and
$\Gamma_{\rm pole}=-2\operatorname{Im}\mathcal E$ is the pole linewidth
parameter. It equals the spectral FWHM for an isolated Lorentzian
resonance. The position $E_{R_{\min}}$ and width of a complete reflectivity
dip can differ from $\operatorname{Re}\mathcal E$ and
$\Gamma_{\rm pole}$ because of interference with a non-resonant background.

"""
    s = between(s, "Here $r_X$ is the amplitude of the reflected wave",
                "The transmission amplitude is related", qw)

    map_text = r"""
At this stage the reflectivity map identifies two spectral features of
the full structure. The quantitative comparison of $2\times2$
eigenenergies, reflectivity minima and complex poles is given in the final
section. Minima for that comparison are found from unbinned spectra.

"""
    s = between(s, "Next, the numerical result is compared",
                "\\clearpage\n\\section{Cavity field and QW position}", map_text)

    s = once(s, "The upper panel shows the four branches themselves, while the lower panel shows their small differences, which are almost invisible on the full energy scale.",
             "Figure~\\ref{fig:te-tm} shows all four branches versus $k_\\parallel$ and their small differences separately in $\\mu$eV.")
    pole_note = ("The complex TE and TM poles are found on the same $k_\\parallel=0\\ldots4\\ \\mu\\mathrm m^{-1}$ "
                 "grid as the other dispersion branches. A separate energy grid is not needed for the poles: "
                 "the complex root is refined directly for every $k_\\parallel$ value.")
    s = once(s, pole_note + "\n\n", "")
    s = after_figure(s, "11_te_tm_polariton_splitting.png", pole_note)
    s = once(s, "On the full scale of the upper panel in Figure~\\ref{fig:te-tm}, the TE and TM branches almost coincide. The lower panel shows that a small splitting is still present and grows to tens of $\\mu$eV.",
             "On the full scale of Figure~\\ref{fig:te-tm}, the TE and TM branches almost coincide. The difference panel shows a splitting that grows to tens of $\\mu$eV.")

    surface = r"""
In Figure~\ref{fig:te-tm-3d}, the upper surfaces show the positions of
transmission-spectrum maxima for a fixed incident tangential field
along $x$: $E_x=E_0$, $E_y=0$. Both transmitted polarizations are collected without an analyzer. Let $k_\parallel=\sqrt{k_x^2+k_y^2}$,
$\varphi=\operatorname{atan2}(k_y,k_x)$ and
$\sin\theta_{\rm ext}=\hbar c k_\parallel/(n_{\rm ext}E)$, where $\theta_{\rm ext}$ is the
entrance-medium incidence angle ($n_{\rm ext}=1$). The matrices $M_j$,
$j=TE,TM$, are evaluated at each energy and $k_\parallel$.
With air on both sides,
\[
t_j=\frac{\det M_j}{(M_j)_{22}},\qquad T_j=|t_j|^2.
\]
In an isotropic planar structure, the TE and TM channels are independent.
For $k_\parallel\ne0$, they form the local eigenpolarization basis
associated with the plane of incidence. At $k_\parallel=0$, the channels
are degenerate and the transverse basis can be chosen arbitrarily.
The fixed incident field sets their contributions to the measured spectrum.

The transversality condition $\mathbf{k}\cdot\mathbf{E}=0$ determines
$E_z$ for fixed $E_x=E_0$. The tangential projection of the TM field is
$E_{TM}^{in}\cos\theta_{\rm ext}=E_0\cos\varphi$, so the amplitudes
along the unit polarization vectors are
\[
E_{TM}^{in}=\frac{E_0\cos\varphi}{\cos\theta_{\rm ext}},\qquad
E_{TE}^{in}=-E_0\sin\varphi.
\]
When both orthogonal output polarizations are collected without an
analyzer, their transmitted powers add without an interference term;
this also holds for coherent incident light.
Normalization to the total incident power gives
\[
T_x(E,k_\parallel,\varphi)=
\frac{\cos^2\varphi\,T_{TM}(E,k_\parallel)
+\cos^2\theta_{\rm ext}\sin^2\varphi\,T_{TE}(E,k_\parallel)}
{\cos^2\varphi+\cos^2\theta_{\rm ext}\sin^2\varphi}.
\]
In the isotropic structure, $M_j$ are independent of azimuth; azimuth
changes the polarization weights. Thus the black $k_y=0$ cut gives
$T_x=T_{TM}$, and the green $k_x=0$ cut gives $T_x=T_{TE}$.

Within each region $\alpha=LP,UP$, the local maxima
$E_{\alpha,m}(k_\parallel,\varphi)$ satisfy
\[
\left.\frac{\partial T_x}{\partial E}\right|_{E_{\alpha,m}}=0,
\qquad
\left.\frac{\partial^2T_x}{\partial E^2}\right|_{E_{\alpha,m}}<0.
\]
The upper panels show $E_{\alpha,m}-E_X(0)$ in meV for all detected
maxima; colour gives $T_x(E_{\alpha,m},k_\parallel,\varphi)$.
The strongest maximum is selected for the lower panels:
\[
\begin{aligned}
m_*&=\operatorname*{arg\,max}_{m}
T_x(E_{\alpha,m},k_\parallel,\varphi),\\
\delta E_{\alpha,x}
&=E_{\alpha,m_*}
-\frac{E_{\alpha,\rm peak}^{TM}+E_{\alpha,\rm peak}^{TE}}{2}.
\end{aligned}
\]
Here $E_{\alpha,\rm peak}^{j}$ is the peak position of the pure
$j$ spectrum at the same $k_\parallel$; $\delta E_{\alpha,x}$ is shown
in $\mu$eV. Gaps in the lower surfaces mark switches of the strongest
peak. These quantities are obtained from transmission spectra;
eigenmode energies are shown separately in Figure~\ref{fig:te-tm}.
\reportfigure{12_fixed_x_transmission_peaks.png}{CAPTION}{fig:te-tm-3d}

""".replace("CAPTION", captions["12_fixed_x_transmission_peaks.png"])
    s = once(s, "\\clearpage\n\\section{Open microcavity: mode energies and decay}",
             surface + "\\clearpage\n\\section{Open microcavity: mode energies and decay}")

    pole = r"""Here $E_{\rm mode}=\operatorname{Re}\mathcal E$ is the real part of
the complex eigenmode energy, and
$\Gamma_{\rm pole}=-2\operatorname{Im}\mathcal E$ is its decay and linewidth
parameter. It equals the spectral FWHM for an isolated Lorentzian
resonance; the position and width of a complete reflectivity dip may
differ because of interference with the non-resonant background. With
$|\psi(t)|^2\propto\exp(-\Gamma_{\rm pole}t/\hbar)$, the characteristic
lifetime is
\[
\tau(k_\parallel)=\frac{\hbar}{\Gamma_{\rm pole}(k_\parallel)}.
\]

"""
    s = between(s, "Here $E=\\operatorname{Re}\\mathcal E$",
                "To find such a mode", pole)
    s = once(s, "A separate figure with the real parts of the complex energies is not needed here because the same dispersion branches have already been shown above.",
             "The real parts of the poles are included in the final quantitative comparison of the three energy definitions.")
    s = between(s, "Reflectivity minima and complex poles describe",
                "\\clearpage\n\\section{Collective coupling of several quantum wells}",
                "The reflectivity minima are compared with the real parts of the poles in the final section.\n\n")

    final = r"""
\clearpage
\section{Comparison of polariton resonance definitions}

For TE polarization of the same 23/26-pair DBR structure with one QW,
we compare three energy definitions at equal real $k_\parallel$. The
effective model gives the eigenvalues of
\[
H(k_\parallel)=
\begin{pmatrix}E_C(k_\parallel)&g\\g&E_X(k_\parallel)\end{pmatrix},
\qquad
E_C(k_\parallel)=\operatorname{Re}\mathcal E_C^{bg}(k_\parallel).
\]
Second, the reflectivity minima $E_{LP,UP}^{R_{\min}}$ are locally
refined from the unbinned real-energy TMM spectrum $R(E,k_\parallel)$.
An unresolved minimum is marked invalid and omitted from statistics;
no model prediction is substituted. Third, the two complex poles
$\mathcal E_{LP,UP}^{pole}$ are separately tracked by solving
\[
M_{22}(\mathcal E,k_\parallel)=0.
\]
Their real parts serve as reference eigenmode energies; their imaginary
parts determine the linewidth parameters.

These are not three independent physical models. The reflectivity
minima and poles are different resonance definitions obtained from the
same full TMM structure, while the $2\times2$ matrix is an effective
two-mode description. At equal real $k_\parallel$ the residuals are
\[
\delta E_\alpha^{2\times2}
=E_\alpha^{2\times2}-\operatorname{Re}\mathcal E_\alpha^{pole},
\qquad
\delta E_\alpha^{R}
=E_\alpha^{R_{\min}}-\operatorname{Re}\mathcal E_\alpha^{pole},
\quad \alpha=LP,UP.
\]
\reportfigure{15_three_methods_comparison.png}{CAPTION}{fig:three-methods}

""".replace("CAPTION", captions["15_three_methods_comparison.png"])
    final += stats_table(False)
    final += r"""
All 241 points on each branch in
$k_\parallel=0\ldots4\ \mu\mathrm m^{-1}$ have resolved minima. The
$2\times2$ model has a lower RMS pole residual for LP, while the
reflectivity minimum has a slightly lower RMS for UP. Agreement therefore
depends quantitatively on the branch. The shift of a reflectivity minimum relative to the real part of the pole
reflects the different resonance definitions.

"""
    s = once(s, "\\clearpage\n\\section{Conclusion}",
             final + "\n\\clearpage\n\\section{Conclusion}")
    conclusion = r"""\section{Conclusion}

The analytical $2\times2$ model explains the anticrossing and the
photon and exciton fractions of LP/UP. Full TMM includes finite DBRs,
the QW position, TE/TM polarization and the open nature of the cavity.
The original $k_\parallel$ plots are complemented by angular views and
reflectivity maps calculated at fixed external angle.

Comparing ideal and finite-DBR photon branches in the $2\times2$ model
shows how the photon-dispersion approximation affects polariton energies.
The final pole-referenced residuals have RMS values of about
$1.93$ and $2.72\ \mu$eV for LP and UP in the $2\times2$ model, and
$6.85$ and $2.01\ \mu$eV for reflectivity minima. These values apply
to the chosen structure, calibration and $k_\parallel$ range; the
different resonance definitions remain distinct.

The changes in pole linewidths are consistent with the evolving
photon and exciton fractions. For several QWs, the model reproduces
the collective scaling $\Omega_R\propto\sqrt{N_{QW}}$.

"""
    s = between(s, "\\section{Conclusion}", "\\end{document}", conclusion)
    return s


if __name__ == "__main__":
    build(True)
    build(False)
