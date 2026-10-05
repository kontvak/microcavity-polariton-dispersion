"""Build the revised, separately executable polariton notebook from the baseline.

The baseline notebook and its saved outputs remain available for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


BASE = Path("microcavity_polaritons.ipynb")
OUTPUT = Path("microcavity_polaritons_v2.ipynb")
notebook = json.loads(BASE.read_text())


def source(index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def put(index: int, value: str) -> None:
    cell = notebook["cells"][index]
    cell["source"] = value.strip("\n").splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def code_cell(value: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(value: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": value.strip("\n").splitlines(keepends=True),
    }


cell = source(1)
cell = cell.replace(
    "from matplotlib.colors import PowerNorm\n",
    "from matplotlib.colors import PowerNorm, TwoSlopeNorm, LogNorm\n"
    "from matplotlib.lines import Line2D\n"
    "import matplotlib.tri as mtri\n"
    "from scipy.signal import find_peaks\n",
)
cell = cell.replace(
    'FIGURE_DIR = Path("figures")\nFIGURE_DIR.mkdir(exist_ok=True)',
    'FIGURE_DIR = Path("figures_v2")\nFIGURE_DIR.mkdir(exist_ok=True)\n'
    'DATA_DIR = Path("data_v2")\nDATA_DIR.mkdir(exist_ok=True)',
)
cell = cell.replace(
    'NUMERICAL_GRID_STYLE = {"marker": ".", "markersize": 2.0}',
    '''def save_csv(filename, columns, arrays):
    np.savetxt(
        DATA_DIR / filename, np.column_stack(arrays), delimiter=",",
        header=",".join(columns), comments="",
    )

NUMERICAL_GRID_STYLE = {"marker": ".", "markersize": 2.0}''',
)
cell = cell.replace(
    '    "map_display_rows": 1200,',
    '    "map_display_rows": 1200,\n'
    '    "angle_max_degrees": 30.0,\n'
    '    "angle_points": 241,',
)
put(1, cell)

cell = source(3).replace(
    'momentum_grid = np.linspace(cfg["momentum_min"], cfg["momentum_max"], cfg["momentum_points"])',
    'momentum_grid = np.linspace(cfg["momentum_min"], cfg["momentum_max"], cfg["momentum_points"])\n'
    'angle_grid = np.linspace(0.0, cfg["angle_max_degrees"], cfg["angle_points"])',
)
put(3, cell)

_, hopfield_part = source(4).split("# 3. Hopfield coefficients", 1)
put(4, r'''
energy_offset = lambda values: mc.to_mev(np.asarray(values) - cfg["energy_zero"])
branch_angle = lambda values: mc.external_angle_degrees(
    momentum_grid, values, n_external=cfg["n_air"]
)

# 2. Analytical polariton dispersion at equal k and as angular branch plots
fig, axes = plt.subplots(1, 2, figsize=(10.2, FIGURE_HEIGHT), layout="constrained")
branches = (
    (ideal_photon_grid, "--", "0.5", "bare photon"),
    (exciton_grid, ":", "0.35", "bare exciton"),
    (analytical_lp, "-", "C0", "lower polariton (LP)"),
    (analytical_up, "-", "C1", "upper polariton (UP)"),
)
for values, style, color, label in branches:
    axes[0].plot(momentum_grid, energy_offset(values), style, color=color, label=label)
    axes[1].plot(branch_angle(values), energy_offset(values), style, color=color, label=label)
axes[0].set(xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
            ylabel=r"energy relative to $E_X(0)$ (meV)")
axes[1].set(xlabel=r"external angle $\theta_{\rm ext}$ (degrees)",
            ylabel=r"energy relative to $E_X(0)$ (meV)")
axes[0].legend(title=r"Analytical $2\times2$ model", fontsize=7)
save_show(fig, "02_analytical_polariton_dispersion.png")

''' + "# 3. Hopfield coefficients" + hopfield_part)

put(10, r'''
# 4. Bare-cavity TE reflectivity maps at fixed k and fixed external angle
bare_display_energy, bare_display_values = minimum_energy_bins(
    bare_map_energy, bare_reflectivity_te, cfg["map_display_rows"]
)
bare_display_bin_mev = mc.to_mev(
    (bare_map_energy[-1] - bare_map_energy[0]) / cfg["map_display_rows"]
)
bare_vmin = float(np.nanmin(bare_reflectivity_te))
del bare_reflectivity_te
bare_reflectivity_angle = mc.angle_resolved_reflectivity_map(
    bare_map_energy, angle_grid, "TE", background_layers,
    n_left=cfg["n_air"], n_right=cfg["n_air"], progress=True,
)
bare_angle_energy, bare_angle_values = minimum_energy_bins(
    bare_map_energy, bare_reflectivity_angle, cfg["map_display_rows"]
)
bare_vmin = min(bare_vmin, float(np.nanmin(bare_reflectivity_angle)))
del bare_reflectivity_angle
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
for ax, values, x, x_label in (
    (axes[0], bare_display_values, momentum_grid,
     r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)"),
    (axes[1], bare_angle_values, angle_grid,
     r"fixed external angle $\theta_{\rm ext}$ (degrees)"),
):
    image = ax.imshow(
        values.T, origin="lower", aspect="auto",
        extent=(x[0], x[-1], bare_display_energy[0], bare_display_energy[-1]),
        interpolation="bilinear", cmap="magma", vmin=bare_vmin, vmax=1.0,
    )
    ax.set(xlabel=x_label, ylabel="energy-bin center E (eV)")
    ax.grid(False)
fig.colorbar(
    image, ax=axes,
    label=rf"minimum TE reflectivity per {bare_display_bin_mev:.3f} meV energy bin",
    pad=0.02,
)
save_show(fig, "04_bare_cavity_reflectivity_map.png", dpi=300)

# 5. Photon pole of finite DBR versus ideal planar-cavity photon
fig, axes = plt.subplots(1, 2, figsize=(10.2, FIGURE_HEIGHT), layout="constrained")
for values, style, label in (
    (bare_branch, "-", "finite-DBR photon pole"),
    (ideal_photon_grid, "--", "ideal planar cavity"),
):
    axes[0].plot(momentum_grid, energy_offset(values), style, label=label)
    axes[1].plot(branch_angle(values), energy_offset(values), style, label=label)
axes[0].set(xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
            ylabel=r"energy relative to $E_X(0)$ (meV)")
axes[1].set(xlabel=r"external angle $\theta_{\rm ext}$ (degrees)",
            ylabel=r"energy relative to $E_X(0)$ (meV)")
axes[0].legend(fontsize=7)
save_show(fig, "05_bare_photon_dispersion_comparison.png")

# 6. Effect of photon-dispersion choice inside the same Hermitian 2x2 model
ideal_model_lp, ideal_model_up = mc.polariton_energies(
    ideal_photon_grid, exciton_grid, cfg["single_qw_rabi_target"]
)
finite_model_lp, finite_model_up = mc.polariton_energies(
    bare_branch, exciton_grid, cfg["single_qw_rabi_target"]
)
photon_model_delta_lp = mc.to_mev(ideal_model_lp - finite_model_lp)
photon_model_delta_up = mc.to_mev(ideal_model_up - finite_model_up)
save_csv(
    "fig06_2x2_photon_model_comparison.csv",
    ("k_um_inverse", "ideal_lp_eV", "ideal_up_eV", "finite_lp_eV",
     "finite_up_eV", "ideal_minus_finite_lp_meV", "ideal_minus_finite_up_meV"),
    (momentum_grid, ideal_model_lp, ideal_model_up, finite_model_lp,
     finite_model_up, photon_model_delta_lp, photon_model_delta_up),
)
fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 5.6),
                         sharex=True, layout="constrained")
for branch, ideal, finite, color in (
    ("LP", ideal_model_lp, finite_model_lp, "C0"),
    ("UP", ideal_model_up, finite_model_up, "C1"),
):
    axes[0].plot(momentum_grid, energy_offset(ideal), "--", color=color,
                 label=f"{branch}: ideal photon")
    axes[0].plot(momentum_grid, energy_offset(finite), "-", color=color,
                 label=f"{branch}: finite-DBR photon")
axes[0].set(ylabel=r"energy relative to $E_X(0)$ (meV)")
axes[0].legend(ncol=2, fontsize=7)
axes[1].plot(momentum_grid, photon_model_delta_lp, label="LP")
axes[1].plot(momentum_grid, photon_model_delta_up, label="UP")
axes[1].axhline(0.0, color="0.5", lw=0.7)
axes[1].set(xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
            ylabel="ideal minus finite (meV)")
axes[1].legend()
save_show(fig, "06_2x2_photon_model_comparison.png")
''')

put(14, r'''
# 7. TE polariton reflectivity maps at fixed k and fixed external angle
polariton_display_energy, polariton_display_values = minimum_energy_bins(
    polariton_map_energy, polariton_reflectivity_te, cfg["map_display_rows"]
)
polariton_display_bin_mev = mc.to_mev(
    (polariton_map_energy[-1] - polariton_map_energy[0]) / cfg["map_display_rows"]
)
polariton_vmin = float(np.nanmin(polariton_reflectivity_te))
del polariton_reflectivity_te
polariton_reflectivity_angle = mc.angle_resolved_reflectivity_map(
    polariton_map_energy, angle_grid, "TE", polariton_layers,
    n_left=cfg["n_air"], n_right=cfg["n_air"], progress=True,
)
polariton_angle_energy, polariton_angle_values = minimum_energy_bins(
    polariton_map_energy, polariton_reflectivity_angle, cfg["map_display_rows"]
)
polariton_vmin = min(polariton_vmin, float(np.nanmin(polariton_reflectivity_angle)))
del polariton_reflectivity_angle
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
for ax, values, x, x_label in (
    (axes[0], polariton_display_values, momentum_grid,
     r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)"),
    (axes[1], polariton_angle_values, angle_grid,
     r"fixed external angle $\theta_{\rm ext}$ (degrees)"),
):
    image = ax.imshow(
        values.T, origin="lower", aspect="auto",
        extent=(x[0], x[-1], polariton_display_energy[0], polariton_display_energy[-1]),
        interpolation="bilinear", cmap="magma",
        norm=PowerNorm(gamma=6.0, vmin=polariton_vmin, vmax=1.0),
    )
    ax.set(xlabel=x_label, ylabel="energy-bin center E (eV)")
    ax.grid(False)
colorbar = fig.colorbar(
    image, ax=axes,
    label=rf"minimum TE reflectivity per {polariton_display_bin_mev:.3f} meV energy bin",
    pad=0.02,
)
colorbar.set_ticks([0.8, 0.95, 1.0])
save_show(fig, "07_polariton_reflectivity_map_te.png", dpi=300)

# 8. Fixed-angle real-energy R, T, and A spectra
representative_angles = (0.0, 15.0, 30.0)
spectral_energy = np.linspace(1.44, 1.47, 6001)
spectral_records = []
fig, axes = plt.subplots(3, 2, figsize=(10.2, 8.2),
                         sharex=True, layout="constrained")
fig.suptitle(
    r"Fixed-angle TE spectra: $R+T+A=1$; UP shifts while LP fades in $R$",
    fontsize=11,
)
for row, angle in enumerate(representative_angles):
    reflectance_ax, loss_ax = axes[row]
    k_at_energy = mc.external_momentum(spectral_energy, angle, n_external=cfg["n_air"])
    reflectance = mc.reflectivity(
        spectral_energy, k_at_energy, "TE", polariton_layers,
        n_left=cfg["n_air"], n_right=cfg["n_air"],
    )
    transmission = mc.transmittance(
        spectral_energy, k_at_energy, "TE", polariton_layers,
        n_left=cfg["n_air"], n_right=cfg["n_air"],
    )
    missing_coherent_fraction = 1.0 - reflectance - transmission
    assert np.min(missing_coherent_fraction) > -1.0e-8
    spectral_records.append(np.column_stack((
        np.full(spectral_energy.shape, angle), spectral_energy, k_at_energy,
        reflectance, transmission, missing_coherent_fraction,
    )))
    reflectance_ax.plot(energy_offset(spectral_energy), reflectance,
                        label="R", color="C0")
    loss_ax.semilogy(energy_offset(spectral_energy), transmission,
                     label="T", color="C1")
    loss_ax.semilogy(energy_offset(spectral_energy), missing_coherent_fraction,
                     label="A = 1 − R − T", color="C2")
    for ax in (reflectance_ax, loss_ax):
        ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    reflectance_dips, _ = find_peaks(
        -reflectance, prominence=max(1e-4, 0.03 * (1.0 - reflectance.min())),
        distance=60,
    )
    transmission_peaks, _ = find_peaks(
        transmission, prominence=max(1e-7, 0.02 * transmission.max()),
        distance=60,
    )
    loss_features, _ = find_peaks(
        missing_coherent_fraction,
        prominence=max(5e-6, 1e-4 * missing_coherent_fraction.max()),
        distance=60,
    )
    assert len(loss_features) == 2, f"expected LP and UP features at {angle:g} degrees"
    reflectance_ax.scatter(energy_offset(spectral_energy[reflectance_dips]),
                           reflectance[reflectance_dips], marker="v", s=25,
                           color="C0", label="R minimum")
    loss_ax.scatter(energy_offset(spectral_energy[transmission_peaks]),
                    transmission[transmission_peaks], marker="^", s=25,
                    color="C1", label="T maximum")
    for branch, index in zip(("LP", "UP"), sorted(loss_features)[:2]):
        loss_ax.annotate(
            branch,
            (energy_offset(spectral_energy[index]), missing_coherent_fraction[index]),
            xytext=(0, -17),
            textcoords="offset points", ha="center", fontsize=8,
            fontweight="bold", color="C2",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.4},
        )
    reflectance_ax.set(ylabel="R (fraction, linear)",
                       title=rf"$\theta_{{\rm ext}}={angle:g}^\circ$")
    reflectance_ax.set_ylim(max(0.0, reflectance.min() - 0.08), 1.01)
    loss_ax.set(ylabel="T, A (fraction, log)",
                title=rf"$\theta_{{\rm ext}}={angle:g}^\circ$")
axes[0, 0].legend(ncol=2, fontsize=7)
axes[0, 1].legend(ncol=3, fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel(r"energy relative to $E_X(0)$ (meV)")
save_show(fig, "08_RTA_fixed_angles.png")
np.savetxt(
    DATA_DIR / "fig08_RTA_fixed_angles.csv", np.vstack(spectral_records),
    delimiter=",",
    header="angle_degrees,energy_eV,k_um_inverse,R,T,A_coherent_loss", comments="",
)
''')

put(16, source(16).replace(
    'save_show(fig, "09_dbr_linewidth_sweep.png")',
    'save_show(fig, "10_dbr_linewidth_sweep.png")',
))
put(18, source(18).replace(
    'save_show(fig, "08_qw_position_coupling.png")',
    'save_show(fig, "09_qw_position_coupling.png")',
).replace(
    r"2g(z)=\Omega_1\sqrt{I(z)/I(z_0)}",
    r"2g_{\rm field}(\Delta z_{QW})=\Omega_1\sqrt{\mathrm{I}(z_0+\Delta z_{QW})/\mathrm{I}(z_0)}",
))

notebook["cells"][19] = markdown_cell(r"""
## 7. TE–TM splitting and polarized transmission

Figure 11 shows the complex-pole dispersion and its TE–TM splitting.
Figure 12 fixes the tangential electric field along x at the entrance
interface and collects both transmitted polarizations without an analyzer.
TE/TM matrices depend on energy and radial momentum; their incident power
weights also depend on azimuth and incidence angle. Actual spectral maxima
are found in separate LP/UP windows, keeping both maxima of a resolved
doublet. The lower panels show the strongest maximum relative to the mean
of the pure TE and TM transmission maxima; switches between two resolved
maxima in the lower panels are drawn with gaps. The upper panels connect
continuing local maxima across one-/two-peak boundaries. A diagnostic
spectral map checks the continuous transmitted intensity directly.
""")

put(20, r'''
# 11. TE/TM pole dispersion and splitting
te_tm_lp = mc.to_mev(poles_te["lp"].real - poles_tm["lp"].real)
te_tm_up = mc.to_mev(poles_te["up"].real - poles_tm["up"].real)
assert abs(te_tm_lp[0]) < 1e-7 and abs(te_tm_up[0]) < 1e-7
fig, axes = plt.subplots(
    2, 1, figsize=(FIGURE_WIDTH, 5.6), sharex=True,
    gridspec_kw={"height_ratios": (1.5, 1.0)}, layout="constrained",
)
for branch, color in (("lp", "C0"), ("up", "C1")):
    axes[0].plot(momentum_grid, energy_offset(poles_te[branch].real),
                 "-", color=color, linewidth=4.0, alpha=0.42,
                 label=f"{branch.upper()} TE: broad solid")
    axes[0].plot(momentum_grid, energy_offset(poles_tm[branch].real),
                 "--", color=color, linewidth=1.6,
                 label=f"{branch.upper()} TM: thin dashed")
axes[0].set(
    ylabel=r"mode energy relative to $E_X(0)$ (meV)",
    title="TE and TM overlap on the meV energy scale",
)
axes[0].legend(ncol=2, fontsize=7)
axes[1].plot(momentum_grid, te_tm_lp * 1e3, color="C0", label="LP: TE − TM")
axes[1].plot(momentum_grid, te_tm_up * 1e3, color="C1", label="UP: TE − TM")
axes[1].axhline(0.0, color="0.5", linewidth=0.7)
axes[1].set(
    xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
    ylabel=r"$E_{\rm TE}-E_{\rm TM}$ ($\mu$eV)",
    title="Polarization splitting at equal real in-plane wave vector",
)
axes[1].legend(fontsize=7)
save_show(fig, "11_te_tm_polariton_splitting.png")

# 12. Transmission maxima for fixed tangential E_x and no output analyzer.
# Every spectrum uses actual TE/TM transfer matrices at the grid's radial k.
# Reusing those matrices across azimuths is exact for this isotropic stack.
half_surface_grid = 80
kx = np.linspace(-cfg["momentum_max"], cfg["momentum_max"],
                 2 * half_surface_grid + 1)
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky)
IX, IY = np.meshgrid(np.arange(-half_surface_grid, half_surface_grid + 1),
                        np.arange(-half_surface_grid, half_surface_grid + 1))
shell = IX**2 + IY**2
KR = np.hypot(KX, KY)
outside_radial_grid = shell > half_surface_grid**2
surface_data = {
    "kx": kx, "ky": ky, "radial_k": KR,
    "outside_radial_grid": outside_radial_grid,
    "input_field_angle_degrees": np.array(0.0),
    "output_analyzer": np.array("none"),
    "relative_peak_prominence": np.array(1.0e-5),
}
sample_spectra = {"azimuth_degrees": np.array([0.0, 45.0, 90.0])}


def transmission_window(momentum, branch, azimuths, refinement=1):
    poles = {
        b: np.array([
            complex(np.interp(momentum, momentum_grid, p[b].real),
                    np.interp(momentum, momentum_grid, p[b].imag))
            for p in (poles_te, poles_tm)
        ])
        for b in ("lp", "up")
    }
    middle = 0.5 * (max(poles["lp"].real) + min(poles["up"].real))
    widths = mc.linewidth_from_pole(poles[branch])
    padding = max(3.0 * max(widths), 2.0 * np.ptp(poles[branch].real), 1.0e-5)
    lower = min(poles[branch].real) - padding
    upper = max(poles[branch].real) + padding
    if branch == "lp":
        upper = min(upper, middle)
    else:
        lower = max(lower, middle)
    samples = max(4001, int(np.ceil((upper - lower) / min(widths) * 120)) + 1)
    energies = np.linspace(lower, upper, refinement * (samples - 1) + 1)
    result = mc.fixed_tangential_polarization_rta(
        energies[:, None], float(momentum), np.asarray(azimuths)[None, :],
        polariton_layers, n_left=cfg["n_air"], n_right=cfg["n_air"],
    )
    return energies, result


for branch in ("lp", "up"):
    for key in (
        "single_energy_eV", "single_height", "lower_energy_eV", "lower_height",
        "upper_energy_eV", "upper_height", "dominant_energy_eV", "dominant_height",
        "tm_peak_eV", "te_peak_eV", "reference_eV", "angular_shift_ueV",
    ):
        surface_data[f"{branch}_{key}"] = np.full(KR.shape, np.nan)
    surface_data[f"{branch}_peak_count"] = np.zeros(KR.shape, dtype=int)

for radial_shell in tqdm(np.unique(shell[~outside_radial_grid]),
                         desc="fixed-E_x transmission spectra"):
    positions = np.where(shell == radial_shell)
    momentum = cfg["momentum_max"] * np.sqrt(float(radial_shell)) / half_surface_grid
    point_azimuths = np.rad2deg(np.arctan2(
        np.abs(KY[positions]), np.abs(KX[positions]),
    ))
    azimuths = np.unique(np.concatenate((point_azimuths, [0.0, 45.0, 90.0])))
    inverse = np.searchsorted(azimuths, point_azimuths)
    for branch in ("lp", "up"):
        energies, response = transmission_window(momentum, branch, azimuths)
        maxima = [mc.sampled_spectral_maxima(energies, response["T"][:, i])
                  for i in range(len(azimuths))]
        counts = np.array([len(p["energy"]) for p in maxima])
        assert np.all((counts >= 1) & (counts <= 2)), (
            f"{branch}: unexpected peak count at k={momentum}: {counts}"
        )
        pure_tm = maxima[0]["energy"]
        pure_te = maxima[-1]["energy"]
        assert len(pure_tm) == len(pure_te) == 1
        reference = 0.5 * (pure_tm[0] + pure_te[0])
        for key, value in (("tm_peak_eV", pure_tm[0]), ("te_peak_eV", pure_te[0]),
                           ("reference_eV", reference)):
            surface_data[f"{branch}_{key}"][positions] = value
        surface_data[f"{branch}_peak_count"][positions] = counts[inverse]
        dominant = np.array([p["energy"][np.argmax(p["height"])] for p in maxima])
        heights = np.array([np.max(p["height"]) for p in maxima])
        surface_data[f"{branch}_dominant_energy_eV"][positions] = dominant[inverse]
        surface_data[f"{branch}_dominant_height"][positions] = heights[inverse]
        surface_data[f"{branch}_angular_shift_ueV"][positions] = (
            1.0e6 * (dominant[inverse] - reference)
        )
        for tag, count, peak_index in (("single", 1, 0), ("lower", 2, 0), ("upper", 2, 1)):
            peak_energies = np.array([
                p["energy"][peak_index] if len(p["energy"]) == count else np.nan
                for p in maxima
            ])
            peak_heights = np.array([
                p["height"][peak_index] if len(p["energy"]) == count else np.nan
                for p in maxima
            ])
            surface_data[f"{branch}_{tag}_energy_eV"][positions] = peak_energies[inverse]
            surface_data[f"{branch}_{tag}_height"][positions] = peak_heights[inverse]
        if radial_shell in (0, (half_surface_grid // 2)**2, half_surface_grid**2):
            tag = f"k{int(round(momentum))}_{branch}"
            selected = np.searchsorted(azimuths, sample_spectra["azimuth_degrees"])
            sample_spectra[f"{tag}_energy_eV"] = energies
            for key in ("R", "T", "A", "p_te", "p_tm"):
                sample_spectra[f"{tag}_{key}"] = response[key][:, selected]

# Check the actual spectral extrema, including the resolved UP doublet.
convergence_errors = []
for momentum in (0.0, 2.0, 4.0):
    for branch in ("lp", "up"):
        tag = f"k{int(momentum)}_{branch}"
        fine_energies, fine_response = transmission_window(
            momentum, branch, sample_spectra["azimuth_degrees"], refinement=2,
        )
        for column in range(3):
            coarse = mc.sampled_spectral_maxima(
                sample_spectra[f"{tag}_energy_eV"],
                sample_spectra[f"{tag}_T"][:, column],
            )
            fine = mc.sampled_spectral_maxima(fine_energies, fine_response["T"][:, column])
            assert len(coarse["energy"]) == len(fine["energy"])
            convergence_errors.extend(1.0e6 * np.abs(coarse["energy"] - fine["energy"]))
surface_data["max_peak_grid_refinement_error_ueV"] = np.max(convergence_errors)
assert surface_data["max_peak_grid_refinement_error_ueV"] < 0.02
assert np.all(surface_data["lp_peak_count"][~outside_radial_grid] == 1)
assert np.any(surface_data["up_peak_count"] == 2)
assert len(mc.sampled_spectral_maxima(
    sample_spectra["k4_up_energy_eV"], sample_spectra["k4_up_T"][:, 1],
)["energy"]) == 2
for branch in ("lp", "up"):
    dominant = surface_data[f"{branch}_dominant_energy_eV"]
    assert np.allclose(dominant[half_surface_grid, :], surface_data[f"{branch}_tm_peak_eV"][half_surface_grid, :],
                       rtol=0, atol=2e-10)
    assert np.allclose(dominant[:, half_surface_grid], surface_data[f"{branch}_te_peak_eV"][:, half_surface_grid],
                       rtol=0, atol=2e-10)
    assert np.allclose(dominant, dominant[::-1, ::-1], rtol=0, atol=2e-10,
                       equal_nan=True)

def transmission_peak_mesh(branch):
    # Match actual local maxima on adjacent k-grid vertices. Subtract the
    # radial mean energy before matching, so the large radial dispersion
    # cannot cause a false match between different members of the doublet.
    # A continuing peak remains connected when another maximum appears.
    # No triangle interpolates between the two peaks at the same vertex.
    single = surface_data[f"{branch}_single_energy_eV"]
    energies = np.column_stack((
        np.where(np.isfinite(single), single,
                 surface_data[f"{branch}_lower_energy_eV"]).ravel(),
        surface_data[f"{branch}_upper_energy_eV"].ravel(),
    ))
    heights = np.column_stack((
        np.where(np.isfinite(single), surface_data[f"{branch}_single_height"],
                 surface_data[f"{branch}_lower_height"]).ravel(),
        surface_data[f"{branch}_upper_height"].ravel(),
    ))
    residual = energies - surface_data[f"{branch}_reference_eV"].ravel()[:, None]
    base = mtri.Triangulation(KX.ravel(), KY.ravel()).triangles
    base = base[np.all(np.isfinite(energies[base, 0]), axis=1)]
    faces = []
    for choices in np.ndindex(2, 2, 2):
        chosen = np.asarray(choices)
        valid = np.all(np.isfinite(residual[base, chosen]), axis=1)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            distance = np.abs(residual[base[:, a], :, None]
                              - residual[base[:, b], None, :])
            distance = np.where(np.isfinite(distance), distance, np.inf)
            selected = distance[:, chosen[a], chosen[b]]
            # Reciprocal nearest neighbours give a consistent local sheet.
            valid &= (selected <= np.min(distance[:, chosen[a], :], axis=1) + 1e-12)
            valid &= (selected <= np.min(distance[:, :, chosen[b]], axis=1) + 1e-12)
        faces.append(2 * base[valid] + chosen)
    faces = np.concatenate(faces)
    assert len(faces) and np.all(np.isfinite(energies.ravel()[faces]))
    assert np.all(np.diff(np.sort(faces // 2, axis=1), axis=1) > 0)
    surface_data[f"{branch}_mesh_triangles"] = faces
    surface_data[f"{branch}_mesh_energy_eV"] = energies
    surface_data[f"{branch}_mesh_height"] = heights
    return energies.ravel(), heights.ravel(), faces


peak_meshes = {branch: transmission_peak_mesh(branch) for branch in ("lp", "up")}
np.savez_compressed(DATA_DIR / "fig12_fixed_x_transmission_peaks.npz", **surface_data)
np.savez_compressed(DATA_DIR / "fig12_sample_transmission_spectra.npz", **sample_spectra)

fig = plt.figure(figsize=(10.2, 9.2), layout="constrained")
grid = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.06)
axes = [[fig.add_subplot(grid[row, column], projection="3d")
         for column in range(2)] for row in range(2)]
fig.suptitle(r"Fixed tangential $E_x$: maxima of the transmitted spectrum",
             fontsize=11)
peak_norm = LogNorm(vmin=1.0e-8, vmax=1.0)
peak_cmap = plt.colormaps["viridis"]
angular_limit = max(
    5.0, 5.0 * np.ceil(max(np.nanmax(np.abs(surface_data[f"{b}_angular_shift_ueV"]))
                           for b in ("lp", "up")) / 5.0),
)
angular_norm = TwoSlopeNorm(vmin=-angular_limit, vcenter=0, vmax=angular_limit)
for column, branch in enumerate(("lp", "up")):
    full_ax = axes[0][column]
    full_ax.computed_zorder = False
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    peak_energy, peak_height, faces = peak_meshes[branch]
    vertices = np.column_stack((np.repeat(KX.ravel(), 2),
                                np.repeat(KY.ravel(), 2), energy_offset(peak_energy)))
    # Geometric-mean peak height colours each triangle on the logarithmic scale.
    face_height = np.exp(np.mean(np.log(peak_height[faces]), axis=1))
    full_ax.add_collection3d(Poly3DCollection(
        vertices[faces], facecolors=peak_cmap(peak_norm(face_height)),
        edgecolors="none", linewidths=0, antialiased=False, alpha=0.96,
    ))
    full_ax.plot(kx, np.zeros_like(kx),
                 energy_offset(surface_data[f"{branch}_tm_peak_eV"][half_surface_grid, :]),
                 color="black", linewidth=1.8, zorder=5)
    full_ax.plot(np.zeros_like(ky), ky,
                 energy_offset(surface_data[f"{branch}_te_peak_eV"][:, half_surface_grid]),
                 color="limegreen", linewidth=1.8, zorder=5)
    full_ax.set(
        xlabel=r"$k_x$ ($\mu$m$^{-1}$)", ylabel=r"$k_y$ ($\mu$m$^{-1}$)",
        zlabel=r"$E_{\rm peak}-E_X(0)$ (meV)",
        xlim=(-4, 4), ylim=(-4, 4), zlim=(-2, 20),
        title=f"{branch.upper()}: all transmission maxima",
    )
    full_ax.set_zticks([-2, 0, 5, 10, 15, 20])

    angular_ax = axes[1][column]
    delta = surface_data[f"{branch}_angular_shift_ueV"].ravel()
    count = surface_data[f"{branch}_peak_count"].ravel()
    splitting = 1.0e6 * np.abs(
        surface_data[f"{branch}_tm_peak_eV"] - surface_data[f"{branch}_te_peak_eV"]
    ).ravel()
    triangulation = mtri.Triangulation(KX.ravel(), KY.ravel())
    triangles = triangulation.triangles
    missing = np.any(~np.isfinite(delta[triangles]), axis=1)
    # Leave a gap at a switch between two resolved peaks; never connect it
    # with an artificial intermediate energy.
    jump = (
        np.any(count[triangles] == 2, axis=1)
        & (np.ptp(delta[triangles], axis=1)
           > np.maximum(0.55 * np.max(splitting[triangles], axis=1), 1.0))
    )
    triangulation.set_mask(missing | jump)
    angular_ax.plot_trisurf(
        triangulation, np.nan_to_num(delta),
        cmap="coolwarm", norm=angular_norm, shade=False, linewidth=0,
    )
    angular_ax.set(
        xlabel=r"$k_x$ ($\mu$m$^{-1}$)", ylabel=r"$k_y$ ($\mu$m$^{-1}$)",
        zlabel=r"$E_{\rm strongest}-\overline{E}_{\rm pure}$ ($\mu$eV)",
        xlim=(-4, 4), ylim=(-4, 4), zlim=(-angular_limit, angular_limit),
        title=f"{branch.upper()}: shift of the strongest maximum",
    )
    angular_ax.set_zticks([-angular_limit, 0, angular_limit])
    for ax in (full_ax, angular_ax):
        ax.tick_params(labelsize=9, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_size(9)
        ax.set_box_aspect((1.0, 1.0, 0.62))
        ax.view_init(elev=25, azim=-58)
fig.colorbar(
    plt.cm.ScalarMappable(norm=peak_norm, cmap=peak_cmap), ax=axes[0],
    label="peak transmittance T", fraction=0.024, pad=0.13,
)
axes[0][0].legend(
    handles=(Line2D([0], [0], color="black", lw=1.8, label=r"$k_y=0$: pure TM"),
             Line2D([0], [0], color="limegreen", lw=1.8, label=r"$k_x=0$: pure TE")),
    loc="upper left", bbox_to_anchor=(0.01, 0.96), fontsize=8,
)
fig.supxlabel(
    r"$\overline{E}_{\rm pure}=(E_{\rm peak}^{TM}+E_{\rm peak}^{TE})/2$. "
    "Upper: all local maxima. Lower: gaps at strongest-peak switches.",
    fontsize=9,
)
save_show(fig, "12_fixed_x_transmission_peaks.png", dpi=220)
print(
    "Figure 12: peak-grid refinement error",
    f"{surface_data['max_peak_grid_refinement_error_ueV']:.5f} µeV;",
    "resolved UP doublet retained at k=4 µm^-1, azimuth=45 degrees.",
)


# Direct spectral diagnostics: intensity is smooth although extrema can appear,
# end, or exchange which one is strongest. These maps use actual spectra,
# independent of the 3D surface triangulation.
diag_k = np.linspace(2.6, 4.0, 281)
diag_offset_ueV = np.linspace(-80.0, 80.0, 1601)
diag_phi = np.linspace(0.0, 90.0, 361)
diag_radial_T = np.empty((len(diag_offset_ueV), len(diag_k)))
diag_radial_peaks, diag_angular_peaks = [], []
diag_pure_reference = np.empty(len(diag_k))
for column, momentum in enumerate(tqdm(diag_k, desc="UP spectral diagnostic")):
    energies, response = transmission_window(momentum, "up", [0.0, 45.0, 90.0])
    pure = [mc.sampled_spectral_maxima(energies, response["T"][:, j])["energy"][0]
            for j in (0, 2)]
    reference = np.mean(pure)
    diag_pure_reference[column] = reference
    grid_energy = reference + 1e-6 * diag_offset_ueV
    diag_radial_T[:, column] = mc.fixed_tangential_polarization_rta(
        grid_energy, float(momentum), 45.0, polariton_layers,
        n_left=cfg["n_air"], n_right=cfg["n_air"],
    )["T"]
    peaks = mc.sampled_spectral_maxima(grid_energy, diag_radial_T[:, column])
    diag_radial_peaks.extend((momentum, 1e6 * (e - reference), h)
                             for e, h in zip(peaks["energy"], peaks["height"]))
reference = 0.5 * (surface_data["up_tm_peak_eV"][half_surface_grid, -1]
                   + surface_data["up_te_peak_eV"][half_surface_grid, -1])
diag_angular_energy = reference + 1e-6 * diag_offset_ueV
diag_angular_T = mc.fixed_tangential_polarization_rta(
    diag_angular_energy[:, None], 4.0, diag_phi[None, :], polariton_layers,
    n_left=cfg["n_air"], n_right=cfg["n_air"],
)["T"]
for column, phi in enumerate(diag_phi):
    peaks = mc.sampled_spectral_maxima(diag_angular_energy, diag_angular_T[:, column])
    diag_angular_peaks.extend((phi, 1e6 * (e - reference), h)
                              for e, h in zip(peaks["energy"], peaks["height"]))
diag_radial_peaks, diag_angular_peaks = map(np.asarray,
                                          (diag_radial_peaks, diag_angular_peaks))
np.savez_compressed(
    DATA_DIR / "fig12_UP_spectral_diagnostic.npz", k=diag_k,
    offset_ueV=diag_offset_ueV, radial_T=diag_radial_T,
    radial_reference_eV=diag_pure_reference, radial_peaks=diag_radial_peaks,
    azimuth_degrees=diag_phi, angular_T=diag_angular_T,
    angular_reference_eV=reference, angular_peaks=diag_angular_peaks,
)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), layout="constrained")
for ax, x, transmission, peaks, title, xlabel in (
    (axes[0], diag_k, diag_radial_T, diag_radial_peaks,
     r"UP spectrum: fixed $\varphi=45^\circ$", r"$k_\parallel$ ($\mu$m$^{-1}$)"),
    (axes[1], diag_phi, diag_angular_T, diag_angular_peaks,
     r"UP spectrum: fixed $k_\parallel=4$ $\mu$m$^{-1}$", r"azimuth $\varphi$ (degrees)"),
):
    image = ax.pcolormesh(x, diag_offset_ueV, transmission, shading="auto",
                         cmap="magma", norm=LogNorm(vmin=1e-4, vmax=0.3), rasterized=True)
    ax.scatter(peaks[:, 0], peaks[:, 1], s=2.0, c="cyan", linewidths=0,
               label="all local maxima")
    ax.axhline(0, color="white", lw=0.6, alpha=0.6)
    ax.set(xlabel=xlabel, ylabel=r"$E-\overline{E}_{\rm pure}(k)$ ($\mu$eV)",
           title=title, ylim=(-80, 80))
    ax.grid(False)
axes[0].legend(loc="upper left", fontsize=8)
fig.colorbar(image, ax=axes, label="transmittance T", fraction=0.025, pad=0.025)
fig.supxlabel("Continuous spectra; the number and relative heights of maxima change.",
              fontsize=9)
save_show(fig, "12_UP_spectral_diagnostic.png", dpi=220)

# 13. Pole linewidths versus k and the visual external angle
te_lp_width = mc.to_mev(mc.linewidth_from_pole(poles_te["lp"]))
te_up_width = mc.to_mev(mc.linewidth_from_pole(poles_te["up"]))
save_csv(
    "fig13_pole_linewidths.csv",
    ("k_um_inverse", "lp_angle_degrees", "up_angle_degrees",
     "lp_pole_width_meV", "up_pole_width_meV"),
    (momentum_grid, branch_angle(poles_te["lp"].real),
     branch_angle(poles_te["up"].real), te_lp_width, te_up_width),
)
fig, axes = plt.subplots(1, 2, figsize=(10.2, FIGURE_HEIGHT), layout="constrained")
for branch, values, widths, color in (
    ("LP", poles_te["lp"].real, te_lp_width, "C0"),
    ("UP", poles_te["up"].real, te_up_width, "C1"),
):
    axes[0].plot(momentum_grid, widths, color=color, label=branch)
    axes[1].plot(branch_angle(values), widths, color=color, label=branch)
axes[0].set(xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
            ylabel="pole linewidth parameter (meV)")
axes[1].set(xlabel=r"external angle $\theta_{\rm ext}$ (degrees)",
            ylabel="pole linewidth parameter (meV)")
axes[0].legend()
save_show(fig, "13_polariton_linewidths.png")
''')

put(22, source(22).replace(
    'save_show(fig, "13_multiple_qw_scaling.png")',
    'save_show(fig, "14_multiple_qw_scaling.png")',
))

comparison_code = r'''
# 15. Effective 2x2 energies, reflectivity minima, and TMM poles at equal k
comparison_arrays = [momentum_grid]
comparison_columns = ["k_um_inverse"]
comparison_stats = {}
for branch in ("lp", "up"):
    model = scalar_lp if branch == "lp" else scalar_up
    dip = dips_te[branch]
    pole = poles_te[branch].real
    valid = dips_te[f"resolved_{branch}"] & np.isfinite(pole)
    model_residual = mc.to_mev(model - pole) * 1e3
    dip_residual = mc.to_mev(dip - pole) * 1e3
    assert np.any(valid), f"no valid reflectivity minima for {branch}"
    comparison_stats[branch] = {
        "n_valid": int(np.count_nonzero(valid)),
        "k_min": float(momentum_grid[valid][0]),
        "k_max": float(momentum_grid[valid][-1]),
        "model_rms_ueV": float(np.sqrt(np.mean(model_residual[valid] ** 2))),
        "model_max_abs_ueV": float(np.max(np.abs(model_residual[valid]))),
        "dip_rms_ueV": float(np.sqrt(np.mean(dip_residual[valid] ** 2))),
        "dip_max_abs_ueV": float(np.max(np.abs(dip_residual[valid]))),
        "model_at_zero_ueV": float(model_residual[0]) if valid[0] else np.nan,
        "dip_at_zero_ueV": float(dip_residual[0]) if valid[0] else np.nan,
        "model_at_last_valid_ueV": float(model_residual[valid][-1]),
        "dip_at_last_valid_ueV": float(dip_residual[valid][-1]),
    }
    comparison_columns.extend((
        f"{branch}_2x2_eV", f"{branch}_R_min_eV", f"{branch}_pole_eV",
        f"{branch}_R_min_valid", f"{branch}_2x2_minus_pole_ueV",
        f"{branch}_R_min_minus_pole_ueV", f"{branch}_2x2_angle_degrees",
        f"{branch}_R_min_angle_degrees", f"{branch}_pole_angle_degrees",
    ))
    comparison_arrays.extend((
        model, dip, pole, valid.astype(int), model_residual, dip_residual,
        branch_angle(model), branch_angle(dip), branch_angle(pole),
    ))
save_csv("fig15_three_methods_comparison.csv", comparison_columns, comparison_arrays)
print("Three-method residual statistics at equal real k (microelectronvolts):")
for branch, result in comparison_stats.items():
    print(branch.upper(), result)

fig, ax = plt.subplots(figsize=(10.2, FIGURE_HEIGHT), layout="constrained")
for branch, color in (("lp", "C0"), ("up", "C1")):
    model = scalar_lp if branch == "lp" else scalar_up
    dip = dips_te[branch]
    pole = poles_te[branch].real
    valid = dips_te[f"resolved_{branch}"] & np.isfinite(pole)
    tag = branch.upper()
    ax.plot(momentum_grid, mc.to_mev(model - pole) * 1e3, "--",
                 color=color, label=f"{tag}: 2×2 − pole")
    ax.scatter(momentum_grid[valid],
                    mc.to_mev(dip[valid] - pole[valid]) * 1e3,
                    marker="o", s=9, facecolors="none", edgecolors=color,
                    label=f"{tag}: R min − pole")
ax.axhline(0.0, color="0.5", lw=0.7)
ax.set(xlabel=r"in-plane wave vector $k_\parallel$ ($\mu$m$^{-1}$)",
            ylabel=r"difference from Re(TMM pole) ($\mu$eV)",
            title="Three resonance-energy definitions at equal in-plane wave vector")
ax.legend(ncol=2, fontsize=8)
save_show(fig, "15_three_methods_comparison.png", dpi=220)
'''

summary_code = source(24).replace(
    '    "machine_limited_pole_count": int(',
    '    "three_method_residual_stats_ueV": comparison_stats,\n'
    '    "machine_limited_pole_count": int(',
)
put(24, summary_code)

# Keep numerical and figure sections in the order used by the revised reports.
cells = notebook["cells"]
cells[15:19] = [cells[17], cells[18], cells[15], cells[16]]
cells[15]["source"] = ["## 5. QW-position sweep at fixed bare detuning\n"]
cells[17]["source"] = ["## 6. DBR-number sweep\n"]
cells[23]["source"] = ["## 10. Summary diagnostics\n"]
cells[23:23] = [
    markdown_cell("## 9. Comparison of three resonance-energy definitions"),
    code_cell(comparison_code),
]

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"Wrote {OUTPUT}")
