"""Numerical tools for planar exciton--microcavity polaritons.

The module follows the conventions fixed in
``microcavity_polaritons_numerical_architecture.md``:

* energies are in eV, lengths in micrometres, and in-plane momenta in
  inverse micrometres;
* the transfer matrix maps amplitudes from left to right,
  ``v_right = M @ v_left``;
* TE amplitudes are tangential electric fields, while TM amplitudes are
  tangential ``c B`` fields;
* outgoing complex modes are zeros of ``M[1, 1]`` and use
  ``E_complex = E - 1j * FWHM / 2``.

Sample-specific parameters deliberately do not live in this file.  Layers and
quantum wells are plain dictionaries so the sole project configuration can
remain a normal ``dict`` in the notebook.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Constants and units
# ---------------------------------------------------------------------------

HBAR_C = 0.1973269804  # eV um
HC = 1.2398419843  # eV um
HBAR = 6.582119569e-16  # eV s
HBAR2_OVER_2M0 = 3.80998212e-8  # eV um^2

EV_TO_MEV = 1.0e3
MEV_TO_EV = 1.0e-3
NM_TO_UM = 1.0e-3

Polarization = str
Layer = dict[str, Any]
MatrixBuilder = Callable[[complex, float, Polarization], np.ndarray]


def mev(value: float | np.ndarray) -> float | np.ndarray:
    """Convert meV to eV."""

    return np.asarray(value) * MEV_TO_EV if np.ndim(value) else float(value) * MEV_TO_EV


def to_mev(value: float | np.ndarray) -> float | np.ndarray:
    """Convert eV to meV."""

    return np.asarray(value) * EV_TO_MEV if np.ndim(value) else float(value) * EV_TO_MEV


# ---------------------------------------------------------------------------
# Analytical dispersion
# ---------------------------------------------------------------------------


def photon_energy(
    k_parallel: float | np.ndarray,
    energy_zero: float,
    refractive_index: float,
) -> np.ndarray:
    """Ideal planar-cavity photon dispersion."""

    k_parallel = np.asarray(k_parallel, dtype=np.float64)
    return np.sqrt(
        energy_zero**2 + (HBAR_C * k_parallel / refractive_index) ** 2
    )


def photon_mass_ratio(energy_zero: float, refractive_index: float) -> float:
    """Return the ideal photon effective mass divided by the electron mass."""

    curvature = HBAR_C**2 / (2.0 * refractive_index**2 * energy_zero)
    return HBAR2_OVER_2M0 / curvature


def exciton_energy(
    k_parallel: float | np.ndarray,
    energy_zero: float,
    mass_ratio: float,
) -> np.ndarray:
    """Parabolic QW-exciton in-plane dispersion."""

    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be positive")
    k_parallel = np.asarray(k_parallel, dtype=np.float64)
    return energy_zero + HBAR2_OVER_2M0 * k_parallel**2 / mass_ratio


def polariton_energies(
    cavity_energy: float | np.ndarray,
    exciton_energy_value: float | np.ndarray,
    rabi_splitting: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower- and upper-polariton energies for the scalar model."""

    cavity_energy = np.asarray(cavity_energy, dtype=np.float64)
    exciton_energy_value = np.asarray(exciton_energy_value, dtype=np.float64)
    detuning = cavity_energy - exciton_energy_value
    splitting = np.sqrt(detuning**2 + rabi_splitting**2)
    average = 0.5 * (cavity_energy + exciton_energy_value)
    return average - 0.5 * splitting, average + 0.5 * splitting


def hopfield_coefficients(
    cavity_energy: float | np.ndarray,
    exciton_energy_value: float | np.ndarray,
    rabi_splitting: float,
) -> dict[str, np.ndarray]:
    """Return photon/exciton fractions for the real two-oscillator model."""

    cavity_energy = np.asarray(cavity_energy, dtype=np.float64)
    exciton_energy_value = np.asarray(exciton_energy_value, dtype=np.float64)
    detuning = cavity_energy - exciton_energy_value
    denominator = np.sqrt(detuning**2 + rabi_splitting**2)
    photon_lp = 0.5 * (1.0 - detuning / denominator)
    exciton_lp = 1.0 - photon_lp
    return {
        "photon_lp": photon_lp,
        "exciton_lp": exciton_lp,
        "photon_up": exciton_lp,
        "exciton_up": photon_lp,
    }


def complex_coupled_oscillator_eigenvalues(
    cavity_energy: float,
    exciton_energy_value: float,
    rabi_splitting: float,
    cavity_fwhm: float,
    exciton_fwhm: float,
) -> np.ndarray:
    """Complex LP/UP eigenvalues, sorted by their real parts."""

    hamiltonian = np.array(
        [
            [cavity_energy - 0.5j * cavity_fwhm, 0.5 * rabi_splitting],
            [0.5 * rabi_splitting, exciton_energy_value - 0.5j * exciton_fwhm],
        ],
        dtype=np.complex128,
    )
    values = np.linalg.eigvals(hamiltonian)
    return values[np.argsort(values.real)]


def linewidth_from_pole(pole: complex | np.ndarray) -> float | np.ndarray:
    """Energy FWHM from a pole using E_complex = E - i Gamma / 2."""

    result = -2.0 * np.imag(np.asarray(pole, dtype=np.complex128))
    return float(result) if result.ndim == 0 else result


def lifetime_from_pole(pole: complex | np.ndarray) -> float | np.ndarray:
    """Intensity lifetime in seconds."""

    linewidth = np.asarray(linewidth_from_pole(pole), dtype=np.float64)
    result = np.where(linewidth <= 0.0, np.inf, HBAR / linewidth)
    return float(result) if result.ndim == 0 else result


def quality_factor_from_pole(pole: complex | np.ndarray) -> float | np.ndarray:
    """Quality factor Re(E) / FWHM."""

    pole_array = np.asarray(pole, dtype=np.complex128)
    linewidth = np.asarray(linewidth_from_pole(pole_array), dtype=np.float64)
    result = np.where(linewidth <= 0.0, np.inf, pole_array.real / linewidth)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------


def make_layer(
    name: str,
    refractive_index: float,
    thickness: float,
    *,
    region: str = "other",
) -> Layer:
    """Create a homogeneous finite layer."""

    if refractive_index <= 0.0:
        raise ValueError("refractive_index must be positive")
    if thickness < 0.0:
        raise ValueError("thickness must be non-negative")
    return {
        "kind": "layer",
        "name": str(name),
        "n": float(refractive_index),
        "d": float(thickness),
        "region": str(region),
    }


def make_qw(
    name: str,
    refractive_index: float,
    thickness: float,
    exciton_energy_zero: float,
    exciton_mass_ratio: float,
    gamma_x: float,
    gamma_0: float,
    *,
    enabled: bool = True,
    region: str = "cavity",
) -> Layer:
    """Create a finite-index QW with a resonant sheet at its centre."""

    if gamma_x < 0.0 or gamma_0 < 0.0:
        raise ValueError("QW linewidth parameters must be non-negative")
    layer = make_layer(name, refractive_index, thickness, region=region)
    layer.update(
        {
            "kind": "qw",
            "exciton_energy": float(exciton_energy_zero),
            "exciton_mass_ratio": float(exciton_mass_ratio),
            "gamma_x": float(gamma_x),
            "gamma_0": float(gamma_0),
            "enabled": bool(enabled),
        }
    )
    return layer


def build_dbr(
    n_h: float,
    n_l: float,
    d_h: float,
    d_l: float,
    pairs: int,
    order: str,
    *,
    region: str,
) -> list[Layer]:
    """Build an ordered quarter-wave DBR list."""

    if pairs < 0:
        raise ValueError("pairs must be non-negative")
    order = order.upper()
    if order not in {"HL", "LH"}:
        raise ValueError("order must be 'HL' or 'LH'")
    materials = {
        "H": (n_h, d_h),
        "L": (n_l, d_l),
    }
    layers: list[Layer] = []
    for pair_index in range(pairs):
        for symbol in order:
            n_value, d_value = materials[symbol]
            layers.append(
                make_layer(
                    f"{region}_{pair_index + 1:02d}_{symbol}",
                    n_value,
                    d_value,
                    region=region,
                )
            )
    return layers


def build_microcavity(
    *,
    n_h: float,
    n_l: float,
    d_h: float,
    d_l: float,
    n_top: int,
    n_bottom: int,
    n_cavity: float,
    cavity_length: float,
    qws: Sequence[dict[str, Any]] = (),
) -> list[Layer]:
    """Build ``(HL)^Ntop | cavity | (LH)^Nbottom``.

    Each QW dictionary must contain a centre ``z`` measured from the left
    cavity boundary and the arguments accepted by :func:`make_qw` (using the
    short keys ``n``, ``d``, ``ex0``, ``mass_ratio``, ``gamma_x``,
    ``gamma_0``, and optionally ``enabled`` and ``name``).  QW material
    replaces equal-thickness cavity material, so the total spacer thickness
    remains ``cavity_length``.
    """

    if cavity_length <= 0.0:
        raise ValueError("cavity_length must be positive")

    ordered_qws = sorted((dict(item) for item in qws), key=lambda item: item["z"])
    left_edge = 0.0
    cavity_layers: list[Layer] = []
    spacer_index = 1

    for qw_index, spec in enumerate(ordered_qws, start=1):
        centre = float(spec["z"])
        thickness = float(spec["d"])
        qw_left = centre - 0.5 * thickness
        qw_right = centre + 0.5 * thickness
        if qw_left < left_edge - 1e-14 or qw_right > cavity_length + 1e-14:
            raise ValueError("QWs overlap or lie outside the cavity")
        spacer_thickness = max(0.0, qw_left - left_edge)
        if spacer_thickness > 0.0:
            cavity_layers.append(
                make_layer(
                    f"C_{spacer_index}",
                    n_cavity,
                    spacer_thickness,
                    region="cavity",
                )
            )
            spacer_index += 1
        cavity_layers.append(
            make_qw(
                spec.get("name", f"QW_{qw_index}"),
                spec["n"],
                thickness,
                spec["ex0"],
                spec["mass_ratio"],
                spec["gamma_x"],
                spec["gamma_0"],
                enabled=spec.get("enabled", True),
            )
        )
        left_edge = qw_right

    if left_edge < cavity_length:
        cavity_layers.append(
            make_layer(
                f"C_{spacer_index}",
                n_cavity,
                cavity_length - left_edge,
                region="cavity",
            )
        )

    return (
        build_dbr(n_h, n_l, d_h, d_l, n_top, "HL", region="top_dbr")
        + cavity_layers
        + build_dbr(n_h, n_l, d_h, d_l, n_bottom, "LH", region="bottom_dbr")
    )


def total_thickness(layers: Sequence[Layer], *, region: str | None = None) -> float:
    """Return physical thickness, optionally restricted to a region."""

    return float(
        sum(
            layer["d"]
            for layer in layers
            if region is None or layer.get("region") == region
        )
    )


def index_profile(
    layers: Sequence[Layer],
    *,
    points_per_layer: int = 2,
    n_left: float | None = None,
    n_right: float | None = None,
    exterior_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Refractive-index profile, optionally with finite exterior padding."""

    if points_per_layer < 2:
        raise ValueError("points_per_layer must be at least two")
    if exterior_width < 0.0:
        raise ValueError("exterior_width must be non-negative")
    if exterior_width and (n_left is None or n_right is None):
        raise ValueError("n_left and n_right are required for exterior padding")
    z_values: list[float] = []
    n_values: list[float] = []
    z_left = 0.0
    for layer in layers:
        local = np.linspace(z_left, z_left + layer["d"], points_per_layer)
        z_values.extend(local.tolist())
        n_values.extend([layer["n"]] * points_per_layer)
        z_left += layer["d"]
    if exterior_width:
        z_values = [-exterior_width, 0.0, *z_values, z_left, z_left + exterior_width]
        n_values = [float(n_left), float(n_left), *n_values, float(n_right), float(n_right)]
    return np.asarray(z_values), np.asarray(n_values)


# ---------------------------------------------------------------------------
# Transfer matrices and QW response
# ---------------------------------------------------------------------------


def _validate_polarization(polarization: Polarization) -> str:
    value = polarization.upper()
    if value not in {"TE", "TM"}:
        raise ValueError("polarization must be 'TE' or 'TM'")
    return value


def longitudinal_wavevector(
    energy: complex | np.ndarray,
    k_parallel: float,
    refractive_index: float,
) -> np.ndarray:
    """Longitudinal wavevector on the deterministic outgoing branch."""

    energy_array = np.asarray(energy, dtype=np.complex128)
    argument = (refractive_index * energy_array / HBAR_C) ** 2 - k_parallel**2
    q_value = np.asarray(np.sqrt(argument), dtype=np.complex128)

    real_scale = np.maximum(1.0, np.abs(q_value))
    near_zero_real = np.abs(q_value.real) <= 64.0 * np.finfo(float).eps * real_scale
    flip = q_value.real < 0.0
    desired_imag_sign = np.sign(energy_array.imag)
    wrong_tie_sign = near_zero_real & (desired_imag_sign != 0.0) & (
        np.sign(q_value.imag) != desired_imag_sign
    )
    q_value = np.where(flip | wrong_tie_sign, -q_value, q_value)
    return q_value


def optical_admittance(
    refractive_index: float,
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
) -> np.ndarray:
    """TE/TM optical admittance in the project's amplitude basis."""

    polarization = _validate_polarization(polarization)
    energy_array = np.asarray(energy, dtype=np.complex128)
    q_value = longitudinal_wavevector(energy_array, k_parallel, refractive_index)
    cos_phi = q_value / (refractive_index * energy_array / HBAR_C)
    if polarization == "TE":
        return refractive_index * cos_phi
    return cos_phi / refractive_index


def _empty_matrix(energy: complex | np.ndarray) -> np.ndarray:
    shape = np.asarray(energy).shape
    return np.broadcast_to(np.eye(2, dtype=np.complex128), shape + (2, 2)).copy()


def _qw_args(layer: Layer) -> dict[str, float]:
    """Arguments shared by all resonant-sheet calculations."""

    return {
        "refractive_index": layer["n"],
        "exciton_energy_zero": layer["exciton_energy"],
        "exciton_mass_ratio": layer["exciton_mass_ratio"],
        "gamma_x": layer["gamma_x"],
        "gamma_0": layer["gamma_0"],
    }


def interface_matrix(
    n_left: float,
    n_right: float,
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
) -> np.ndarray:
    """Forward matrix of an interface from ``n_left`` to ``n_right``."""

    y_left = optical_admittance(n_left, energy, k_parallel, polarization)
    y_right = optical_admittance(n_right, energy, k_parallel, polarization)
    matrix = np.empty(np.asarray(energy).shape + (2, 2), dtype=np.complex128)
    diagonal = (y_left + y_right) / (2.0 * y_right)
    off_diagonal = (y_right - y_left) / (2.0 * y_right)
    matrix[..., 0, 0] = diagonal
    matrix[..., 0, 1] = off_diagonal
    matrix[..., 1, 0] = off_diagonal
    matrix[..., 1, 1] = diagonal
    return matrix


def propagation_matrix(
    refractive_index: float,
    thickness: float,
    energy: complex | np.ndarray,
    k_parallel: float,
) -> np.ndarray:
    """Forward propagation through a homogeneous layer."""

    q_value = longitudinal_wavevector(energy, k_parallel, refractive_index)
    phase = q_value * thickness
    matrix = np.zeros(np.asarray(energy).shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = np.exp(1j * phase)
    matrix[..., 1, 1] = np.exp(-1j * phase)
    return matrix


def qw_scattering_amplitudes(
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    *,
    refractive_index: float,
    exciton_energy_zero: float,
    exciton_mass_ratio: float,
    gamma_x: float,
    gamma_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflection and transmission amplitudes of a resonant QW sheet."""

    energy_array = np.asarray(energy, dtype=np.complex128)
    if gamma_0 == 0.0:
        return np.zeros_like(energy_array), np.ones_like(energy_array)
    q_value = longitudinal_wavevector(energy_array, k_parallel, refractive_index)
    cos_phi = q_value / (refractive_index * energy_array / HBAR_C)
    polarization = _validate_polarization(polarization)
    gamma_pol = gamma_0 / cos_phi if polarization == "TE" else gamma_0 * cos_phi
    ex_energy = exciton_energy(k_parallel, exciton_energy_zero, exciton_mass_ratio)
    denominator = ex_energy - energy_array - 1j * (gamma_pol + gamma_x)
    electric_reflection = 1j * gamma_pol / denominator
    transmission = 1.0 + electric_reflection
    # Reversing propagation reverses the sign relating reflected electric and
    # magnetic amplitudes.  The TM state vector stores magnetic amplitudes.
    amplitude_reflection = (
        electric_reflection if polarization == "TE" else -electric_reflection
    )
    return amplitude_reflection, transmission


def qw_transfer_matrix(
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    *,
    refractive_index: float,
    exciton_energy_zero: float,
    exciton_mass_ratio: float,
    gamma_x: float,
    gamma_0: float,
) -> np.ndarray:
    """Forward resonant-sheet matrix.

    The algebraically reduced form avoids cancellation in ``(r, t)`` near
    resonance and is exactly equivalent to the matrix specified in the
    architecture document.
    """

    energy_array = np.asarray(energy, dtype=np.complex128)
    if gamma_0 == 0.0:
        return _empty_matrix(energy_array)

    q_value = longitudinal_wavevector(energy_array, k_parallel, refractive_index)
    cos_phi = q_value / (refractive_index * energy_array / HBAR_C)
    polarization = _validate_polarization(polarization)
    gamma_pol = gamma_0 / cos_phi if polarization == "TE" else gamma_0 * cos_phi
    ex_energy = exciton_energy(k_parallel, exciton_energy_zero, exciton_mass_ratio)
    beta = 1j * gamma_pol / (ex_energy - energy_array - 1j * gamma_x)

    matrix = np.empty(energy_array.shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = 1.0 + beta
    off_diagonal_sign = 1.0 if polarization == "TE" else -1.0
    matrix[..., 0, 1] = off_diagonal_sign * beta
    matrix[..., 1, 0] = -off_diagonal_sign * beta
    matrix[..., 1, 1] = 1.0 - beta
    return matrix


def _layer_matrix(
    layer: Layer,
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
) -> np.ndarray:
    """Propagation through one layer, including an active QW sheet."""

    n, d = layer["n"], layer["d"]
    kind = layer["kind"]
    if kind not in {"layer", "qw"}:
        raise ValueError(f"unknown layer kind: {kind!r}")
    if kind == "layer" or not layer.get("enabled", True):
        return propagation_matrix(n, d, energy, k_parallel)
    half = propagation_matrix(n, d / 2.0, energy, k_parallel)
    sheet = qw_transfer_matrix(
        energy, k_parallel, polarization, **_qw_args(layer)
    )
    return half @ sheet @ half


def stack_matrix(
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    layers: Sequence[Layer],
    *,
    n_left: float,
    n_right: float,
) -> np.ndarray:
    """Full forward transfer matrix of an ordered layer list."""

    polarization = _validate_polarization(polarization)
    energy_array = np.asarray(energy, dtype=np.complex128)
    wavevectors: dict[float, np.ndarray] = {}
    admittances: dict[float, np.ndarray] = {}

    def wavevector(refractive_index: float) -> np.ndarray:
        if refractive_index not in wavevectors:
            wavevectors[refractive_index] = longitudinal_wavevector(
                energy_array, k_parallel, refractive_index
            )
        return wavevectors[refractive_index]

    def admittance(refractive_index: float) -> np.ndarray:
        if refractive_index not in admittances:
            cos_phi = wavevector(refractive_index) / (
                refractive_index * energy_array / HBAR_C
            )
            admittances[refractive_index] = (
                refractive_index * cos_phi
                if polarization == "TE"
                else cos_phi / refractive_index
            )
        return admittances[refractive_index]

    one = np.ones_like(energy_array)
    zero = np.zeros_like(energy_array)
    total_00, total_01 = one, zero
    total_10, total_11 = zero, one

    def apply_interface(left_index: float, right_index: float) -> None:
        nonlocal total_00, total_01, total_10, total_11
        left_y = admittance(left_index)
        right_y = admittance(right_index)
        diagonal = (left_y + right_y) / (2.0 * right_y)
        off_diagonal = (right_y - left_y) / (2.0 * right_y)
        old_00, old_01 = total_00, total_01
        old_10, old_11 = total_10, total_11
        total_00 = diagonal * old_00 + off_diagonal * old_10
        total_01 = diagonal * old_01 + off_diagonal * old_11
        total_10 = off_diagonal * old_00 + diagonal * old_10
        total_11 = off_diagonal * old_01 + diagonal * old_11

    def apply_propagation(refractive_index: float, thickness: float) -> None:
        nonlocal total_00, total_01, total_10, total_11
        forward = np.exp(1j * wavevector(refractive_index) * thickness)
        backward = 1.0 / forward
        total_00 = forward * total_00
        total_01 = forward * total_01
        total_10 = backward * total_10
        total_11 = backward * total_11

    def apply_qw_sheet(layer: Layer) -> None:
        nonlocal total_00, total_01, total_10, total_11
        layer_index = float(layer["n"])
        cos_phi = wavevector(layer_index) / (
            layer_index * energy_array / HBAR_C
        )
        gamma_pol = (
            layer["gamma_0"] / cos_phi
            if polarization == "TE"
            else layer["gamma_0"] * cos_phi
        )
        exciton_value = exciton_energy(
            k_parallel, layer["exciton_energy"], layer["exciton_mass_ratio"]
        )
        beta = 1j * gamma_pol / (
            exciton_value - energy_array - 1j * layer["gamma_x"]
        )
        sign = 1.0 if polarization == "TE" else -1.0
        old_00, old_01 = total_00, total_01
        old_10, old_11 = total_10, total_11
        total_00 = (1.0 + beta) * old_00 + sign * beta * old_10
        total_01 = (1.0 + beta) * old_01 + sign * beta * old_11
        total_10 = -sign * beta * old_00 + (1.0 - beta) * old_10
        total_11 = -sign * beta * old_01 + (1.0 - beta) * old_11

    current_index = float(n_left)
    for layer in layers:
        layer_index = float(layer["n"])
        apply_interface(current_index, layer_index)
        if layer["kind"] == "qw" and layer.get("enabled", True):
            apply_propagation(layer_index, float(layer["d"]) / 2.0)
            apply_qw_sheet(layer)
            apply_propagation(layer_index, float(layer["d"]) / 2.0)
        else:
            apply_propagation(layer_index, float(layer["d"]))
        current_index = layer_index
    apply_interface(current_index, float(n_right))

    matrix = np.empty(energy_array.shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = total_00
    matrix[..., 0, 1] = total_01
    matrix[..., 1, 0] = total_10
    matrix[..., 1, 1] = total_11
    return matrix


def reflection_transmission(
    energy: complex | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    layers: Sequence[Layer],
    *,
    n_left: float,
    n_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Complex reflection and transmission amplitudes."""

    matrix = stack_matrix(
        energy,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )
    m22 = matrix[..., 1, 1]
    reflection = -matrix[..., 1, 0] / m22
    determinant = matrix[..., 0, 0] * m22 - matrix[..., 0, 1] * matrix[..., 1, 0]
    transmission = determinant / m22
    return reflection, transmission


def reflectivity(
    energy: float | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    layers: Sequence[Layer],
    *,
    n_left: float,
    n_right: float,
) -> np.ndarray:
    """Power reflectivity for real energies."""

    reflection, _ = reflection_transmission(
        energy,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )
    return np.abs(reflection) ** 2


def transmittance(
    energy: float | np.ndarray,
    k_parallel: float,
    polarization: Polarization,
    layers: Sequence[Layer],
    *,
    n_left: float,
    n_right: float,
) -> np.ndarray:
    """Power transmittance for real-energy propagating external waves."""

    _, transmission = reflection_transmission(
        energy,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )
    y_left = optical_admittance(n_left, energy, k_parallel, polarization)
    y_right = optical_admittance(n_right, energy, k_parallel, polarization)
    return np.real(y_right) / np.real(y_left) * np.abs(transmission) ** 2


def reflectivity_map(
    energies: np.ndarray,
    momenta: np.ndarray,
    polarization: Polarization,
    layers: Sequence[Layer],
    *,
    n_left: float,
    n_right: float,
    progress: bool = False,
) -> np.ndarray:
    """Reflectivity map using a Python momentum loop and vectorized energy."""

    iterator: Iterable[float] = np.asarray(momenta, dtype=float)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=f"{polarization.upper()} reflectivity")
    rows = [
        reflectivity(
            energies,
            float(momentum),
            polarization,
            layers,
            n_left=n_left,
            n_right=n_right,
        )
        for momentum in iterator
    ]
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Resonance refinement and cavity tuning
# ---------------------------------------------------------------------------


def refine_reflectivity_minimum(
    layers: Sequence[Layer],
    energy_guess: float,
    k_parallel: float,
    polarization: Polarization,
    *,
    n_left: float,
    n_right: float,
    half_window: float = 0.01,
    coarse_points: int = 401,
    xatol: float = 1.0e-9,
) -> dict[str, Any]:
    """Locate and refine the reflectivity dip nearest ``energy_guess``."""

    if half_window <= 0.0 or coarse_points < 5:
        raise ValueError("half_window must be positive and coarse_points >= 5")
    energies = np.linspace(
        energy_guess - half_window,
        energy_guess + half_window,
        coarse_points,
    )
    values = reflectivity(
        energies,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )
    candidates, _ = find_peaks(-values)
    if candidates.size:
        coarse_index = int(candidates[np.argmin(np.abs(energies[candidates] - energy_guess))])
    else:
        coarse_index = int(np.argmin(values))
    left_index = max(0, coarse_index - 1)
    right_index = min(coarse_points - 1, coarse_index + 1)
    if left_index == right_index:
        raise RuntimeError("could not bracket a reflectivity minimum")

    objective = lambda value: float(
        reflectivity(
            value,
            k_parallel,
            polarization,
            layers,
            n_left=n_left,
            n_right=n_right,
        )
    )
    result = minimize_scalar(
        objective,
        bounds=(float(energies[left_index]), float(energies[right_index])),
        method="bounded",
        options={"xatol": xatol},
    )
    energy_minimum = float(result.x)
    bracket = (float(energies[left_index]), float(energies[right_index]))
    interior = bracket[0] < energy_minimum < bracket[1]
    return {
        "energy": energy_minimum,
        "reflectivity": float(result.fun),
        "success": bool(result.success and interior),
        "bracket": bracket,
        "coarse_energy": float(energies[coarse_index]),
    }


def tune_cavity_length(
    layer_builder: Callable[[float], Sequence[Layer]],
    target_energy: float,
    initial_length: float,
    *,
    n_left: float,
    n_right: float,
    polarization: Polarization = "TE",
    length_bracket: tuple[float, float] | None = None,
    initial_bracket_fraction: float = 0.01,
    maximum_bracket_fraction: float = 0.20,
    resonance_half_window: float = 0.05,
    resonance_coarse_points: int = 801,
    energy_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Tune the photon-reference cavity resonance to ``target_energy``.

    If a bracket is not supplied, a symmetric bracket around
    ``initial_length`` is expanded until the tracked resonance changes sign
    relative to the target.  The approximate inverse-length scaling is used
    only to centre each local resonance search.
    """

    if initial_length <= 0.0:
        raise ValueError("initial_length must be positive")
    evaluations: dict[float, tuple[float, dict[str, Any]]] = {}

    def resonance_at(length: float) -> float:
        key = float(length)
        if key in evaluations:
            return evaluations[key][0]
        predicted = target_energy * initial_length / length
        result = refine_reflectivity_minimum(
            layer_builder(length),
            predicted,
            0.0,
            polarization,
            n_left=n_left,
            n_right=n_right,
            half_window=resonance_half_window,
            coarse_points=resonance_coarse_points,
        )
        if not result["success"]:
            raise RuntimeError(f"cavity resonance refinement failed at L={length}")
        evaluations[key] = (result["energy"], result)
        return float(result["energy"])

    def objective(length: float) -> float:
        return resonance_at(length) - target_energy

    if length_bracket is None:
        fraction = initial_bracket_fraction
        while True:
            lower = initial_length * (1.0 - fraction)
            upper = initial_length * (1.0 + fraction)
            f_lower = objective(lower)
            f_upper = objective(upper)
            if f_lower == 0.0 or f_upper == 0.0 or f_lower * f_upper < 0.0:
                length_bracket = (lower, upper)
                break
            fraction *= 1.8
            if fraction > maximum_bracket_fraction:
                raise RuntimeError(
                    "could not bracket the tuned cavity length; provide length_bracket"
                )
    lower, upper = map(float, length_bracket)
    if not 0.0 < lower < upper:
        raise ValueError("length_bracket must satisfy 0 < lower < upper")
    f_lower = objective(lower)
    f_upper = objective(upper)
    if f_lower * f_upper > 0.0:
        raise ValueError("length_bracket does not bracket the target resonance")

    tuned_length = float(
        brentq(objective, lower, upper, xtol=1.0e-12, rtol=1.0e-12)
    )
    resonance = resonance_at(tuned_length)
    error = resonance - target_energy
    if abs(error) >= energy_tolerance:
        raise RuntimeError(
            f"cavity tuning missed the target by {error:.3e} eV"
        )
    return {
        "length": tuned_length,
        "resonance_energy": resonance,
        "error": error,
        "bracket": (lower, upper),
        "evaluations": len(evaluations),
    }


# ---------------------------------------------------------------------------
# Field reconstruction and antinodes
# ---------------------------------------------------------------------------


def _plane_wave(amplitudes: np.ndarray, q: complex, z: np.ndarray) -> np.ndarray:
    """Electric field from right- and left-going amplitudes."""

    return amplitudes[0] * np.exp(1j * q * z) + amplitudes[1] * np.exp(-1j * q * z)


def _field_in_layer(
    layer: Layer,
    amplitudes: np.ndarray,
    z: np.ndarray,
    energy: float,
    k_parallel: float,
    polarization: Polarization,
) -> np.ndarray:
    """Sample the field inside one homogeneous or QW layer."""

    n, d = layer["n"], layer["d"]
    q = complex(longitudinal_wavevector(energy, k_parallel, n))
    if layer["kind"] != "qw" or not layer.get("enabled", True):
        return _plane_wave(amplitudes, q, z)

    half = d / 2.0
    left = z < half
    field = np.empty(z.shape, dtype=np.complex128)
    field[left] = _plane_wave(amplitudes, q, z[left])
    middle = (
        qw_transfer_matrix(energy, k_parallel, polarization, **_qw_args(layer))
        @ propagation_matrix(n, half, energy, k_parallel)
        @ amplitudes
    )
    field[~left] = _plane_wave(middle, q, z[~left] - half)
    return field


def field_profile(
    layers: Sequence[Layer],
    energy: float,
    k_parallel: float = 0.0,
    polarization: Polarization = "TE",
    *,
    n_left: float,
    n_right: float,
    dz: float = 5.0e-4,
    region: str = "cavity",
) -> dict[str, np.ndarray]:
    """Reconstruct the driven TE electric field in one stack region."""

    polarization = _validate_polarization(polarization)
    if polarization != "TE":
        raise ValueError("the project reconstructs electric-field profiles for TE only")
    if dz <= 0.0:
        raise ValueError("dz must be positive")

    reflection = reflection_transmission(
        energy,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )[0]
    amplitudes = np.array([1.0 + 0.0j, complex(np.asarray(reflection))])
    current_n, global_left = float(n_left), 0.0
    region_left: float | None = None
    z_values: list[np.ndarray] = []
    field_values: list[np.ndarray] = []

    for layer in layers:
        layer_n, thickness = float(layer["n"]), float(layer["d"])
        amplitudes = (
            interface_matrix(current_n, layer_n, energy, k_parallel, polarization)
            @ amplitudes
        )
        z_local = np.arange(0.0, thickness, dz)
        if not z_local.size:
            z_local = np.array([0.0])

        if layer.get("region") == region:
            if region_left is None:
                region_left = global_left
            z_values.append(global_left + z_local)
            field_values.append(
                _field_in_layer(
                    layer, amplitudes, z_local, energy, k_parallel, polarization
                )
            )
        amplitudes = (
            _layer_matrix(layer, energy, k_parallel, polarization) @ amplitudes
        )
        current_n = layer_n
        global_left += thickness

    if not z_values or region_left is None:
        raise ValueError(f"no layers found in region {region!r}")
    z_global, field = np.concatenate(z_values), np.concatenate(field_values)
    intensity = np.abs(field) ** 2
    if intensity.max() > 0.0:
        intensity /= intensity.max()
    return {
        "z": z_global - region_left,
        "z_global": z_global,
        "field": field,
        "intensity": intensity,
    }


def find_field_antinodes(
    z: np.ndarray,
    intensity: np.ndarray,
    *,
    minimum_separation: float,
) -> dict[str, np.ndarray]:
    """Find deterministic local field maxima with a physical separation."""

    z = np.asarray(z, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if z.ndim != 1 or intensity.shape != z.shape or z.size < 3:
        raise ValueError("z and intensity must be matching one-dimensional arrays")
    if minimum_separation <= 0.0:
        raise ValueError("minimum_separation must be positive")
    median_step = float(np.median(np.diff(z)))
    sample_distance = max(1, int(np.floor(minimum_separation / median_step)))
    indices, _ = find_peaks(intensity, distance=sample_distance)
    if indices.size == 0:
        raise RuntimeError("no cavity-field antinodes detected")
    return {
        "indices": indices,
        "z": z[indices],
        "intensity": intensity[indices],
    }


def tune_qw_to_central_antinode(
    homogeneous_builder: Callable[[float], Sequence[Layer]],
    background_builder: Callable[[float, float], Sequence[Layer]],
    target_energy: float,
    initial_length: float,
    *,
    n_left: float,
    n_right: float,
    minimum_peak_separation: float,
    dz: float = 5.0e-4,
    position_tolerance: float | None = None,
    maximum_iterations: int = 8,
    tuning_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune the passive QW and its nearest antinode to a fixed point."""

    options = dict(tuning_options or {})
    position_tolerance = dz if position_tolerance is None else position_tolerance

    def field_and_offset(
        layers: Sequence[Layer], length: float, position_guess: float
    ) -> tuple[dict[str, np.ndarray], float]:
        field = field_profile(
            layers, target_energy, n_left=n_left, n_right=n_right, dz=dz
        )
        peaks = find_field_antinodes(
            field["z"],
            field["intensity"],
            minimum_separation=minimum_peak_separation,
        )["z"]
        position = float(peaks[np.argmin(np.abs(peaks - position_guess))])
        return field, position - length / 2.0

    homogeneous_tuning = tune_cavity_length(
        homogeneous_builder,
        target_energy,
        initial_length,
        n_left=n_left,
        n_right=n_right,
        **options,
    )
    length = homogeneous_tuning["length"]
    field, offset = field_and_offset(
        homogeneous_builder(length), length, length / 2.0
    )

    history: list[dict[str, float]] = []
    for iteration in range(1, maximum_iterations + 1):
        builder = lambda length, fixed_offset=offset: background_builder(
            length, fixed_offset
        )
        tuning = tune_cavity_length(
            builder,
            target_energy,
            length,
            n_left=n_left,
            n_right=n_right,
            **options,
        )
        length = tuning["length"]
        layers = background_builder(length, offset)
        field, new_offset = field_and_offset(
            layers, length, length / 2.0 + offset
        )
        history.append(
            {
                "iteration": float(iteration),
                "length": length,
                "offset": offset,
                "new_offset": new_offset,
            }
        )
        if abs(new_offset - offset) <= position_tolerance:
            offset = new_offset
            break
        offset = new_offset
    else:
        raise RuntimeError("passive-QW antinode fixed-point iteration did not converge")

    final_builder = lambda length: background_builder(length, offset)
    tuning = tune_cavity_length(
        final_builder,
        target_energy,
        length,
        n_left=n_left,
        n_right=n_right,
        **options,
    )
    length = tuning["length"]
    layers = background_builder(length, offset)
    field = field_profile(
        layers, target_energy, n_left=n_left, n_right=n_right, dz=dz
    )
    return {
        "length": length,
        "qw_offset": offset,
        "qw_position": length / 2.0 + offset,
        "layers": layers,
        "field": field,
        "tuning": tuning,
        "homogeneous_tuning": homogeneous_tuning,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Reflectivity branch extraction and diagnostic FWHM
# ---------------------------------------------------------------------------


def _bounded_reflectivity_dip(
    reflectivity_function: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    xatol: float,
) -> dict[str, Any]:
    lower, upper = bracket
    if lower >= upper:
        return {"success": False, "energy": np.nan, "value": np.nan}
    result = minimize_scalar(
        reflectivity_function,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": xatol},
    )
    margin = max(xatol, 1.0e-6 * (upper - lower))
    interior = lower + margin < result.x < upper - margin
    endpoint_minimum = min(reflectivity_function(lower), reflectivity_function(upper))
    resolved = bool(result.success and interior and result.fun < endpoint_minimum)
    return {
        "success": resolved,
        "energy": float(result.x) if resolved else np.nan,
        "value": float(result.fun) if resolved else np.nan,
        "bracket": bracket,
    }


def trace_reflectivity_branches(
    momenta: np.ndarray,
    lp_predictions: np.ndarray,
    up_predictions: np.ndarray,
    reflectivity_function: Callable[[float, float], float],
    *,
    half_window: float = 0.0025,
    separation_margin: float = 5.0e-5,
    xatol: float = 1.0e-9,
) -> dict[str, np.ndarray]:
    """Trace resolved LP/UP reflectivity dips in non-overlapping brackets."""

    momenta = np.asarray(momenta, dtype=np.float64)
    lp_predictions = np.asarray(lp_predictions, dtype=np.float64)
    up_predictions = np.asarray(up_predictions, dtype=np.float64)
    if not (momenta.shape == lp_predictions.shape == up_predictions.shape):
        raise ValueError("momenta and predictions must have matching shapes")
    lp = np.full(momenta.shape, np.nan)
    up = np.full(momenta.shape, np.nan)
    resolved = np.zeros(momenta.shape, dtype=bool)

    for index, momentum in enumerate(momenta):
        if index == 0:
            guess_lp = lp_predictions[index]
            guess_up = up_predictions[index]
        else:
            guess_lp = lp[index - 1] if np.isfinite(lp[index - 1]) else lp_predictions[index]
            guess_up = up[index - 1] if np.isfinite(up[index - 1]) else up_predictions[index]
            if index >= 2 and np.isfinite(lp[index - 2]) and np.isfinite(lp[index - 1]):
                guess_lp += lp[index - 1] - lp[index - 2]
            if index >= 2 and np.isfinite(up[index - 2]) and np.isfinite(up[index - 1]):
                guess_up += up[index - 1] - up[index - 2]
        if guess_lp >= guess_up:
            guess_lp = lp_predictions[index]
            guess_up = up_predictions[index]
        midpoint = 0.5 * (guess_lp + guess_up)
        lp_bracket = (
            guess_lp - half_window,
            min(guess_lp + half_window, midpoint - separation_margin),
        )
        up_bracket = (
            max(guess_up - half_window, midpoint + separation_margin),
            guess_up + half_window,
        )
        objective = lambda energy, k=float(momentum): float(
            reflectivity_function(energy, k)
        )
        lp_result = _bounded_reflectivity_dip(objective, lp_bracket, xatol=xatol)
        up_result = _bounded_reflectivity_dip(objective, up_bracket, xatol=xatol)
        if (
            lp_result["success"]
            and up_result["success"]
            and lp_result["energy"] < up_result["energy"]
        ):
            lp[index] = lp_result["energy"]
            up[index] = up_result["energy"]
            resolved[index] = True
    return {"lp": lp, "up": up, "resolved": resolved}


def reflectivity_fwhm(
    energy_minimum: float,
    reflectivity_function: Callable[[float], float],
    *,
    half_window: float = 0.005,
    samples: int = 401,
) -> dict[str, Any]:
    """Diagnostic half-depth FWHM using nearest bracketing maxima."""

    energies = np.linspace(
        energy_minimum - half_window,
        energy_minimum + half_window,
        samples,
    )
    values = np.asarray([reflectivity_function(value) for value in energies])
    maxima, _ = find_peaks(values)
    left_maxima = maxima[energies[maxima] < energy_minimum]
    right_maxima = maxima[energies[maxima] > energy_minimum]
    if left_maxima.size == 0 or right_maxima.size == 0:
        return {"success": False, "fwhm": np.nan}
    left_maximum = int(left_maxima[-1])
    right_maximum = int(right_maxima[0])
    r_minimum = float(reflectivity_function(energy_minimum))
    background = 0.5 * (values[left_maximum] + values[right_maximum])
    half_depth = 0.5 * (background + r_minimum)
    shifted = values - half_depth

    left_indices = np.where(np.signbit(shifted[:-1]) != np.signbit(shifted[1:]))[0]
    left_indices = left_indices[energies[left_indices] < energy_minimum]
    right_indices = np.where(np.signbit(shifted[:-1]) != np.signbit(shifted[1:]))[0]
    right_indices = right_indices[energies[right_indices + 1] > energy_minimum]
    if left_indices.size == 0 or right_indices.size == 0:
        return {"success": False, "fwhm": np.nan}
    left_crossing_index = int(left_indices[-1])
    right_crossing_index = int(right_indices[0])
    equation = lambda value: float(reflectivity_function(value) - half_depth)
    left_energy = float(
        brentq(
            equation,
            energies[left_crossing_index],
            energies[left_crossing_index + 1],
        )
    )
    right_energy = float(
        brentq(
            equation,
            energies[right_crossing_index],
            energies[right_crossing_index + 1],
        )
    )
    if not left_energy < energy_minimum < right_energy:
        return {"success": False, "fwhm": np.nan}
    return {
        "success": True,
        "fwhm": right_energy - left_energy,
        "left": left_energy,
        "right": right_energy,
        "background": float(background),
        "half_depth": float(half_depth),
    }


# ---------------------------------------------------------------------------
# Complex eigenmodes
# ---------------------------------------------------------------------------


def normalized_m22_residual(matrix: np.ndarray) -> float:
    """Scale-independent residual used to accept a complex pole."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    magnitudes = np.abs(matrix)
    scale = float(np.max(magnitudes))
    if scale == 0.0 or not np.isfinite(scale):
        return np.inf
    scaled_norm = float(np.sqrt(np.sum((magnitudes / scale) ** 2)))
    return float((abs(matrix[1, 1]) / scale) / scaled_norm)


def solve_complex_pole(
    matrix_builder: Callable[[complex], np.ndarray],
    initial_pole: complex,
    *,
    residual_tolerance: float = 1.0e-9,
    machine_residual_ceiling: float = 5.0e-8,
    backward_error_tolerance: float = 1.0e-10,
    root_tolerance: float = 1.0e-11,
    maximum_evaluations: int = 300,
    require_decaying: bool = True,
) -> dict[str, Any]:
    """Solve ``M22(E_complex)=0`` as two real equations."""

    initial_pole = complex(initial_pole)
    initial_matrix = np.asarray(matrix_builder(initial_pole), dtype=np.complex128)
    fixed_scale = max(1.0, abs(initial_matrix[1, 1]))

    def equation(coordinates: np.ndarray) -> np.ndarray:
        energy = complex(coordinates[0], coordinates[1])
        with np.errstate(all="ignore"):
            value = matrix_builder(energy)[1, 1] / fixed_scale
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            return np.array([1.0e12, 1.0e12])
        return np.array([value.real, value.imag])

    result = root(
        equation,
        np.array([initial_pole.real, initial_pole.imag]),
        method="hybr",
        options={"xtol": root_tolerance, "maxfev": maximum_evaluations},
    )
    pole = complex(result.x[0], result.x[1])
    # A few complex-Newton polishing steps reduce the residual left by the
    # two-real-variable solver, especially for exciton-like poles.
    derivative_step = 1.0e-8
    for _ in range(5):
        with np.errstate(all="ignore"):
            value = matrix_builder(pole)[1, 1]
            derivative = (
                matrix_builder(pole + derivative_step)[1, 1]
                - matrix_builder(pole - derivative_step)[1, 1]
            ) / (2.0 * derivative_step)
        if not np.isfinite(value) or not np.isfinite(derivative) or derivative == 0.0:
            break
        candidate = pole - value / derivative
        current_residual = normalized_m22_residual(matrix_builder(pole))
        candidate_residual = normalized_m22_residual(matrix_builder(candidate))
        if candidate_residual <= current_residual:
            pole = complex(candidate)
        else:
            break
    matrix = np.asarray(matrix_builder(pole), dtype=np.complex128)
    residual = normalized_m22_residual(matrix)
    with np.errstate(all="ignore"):
        derivative = (
            matrix_builder(pole + derivative_step)[1, 1]
            - matrix_builder(pole - derivative_step)[1, 1]
        ) / (2.0 * derivative_step)
    backward_error = (
        float(abs(matrix[1, 1] / derivative))
        if np.isfinite(derivative) and derivative != 0.0
        else np.inf
    )
    decaying = pole.imag < 0.0 or not require_decaying
    strict_residual = residual < residual_tolerance
    machine_limited = (
        not strict_residual
        and residual < machine_residual_ceiling
        and backward_error < backward_error_tolerance
    )
    success = bool((result.success or strict_residual) and (strict_residual or machine_limited) and decaying)
    return {
        "pole": pole,
        "success": success,
        "residual": residual,
        "backward_error": backward_error,
        "machine_limited": bool(machine_limited),
        "message": str(result.message),
        "evaluations": int(result.nfev),
    }


def discover_complex_poles(
    matrix_builder: Callable[[complex], np.ndarray],
    seeds: Sequence[complex],
    *,
    real_window: tuple[float, float] | None = None,
    uniqueness_tolerance: float = 1.0e-7,
    residual_tolerance: float = 1.0e-9,
) -> list[dict[str, Any]]:
    """Solve from several seeds and deduplicate accepted poles."""

    solutions: list[dict[str, Any]] = []
    for seed in seeds:
        result = solve_complex_pole(
            matrix_builder,
            seed,
            residual_tolerance=residual_tolerance,
        )
        pole = result["pole"]
        if not result["success"]:
            continue
        if real_window is not None and not real_window[0] <= pole.real <= real_window[1]:
            continue
        if any(abs(pole - existing["pole"]) < uniqueness_tolerance for existing in solutions):
            continue
        solutions.append(result)
    solutions.sort(key=lambda item: (item["pole"].real, item["pole"].imag))
    return solutions


def solve_bare_cavity_pole(
    layers: Sequence[Layer],
    energy_guess: float,
    *,
    n_left: float,
    n_right: float,
    polarization: Polarization = "TE",
    k_parallel: float = 0.0,
    minimum_half_window: float = 0.01,
    default_fwhm: float = 2.0e-4,
) -> dict[str, Any]:
    """Find a reflectivity resonance and refine it to the bare complex pole."""

    dip = refine_reflectivity_minimum(
        layers,
        energy_guess,
        k_parallel,
        polarization,
        n_left=n_left,
        n_right=n_right,
        half_window=minimum_half_window,
    )
    spectrum = lambda energy: float(
        reflectivity(
            energy,
            k_parallel,
            polarization,
            layers,
            n_left=n_left,
            n_right=n_right,
        )
    )
    fwhm = reflectivity_fwhm(dip["energy"], spectrum)
    width_guess = fwhm["fwhm"] if fwhm["success"] else default_fwhm
    matrix_builder = lambda energy: stack_matrix(
        energy,
        k_parallel,
        polarization,
        layers,
        n_left=n_left,
        n_right=n_right,
    )
    pole = solve_complex_pole(
        matrix_builder,
        complex(dip["energy"], -0.5 * width_guess),
    )
    if not pole["success"]:
        raise RuntimeError(f"bare cavity pole failed: {pole['message']}")
    return {"dip": dip, "reflectivity_fwhm": fwhm, **pole}


def trace_two_complex_poles(
    momenta: np.ndarray,
    matrix_builder: Callable[[float], Callable[[complex], np.ndarray]],
    initial_poles: Sequence[complex],
    *,
    fallback_predictions: Callable[[float], Sequence[complex]] | None = None,
    uniqueness_tolerance: float = 1.0e-7,
    prediction_window: float = 0.01,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Trace two bright poles, using a multi-seed fallback when needed."""

    momenta = np.asarray(momenta, dtype=np.float64)
    if len(initial_poles) != 2:
        raise ValueError("initial_poles must contain LP and UP seeds")
    output = {
        "lp": np.full(momenta.shape, np.nan + 1j * np.nan, dtype=np.complex128),
        "up": np.full(momenta.shape, np.nan + 1j * np.nan, dtype=np.complex128),
        "residual_lp": np.full(momenta.shape, np.nan),
        "residual_up": np.full(momenta.shape, np.nan),
        "machine_limited_lp": np.zeros(momenta.shape, dtype=bool),
        "machine_limited_up": np.zeros(momenta.shape, dtype=bool),
        "success": np.zeros(momenta.shape, dtype=bool),
    }
    iterator: Iterable[tuple[int, float]] = enumerate(momenta)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, total=momenta.size, desc="complex poles")

    previous = [complex(initial_poles[0]), complex(initial_poles[1])]
    for index, momentum in iterator:
        builder = matrix_builder(float(momentum))
        predicted = (
            list(map(complex, fallback_predictions(float(momentum))))
            if fallback_predictions is not None
            else list(previous)
        )
        if len(predicted) != 2:
            raise ValueError("fallback_predictions must return exactly two poles")
        predicted.sort(key=lambda value: value.real)

        direct = [
            solve_complex_pole(builder, previous[0]),
            solve_complex_pole(builder, previous[1]),
        ]
        direct.sort(key=lambda item: item["pole"].real)
        valid = (
            all(item["success"] for item in direct)
            and abs(direct[1]["pole"] - direct[0]["pole"]) >= uniqueness_tolerance
            and all(
                abs(item["pole"].real - guess.real) <= prediction_window
                for item, guess in zip(direct, predicted)
            )
        )
        if valid:
            selected = direct
        else:
            nearby = [
                pole + shift - 2.0e-5j
                for pole in predicted
                for shift in (-2.0e-4, 2.0e-4)
            ]
            found = discover_complex_poles(
                builder,
                [*previous, *predicted, *nearby],
                real_window=(
                    min(value.real for value in predicted) - prediction_window,
                    max(value.real for value in predicted) + prediction_window,
                ),
                uniqueness_tolerance=uniqueness_tolerance,
            )
            if len(found) < 2:
                continue
            selected = min(
                combinations(found, 2),
                key=lambda pair: sum(
                    abs(item["pole"] - guess)
                    for item, guess in zip(pair, predicted)
                ),
            )

        for branch, result in zip(("lp", "up"), selected):
            output[branch][index] = result["pole"]
            output[f"residual_{branch}"][index] = result["residual"]
            output[f"machine_limited_{branch}"][index] = result["machine_limited"]
        output["success"][index] = True
        previous = [output["lp"][index], output["up"][index]]
    return output


# ---------------------------------------------------------------------------
# Radiative-width calibration and checks
# ---------------------------------------------------------------------------


def calibrate_radiative_width(
    matrix_builder: Callable[[float], Callable[[complex], np.ndarray]],
    target_splitting: float,
    exciton_energy_zero: float,
    cavity_pole: complex,
    gamma_x: float,
    *,
    gamma_bracket: tuple[float, float] | None = None,
    gamma_scan_bounds: tuple[float, float] = (1.0e-6, 5.0e-3),
    gamma_scan_points: int = 20,
    pole_window: float = 0.03,
) -> dict[str, Any]:
    """Calibrate ``Gamma_0`` from the zero-detuning complex-pole gap.

    The lower endpoint is always strictly positive: at exactly zero radiative
    width the QW transfer matrix is the identity and the optical exciton pole
    is a removable singularity.  Pole pairs are continued from the nearest
    cached ``Gamma_0`` evaluation during scalar root finding.
    """

    if target_splitting <= 0.0:
        raise ValueError("target_splitting must be positive")
    cache: dict[float, tuple[complex, complex]] = {}

    average_imag = 0.5 * (cavity_pole.imag - gamma_x)
    analytical_seeds = [
        complex(exciton_energy_zero - 0.5 * target_splitting, average_imag),
        complex(exciton_energy_zero + 0.5 * target_splitting, average_imag),
        complex(cavity_pole),
        complex(exciton_energy_zero, -0.75 * gamma_x),
        complex(exciton_energy_zero, -1.25 * gamma_x),
        complex(exciton_energy_zero - target_splitting, average_imag),
        complex(exciton_energy_zero + target_splitting, average_imag),
    ]

    def pole_pair(gamma_0: float) -> tuple[complex, complex]:
        key = float(gamma_0)
        if key in cache:
            return cache[key]
        seeds = list(analytical_seeds)
        if cache:
            nearest_key = min(cache, key=lambda value: abs(np.log(value / key)))
            seeds = list(cache[nearest_key]) + seeds
        found = discover_complex_poles(
            matrix_builder(key),
            seeds,
            real_window=(exciton_energy_zero - pole_window, exciton_energy_zero + pole_window),
            uniqueness_tolerance=1.0e-8,
        )
        if len(found) < 2:
            raise RuntimeError(f"could not find two polariton poles at Gamma_0={key:g} eV")
        pair = (found[0]["pole"], found[-1]["pole"])
        cache[key] = pair
        return pair

    def objective(gamma_0: float) -> float:
        lower_pole, upper_pole = pole_pair(gamma_0)
        return upper_pole.real - lower_pole.real - target_splitting

    if gamma_bracket is None:
        scan = np.geomspace(gamma_scan_bounds[0], gamma_scan_bounds[1], gamma_scan_points)
        previous_gamma = float(scan[0])
        previous_value = objective(previous_gamma)
        gamma_bracket = None
        for gamma_value in scan[1:]:
            gamma_value = float(gamma_value)
            current_value = objective(gamma_value)
            if previous_value == 0.0 or previous_value * current_value <= 0.0:
                gamma_bracket = (previous_gamma, gamma_value)
                break
            previous_gamma = gamma_value
            previous_value = current_value
        if gamma_bracket is None:
            raise RuntimeError("automatic Gamma_0 scan did not bracket the target splitting")
    lower, upper = map(float, gamma_bracket)
    if lower <= 0.0 or lower >= upper:
        raise ValueError("gamma_bracket must satisfy 0 < lower < upper")
    f_lower = objective(lower)
    f_upper = objective(upper)
    if f_lower * f_upper > 0.0:
        raise ValueError("gamma_bracket does not bracket the target splitting")
    gamma_0 = float(brentq(objective, lower, upper, xtol=1.0e-12, rtol=1.0e-10))
    poles = pole_pair(gamma_0)
    return {
        "gamma_0": gamma_0,
        "poles": np.asarray(poles, dtype=np.complex128),
        "splitting": poles[1].real - poles[0].real,
        "bracket": (lower, upper),
        "evaluations": len(cache),
    }


def basic_tmm_checks(
    layers: Sequence[Layer],
    energies: np.ndarray,
    *,
    n_left: float,
    n_right: float,
    n_test_a: float,
    n_test_b: float,
) -> dict[str, float]:
    """Return the core algebraic and lossless-stack diagnostics."""

    energies = np.asarray(energies, dtype=np.float64)
    identity = np.eye(2)
    zero_propagation = propagation_matrix(n_test_a, 0.0, energies, 0.0)
    forward = interface_matrix(n_test_a, n_test_b, energies, 0.0, "TE")
    backward = interface_matrix(n_test_b, n_test_a, energies, 0.0, "TE")
    r_te = reflectivity(
        energies, 0.0, "TE", layers, n_left=n_left, n_right=n_right
    )
    t_te = transmittance(
        energies, 0.0, "TE", layers, n_left=n_left, n_right=n_right
    )
    r_tm = reflectivity(
        energies, 0.0, "TM", layers, n_left=n_left, n_right=n_right
    )
    return {
        "zero_thickness_error": float(np.max(np.abs(zero_propagation - identity))),
        "interface_inverse_error": float(np.max(np.abs(backward @ forward - identity))),
        "energy_conservation_error": float(np.max(np.abs(r_te + t_te - 1.0))),
        "normal_incidence_te_tm_error": float(np.max(np.abs(r_te - r_tm))),
    }


def fit_parabolic_photon_mass(
    momenta: np.ndarray,
    energies: np.ndarray,
    *,
    maximum_momentum: float,
) -> dict[str, float]:
    """Fit ``E(k) = E0 + a k^2`` and return ``m_eff / m0``."""

    momenta = np.asarray(momenta, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    mask = (
        np.isfinite(momenta)
        & np.isfinite(energies)
        & (np.abs(momenta) <= maximum_momentum)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("at least three finite low-momentum points are required")
    coefficient, intercept = np.polyfit(momenta[mask] ** 2, energies[mask], 1)
    if coefficient <= 0.0:
        raise RuntimeError("fitted photon curvature is not positive")
    return {
        "energy_zero": float(intercept),
        "curvature": float(coefficient),
        "mass_ratio": float(HBAR2_OVER_2M0 / coefficient),
    }


def select_strongest_antinodes(
    antinode_z: np.ndarray,
    antinode_intensity: np.ndarray,
    count: int,
    *,
    cavity_length: float,
) -> np.ndarray:
    """Select antinodes by intensity, centrality, then left-to-right position."""

    antinode_z = np.asarray(antinode_z, dtype=np.float64)
    antinode_intensity = np.asarray(antinode_intensity, dtype=np.float64)
    if antinode_z.shape != antinode_intensity.shape:
        raise ValueError("antinode positions and intensities must have matching shapes")
    if count <= 0 or count > antinode_z.size:
        raise ValueError("count must select one or more available antinodes")
    # np.lexsort uses the final key as primary: highest intensity, smallest
    # centre distance, then the leftmost coordinate resolve all ties.
    order = np.lexsort(
        (
            antinode_z,
            np.abs(antinode_z - 0.5 * cavity_length),
            -antinode_intensity,
        )
    )
    return np.sort(antinode_z[order[:count]])
