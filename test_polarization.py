"""Physical checks for polarized transmission and complex two-mode comparison."""
import unittest

import numpy as np

import microcavity as mc


class PolarizationPhysicsTests(unittest.TestCase):
    def test_oblique_incident_power_is_not_geometric_field_weight(self):
        energy = 1.45
        k = mc.external_momentum(energy, 30)
        p_te, p_tm = mc.tangential_polarization_weights(energy, k, 45)
        self.assertAlmostEqual(float(p_tm), 4 / 7, places=14)
        self.assertAlmostEqual(float(p_te), 3 / 7, places=14)

    def test_fresnel_interface_matches_vector_poynting_flux(self):
        energy, n_left, n_right = 1.45, 1.0, 1.5
        theta, phi = np.deg2rad([30, 45])
        cos_left = np.cos(theta)
        cos_right = np.sqrt(1 - (n_left * np.sin(theta) / n_right) ** 2)
        t_s = 2 * n_left * cos_left / (n_left * cos_left + n_right * cos_right)
        t_p = 2 * n_left * cos_left / (n_right * cos_left + n_left * cos_right)
        e_p, e_s = np.cos(phi) / cos_left, -np.sin(phi)
        expected = (
            n_right * cos_right * ((t_p * e_p) ** 2 + (t_s * e_s) ** 2)
            / (n_left * cos_left * (e_p**2 + e_s**2))
        )
        result = mc.fixed_tangential_polarization_rta(
            energy, mc.external_momentum(energy, 30), 45, [],
            n_left=n_left, n_right=n_right,
        )
        self.assertAlmostEqual(float(result["T"]), expected, places=13)
        self.assertAlmostEqual(float(result["A"]), 0, places=13)

    def test_pure_cuts_and_lossless_stack_conserve_power(self):
        energies = np.linspace(1.3, 1.6, 51)[:, None]
        layers = [{"kind": "dielectric", "n": 2.1, "d": 0.17},
                  {"kind": "dielectric", "n": 1.5, "d": 0.23}]
        result = mc.fixed_tangential_polarization_rta(
            energies, 2.0, np.array([0, 23, 45, 90])[None, :], layers,
            n_left=1, n_right=1,
        )
        np.testing.assert_allclose(result["T"][:, 0], result["T_tm"][:, 0], atol=1e-13)
        np.testing.assert_allclose(result["T"][:, -1], result["T_te"][:, 0], atol=1e-13)
        np.testing.assert_allclose(result["R"] + result["T"], 1, atol=2e-13)
        np.testing.assert_allclose(result["p_te"] + result["p_tm"], 1, atol=2e-15)

    def test_normal_incidence_and_rotation_symmetry(self):
        energies = np.linspace(1.3, 1.6, 31)[:, None]
        layers = [{"kind": "dielectric", "n": 2.1, "d": 0.17}]
        result = mc.fixed_tangential_polarization_rta(
            energies, 0, np.array([0, 30, 60, 90])[None, :], layers,
            n_left=1, n_right=1,
        )
        np.testing.assert_allclose(
            result["T"], np.broadcast_to(result["T"][:, :1], result["T"].shape),
            atol=1e-13,
        )
        original = mc.tangential_polarization_weights(1.45, 3, 31)
        rotated = mc.tangential_polarization_weights(1.45, 3, 68, field_angle_degrees=37)
        np.testing.assert_allclose(original, rotated, atol=2e-15)

    def test_high_reflectivity_dbr_matches_quarter_wave_solution(self):
        energy, n_h, n_l, pairs = 1.45, 3.6, 2.95, 70
        wavelength = 2 * np.pi * mc.HBAR_C / energy
        layers = [
            {"kind": "dielectric", "n": n, "d": wavelength / (4 * n)}
            for _ in range(pairs) for n in (n_h, n_l)
        ]
        ratio = (n_h / n_l) ** pairs
        expected_t = (-1) ** pairs * 2 / (ratio + 1 / ratio)
        for polarization in ("TE", "TM"):
            _, t = mc.reflection_transmission(
                energy, 0, polarization, layers, n_left=1, n_right=1,
            )
            np.testing.assert_allclose(t, expected_t, atol=1e-16, rtol=2e-13)

    def test_evanescent_incidence_is_rejected(self):
        with self.assertRaises(ValueError):
            mc.tangential_polarization_weights(1.45, 8, 45)

    def test_resolved_peaks_are_both_retained_and_refined(self):
        energy = np.linspace(1.4488, 1.4512, 12001)
        centers = np.array([1.4490173, 1.4510137])
        width = 2.0e-5
        values = sum(
            height / (1 + ((energy - center) / width) ** 2)
            for center, height in zip(centers, [0.4, 1.0])
        )
        peaks = mc.sampled_spectral_maxima(energy, values)
        self.assertEqual(len(peaks["energy"]), 2)
        np.testing.assert_allclose(peaks["energy"], centers, atol=2e-8, rtol=0)

    def test_unresolved_peak_is_not_an_average_of_energies(self):
        energy = np.linspace(1.449, 1.451, 4001)
        first, second, width = 1.44994, 1.45006, 0.0002
        values = (
            0.9 / (1 + ((energy - first) / width) ** 2)
            + 0.1 / (1 + ((energy - second) / width) ** 2)
        )
        peaks = mc.sampled_spectral_maxima(energy, values)
        self.assertEqual(len(peaks["energy"]), 1)
        self.assertGreater(abs(peaks["energy"][0] - (0.9 * first + 0.1 * second)), 1e-6)


class ComplexComparisonTests(unittest.TestCase):
    def inputs(self):
        k = np.linspace(0., 4., 31)
        photon = 1.45 + .001*k*k - 2e-6j
        exciton = mc.exciton_energy(k, 1.45, .5)
        omega = mc.mev(8/np.sqrt(6)); gamma_x = mc.mev(.25)
        g = np.sqrt((omega/2)**2+((-2*photon[0].imag-2*gamma_x)/4)**2)
        poles = np.array([mc.complex_coupled_oscillator_eigenvalues(
            c.real, x, 2*g, -2*c.imag, 2*gamma_x)
            for c,x in zip(photon,exciton)])
        return k, photon, exciton, gamma_x, omega, poles

    def test_known_complex_modes_and_common_validity_mask(self):
        k,c,x,gamma,omega,poles = self.inputs()
        dips = poles.real.copy(); dips[5,0] = np.nan
        resolved = np.ones(poles.shape, dtype=bool)
        resolved[7,1] = False
        result = mc.compare_complex_modes(k,c,x,gamma,omega,poles,dips,resolved)
        np.testing.assert_allclose(result['effective_poles_eV'],poles,atol=1e-14,rtol=0)
        self.assertAlmostEqual(result['parameters']['rabi_argument_eV'],
                               2*result['parameters']['g_eff_eV'],places=15)
        self.assertLess(result['checks']['zero_k_splitting_error_ueV'],1e-6)
        self.assertEqual(result['statistics']['energy_effective']['lp']['n_valid'],30)
        self.assertEqual(result['statistics']['energy_reflectivity']['lp']['n_valid'],30)
        self.assertEqual(result['statistics']['energy_effective']['up']['n_valid'],30)
        self.assertEqual(result['statistics']['linewidth_effective']['lp']['n_valid'],31)

    def test_detuning_and_nonpassive_photons_rejected(self):
        k,c,x,gamma,omega,poles = self.inputs()
        resolved = np.ones(poles.shape,dtype=bool)
        with self.assertRaises(ValueError):
            mc.compare_complex_modes(k,c+1e-6,x,gamma,omega,poles,poles.real,resolved)
        with self.assertRaises(ValueError):
            mc.compare_complex_modes(k,c.conjugate(),x,gamma,omega,poles,poles.real,resolved)
        with self.assertRaises(ValueError):
            mc.compare_complex_modes(k,c[:-1],x,gamma,omega,poles,poles.real,resolved)

    def test_uncoupled_degenerate_real_energies(self):
        k,c,x,gamma,omega,poles = self.inputs()
        # Equal real energies with unequal losses must preserve both roots.
        uncoupled = mc.complex_coupled_oscillator_eigenvalues(1.45,1.45,0.,4e-6,2*gamma)
        np.testing.assert_allclose(np.sort_complex(uncoupled),
                                   np.sort_complex([1.45-2e-6j,1.45-1j*gamma]),atol=1e-15)


if __name__ == "__main__":
    unittest.main()
