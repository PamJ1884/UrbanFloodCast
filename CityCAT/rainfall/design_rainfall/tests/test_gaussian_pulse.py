"""Synthetic Gaussian-family tests; illustrative values are not ensemble defaults."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    GaussianPulse,
    TemporalConfiguration,
    export_citycat_rainfall,
    generate_gaussian_pulse,
    generate_timeline,
    validate_citycat_rainfall,
    validate_rainfall_event,
)
from CityCAT.rainfall.design_rainfall.gaussian_pulse import (
    _rainfall_diagnostics,
    _standardized_mode,
)


TEST_CONFIGURATION = TemporalConfiguration(300, 7200, 5400, depth_tolerance_mm=1e-9)
TEST_PULSE = GaussianPulse(0, 3600, 20.0, 0.5, 0.18, 0.0, 1.0 / 12.0)


def generate_test_event(pulse=TEST_PULSE, configuration=TEST_CONFIGURATION, **identity):
    arguments = dict(
        rainfall_index=901, phase_event_id="TEST_GAUSSIAN_EVENT",
        storm_id="TEST_GAUSSIAN_STORM", role="TEST_ROLE",
        profile_family="TEST_GAUSSIAN_FAMILY",
    )
    arguments.update(identity)
    return generate_gaussian_pulse(configuration, pulse, **arguments)


def integrate_linear_series(times, values, *, power=0, origin=0.0, stop=None):
    """Independent two-node Gauss-Legendre quadrature, exact through degree 3.

    A linear intensity times (t-origin)**power, power<=2, has degree <=3.
    Split at every sample and at stop, then evaluate intensity with np.interp.
    """
    nodes, weights = np.polynomial.legendre.leggauss(2)
    terms = []
    for left, right in zip(times[:-1], times[1:]):
        if stop is not None:
            right = min(right, stop)
        if right <= left:
            continue
        x = (left + right) / 2 + nodes * (right - left) / 2
        y = np.interp(x, times, values)
        terms.append(float(np.dot(weights, y * (x - origin) ** power) * (right - left) / 2))
    return math.fsum(terms)


class GaussianPulseTests(unittest.TestCase):
    def test_all_seven_parameters_are_required(self):
        parameters = {field.name: getattr(TEST_PULSE, field.name) for field in fields(TEST_PULSE)}
        self.assertEqual(len(parameters), 7)
        for missing in parameters:
            with self.subTest(missing=missing), self.assertRaises(TypeError):
                GaussianPulse(**{key: value for key, value in parameters.items() if key != missing})

    def test_zero_alpha_matches_independent_normal_and_taper(self):
        pulse = replace(TEST_PULSE, edge_taper_ratio=0.25)
        event = generate_test_event(pulse)
        times = generate_timeline(TEST_CONFIGURATION)
        expected = np.zeros(times.size)
        for i, t in enumerate(times):
            if 0 < t < 3600:
                u = t / 3600
                window = math.sin(math.pi / 2 * min(u / 0.25, (1 - u) / 0.25, 1)) ** 2
                expected[i] = math.exp(-((t - 1800) / 648) ** 2 / 2) / math.sqrt(2 * math.pi) * window
        raw_area = integrate_linear_series(times, expected) / 3600
        expected *= 20 / raw_area
        np.testing.assert_allclose(event.intensities_mm_per_hour, expected, rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(expected[:13], expected[:13][::-1], atol=2e-14)
        metadata = event.peak_metadata
        self.assertEqual(metadata["standardized_mode"], 0)
        self.assertEqual(metadata["distribution_location_seconds"], 1800)
        self.assertAlmostEqual(metadata["temporal_centroid_seconds"], 1800)
        self.assertAlmostEqual(metadata["depth_fraction_up_to_requested_peak"], 0.5)
        self.assertAlmostEqual(metadata["sampled_shape_integral_hours"], raw_area)

    def test_opposite_alpha_and_peak_ratios_mirror(self):
        for alpha in (0.2, 2.0, 15.0):
            with self.subTest(alpha=alpha):
                first = generate_test_event(replace(TEST_PULSE, skewness_parameter=alpha, peak_position_ratio=0.35))
                second = generate_test_event(replace(TEST_PULSE, skewness_parameter=-alpha, peak_position_ratio=0.65))
                np.testing.assert_allclose(first.intensities_mm_per_hour[:13],
                                           second.intensities_mm_per_hour[:13][::-1],
                                           rtol=2e-13, atol=1e-13)
                self.assertAlmostEqual(first.peak_metadata["temporal_centroid_seconds"]
                                       + second.peak_metadata["temporal_centroid_seconds"], 3600)

    def test_illustrative_configurations_conserve_depth(self):
        for alpha, scale_ratio in ((-2, 0.18), (0, 0.15), (2, 0.18)):
            with self.subTest(alpha=alpha, scale_ratio=scale_ratio):
                pulse = replace(TEST_PULSE, skewness_parameter=alpha, scale_ratio=scale_ratio)
                event = generate_test_event(pulse)
                values = event.intensities_mm_per_hour
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.all(values >= 0))
                self.assertEqual(values.size, 25)
                area = integrate_linear_series(generate_timeline(TEST_CONFIGURATION), values) / 3600
                self.assertAlmostEqual(area, 20, delta=TEST_CONFIGURATION.depth_tolerance_mm)
                validate_rainfall_event(event, TEST_CONFIGURATION)

    def test_mode_satisfies_independent_stationarity_equation(self):
        for alpha in (0.01, 0.2, 1, 1.01, 2, 10, 100):
            with self.subTest(alpha=alpha):
                event = generate_test_event(replace(TEST_PULSE, skewness_parameter=alpha))
                mode = event.peak_metadata["standardized_mode"]
                v = alpha * mode
                phi = math.exp(-v * v / 2) / math.sqrt(2 * math.pi)
                cdf = (1 + math.erf(v / math.sqrt(2))) / 2
                self.assertAlmostEqual(mode * cdf / (alpha * phi), 1.0, delta=2e-11)
                location = event.peak_metadata["distribution_location_seconds"]
                self.assertAlmostEqual(location + 648 * mode, 1800)
                self.assertNotEqual(location, 1800)

    def test_controlled_modes_for_extreme_finite_alpha(self):
        for alpha in (1e-200, 1e200, 1e308):
            with self.subTest(alpha=alpha):
                mode = _standardized_mode(alpha)
                self.assertTrue(math.isfinite(mode))
                self.assertGreater(mode, 0)
                self.assertEqual(_standardized_mode(-alpha), -mode)
                if alpha < 1:
                    self.assertAlmostEqual(mode / alpha, math.sqrt(2 / math.pi), delta=1e-13)
                else:
                    v = alpha * mode
                    residual = math.log(mode) - math.log(alpha) + v * v / 2 + math.log(2 * math.pi) / 2
                    self.assertAlmostEqual(residual, 0, delta=1e-9)

    def test_mode_failure_is_explicit(self):
        with patch("CityCAT.rainfall.design_rainfall.gaussian_pulse.MODE_MAX_ITERATIONS", 0):
            with self.assertRaisesRegex(ValueError, "mode calculation failed.*converge"):
                generate_test_event(replace(TEST_PULSE, skewness_parameter=2))

    def test_delayed_start_and_absolute_endpoint(self):
        pulse = replace(TEST_PULSE, start_time_seconds=600, skewness_parameter=-2)
        event = generate_test_event(pulse)
        times = generate_timeline(TEST_CONFIGURATION)
        self.assertTrue(np.all(event.intensities_mm_per_hour[times <= 600] == 0))
        self.assertTrue(np.all(event.intensities_mm_per_hour[times >= 4200] == 0))
        self.assertEqual(event.active_duration_seconds, 4200)
        self.assertEqual(event.peak_metadata["pulse_end_time_seconds"], 4200)

    def test_off_grid_peak_and_non_25_record_timeline(self):
        configuration = TemporalConfiguration(120, 3600, 3000, depth_tolerance_mm=1e-9)
        pulse = GaussianPulse(240, 1200, 7.5, 0.37, 0.2, 2, 0.1)
        event = generate_test_event(pulse, configuration)
        self.assertEqual(event.intensities_mm_per_hour.size, 31)
        self.assertEqual(event.peak_metadata["requested_peak_time_seconds"], 684)
        self.assertNotIn(684, generate_timeline(configuration))
        self.assertNotIn("peak_time_seconds", event.peak_metadata)
        self.assertEqual(event.active_duration_seconds, 1440)
        validate_rainfall_event(event, configuration)

    def test_all_sampled_maximum_times_are_retained(self):
        event = generate_test_event(replace(TEST_PULSE, duration_seconds=3300))
        self.assertEqual(event.peak_metadata["sampled_maximum_times_seconds"], (1500, 1800))
        self.assertEqual(event.peak_metadata["requested_peak_time_seconds"], 1650)

    def test_taper_can_move_peak(self):
        pulse = replace(TEST_PULSE, peak_position_ratio=1 / 12, edge_taper_ratio=0.4)
        event = generate_test_event(pulse)
        self.assertEqual(event.peak_metadata["requested_peak_time_seconds"], 300)
        self.assertGreater(event.peak_metadata["sampled_maximum_times_seconds"][0], 300)

    def test_diagnostics_against_independent_quadrature(self):
        for alpha in (-2, 0, 2):
            pulse = replace(TEST_PULSE, start_time_seconds=600, peak_position_ratio=0.37,
                            skewness_parameter=alpha, edge_taper_ratio=0.2)
            event = generate_test_event(pulse)
            times = generate_timeline(TEST_CONFIGURATION)
            values = event.intensities_mm_per_hour
            area = integrate_linear_series(times, values)
            centroid = integrate_linear_series(times, values, power=1) / area
            variance = integrate_linear_series(times, values, power=2, origin=centroid) / area
            partial = integrate_linear_series(times, values, stop=pulse.peak_time_seconds) / area
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(event.peak_metadata["temporal_centroid_seconds"], centroid, places=10)
                self.assertAlmostEqual(event.peak_metadata["effective_temporal_standard_deviation_seconds"],
                                       math.sqrt(variance), places=10)
                self.assertAlmostEqual(event.peak_metadata["depth_fraction_up_to_requested_peak"], partial, places=14)
                self.assertNotAlmostEqual(math.sqrt(variance), pulse.scale_ratio * pulse.duration_seconds, places=3)

    def test_analytic_triangle_moments_and_off_grid_accumulation(self):
        # Symmetric linear triangle: centroid=start+D/2, variance=D**2/24;
        # at ratio r<1/2 its accumulated fraction is 2*r**2.
        pulse = GaussianPulse(600, 1200, 1, 0.375, 0.2, 0, 0.1)
        centroid, deviation, fraction = _rainfall_diagnostics(
            np.array([0, 600, 1200, 1800, 2400]), np.array([0, 0, 2, 0, 0]), pulse,
        )
        self.assertAlmostEqual(centroid, 1200)
        self.assertAlmostEqual(deviation, 1200 / math.sqrt(24))
        self.assertAlmostEqual(fraction, 0.28125)

    def test_off_grid_partial_interval_is_not_rounded_to_a_sample(self):
        pulse = replace(TEST_PULSE, peak_position_ratio=0.37, skewness_parameter=2)
        event = generate_test_event(pulse)
        times = generate_timeline(TEST_CONFIGURATION)
        values = event.intensities_mm_per_hour
        index = int(pulse.peak_time_seconds // 300)
        elapsed = pulse.peak_time_seconds - times[index]
        y_peak = values[index] + (values[index + 1] - values[index]) * elapsed / 300
        before = np.trapz(values[:index + 1], times[:index + 1])
        partial = elapsed * (values[index] + y_peak) / 2
        total = np.trapz(values, times)
        fraction = event.peak_metadata["depth_fraction_up_to_requested_peak"]
        self.assertAlmostEqual(fraction, (before + partial) / total, places=14)
        self.assertGreater(fraction, before / total)
        self.assertLess(fraction, np.trapz(values[:index + 2], times[:index + 2]) / total)

    def test_metadata_identity_and_no_frequency_inference(self):
        pulse = replace(TEST_PULSE, skewness_parameter=2)
        event = generate_test_event(pulse)
        metadata = event.peak_metadata
        self.assertEqual(event.rainfall_index, 901)
        self.assertEqual(event.phase_event_id, "TEST_GAUSSIAN_EVENT")
        self.assertEqual(event.storm_id, "TEST_GAUSSIAN_STORM")
        self.assertEqual(event.role, "TEST_ROLE")
        self.assertEqual(event.profile_family, "TEST_GAUSSIAN_FAMILY")
        for key, expected in {
            "pulse_start_time_seconds": 0, "pulse_duration_seconds": 3600,
            "pulse_end_time_seconds": 3600, "target_depth_mm": 20,
            "requested_peak_position_ratio": 0.5, "scale_ratio": 0.18,
            "skewness_parameter": 2, "edge_taper_ratio": 1 / 12,
            "scale_seconds": 648, "rainfall_timestep_seconds": 300,
            "depth_integration_method": "trapezoidal_sampled_piecewise_linear",
        }.items():
            self.assertEqual(metadata[key], expected)
        self.assertEqual(metadata["positive_interior_samples"],
                         np.count_nonzero(event.intensities_mm_per_hour > 0))
        self.assertAlmostEqual(metadata["integrated_depth_mm"], 20)
        self.assertAlmostEqual(metadata["sampled_shape_integral_hours"]
                               * metadata["shape_normalisation_factor_mm_per_hour"], 20)
        self.assertIn("sin((pi/2)", metadata["taper_definition"])
        self.assertIn("millimetres per hour", metadata["shape_normalisation_factor_meaning"])
        for absent in ("return_period_years", "uplift_factor", "climate_scenario", "measured_skewness"):
            self.assertNotIn(absent, metadata)
        with self.assertRaises(TypeError):
            metadata["target_depth_mm"] = 10

    def test_invalid_parameters(self):
        invalid = {
            "start_time_seconds": [-1, True, 0.0],
            "duration_seconds": [0, -300, False, 3600.0],
            "target_depth_mm": [0, -1], "peak_position_ratio": [0, 1, -0.1, 1.1],
            "scale_ratio": [0, -1], "edge_taper_ratio": [0, 0.5, -0.1, 0.7],
            "skewness_parameter": [],
        }
        for field, values in invalid.items():
            for value in values + [math.nan, math.inf, -math.inf, True, "1", None, 1 + 2j]:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    replace(TEST_PULSE, **{field: value})

    def test_invalid_timing_and_endpoints(self):
        for pulse, message in (
            (replace(TEST_PULSE, start_time_seconds=1), "start_time_seconds.*align"),
            (replace(TEST_PULSE, duration_seconds=3601), "duration_seconds.*align"),
            (replace(TEST_PULSE, start_time_seconds=2100), "maximum active endpoint"),
            (replace(TEST_PULSE, start_time_seconds=3900), "simulation horizon"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                generate_test_event(pulse)
        endpoint = generate_test_event(replace(TEST_PULSE, start_time_seconds=1800))
        self.assertEqual(endpoint.active_duration_seconds, 5400)

    def test_inadequate_sampling_and_underflow(self):
        for changes in ({"duration_seconds": 300}, {"duration_seconds": 600},
                        {"scale_ratio": 1e-6}, {"scale_ratio": 1e-200, "peak_position_ratio": 0.37}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "temporal-resolution.*two"):
                generate_test_event(replace(TEST_PULSE, **changes))
        event = generate_test_event(replace(TEST_PULSE, duration_seconds=900))
        self.assertEqual(event.peak_metadata["positive_interior_samples"], 2)

    def test_nonfinite_derived_scale_and_normalisation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale/location"):
            generate_test_event(replace(TEST_PULSE, scale_ratio=1e308))
        with self.assertRaisesRegex(ValueError, "normalisation|integrated depth"):
            generate_test_event(replace(TEST_PULSE, target_depth_mm=1e308))

    def test_no_intensity_cap_or_depth_dependent_shape_change(self):
        small = generate_test_event()
        large = generate_test_event(replace(TEST_PULSE, target_depth_mm=2000))
        np.testing.assert_allclose(large.intensities_mm_per_hour, 100 * small.intensities_mm_per_hour)
        self.assertGreater(float(np.max(large.intensities_mm_per_hour)), 1000)

    def test_input_types_and_identity_use_existing_validation(self):
        with self.assertRaisesRegex(ValueError, "configuration"):
            generate_test_event(configuration=None)
        with self.assertRaisesRegex(ValueError, "pulse"):
            generate_test_event(pulse=None)
        for arguments in ({"rainfall_index": True}, {"phase_event_id": ""},
                          {"storm_id": ""}, {"role": " "}, {"profile_family": ""}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generate_test_event(**arguments)

    def test_temporary_citycat_export_and_validation(self):
        event = generate_test_event(replace(TEST_PULSE, skewness_parameter=-2))
        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(directory, event, TEST_CONFIGURATION)
            validate_citycat_rainfall(path, event, TEST_CONFIGURATION)
            self.assertEqual(path.name, "Rainfall_Data_901.txt")
            records = path.read_text().splitlines()[5:]
            self.assertEqual(len(records), 25)
            self.assertEqual([int(row.split()[0]) for row in records], list(range(0, 7201, 300)))


if __name__ == "__main__":
    unittest.main()
