"""Synthetic TEST checks of the reference and experimental nodal adaptation."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace

import numpy as np

import CityCAT.rainfall.design_rainfall as public_api
from CityCAT.rainfall.design_rainfall import (
    TemporalConfiguration,
    build_fsr_feh_reference,
    central_depth_fraction,
    chronological_cumulative_fraction,
    export_citycat_rainfall,
    validate_citycat_rainfall,
    validate_rainfall_event,
)
from CityCAT.rainfall.design_rainfall.experimental_fsr_feh import (
    adapt_fsr_feh_reference_experimental,
)


# TEST only: the maximum endpoint here is not an ensemble approval.
TEST_CONFIGURATION = TemporalConfiguration(300, 7200, 7200, depth_tolerance_mm=1e-9)
TEST_DURATIONS_MINUTES = (15, 30, 45, 60, 90, 120)


def reference(profile="summer", duration=3600, start=0, depth=20.0,
              configuration=TEST_CONFIGURATION):
    return build_fsr_feh_reference(
        configuration, profile_name=profile, start_time_seconds=start,
        duration_seconds=duration, target_depth_mm=depth,
    )


def adapt(source):
    return adapt_fsr_feh_reference_experimental(
        source, rainfall_index=901, phase_event_id="TEST_FSR_FEH",
        storm_id="TEST_EXPERIMENTAL", role="TEST_REVIEW",
        profile_family="TEST_EXPERIMENTAL_FSR_FEH",
    )


class FSRFEHReferenceTests(unittest.TestCase):
    def test_exact_boundaries_and_outside_event(self):
        for profile in ("summer", "winter"):
            with self.subTest(profile=profile):
                self.assertEqual(central_depth_fraction(0, profile_name=profile), 0)
                self.assertEqual(central_depth_fraction(1, profile_name=profile), 1)
                for u, expected in ((-2, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1)):
                    self.assertEqual(chronological_cumulative_fraction(u, profile_name=profile), expected)

    def test_independent_high_precision_interior_reference_values(self):
        # Rounded from 60-digit Decimal evaluation of the published equation,
        # independent of the production expm1/log1p evaluation.
        expected = {
            "summer": (0.5830681194662652934, 0.8109537462629626562, 0.9313232686859165705),
            "winter": (0.5240409800614142565, 0.7966158364370766565, 0.9328168023522275731),
        }
        for profile, values in expected.items():
            for x, y in zip((0.25, 0.5, 0.75), values):
                with self.subTest(profile=profile, x=x):
                    self.assertAlmostEqual(central_depth_fraction(x, profile_name=profile), y, places=15)
                    self.assertAlmostEqual(chronological_cumulative_fraction((1-x)/2, profile_name=profile),
                                           (1-y)/2, places=15)
                    self.assertAlmostEqual(chronological_cumulative_fraction((1+x)/2, profile_name=profile),
                                           (1+y)/2, places=15)
        self.assertNotAlmostEqual(central_depth_fraction(0.25, profile_name="summer"),
                                  chronological_cumulative_fraction(0.25, profile_name="summer"))

    def test_stable_small_fractions(self):
        for profile, a, b in (("summer", 0.1, 0.815), ("winter", 0.06, 1.026)):
            x = 1e-20
            y = central_depth_fraction(x, profile_name=profile)
            self.assertAlmostEqual(y / (-math.log(a)*x**b/(1-a)), 1, places=12)
            u = 1e-20
            c = chronological_cumulative_fraction(u, profile_name=profile)
            self.assertGreater(c, 0)
            self.assertAlmostEqual(c / (-a*b*math.log(a)*u/(1-a)), 1, places=12)

    def test_monotonicity_symmetry_and_central_depth_identity(self):
        for profile in ("summer", "winter"):
            grid = np.linspace(0, 1, 1001)
            values = np.array([chronological_cumulative_fraction(float(u), profile_name=profile) for u in grid])
            self.assertTrue(np.all(np.diff(values) >= 0))
            np.testing.assert_allclose(values + values[::-1], 1, atol=3e-15, rtol=0)
            for x in (0.1, 0.25, 0.5, 0.75, 0.9):
                central = (chronological_cumulative_fraction((1+x)/2, profile_name=profile)
                           - chronological_cumulative_fraction((1-x)/2, profile_name=profile))
                self.assertAlmostEqual(central, central_depth_fraction(x, profile_name=profile), places=14)

    def test_intervals_units_odd_even_counts_and_off_grid_centres(self):
        for profile in ("summer", "winter"):
            for minutes in TEST_DURATIONS_MINUTES:
                with self.subTest(profile=profile, minutes=minutes):
                    source = reference(profile, minutes*60)
                    count = minutes // 5
                    self.assertEqual(source.interval_edges_seconds, tuple(range(0, 7201, 300)))
                    self.assertEqual(len(source.interval_depths_mm), 24)
                    depths = np.array(source.interval_depths_mm)
                    self.assertTrue(np.all(depths >= 0))
                    self.assertAlmostEqual(math.fsum(depths), 20, places=13)
                    self.assertEqual(np.count_nonzero(depths), count)
                    # Differences of cumulative values have rounding error on
                    # the scale of P, rather than the smallest interval depth.
                    np.testing.assert_allclose(
                        depths[:count], depths[:count][::-1], rtol=0,
                        atol=4*np.finfo(float).eps*source.target_depth_mm,
                    )
                    np.testing.assert_allclose(source.interval_average_intensities_mm_per_hour,
                                               depths*12, atol=1e-13)
                    self.assertEqual((source.depth_units, source.intensity_units, source.time_units),
                                     ("mm", "mm/hour", "seconds"))
                    self.assertEqual((minutes*30 in source.interval_edges_seconds), count % 2 == 0)
                    self.assertEqual(source.cumulative_depth_at(minutes*30), 10)

    def test_delayed_start_and_unmodified_reference_depth(self):
        for profile in ("summer", "winter"):
            for start, duration in ((900, 900), (1800, 3600)):
                source = reference(profile, duration, start)
                first, last = start//300, (start+duration)//300
                self.assertTrue(all(v == 0 for v in source.interval_depths_mm[:first]))
                self.assertTrue(all(v == 0 for v in source.interval_depths_mm[last:]))
                for t in (-100, 0, start):
                    self.assertEqual(source.cumulative_depth_at(t), 0)
                for t in (start+duration, 7200, 8000):
                    self.assertEqual(source.cumulative_depth_at(t), 20)
                self.assertEqual(source.cumulative_depth_at(start+duration/2), 10)
                doubled = reference(profile, duration, start, depth=40)
                np.testing.assert_array_equal(np.array(source.interval_depths_mm)*2, doubled.interval_depths_mm)

    def test_source_provenance(self):
        for profile, pair in (("summer", (0.1, 0.815)), ("winter", (0.06, 1.026))):
            source = reference(profile)
            self.assertEqual((source.parameter_a, source.parameter_b), pair)
            self.assertEqual(source.profile_name, profile)
            self.assertIn("Design-Storm-Profiles", source.source_reference)
            self.assertEqual(source.target_depth_mm, 20)

    def test_invalid_values_and_names(self):
        for invalid in ("autumn", "Summer", "", None, True, []):
            with self.subTest(profile=invalid), self.assertRaisesRegex(ValueError, "profile_name"):
                reference(invalid)
        for function in (central_depth_fraction, chronological_cumulative_fraction):
            for invalid in (True, "0.5", math.nan, math.inf, -math.inf, 1j):
                with self.subTest(function=function, value=invalid), self.assertRaises(ValueError):
                    function(invalid, profile_name="summer")
        for x in (-0.01, 1.01):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                central_depth_fraction(x, profile_name="summer")
        for depth in (0, -1, True, "20", math.nan, math.inf):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                reference(depth=depth)

    def test_invalid_timing_and_configuration(self):
        for value in (-1, True, 0.0, 100):
            with self.subTest(start=value), self.assertRaisesRegex(ValueError, "start_time_seconds"):
                reference(start=value)
        for value in (0, -300, True, 3600.0, 1000):
            with self.subTest(duration=value), self.assertRaisesRegex(ValueError, "duration_seconds"):
                reference(duration=value)
        with self.assertRaisesRegex(ValueError, "simulation horizon"):
            reference(duration=7200, start=300)
        with self.assertRaisesRegex(ValueError, "maximum active endpoint"):
            reference(duration=6000, configuration=TemporalConfiguration(300, 7200, 5400))
        with self.assertRaisesRegex(ValueError, "configuration"):
            reference(configuration=None)

    def test_required_arguments_and_short_reference(self):
        arguments = dict(configuration=TEST_CONFIGURATION, profile_name="summer",
                         start_time_seconds=0, duration_seconds=900, target_depth_mm=20)
        for missing in arguments:
            with self.subTest(missing=missing), self.assertRaises(TypeError):
                build_fsr_feh_reference(**{k: v for k, v in arguments.items() if k != missing})
        source = reference(duration=300)
        self.assertEqual(source.interval_depths_mm[0], 20)
        self.assertEqual(source.interval_average_intensities_mm_per_hour[0], 240)


class ExperimentalFSRFEHTests(unittest.TestCase):
    def test_experimental_api_is_not_publicly_exported(self):
        self.assertNotIn("adapt_fsr_feh_reference_experimental", public_api.__all__)
        self.assertFalse(hasattr(public_api, "adapt_fsr_feh_reference_experimental"))
        self.assertFalse(hasattr(public_api, "ExperimentalFSRFEHAdaptation"))

    def test_three_interval_independent_adaptation(self):
        # Symmetric three-interval reference [d0,d1,d0] produces two equal
        # positive nodes. After scaling its interval depths must be [P/4,P/2,P/4].
        for profile in ("summer", "winter"):
            source = reference(profile, duration=900)
            result = adapt(source)
            np.testing.assert_allclose(result.event.intensities_mm_per_hour[:4], [0, 120, 120, 0], atol=1e-12)
            np.testing.assert_allclose(result.adapted_interval_depths_mm[:3], [5, 10, 5], atol=1e-13)
            expected_before = source.interval_depths_mm[0] + source.interval_depths_mm[1]
            self.assertAlmostEqual(result.depth_before_normalization_mm, expected_before)
            self.assertAlmostEqual(result.normalization_factor, 20/expected_before)
            self.assertGreater(result.maximum_absolute_cumulative_depth_error_mm, 0)
            self.assertFalse(np.allclose(source.interval_depths_mm, result.adapted_interval_depths_mm))

    def test_adaptation_diagnostics_independent_calculations(self):
        for profile in ("summer", "winter"):
            for minutes in TEST_DURATIONS_MINUTES:
                with self.subTest(profile=profile, minutes=minutes):
                    source = reference(profile, minutes*60)
                    original_depths = source.interval_depths_mm
                    result = adapt(source)
                    count = minutes//5
                    nodal = [0.0]*25
                    for j in range(1, count):
                        nodal[j] = (source.interval_depths_mm[j-1]+source.interval_depths_mm[j])*6
                    before = sum((nodal[j]+nodal[j+1])/24 for j in range(24))
                    scaled = np.array(nodal)*(20/before)
                    intervals = np.array([(scaled[j]+scaled[j+1])/24 for j in range(24)])
                    cumulative = np.array([sum(intervals[:j]) for j in range(25)])
                    error = max(abs(cumulative-np.array(source.cumulative_depths_mm)))
                    self.assertAlmostEqual(result.depth_before_normalization_mm, before, places=12)
                    self.assertAlmostEqual(result.final_total_depth_mm, 20, places=12)
                    self.assertAlmostEqual(result.normalization_factor, 20/before, places=13)
                    np.testing.assert_allclose(result.adapted_interval_depths_mm, intervals, atol=1e-13)
                    np.testing.assert_allclose(result.adapted_cumulative_depths_mm, cumulative, atol=1e-13)
                    self.assertAlmostEqual(result.maximum_absolute_cumulative_depth_error_mm, error, places=12)
                    self.assertAlmostEqual(result.maximum_absolute_cumulative_depth_error_fraction, error/20, places=13)
                    self.assertEqual(result.cumulative_error_evaluation, "reference_interval_edges")
                    self.assertAlmostEqual(result.reference_maximum_interval_average_intensity_mm_per_hour,
                                           max(source.interval_depths_mm)*12, places=12)
                    self.assertAlmostEqual(result.adapted_maximum_nodal_intensity_mm_per_hour, max(scaled), places=12)
                    for time in result.adapted_maximum_nodal_times_seconds:
                        self.assertEqual(result.event.intensities_mm_per_hour[time//300],
                                         result.adapted_maximum_nodal_intensity_mm_per_hour)
                    self.assertIs(result.reference, source)
                    self.assertEqual(source.interval_depths_mm, original_depths)
                    validate_rainfall_event(result.event, TEST_CONFIGURATION)

    def test_delayed_zero_endpoints_and_conservation(self):
        for profile in ("summer", "winter"):
            for start, duration in ((900, 900), (1800, 3600)):
                source = reference(profile, duration, start)
                result = adapt(source)
                times = np.array(source.interval_edges_seconds)
                values = result.event.intensities_mm_per_hour
                self.assertTrue(np.all(values[times <= start] == 0))
                self.assertTrue(np.all(values[times >= start+duration] == 0))
                self.assertAlmostEqual(float(np.trapz(values, times/3600)), 20, places=12)
                self.assertEqual(result.event.active_duration_seconds, start+duration)
                for time in (-1, 0, start):
                    self.assertEqual(result.cumulative_depth_at(time), 0)
                for time in (start+duration, 7200, 9000):
                    self.assertAlmostEqual(result.cumulative_depth_at(time), 20, places=12)
                self.assertAlmostEqual(result.cumulative_depth_at(start+duration/2), 10, places=12)

    def test_partial_interval_cumulative_is_quadratic(self):
        result = adapt(reference(duration=900))
        self.assertAlmostEqual(result.cumulative_depth_at(150), 1.25)
        self.assertAlmostEqual(result.cumulative_depth_at(300), 5)
        self.assertAlmostEqual(result.cumulative_depth_at(450), 10)
        self.assertAlmostEqual(result.cumulative_depth_at(750), 18.75)
        self.assertNotAlmostEqual(result.cumulative_depth_at(150), 2.5)
        # Independent midpoint integration is exact for a linear intensity,
        # provided each interval and the final partial interval are separate.
        times = np.array(result.reference.interval_edges_seconds)
        values = result.event.intensities_mm_per_hour
        for stop in (17.5, 425, 899.5, 1800):
            expected = 0.0
            for a, b in zip(times[:-1], times[1:]):
                b = min(b, stop)
                if b > a:
                    expected += (b-a)*float(np.interp((a+b)/2, times, values))/3600
            self.assertAlmostEqual(result.cumulative_depth_at(stop), expected, places=12)

    def test_rejects_insufficient_support_or_invalid_integrals(self):
        with self.assertRaisesRegex(ValueError, "insufficient temporal support"):
            adapt(reference(duration=300))
        valid = reference()
        with self.assertRaisesRegex(ValueError, "integral.*finite and positive"):
            adapt(replace(valid, interval_average_intensities_mm_per_hour=(0.0,)*24))
        for bad in ((math.nan,)*24, (-1.0,)*24, (1.0,)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "interval intensities"):
                adapt(replace(valid, interval_average_intensities_mm_per_hour=bad))
        with self.assertRaisesRegex(ValueError, "reference"):
            adapt(None)

    def test_metadata_and_temporary_core_export(self):
        result = adapt(reference("winter"))
        self.assertIn("experimental", result.event.peak_metadata["status"])
        self.assertEqual(result.event.peak_metadata["normalization_factor_units"], "dimensionless")
        self.assertEqual(result.event.phase_event_id, "TEST_FSR_FEH")
        for absent in ("return_period_years", "arf", "seasonal_correction_factor", "uplift_factor"):
            self.assertNotIn(absent, result.event.peak_metadata)
        with self.assertRaises(ValueError):
            result.event.intensities_mm_per_hour[1] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(directory, result.event, TEST_CONFIGURATION)
            validate_citycat_rainfall(path, result.event, TEST_CONFIGURATION)


if __name__ == "__main__":
    unittest.main()
