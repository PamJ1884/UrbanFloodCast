"""Synthetic TEST cases for two-pulse composite triangular rainfall events."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    CompositeTriangularPulse,
    RainfallEvent,
    TemporalConfiguration,
    TriangularPulse,
    export_citycat_rainfall,
    generate_composite_triangular_pulse,
    generate_timeline,
    generate_triangular_pulse,
    validate_citycat_rainfall,
    validate_rainfall_event,
)


CONFIGURATION = TemporalConfiguration(
    rainfall_timestep_seconds=300,
    simulation_horizon_seconds=7200,
    maximum_active_duration_seconds=5400,
    depth_tolerance_mm=1e-9,
    citycat_value_tolerance=1e-15,
)


def generate_test_event(
    composite: CompositeTriangularPulse,
) -> RainfallEvent:
    return generate_composite_triangular_pulse(
        CONFIGURATION,
        composite,
        rainfall_index=951,
        phase_event_id="TEST-COMPOSITE",
        storm_id="TEST-COMPOSITE-STORM",
        role="TEST_ROLE",
        profile_family="TEST_COMPOSITE_TRIANGULAR",
    )


def integrated_depth_mm(event: RainfallEvent) -> float:
    return float(
        np.trapz(
            event.intensities_mm_per_hour,
            generate_timeline(CONFIGURATION) / 3600.0,
        )
    )


class CompositeTriangularPulseTests(unittest.TestCase):
    def test_generates_valid_rainfall_event(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.40),
            TriangularPulse(3000, 1500, 18.0, 0.65),
        )

        event = generate_test_event(composite)

        self.assertIsInstance(event, RainfallEvent)
        self.assertEqual(event.target_depth_mm, 42.0)
        self.assertEqual(event.active_duration_seconds, 4500)
        validate_rainfall_event(event, CONFIGURATION)

    def test_total_depth_is_sum_of_constituent_depths(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.40),
            TriangularPulse(3000, 1500, 18.0, 0.65),
        )

        event = generate_test_event(composite)

        self.assertAlmostEqual(
            composite.target_depth_mm,
            42.0,
        )
        self.assertAlmostEqual(
            integrated_depth_mm(event),
            42.0,
        )

    def test_constituent_depths_are_preserved(self) -> None:
        first = TriangularPulse(0, 1800, 24.0, 0.40)
        second = TriangularPulse(3000, 1500, 18.0, 0.65)

        composite_event = generate_test_event(
            CompositeTriangularPulse(first, second)
        )

        first_event = generate_triangular_pulse(
            CONFIGURATION,
            first,
            rainfall_index=1,
            phase_event_id="FIRST",
            storm_id="FIRST",
            role="TEST",
            profile_family="FIRST",
        )
        second_event = generate_triangular_pulse(
            CONFIGURATION,
            second,
            rainfall_index=2,
            phase_event_id="SECOND",
            storm_id="SECOND",
            role="TEST",
            profile_family="SECOND",
        )

        expected = (
            first_event.intensities_mm_per_hour
            + second_event.intensities_mm_per_hour
        )

        np.testing.assert_allclose(
            composite_event.intensities_mm_per_hour,
            expected,
        )

        self.assertAlmostEqual(
            integrated_depth_mm(first_event),
            24.0,
        )
        self.assertAlmostEqual(
            integrated_depth_mm(second_event),
            18.0,
        )

    def test_complete_constituent_stage3a_metadata_is_preserved(self) -> None:
        first_pulse = TriangularPulse(0, 1800, 24.0, 0.5)
        weaker_second_pulse = TriangularPulse(3000, 1200, 4.0, 0.5)
        composite_event = generate_test_event(
            CompositeTriangularPulse(first_pulse, weaker_second_pulse)
        )
        first_event = generate_triangular_pulse(
            CONFIGURATION,
            first_pulse,
            rainfall_index=1,
            phase_event_id="TEST-FIRST",
            storm_id="TEST-FIRST",
            role="TEST",
            profile_family="TEST-FIRST",
        )
        second_event = generate_triangular_pulse(
            CONFIGURATION,
            weaker_second_pulse,
            rainfall_index=2,
            phase_event_id="TEST-SECOND",
            storm_id="TEST-SECOND",
            role="TEST",
            profile_family="TEST-SECOND",
        )

        composite_metadata = composite_event.peak_metadata
        first_metadata = first_event.peak_metadata
        second_metadata = second_event.peak_metadata
        assert composite_metadata is not None
        assert first_metadata is not None
        assert second_metadata is not None
        stored_first = composite_metadata["first_pulse_peak_metadata"]
        stored_second = composite_metadata["second_pulse_peak_metadata"]
        self.assertEqual(dict(stored_first), dict(first_metadata))
        self.assertEqual(dict(stored_second), dict(second_metadata))

        required_stage3a_fields = (
            "requested_peak_position_ratio",
            "requested_peak_time_seconds",
            "sampled_maximum_intensity_mm_per_hour",
            "sampled_maximum_times_seconds",
            "sampled_shape_integral_hours",
            "shape_normalisation_factor_mm_per_hour",
            "shape_normalisation_factor_meaning",
            "depth_integration_method",
        )
        for field in required_stage3a_fields:
            with self.subTest(field=field):
                self.assertEqual(stored_first[field], first_metadata[field])
                self.assertEqual(stored_second[field], second_metadata[field])

        self.assertLess(
            stored_second["sampled_maximum_intensity_mm_per_hour"],
            composite_metadata["sampled_maximum_intensity_mm_per_hour"],
        )
        self.assertTrue(
            set(stored_second["sampled_maximum_times_seconds"]).isdisjoint(
                composite_metadata["sampled_maximum_times_seconds"]
            )
        )

    def test_independent_example_and_citycat_export(self) -> None:
        first_pulse = TriangularPulse(0, 1800, 12.0, 0.5)
        second_pulse = TriangularPulse(3000, 900, 8.0, 0.5)
        event = generate_test_event(
            CompositeTriangularPulse(first_pulse, second_pulse)
        )
        timeline = generate_timeline(CONFIGURATION)
        expected_first = np.zeros(CONFIGURATION.number_of_records)
        expected_second = np.zeros(CONFIGURATION.number_of_records)
        expected_first[1:6] = [16.0, 32.0, 48.0, 32.0, 16.0]
        expected_second[11:13] = [48.0, 48.0]
        expected_composite = expected_first + expected_second

        np.testing.assert_allclose(
            event.intensities_mm_per_hour,
            expected_composite,
        )
        self.assertAlmostEqual(
            float(np.trapz(expected_first, timeline / 3600.0)),
            12.0,
        )
        self.assertAlmostEqual(
            float(np.trapz(expected_second, timeline / 3600.0)),
            8.0,
        )
        self.assertAlmostEqual(integrated_depth_mm(event), 20.0)
        dry_gap = (timeline >= 1800) & (timeline <= 3000)
        self.assertTrue(np.all(event.intensities_mm_per_hour[dry_gap] == 0.0))
        self.assertEqual(event.active_duration_seconds, 3900)
        self.assertTrue(
            np.all(event.intensities_mm_per_hour[timeline >= 3900] == 0.0)
        )

        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(directory, event, CONFIGURATION)
            validate_citycat_rainfall(path, event, CONFIGURATION)

    def test_dry_gap_contains_zero_rainfall(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.40),
            TriangularPulse(3000, 1500, 18.0, 0.65),
        )

        event = generate_test_event(composite)
        timeline = generate_timeline(CONFIGURATION)

        gap = (
            (timeline >= 1800)
            & (timeline <= 3000)
        )

        self.assertTrue(
            np.all(event.intensities_mm_per_hour[gap] == 0.0)
        )
        self.assertEqual(composite.dry_gap_seconds, 1200)

    def test_rejects_touching_pulses_without_positive_gap(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "strictly positive dry gap",
        ):
            CompositeTriangularPulse(
                TriangularPulse(0, 1800, 20.0, 0.5),
                TriangularPulse(1800, 1200, 10.0, 0.5),
            )

    def test_rejects_overlapping_pulses(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "strictly positive dry gap",
        ):
            CompositeTriangularPulse(
                TriangularPulse(0, 1800, 20.0, 0.5),
                TriangularPulse(1500, 1200, 10.0, 0.5),
            )

    def test_rejects_second_pulse_before_first(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "strictly positive dry gap",
        ):
            CompositeTriangularPulse(
                TriangularPulse(1800, 1200, 20.0, 0.5),
                TriangularPulse(0, 1200, 10.0, 0.5),
            )

    def test_rejects_second_endpoint_beyond_active_limit(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 20.0, 0.5),
            TriangularPulse(4800, 1200, 10.0, 0.5),
        )

        with self.assertRaisesRegex(
            ValueError,
            "maximum active endpoint",
        ):
            generate_test_event(composite)

    def test_off_grid_peaks_are_preserved_in_metadata(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.37),
            TriangularPulse(3000, 1500, 18.0, 0.63),
        )

        event = generate_test_event(composite)
        metadata = event.peak_metadata
        assert metadata is not None

        self.assertAlmostEqual(
            metadata["first_pulse_requested_peak_time_seconds"],
            666.0,
        )
        self.assertAlmostEqual(
            metadata["second_pulse_requested_peak_time_seconds"],
            3945.0,
        )

    def test_metadata_describes_both_pulses_and_gap(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.40),
            TriangularPulse(3000, 1500, 18.0, 0.65),
        )

        event = generate_test_event(composite)
        metadata = event.peak_metadata
        assert metadata is not None

        self.assertEqual(metadata["number_of_pulses"], 2)
        self.assertEqual(
            metadata["first_pulse_target_depth_mm"],
            24.0,
        )
        self.assertEqual(
            metadata["second_pulse_target_depth_mm"],
            18.0,
        )
        self.assertEqual(metadata["dry_gap_seconds"], 1200)
        self.assertEqual(metadata["active_endpoint_seconds"], 4500)
        self.assertAlmostEqual(metadata["integrated_depth_mm"], 42.0)

    def test_identical_inputs_are_deterministic(self) -> None:
        composite = CompositeTriangularPulse(
            TriangularPulse(0, 1800, 24.0, 0.40),
            TriangularPulse(3000, 1500, 18.0, 0.65),
        )

        first = generate_test_event(composite)
        second = generate_test_event(composite)

        np.testing.assert_array_equal(
            first.intensities_mm_per_hour,
            second.intensities_mm_per_hour,
        )
        self.assertEqual(
            dict(first.peak_metadata),
            dict(second.peak_metadata),
        )


if __name__ == "__main__":
    unittest.main()
