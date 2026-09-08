"""Synthetic TEST cases for reusable single triangular rainfall pulses."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    RainfallEvent,
    TemporalConfiguration,
    TriangularPulse,
    export_citycat_rainfall,
    generate_timeline,
    generate_triangular_pulse,
    validate_citycat_rainfall,
    validate_rainfall_event,
)


PHASE2_COMPATIBLE_CONFIGURATION = TemporalConfiguration(
    rainfall_timestep_seconds=300,
    simulation_horizon_seconds=7200,
    maximum_active_duration_seconds=5400,
    depth_tolerance_mm=1e-9,
    citycat_value_tolerance=1e-15,
)
FUTURE_CONFIGURATION = TemporalConfiguration(
    rainfall_timestep_seconds=120,
    simulation_horizon_seconds=3600,
    maximum_active_duration_seconds=3000,
    depth_tolerance_mm=1e-9,
    citycat_value_tolerance=1e-15,
)


def generate_test_event(
    pulse: TriangularPulse,
    configuration: TemporalConfiguration = PHASE2_COMPATIBLE_CONFIGURATION,
    *,
    rainfall_index: int = 901,
) -> RainfallEvent:
    return generate_triangular_pulse(
        configuration,
        pulse,
        rainfall_index=rainfall_index,
        phase_event_id="TEST-PHASE-EVENT",
        storm_id="TEST-STORM",
        role="TEST_ROLE",
        profile_family="TEST_TRIANGULAR_PULSE",
    )


def integrated_depth_mm(
    event: RainfallEvent,
    configuration: TemporalConfiguration,
) -> float:
    return float(
        np.trapz(
            event.intensities_mm_per_hour,
            generate_timeline(configuration) / 3600.0,
        )
    )


class TriangularPulseTests(unittest.TestCase):
    def test_independently_calculable_symmetric_triangle(self) -> None:
        pulse = TriangularPulse(
            start_time_seconds=0,
            duration_seconds=1200,
            target_depth_mm=10.0,
            peak_position_ratio=0.5,
        )
        event = generate_test_event(pulse)
        expected = np.zeros(PHASE2_COMPATIBLE_CONFIGURATION.number_of_records)
        expected[:5] = [0.0, 30.0, 60.0, 30.0, 0.0]
        np.testing.assert_allclose(event.intensities_mm_per_hour, expected)
        self.assertAlmostEqual(
            integrated_depth_mm(event, PHASE2_COMPATIBLE_CONFIGURATION),
            10.0,
        )
        self.assertEqual(event.active_duration_seconds, 1200)
        self.assertEqual(event.phase_event_id, "TEST-PHASE-EVENT")
        self.assertEqual(event.storm_id, "TEST-STORM")
        self.assertEqual(event.role, "TEST_ROLE")
        self.assertEqual(event.profile_family, "TEST_TRIANGULAR_PULSE")

        metadata = event.peak_metadata
        assert metadata is not None
        self.assertEqual(metadata["pulse_start_time_seconds"], 0)
        self.assertEqual(metadata["pulse_duration_seconds"], 1200)
        self.assertEqual(metadata["pulse_end_time_seconds"], 1200)
        self.assertEqual(metadata["requested_peak_position_ratio"], 0.5)
        self.assertEqual(metadata["requested_peak_time_seconds"], 600.0)
        self.assertEqual(metadata["sampled_maximum_times_seconds"], (600,))
        self.assertEqual(metadata["sampled_maximum_intensity_mm_per_hour"], 60.0)
        self.assertEqual(metadata["shape_normalisation_factor_mm_per_hour"], 60.0)
        self.assertAlmostEqual(metadata["integrated_depth_mm"], 10.0)
        self.assertIn(
            "dimensionless sampled triangular shape",
            metadata["shape_normalisation_factor_meaning"],
        )
        validate_rainfall_event(event, PHASE2_COMPATIBLE_CONFIGURATION)

    def test_delayed_symmetric_triangle_preserves_shape_and_depth(self) -> None:
        pulse = TriangularPulse(600, 1200, 10.0, 0.5)
        event = generate_test_event(pulse)
        expected = np.zeros(PHASE2_COMPATIBLE_CONFIGURATION.number_of_records)
        expected[2:7] = [0.0, 30.0, 60.0, 30.0, 0.0]
        np.testing.assert_allclose(event.intensities_mm_per_hour, expected)
        self.assertAlmostEqual(
            integrated_depth_mm(event, PHASE2_COMPATIBLE_CONFIGURATION),
            10.0,
        )
        self.assertTrue(np.all(event.intensities_mm_per_hour[:3] == 0.0))
        self.assertTrue(np.all(event.intensities_mm_per_hour[6:] == 0.0))
        metadata = event.peak_metadata
        assert metadata is not None
        self.assertEqual(metadata["pulse_start_time_seconds"], 600)
        self.assertEqual(metadata["pulse_duration_seconds"], 1200)
        self.assertEqual(metadata["pulse_end_time_seconds"], 1800)
        self.assertEqual(metadata["requested_peak_time_seconds"], 1200.0)
        self.assertEqual(metadata["sampled_maximum_times_seconds"], (1200,))

    def test_grid_aligned_asymmetric_peak(self) -> None:
        pulse = TriangularPulse(300, 1800, 12.5, 1.0 / 3.0)
        event = generate_test_event(pulse)
        metadata = event.peak_metadata
        assert metadata is not None
        self.assertEqual(pulse.peak_time_seconds, 900.0)
        self.assertEqual(metadata["requested_peak_time_seconds"], 900.0)
        self.assertEqual(metadata["sampled_maximum_times_seconds"], (900,))
        self.assertGreater(event.intensities_mm_per_hour[2], 0.0)
        self.assertGreater(
            event.intensities_mm_per_hour[3],
            event.intensities_mm_per_hour[2],
        )
        self.assertGreater(
            event.intensities_mm_per_hour[3],
            event.intensities_mm_per_hour[4],
        )
        self.assertAlmostEqual(
            integrated_depth_mm(event, PHASE2_COMPATIBLE_CONFIGURATION),
            12.5,
        )

    def test_off_grid_peak_and_future_temporal_configuration(self) -> None:
        pulse = TriangularPulse(240, 600, 7.5, 0.25)
        event = generate_test_event(pulse, FUTURE_CONFIGURATION)
        timeline = generate_timeline(FUTURE_CONFIGURATION)
        np.testing.assert_array_equal(timeline, np.arange(0, 3601, 120))
        self.assertEqual(event.intensities_mm_per_hour.size, 31)
        self.assertEqual(event.active_duration_seconds, 840)
        self.assertTrue(
            np.all(event.intensities_mm_per_hour[timeline <= 240] == 0.0)
        )
        self.assertTrue(
            np.all(event.intensities_mm_per_hour[timeline >= 840] == 0.0)
        )
        self.assertAlmostEqual(
            integrated_depth_mm(event, FUTURE_CONFIGURATION),
            pulse.target_depth_mm,
        )

        metadata = event.peak_metadata
        assert metadata is not None
        self.assertEqual(metadata["requested_peak_time_seconds"], 390.0)
        self.assertEqual(metadata["sampled_maximum_times_seconds"], (360, 480))
        self.assertNotIn("peak_time_seconds", metadata)
        self.assertEqual(metadata["rainfall_timestep_seconds"], 120)
        validate_rainfall_event(event, FUTURE_CONFIGURATION)

    def test_intensity_is_not_capped_or_clipped(self) -> None:
        event = generate_test_event(TriangularPulse(0, 1200, 30.0, 0.5))
        self.assertGreater(float(np.max(event.intensities_mm_per_hour)), 120.0)
        self.assertAlmostEqual(
            integrated_depth_mm(event, PHASE2_COMPATIBLE_CONFIGURATION),
            30.0,
        )

    def test_invalid_pulse_fields_are_rejected(self) -> None:
        valid = TriangularPulse(0, 1200, 10.0, 0.5)
        invalid_values = {
            "start_time_seconds": (-1, True, 0.0),
            "duration_seconds": (0, -300, True, 1200.0),
            "target_depth_mm": (0.0, -1.0, math.nan, math.inf, True),
            "peak_position_ratio": (
                0.0,
                1.0,
                -0.1,
                1.1,
                math.nan,
                math.inf,
                True,
            ),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(valid, **{field: value})

    def test_start_and_duration_must_align_with_timestep(self) -> None:
        for pulse, field in (
            (TriangularPulse(100, 1200, 10.0, 0.5), "start_time_seconds"),
            (TriangularPulse(0, 1000, 10.0, 0.5), "duration_seconds"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field}.*align"):
                    generate_test_event(pulse)

    def test_endpoint_beyond_maximum_or_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum active endpoint"):
            generate_test_event(TriangularPulse(4800, 900, 10.0, 0.5))

        horizon_configuration = TemporalConfiguration(300, 7200, 7200)
        with self.assertRaisesRegex(ValueError, "simulation horizon"):
            generate_test_event(
                TriangularPulse(6900, 600, 10.0, 0.5),
                horizon_configuration,
            )

    def test_one_timestep_pulse_with_no_interior_sample_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero integral|positive interior"):
            generate_test_event(TriangularPulse(600, 300, 10.0, 0.5))

    def test_endpoint_exactly_at_maximum_is_accepted(self) -> None:
        pulse = TriangularPulse(4200, 1200, 10.0, 0.5)
        event = generate_test_event(pulse)
        self.assertEqual(event.active_duration_seconds, 5400)
        endpoint_index = 5400 // 300
        self.assertTrue(np.all(event.intensities_mm_per_hour[endpoint_index:] == 0.0))
        validate_rainfall_event(event, PHASE2_COMPATIBLE_CONFIGURATION)

    def test_temporary_citycat_export_and_validation(self) -> None:
        event = generate_test_event(
            TriangularPulse(600, 1800, 11.0, 0.45),
            rainfall_index=901,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(
                directory,
                event,
                PHASE2_COMPATIBLE_CONFIGURATION,
            )
            self.assertEqual(path.name, "Rainfall_Data_901.txt")
            validate_citycat_rainfall(
                path,
                event,
                PHASE2_COMPATIBLE_CONFIGURATION,
            )
            records = path.read_text(encoding="utf-8").splitlines()[5:]
            self.assertEqual(len(records), 25)
            self.assertEqual(
                [int(record.split()[0]) for record in records],
                list(range(0, 7201, 300)),
            )


if __name__ == "__main__":
    unittest.main()
