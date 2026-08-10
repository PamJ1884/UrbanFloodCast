"""Unit tests using synthetic TEST rainfall events only."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    RainfallEvent,
    TemporalConfiguration,
    export_citycat_rainfall,
    generate_timeline,
    validate_citycat_rainfall,
    validate_rainfall_event,
)


TEST_CONFIGURATION = TemporalConfiguration(300, 7200, 7200, 1e-9, 1e-15)


def synthetic_test_event(
    configuration: TemporalConfiguration = TEST_CONFIGURATION,
    *,
    duration_seconds: int = 1800,
    target_depth_mm: float = 10.0,
) -> RainfallEvent:
    """Build a labelled asymmetric triangular TEST event."""

    timeline = generate_timeline(configuration)
    peak = duration_seconds // 2
    shape = np.zeros(configuration.number_of_records)
    rising = (timeline > 0) & (timeline <= peak)
    falling = (timeline > peak) & (timeline < duration_seconds)
    shape[rising] = timeline[rising] / peak
    shape[falling] = (duration_seconds - timeline[falling]) / (duration_seconds - peak)
    area = float(np.trapz(shape, timeline / 3600.0))
    if area > 0.0:
        shape *= target_depth_mm / area
    return RainfallEvent(
        999,
        "TEST-PHASE-EVENT-999",
        "TEST-STORM-999",
        "test_fixture",
        "TEST_asymmetric_triangular",
        target_depth_mm,
        duration_seconds,
        shape,
        {"peak_time_seconds": peak},
    )


class TemporalConfigurationTests(unittest.TestCase):
    def test_phase2_compatible_grid_has_25_records(self) -> None:
        self.assertEqual(TEST_CONFIGURATION.number_of_records, 25)
        np.testing.assert_array_equal(
            generate_timeline(TEST_CONFIGURATION), np.arange(0, 7201, 300)
        )

    def test_future_configuration_needs_no_source_change(self) -> None:
        future = TemporalConfiguration(600, 18000, 14400)
        self.assertEqual(future.number_of_records, 31)
        self.assertEqual(generate_timeline(future)[-1], 18000)

    def test_invalid_duration_alignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            validate_rainfall_event(
                replace(synthetic_test_event(), active_duration_seconds=1700),
                TEST_CONFIGURATION,
            )

    def test_duration_beyond_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds|horizon"):
            validate_rainfall_event(
                replace(synthetic_test_event(), active_duration_seconds=7500),
                TEST_CONFIGURATION,
            )


class RainfallEventTests(unittest.TestCase):
    def test_rainfall_index_must_be_a_positive_non_boolean_integer(self) -> None:
        for invalid_index in (0, -1, True, False, 1.0, "1"):
            with self.subTest(invalid_index=invalid_index):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_rainfall_event(
                        replace(synthetic_test_event(), rainfall_index=invalid_index),
                        TEST_CONFIGURATION,
                    )

    def test_identity_strings_must_be_non_empty(self) -> None:
        for field_name in ("phase_event_id", "storm_id", "role", "profile_family"):
            for invalid_value in ("", "   ", None, 101):
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    with self.assertRaisesRegex(ValueError, "non-empty string"):
                        validate_rainfall_event(
                            replace(
                                synthetic_test_event(),
                                **{field_name: invalid_value},
                            ),
                            TEST_CONFIGURATION,
                        )

    def test_valid_triangle_integrates_to_target(self) -> None:
        event = synthetic_test_event()
        validate_rainfall_event(event, TEST_CONFIGURATION)
        self.assertAlmostEqual(
            float(
                np.trapz(
                    event.intensities_mm_per_hour,
                    generate_timeline(TEST_CONFIGURATION) / 3600.0,
                )
            ),
            event.target_depth_mm,
        )

    def test_negative_and_non_finite_rainfall_are_rejected(self) -> None:
        for value in (-1.0, np.nan, np.inf):
            with self.subTest(value=value):
                event = synthetic_test_event()
                values = event.intensities_mm_per_hour.copy()
                values[2] = value
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    validate_rainfall_event(
                        replace(event, intensities_mm_per_hour=values),
                        TEST_CONFIGURATION,
                    )

    def test_truncated_rainfall_at_horizon_is_rejected(self) -> None:
        configuration = TemporalConfiguration(300, 1800, 1800)
        values = np.zeros(configuration.number_of_records)
        values[-1] = 1.0
        event = RainfallEvent(
            998,
            "TEST-PHASE-EVENT-998",
            "TEST-STORM-998",
            "test_fixture",
            "TEST_truncated",
            0.0,
            1800,
            values,
        )
        with self.assertRaisesRegex(ValueError, "at and after"):
            validate_rainfall_event(event, configuration)

    def test_citycat_export_structure_units_and_validation(self) -> None:
        event = synthetic_test_event()
        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(directory, event, TEST_CONFIGURATION)
            self.assertEqual(path.name, "Rainfall_Data_999.txt")
            lines = path.read_text().splitlines()
            self.assertEqual(
                lines[:5],
                [
                    "* * * TEST-STORM-999",
                    "* * * rainfall ***",
                    "* * *",
                    "25",
                    "* * *",
                ],
            )
            timestamp, value = lines[6].split()
            self.assertEqual(int(timestamp), 300)
            self.assertAlmostEqual(
                float(value), event.intensities_mm_per_hour[1] / 1000.0 / 3600.0
            )
            validate_citycat_rainfall(path, event, TEST_CONFIGURATION)


class CitycatValidatorTests(unittest.TestCase):
    def assert_corrupt_file_rejected(
        self, mutate_lines, expected_message: str
    ) -> None:
        event = synthetic_test_event()
        with tempfile.TemporaryDirectory() as directory:
            path = export_citycat_rainfall(directory, event, TEST_CONFIGURATION)
            lines = path.read_text(encoding="utf-8").splitlines()
            mutate_lines(lines)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, expected_message):
                validate_citycat_rainfall(path, event, TEST_CONFIGURATION)

    def test_incorrect_timestamp_is_rejected(self) -> None:
        def change_timestamp(lines: list[str]) -> None:
            _, value = lines[6].split()
            lines[6] = f"301 {value}"

        self.assert_corrupt_file_rejected(change_timestamp, "timestamps")

    def test_value_outside_tolerance_is_rejected(self) -> None:
        def change_value(lines: list[str]) -> None:
            timestamp, _ = lines[6].split()
            lines[6] = f"{timestamp} 0.001"

        self.assert_corrupt_file_rejected(change_value, "conversion")

    def test_malformed_header_is_rejected(self) -> None:
        self.assert_corrupt_file_rejected(
            lambda lines: lines.__setitem__(1, "* * * rain ***"), "header"
        )

    def test_missing_record_is_rejected(self) -> None:
        self.assert_corrupt_file_rejected(lambda lines: lines.pop(), "exactly 25")

    def test_negative_and_non_finite_values_are_rejected(self) -> None:
        for invalid_value in ("-1", "nan", "inf", "-inf"):
            with self.subTest(invalid_value=invalid_value):
                self.assert_corrupt_file_rejected(
                    lambda lines, value=invalid_value: lines.__setitem__(
                        6, f"300 {value}"
                    ),
                    "finite and non-negative",
                )


if __name__ == "__main__":
    unittest.main()
