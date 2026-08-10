"""Synthetic TEST cases for external experiment configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from CityCAT.rainfall.design_rainfall import load_experiment_configuration


def test_configuration_payload() -> dict[str, object]:
    """Return a valid, deliberately non-production TEST configuration."""

    return {
        "experiment_name": "TEST_reusable_rainfall",
        "experiment_version": "TEST_v1",
        "phase": "TEST_phase",
        "rainfall_index_start": 11,
        "number_of_events": 4,
        "random_seed": 2468,
        "temporal": {
            "rainfall_timestep_seconds": 300,
            "simulation_horizon_seconds": 7200,
            "maximum_active_duration_seconds": 5400,
            "depth_tolerance_mm": 1e-6,
            "citycat_value_tolerance": 1e-15,
        },
        "requested_durations": {"values_seconds": [1800, 3600, 5400]},
        "requested_return_periods_years": [2, 10],
        "permitted_profile_families": [
            "TEST_triangle",
            "TEST_generic_tabulated",
        ],
        "evaluation_roles": {
            "strategy": "TEST_explicit_roles",
            "roles": ["TEST_calibration", "TEST_evaluation"],
            "provisional": True,
        },
        "source_ddf_table_path": "TEST_inputs/TEST_ddf.csv",
        "output_root": "TEST_outputs",
        "data_status": "provisional",
    }


class ExperimentConfigurationTests(unittest.TestCase):
    def load_payload(self, payload: dict[str, object]):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "TEST_experiment.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_experiment_configuration(path), path.parent

    def test_valid_configuration_and_temporal_reconstruction(self) -> None:
        configuration, base_directory = self.load_payload(
            test_configuration_payload()
        )
        self.assertEqual(configuration.experiment_name, "TEST_reusable_rainfall")
        self.assertEqual(configuration.rainfall_index_end, 14)
        self.assertEqual(configuration.temporal.number_of_records, 25)
        self.assertEqual(configuration.temporal.rainfall_timestep_seconds, 300)
        self.assertEqual(
            configuration.requested_durations.durations_seconds,
            (1800, 3600, 5400),
        )
        self.assertEqual(
            configuration.source_ddf_table_path,
            (base_directory / "TEST_inputs/TEST_ddf.csv").resolve(),
        )
        self.assertEqual(
            configuration.output_root, (base_directory / "TEST_outputs").resolve()
        )

    def test_inclusive_duration_range_is_loaded(self) -> None:
        payload = test_configuration_payload()
        payload["requested_durations"] = {
            "range_seconds": {"start": 1800, "stop": 5400, "step": 1800}
        }
        configuration, _ = self.load_payload(payload)
        self.assertEqual(
            configuration.requested_durations.durations_seconds,
            (1800, 3600, 5400),
        )

    def test_duplicate_durations_are_rejected(self) -> None:
        payload = test_configuration_payload()
        payload["requested_durations"] = {
            "values_seconds": [1800, 1800]
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load_payload(payload)

    def test_duration_misalignment_is_rejected(self) -> None:
        payload = test_configuration_payload()
        payload["requested_durations"] = {"values_seconds": [1900]}
        with self.assertRaisesRegex(ValueError, "align"):
            self.load_payload(payload)

    def test_duration_beyond_maximum_is_rejected(self) -> None:
        payload = test_configuration_payload()
        payload["requested_durations"] = {"values_seconds": [5700]}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.load_payload(payload)

    def test_invalid_rainfall_index_start_is_rejected(self) -> None:
        for invalid_value in (0, -1, True):
            with self.subTest(invalid_value=invalid_value):
                payload = test_configuration_payload()
                payload["rainfall_index_start"] = invalid_value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.load_payload(payload)

    def test_duplicate_or_invalid_return_periods_are_rejected(self) -> None:
        for values, message in (
            ([2, 2.0], "unique"),
            ([0], "finite and positive"),
            ([float("nan")], "finite and positive"),
        ):
            with self.subTest(values=values):
                payload = test_configuration_payload()
                payload["requested_return_periods_years"] = values
                with self.assertRaisesRegex(ValueError, message):
                    self.load_payload(payload)

    def test_phase2_compatible_and_future_longer_horizons(self) -> None:
        phase2, _ = self.load_payload(test_configuration_payload())
        self.assertEqual(phase2.temporal.simulation_horizon_seconds, 7200)
        self.assertEqual(phase2.temporal.number_of_records, 25)

        payload = test_configuration_payload()
        payload["temporal"] = {
            "rainfall_timestep_seconds": 600,
            "simulation_horizon_seconds": 18000,
            "maximum_active_duration_seconds": 14400,
        }
        payload["requested_durations"] = {
            "values_seconds": [7200, 14400]
        }
        future, _ = self.load_payload(payload)
        self.assertEqual(future.temporal.simulation_horizon_seconds, 18000)
        self.assertEqual(future.temporal.number_of_records, 31)

    def test_provisional_phase3_example_loads_without_final_choices(self) -> None:
        example_path = (
            Path(__file__).parents[1] / "examples" / "phase3_EXAMPLE.json"
        )
        configuration = load_experiment_configuration(example_path)
        self.assertEqual(configuration.rainfall_index_start, 101)
        self.assertEqual(configuration.number_of_events, 200)
        self.assertEqual(configuration.requested_return_periods_years, ())
        self.assertTrue(configuration.evaluation_roles.provisional)
        self.assertEqual(configuration.evaluation_roles.roles, ())


if __name__ == "__main__":
    unittest.main()
