"""Synthetic TEST cases for normalised DDF ingestion and interpolation."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from CityCAT.rainfall.design_rainfall import apply_climate_uplift, load_ddf_csv


TEST_COLUMNS = (
    "location_id",
    "duration_minutes",
    "return_period_years",
    "depth_mm",
    "source",
    "source_version",
    "extraction_date",
    "climate_scenario",
    "uplift_factor",
    "notes",
    "TEST_custom_provenance",
)


def test_ddf_rows() -> list[dict[str, object]]:
    """Return a small synthetic long-format TEST DDF surface."""

    common = {
        "location_id": "TEST_SITE_A",
        "source": "TEST_SYNTHETIC_SOURCE",
        "source_version": "TEST_v1",
        "extraction_date": "2099-01-01",
        "climate_scenario": "TEST_BASELINE",
        "uplift_factor": "",
        "notes": "TEST values are not real rainfall data",
        "TEST_custom_provenance": "TEST_preserved",
    }
    return [
        {**common, "duration_minutes": 15, "return_period_years": 2, "depth_mm": 10},
        {**common, "duration_minutes": 60, "return_period_years": 2, "depth_mm": 40},
        {**common, "duration_minutes": 15, "return_period_years": 10, "depth_mm": 30},
        {**common, "duration_minutes": 60, "return_period_years": 10, "depth_mm": 90},
    ]


class DDFTestCase(unittest.TestCase):
    def load_rows(
        self,
        rows: list[dict[str, object]],
        *,
        columns=TEST_COLUMNS,
        location_id: str | None = None,
    ):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "TEST_ddf.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return load_ddf_csv(path, location_id=location_id)

    def test_missing_required_columns_are_rejected(self) -> None:
        columns = tuple(name for name in TEST_COLUMNS if name != "depth_mm")
        rows = [
            {name: value for name, value in test_ddf_rows()[0].items() if name in columns}
        ]
        with self.assertRaisesRegex(ValueError, "missing required columns.*depth_mm"):
            self.load_rows(rows, columns=columns)

    def test_empty_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no data records"):
            self.load_rows([])

    def test_duplicate_keys_are_rejected(self) -> None:
        rows = test_ddf_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate DDF key"):
            self.load_rows(rows)

    def test_non_finite_and_non_positive_numeric_values_are_rejected(self) -> None:
        for field in ("duration_minutes", "return_period_years", "depth_mm"):
            for invalid_value in ("nan", "inf", "-inf", 0, -1):
                with self.subTest(field=field, invalid_value=invalid_value):
                    rows = test_ddf_rows()
                    rows[0] = {**rows[0], field: invalid_value}
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        self.load_rows(rows)

    def test_duration_must_convert_to_whole_seconds(self) -> None:
        rows = test_ddf_rows()
        rows[0] = {**rows[0], "duration_minutes": "15.0001"}
        with self.assertRaisesRegex(ValueError, "whole number of seconds"):
            self.load_rows(rows)

    def test_exact_lookup_is_traceable(self) -> None:
        table = self.load_rows(test_ddf_rows())
        estimate = table.lookup_exact("TEST_SITE_A", 15, 2)
        self.assertEqual(estimate.estimate_kind, "exact")
        self.assertEqual(estimate.interpolation_method, "exact")
        self.assertEqual(estimate.depth_mm, 10.0)
        self.assertEqual(estimate.location_id, "TEST_SITE_A")
        self.assertEqual(len(estimate.source_points), 1)

    def test_log_log_duration_interpolation_is_traceable(self) -> None:
        table = self.load_rows(test_ddf_rows())
        estimate = table.interpolate_duration(
            "TEST_SITE_A", 30, 2, method="log_log"
        )
        self.assertEqual(estimate.estimate_kind, "interpolated")
        self.assertEqual(estimate.interpolation_method, "log_log")
        self.assertAlmostEqual(estimate.depth_mm, 20.0)
        self.assertEqual(
            [point.duration_minutes for point in estimate.source_points],
            [15.0, 60.0],
        )

    def test_gumbel_return_period_interpolation_is_traceable(self) -> None:
        table = self.load_rows(test_ddf_rows())
        y_lower = -math.log(-math.log(1.0 - 1.0 / 2.0))
        y_upper = -math.log(-math.log(1.0 - 1.0 / 10.0))
        y_target = 0.5 * (y_lower + y_upper)
        target_return_period = 1.0 / (1.0 - math.exp(-math.exp(-y_target)))
        estimate = table.interpolate_return_period(
            "TEST_SITE_A",
            60,
            target_return_period,
            method="gumbel_log_depth",
        )
        self.assertEqual(estimate.interpolation_method, "gumbel_log_depth")
        self.assertAlmostEqual(estimate.depth_mm, 60.0)
        self.assertEqual(
            [point.return_period_years for point in estimate.source_points],
            [2.0, 10.0],
        )

    def test_extrapolation_is_rejected(self) -> None:
        table = self.load_rows(test_ddf_rows())
        with self.assertRaisesRegex(ValueError, "extrapolation is disabled"):
            table.interpolate_duration("TEST_SITE_A", 10, 2, method="log_log")
        with self.assertRaisesRegex(ValueError, "extrapolation is disabled"):
            table.interpolate_return_period(
                "TEST_SITE_A", 60, 100, method="gumbel_log_depth"
            )

    def test_multiple_locations_require_explicit_selection(self) -> None:
        rows = test_ddf_rows()
        rows.append(
            {
                **rows[0],
                "location_id": "TEST_SITE_B",
                "depth_mm": 12,
            }
        )
        with self.assertRaisesRegex(ValueError, "select location_id explicitly"):
            self.load_rows(rows)
        table = self.load_rows(rows, location_id="TEST_SITE_B")
        self.assertEqual(table.location_id, "TEST_SITE_B")
        self.assertEqual(len(table.records), 1)

    def test_different_non_empty_climate_scenarios_are_rejected(self) -> None:
        rows = test_ddf_rows()
        rows[1] = {**rows[1], "climate_scenario": "TEST_FUTURE_SCENARIO"}
        with self.assertRaisesRegex(
            ValueError,
            "interpolation across climate scenarios is not permitted.*coherent",
        ):
            self.load_rows(rows)

    def test_one_non_empty_climate_scenario_is_accepted(self) -> None:
        table = self.load_rows(test_ddf_rows())
        self.assertEqual(table.location_id, "TEST_SITE_A")
        self.assertEqual(
            {
                record.provenance["climate_scenario"]
                for record in table.records
            },
            {"TEST_BASELINE"},
        )

    def test_absent_or_empty_climate_scenarios_do_not_conflict(self) -> None:
        rows = test_ddf_rows()
        rows[0] = {**rows[0], "climate_scenario": ""}
        rows[1] = {**rows[1], "climate_scenario": "   "}
        rows[2].pop("climate_scenario")
        table = self.load_rows(rows)
        self.assertEqual(len(table.records), 4)

        columns_without_scenario = tuple(
            name for name in TEST_COLUMNS if name != "climate_scenario"
        )
        rows_without_scenario = [
            {name: value for name, value in row.items() if name in columns_without_scenario}
            for row in test_ddf_rows()
        ]
        table = self.load_rows(
            rows_without_scenario, columns=columns_without_scenario
        )
        self.assertEqual(len(table.records), 4)

    def test_unselected_location_scenario_does_not_create_conflict(self) -> None:
        rows = test_ddf_rows()
        rows.append(
            {
                **rows[0],
                "location_id": "TEST_SITE_B",
                "climate_scenario": "TEST_FUTURE_SCENARIO",
            }
        )
        table = self.load_rows(rows, location_id="TEST_SITE_A")
        self.assertEqual(table.location_id, "TEST_SITE_A")
        self.assertEqual(len(table.records), 4)

    def test_interpolation_cannot_mix_baseline_and_adjusted_points(self) -> None:
        rows = [test_ddf_rows()[0], test_ddf_rows()[1]]
        rows[1] = {**rows[1], "climate_scenario": "TEST_CLIMATE_ADJUSTED"}
        with self.assertRaisesRegex(
            ValueError, "interpolation across climate scenarios is not permitted"
        ):
            self.load_rows(rows)

    def test_provenance_fields_are_preserved(self) -> None:
        rows = test_ddf_rows()
        rows[0] = {**rows[0], "uplift_factor": "1.20"}
        table = self.load_rows(rows)
        provenance = table.lookup_exact(
            "TEST_SITE_A", 15, 2
        ).source_points[0].provenance
        self.assertEqual(provenance["source"], "TEST_SYNTHETIC_SOURCE")
        self.assertEqual(provenance["source_version"], "TEST_v1")
        self.assertEqual(provenance["uplift_factor"], "1.20")
        self.assertEqual(provenance["TEST_custom_provenance"], "TEST_preserved")

    def test_invalid_provenance_uplift_factor_is_rejected(self) -> None:
        for invalid_value in ("nan", "inf", "0", "-1"):
            with self.subTest(invalid_value=invalid_value):
                rows = test_ddf_rows()
                rows[0] = {**rows[0], "uplift_factor": invalid_value}
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.load_rows(rows)

    def test_gumbel_rejects_requested_return_period_at_or_below_one(self) -> None:
        table = self.load_rows(test_ddf_rows())
        for requested_return_period in (0.5, 1.0):
            with self.subTest(requested_return_period=requested_return_period):
                with self.assertRaisesRegex(ValueError, "greater than one year"):
                    table.interpolate_return_period(
                        "TEST_SITE_A",
                        60,
                        requested_return_period,
                        method="gumbel_log_depth",
                    )

    def test_gumbel_rejects_source_return_period_at_or_below_one(self) -> None:
        for source_return_period in (0.5, 1.0):
            with self.subTest(source_return_period=source_return_period):
                rows = [dict(test_ddf_rows()[1]), dict(test_ddf_rows()[3])]
                rows[0]["return_period_years"] = source_return_period
                with self.assertRaisesRegex(ValueError, "greater than one year"):
                    self.load_rows(rows).interpolate_return_period(
                        "TEST_SITE_A", 60, 2, method="gumbel_log_depth"
                    )

    def test_linear_return_period_allows_values_at_or_below_one(self) -> None:
        rows = [dict(test_ddf_rows()[1]), dict(test_ddf_rows()[3])]
        rows[0].update(return_period_years=0.5, depth_mm=10)
        rows[1].update(return_period_years=2, depth_mm=40)
        estimate = self.load_rows(rows).interpolate_return_period(
            "TEST_SITE_A", 60, 1, method="linear"
        )
        self.assertEqual(estimate.interpolation_method, "linear")
        self.assertAlmostEqual(estimate.depth_mm, 20.0)

    def test_exact_return_period_does_not_invoke_gumbel_transform(self) -> None:
        rows = [dict(test_ddf_rows()[1]), dict(test_ddf_rows()[3])]
        rows[0]["return_period_years"] = 1
        table = self.load_rows(rows)
        direct = table.lookup_exact("TEST_SITE_A", 60, 1)
        through_interpolator = table.interpolate_return_period(
            "TEST_SITE_A", 60, 1, method="gumbel_log_depth"
        )
        self.assertEqual(direct.estimate_kind, "exact")
        self.assertEqual(through_interpolator.estimate_kind, "exact")
        self.assertEqual(through_interpolator.interpolation_method, "exact")
        self.assertEqual(through_interpolator.depth_mm, direct.depth_mm)

    def test_climate_uplift_is_correct_and_traceable(self) -> None:
        result = apply_climate_uplift(25.0, 1.2)
        self.assertEqual(result.base_depth_mm, 25.0)
        self.assertEqual(result.uplift_factor, 1.2)
        self.assertEqual(result.uplifted_depth_mm, 30.0)
        for invalid_factor in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(invalid_factor=invalid_factor):
                with self.assertRaisesRegex(
                    ValueError, "finite and positive|must not be bool"
                ):
                    apply_climate_uplift(25.0, invalid_factor)


if __name__ == "__main__":
    unittest.main()
