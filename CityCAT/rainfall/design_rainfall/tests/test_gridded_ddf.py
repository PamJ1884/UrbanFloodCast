"""Synthetic TEST cases for explicit gridded-DDF point extraction."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    DDFTable,
    extract_ddf_at_point,
    load_gridded_ddf_catalogue,
    read_arcinfo_ascii_grid,
)
from CityCAT.rainfall.design_rainfall.gridded_ddf import (
    GRID_CATALOGUE_COLUMNS,
)


class GriddedDDFTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def write_grid(
        self,
        name: str,
        values: list[list[object]],
        *,
        x_origin: float = 100.0,
        y_origin: float = 200.0,
        cellsize: float = 10.0,
        convention: str = "center",
        nodata_value: float = -9999.0,
    ) -> Path:
        nrows = len(values)
        ncols = len(values[0])
        if convention == "center":
            x_header = "XLLCENTER"
            y_header = "YLLCENTER"
        elif convention == "corner":
            x_header = "XLLCORNER"
            y_header = "YLLCORNER"
        else:
            raise AssertionError("unsupported TEST convention")
        lines = [
            f"NCOLS {ncols}",
            f"NROWS {nrows}",
            f"{x_header} {x_origin}",
            f"{y_header} {y_origin}",
            f"CELLSIZE {cellsize}",
            f"NODATA_VALUE {nodata_value}",
        ]
        lines.extend(" ".join(str(value) for value in row) for row in values)
        path = self.root / "grids" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_catalogue(self, rows: list[dict[str, object]]) -> Path:
        path = self.root / "catalogues" / "TEST_catalogue.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        extra_fields = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in GRID_CATALOGUE_COLUMNS
            }
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[*GRID_CATALOGUE_COLUMNS, *extra_fields],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def catalogue_row(
        self,
        grid_path: Path | str,
        *,
        duration_minutes: object = 60,
        return_period_years: object = 50,
        value_scale_to_mm: object = 0.1,
        crs: object = "TEST:GRID",
        climate_scenario: object = "TEST_BASELINE",
        **extra: object,
    ) -> dict[str, object]:
        row = {
            "grid_path": grid_path,
            "duration_minutes": duration_minutes,
            "return_period_years": return_period_years,
            "value_scale_to_mm": value_scale_to_mm,
            "crs": crs,
            "source": "TEST_SYNTHETIC_GRID",
            "source_version": "TEST_v1",
            "extraction_date": "2099-01-01",
            "climate_scenario": climate_scenario,
            "notes": "Synthetic TEST data only",
        }
        row.update(extra)
        return row

    def relative_to_catalogue(self, grid_path: Path) -> str:
        return "../grids/" + grid_path.name

    def test_center_header_and_north_to_south_rows_are_preserved(self) -> None:
        path = self.write_grid("orientation.asc", [[1, 2], [3, 4]])
        grid = read_arcinfo_ascii_grid(path)
        self.assertEqual(grid.path, path.resolve())
        self.assertEqual((grid.ncols, grid.nrows), (2, 2))
        self.assertEqual(grid.origin_convention, "center")
        self.assertEqual((grid.xllcenter, grid.yllcenter), (100.0, 200.0))
        self.assertEqual(grid.coordinates_for_index(0, 0), (100.0, 210.0))
        self.assertEqual(grid.coordinates_for_index(1, 0), (100.0, 200.0))
        np.testing.assert_array_equal(grid.values, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(grid.header_metadata["XLLCENTER"], "100.0")

    def test_corner_origin_is_converted_to_cell_centres(self) -> None:
        path = self.write_grid(
            "corner.asc",
            [[1, 2], [3, 4]],
            x_origin=95.0,
            y_origin=195.0,
            convention="corner",
        )
        grid = read_arcinfo_ascii_grid(path)
        self.assertEqual(grid.origin_convention, "corner")
        self.assertEqual((grid.xllcenter, grid.yllcenter), (100.0, 200.0))
        self.assertEqual(grid.coordinates_for_index(0, 1), (110.0, 210.0))
        self.assertEqual(grid.header_metadata["XLLCORNER"], "95.0")

    def test_nodata_is_identified_without_changing_stored_values(self) -> None:
        path = self.write_grid("nodata.asc", [[1, -9999], [3, 4]])
        grid = read_arcinfo_ascii_grid(path)
        self.assertTrue(grid.nodata_mask[0, 1])
        self.assertFalse(grid.nodata_mask[1, 1])
        self.assertEqual(grid.values[0, 1], -9999.0)

    def test_malformed_header_dimensions_and_values_are_rejected(self) -> None:
        valid = self.write_grid("malformed.asc", [[1, 2], [3, 4]])
        original = valid.read_text(encoding="utf-8")
        mutations = (
            (original.replace("NCOLS 2", "NCOLS 0"), "positive integer"),
            (original.replace("NCOLS 2", "NCOLS 2.0"), "positive integer"),
            (original.replace("CELLSIZE 10.0", "CELLSIZE 0"), "positive"),
            (original.replace("XLLCENTER", "XLLMIDDLE"), "unexpected"),
            (original.replace("YLLCENTER", "YLLCORNER"), "CENTER or CORNER"),
            (original.replace("3 4", "3"), "exactly 2 values"),
            (original.replace("3 4", "3 nan"), "must be finite"),
        )
        for index, (text, message) in enumerate(mutations):
            with self.subTest(index=index):
                path = self.root / f"invalid_{index}.asc"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_arcinfo_ascii_grid(path)

        missing_row = self.root / "missing_row.asc"
        missing_row.write_text(
            "\n".join(original.splitlines()[:-1]) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "exactly 2 raster rows"):
            read_arcinfo_ascii_grid(missing_row)

    def test_catalogue_relative_paths_and_provenance_are_preserved(self) -> None:
        grid_path = self.write_grid("relative.asc", [[100, 200], [382, 400]])
        row = self.catalogue_row(
            self.relative_to_catalogue(grid_path),
            TEST_custom_provenance="TEST_preserved",
        )
        catalogue = load_gridded_ddf_catalogue(self.write_catalogue([row]))
        entry = catalogue.entries[0]
        self.assertEqual(entry.grid_path, grid_path.resolve())
        self.assertEqual(
            entry.additional_provenance["TEST_custom_provenance"],
            "TEST_preserved",
        )
        self.assertEqual(catalogue.crs, "TEST:GRID")

    def test_catalogue_rejects_duplicate_keys_and_bad_numeric_metadata(self) -> None:
        grid_path = self.write_grid("duplicate.asc", [[1, 2], [3, 4]])
        relative_path = self.relative_to_catalogue(grid_path)
        row = self.catalogue_row(relative_path)
        with self.assertRaisesRegex(ValueError, "duplicate duration"):
            load_gridded_ddf_catalogue(self.write_catalogue([row, dict(row)]))

        for field, value in (
            ("duration_minutes", 0),
            ("return_period_years", float("nan")),
            ("value_scale_to_mm", -1),
            ("crs", ""),
        ):
            with self.subTest(field=field):
                invalid = {**row, field: value}
                with self.assertRaisesRegex(ValueError, "positive|finite|non-empty"):
                    load_gridded_ddf_catalogue(
                        self.write_catalogue([invalid])
                    )

    def test_catalogue_rejects_inconsistent_geometry_crs_and_scenarios(self) -> None:
        first = self.write_grid("first.asc", [[1, 2], [3, 4]])
        second = self.write_grid(
            "second.asc", [[5, 6], [7, 8]], cellsize=20.0
        )
        rows = [
            self.catalogue_row(
                self.relative_to_catalogue(first), return_period_years=10
            ),
            self.catalogue_row(
                self.relative_to_catalogue(second), return_period_years=20
            ),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent grid geometry"):
            load_gridded_ddf_catalogue(self.write_catalogue(rows))

        same_geometry = self.write_grid("same.asc", [[5, 6], [7, 8]])
        rows[1] = self.catalogue_row(
            self.relative_to_catalogue(same_geometry),
            return_period_years=20,
            crs="TEST:OTHER",
        )
        with self.assertRaisesRegex(ValueError, "consistent CRS"):
            load_gridded_ddf_catalogue(self.write_catalogue(rows))

        rows[1] = self.catalogue_row(
            self.relative_to_catalogue(same_geometry),
            return_period_years=20,
            climate_scenario="TEST_FUTURE",
        )
        with self.assertRaisesRegex(ValueError, "one non-empty climate_scenario"):
            load_gridded_ddf_catalogue(self.write_catalogue(rows))

    def test_exact_extraction_orientation_scale_and_ddf_conversion(self) -> None:
        grid_path = self.write_grid("depth.asc", [[100, 200], [382, 400]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [
                    self.catalogue_row(
                        self.relative_to_catalogue(grid_path),
                        TEST_custom_provenance="TEST_preserved",
                    )
                ]
            )
        )
        result = extract_ddf_at_point(
            catalogue,
            x=100.0,
            y=200.0,
            crs="TEST:GRID",
            location_id="TEST_SITE",
        )
        self.assertEqual(result.extraction_method, "nearest_grid_point")
        self.assertEqual(
            (result.selected_grid_x, result.selected_grid_y),
            (100.0, 200.0),
        )
        self.assertEqual(result.distance_to_grid_point, 0.0)
        extraction = result.extractions[0]
        self.assertEqual(
            (extraction.selected_row, extraction.selected_column),
            (1, 0),
        )
        self.assertEqual(extraction.raw_grid_value, 382.0)
        self.assertAlmostEqual(extraction.depth_mm, 38.2)
        self.assertEqual(extraction.value_scale_to_mm, 0.1)
        self.assertEqual(extraction.source_grid_path, grid_path.resolve())
        self.assertEqual(extraction.grid_header_metadata["NROWS"], "2")
        self.assertEqual(
            extraction.provenance["source_grid_path"],
            str(grid_path.resolve()),
        )
        self.assertEqual(
            extraction.provenance["grid_catalogue_path"],
            str(catalogue.path),
        )
        self.assertEqual(extraction.provenance["arc_ascii_nrows"], "2")
        self.assertEqual(extraction.provenance["raw_grid_value"], "382")
        self.assertEqual(extraction.provenance["value_scale_to_mm"], "0.1")
        self.assertEqual(
            extraction.provenance["TEST_custom_provenance"],
            "TEST_preserved",
        )
        self.assertNotIn("uplift_factor", extraction.provenance)
        self.assertIsInstance(result.ddf_table, DDFTable)
        estimate = result.ddf_table.lookup_exact("TEST_SITE", 60, 50)
        self.assertAlmostEqual(estimate.depth_mm, 38.2)

    def test_exact_boundaries_and_immediate_interior_are_accepted(self) -> None:
        grid_path = self.write_grid("boundaries.asc", [[1, 2], [3, 4]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [self.catalogue_row(self.relative_to_catalogue(grid_path))]
            )
        )
        accepted_points = (
            ("minimum_x", 100.0, 207.0, (100.0, 210.0)),
            ("maximum_x", 110.0, 207.0, (110.0, 210.0)),
            ("minimum_y", 107.0, 200.0, (110.0, 200.0)),
            ("maximum_y", 107.0, 210.0, (110.0, 210.0)),
            (
                "immediately_inside_west",
                math.nextafter(100.0, math.inf),
                207.0,
                (100.0, 210.0),
            ),
            (
                "immediately_inside_east",
                math.nextafter(110.0, -math.inf),
                207.0,
                (110.0, 210.0),
            ),
            (
                "immediately_inside_south",
                107.0,
                math.nextafter(200.0, math.inf),
                (110.0, 200.0),
            ),
            (
                "immediately_inside_north",
                107.0,
                math.nextafter(210.0, -math.inf),
                (110.0, 210.0),
            ),
        )
        for label, x, y, expected_grid_point in accepted_points:
            with self.subTest(boundary=label):
                result = extract_ddf_at_point(
                    catalogue,
                    x=x,
                    y=y,
                    crs="TEST:GRID",
                    location_id=f"TEST_{label.upper()}",
                )
                self.assertEqual((result.requested_x, result.requested_y), (x, y))
                self.assertEqual(
                    (result.selected_grid_x, result.selected_grid_y),
                    expected_grid_point,
                )

    def test_nextafter_outside_each_boundary_is_rejected(self) -> None:
        grid_path = self.write_grid("outside_boundaries.asc", [[1, 2], [3, 4]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [self.catalogue_row(self.relative_to_catalogue(grid_path))]
            )
        )
        outside_points = (
            ("west", math.nextafter(100.0, -math.inf), 207.0),
            ("east", math.nextafter(110.0, math.inf), 207.0),
            ("south", 107.0, math.nextafter(200.0, -math.inf)),
            ("north", 107.0, math.nextafter(210.0, math.inf)),
        )
        for side, x, y in outside_points:
            with self.subTest(side=side):
                with self.assertRaisesRegex(
                    ValueError, "outside the grid-point domain"
                ):
                    extract_ddf_at_point(
                        catalogue,
                        x=x,
                        y=y,
                        crs="TEST:GRID",
                        location_id=f"TEST_OUTSIDE_{side.upper()}",
                    )

    def test_nearest_selection_is_deterministic(self) -> None:
        grid_path = self.write_grid("nearest.asc", [[100, 200], [382, 400]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [self.catalogue_row(self.relative_to_catalogue(grid_path))]
            )
        )
        nearest = extract_ddf_at_point(
            catalogue,
            x=109.0,
            y=209.0,
            crs="TEST:GRID",
            location_id="TEST_NEAREST",
        )
        self.assertEqual(
            (nearest.selected_grid_x, nearest.selected_grid_y),
            (110.0, 210.0),
        )
        self.assertAlmostEqual(nearest.distance_to_grid_point, math.sqrt(2.0))
        self.assertEqual(nearest.extractions[0].raw_grid_value, 200.0)

    def test_midpoint_ties_select_west_south_and_southwest(self) -> None:
        grid_path = self.write_grid("ties.asc", [[1, 2], [3, 4]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [self.catalogue_row(self.relative_to_catalogue(grid_path))]
            )
        )
        tie_points = (
            ("x_only_selects_west", 105.0, 209.0, (100.0, 210.0), 1.0),
            ("y_only_selects_south", 109.0, 205.0, (110.0, 200.0), 4.0),
            ("xy_selects_southwest", 105.0, 205.0, (100.0, 200.0), 3.0),
        )
        for label, x, y, expected_grid_point, expected_raw_value in tie_points:
            with self.subTest(tie=label):
                result = extract_ddf_at_point(
                    catalogue,
                    x=x,
                    y=y,
                    crs="TEST:GRID",
                    location_id=f"TEST_{label.upper()}",
                )
                self.assertEqual(
                    (result.selected_grid_x, result.selected_grid_y),
                    expected_grid_point,
                )
                self.assertEqual(
                    result.extractions[0].raw_grid_value,
                    expected_raw_value,
                )

    def test_outside_crs_and_nodata_extractions_are_rejected(self) -> None:
        grid_path = self.write_grid("reject.asc", [[1, -9999], [3, 4]])
        catalogue = load_gridded_ddf_catalogue(
            self.write_catalogue(
                [self.catalogue_row(self.relative_to_catalogue(grid_path))]
            )
        )
        with self.assertRaisesRegex(ValueError, "outside the grid-point domain"):
            extract_ddf_at_point(
                catalogue,
                x=99.0,
                y=200.0,
                crs="TEST:GRID",
                location_id="TEST_OUTSIDE",
            )
        with self.assertRaisesRegex(ValueError, "does not match catalogue CRS"):
            extract_ddf_at_point(
                catalogue,
                x=100.0,
                y=200.0,
                crs="TEST:OTHER",
                location_id="TEST_CRS",
            )
        with self.assertRaisesRegex(ValueError, "NODATA"):
            extract_ddf_at_point(
                catalogue,
                x=110.0,
                y=210.0,
                crs="TEST:GRID",
                location_id="TEST_NODATA",
            )


if __name__ == "__main__":
    unittest.main()
