"""Optional regression against the external UKCEH DDF sample grids."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from CityCAT.rainfall.design_rainfall import (
    extract_ddf_at_point,
    load_gridded_ddf_catalogue,
)
from CityCAT.rainfall.design_rainfall.gridded_ddf import (
    GRID_CATALOGUE_COLUMNS,
)


SAMPLE_ROOT_ENVIRONMENT_VARIABLE = "UKCEH_DDF_SAMPLE_ROOT"
TEST_SAMPLE_CRS = "TEST_UKCEH_SAMPLE_NATIVE_GRID_CRS_UNVERIFIED"
TEST_LOCATION_ID = "TEST_UKCEH_SAMPLE_GRID_POINT"
SAMPLE_GRID_SPECIFICATIONS = (
    ("KAD", 60, 5, "2015-01-13"),
    ("KAE", 60, 10, "2015-01-13"),
    ("KAI", 60, 30, "2015-01-13"),
    ("KAK", 60, 50, "2015-01-13"),
    ("KAM", 60, 100, "2015-01-13"),
    ("KAO", 60, 200, "2015-01-13"),
    ("KCD", 180, 5, "2015-01-14"),
    ("KCE", 180, 10, "2015-01-14"),
    ("KCI", 180, 30, "2015-01-14"),
    ("KCK", 180, 50, "2015-01-14"),
    ("KCM", 180, 100, "2015-01-14"),
    ("KCO", 180, 200, "2015-01-14"),
)


class ExternalUKCEHSampleRegressionTests(unittest.TestCase):
    def configured_sample_root(self) -> Path:
        raw_path = os.environ.get(SAMPLE_ROOT_ENVIRONMENT_VARIABLE)
        if not raw_path:
            self.skipTest(
                f"external UKCEH sample unavailable: set "
                f"{SAMPLE_ROOT_ENVIRONMENT_VARIABLE}"
            )
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            self.skipTest(f"external UKCEH sample directory unavailable: {root}")
        return root

    def explicit_sample_rows(self, sample_root: Path) -> list[dict[str, object]]:
        all_grids = sorted(sample_root.rglob("*.asc"))
        self.assertEqual(
            len(all_grids),
            12,
            f"configured sample root must contain exactly 12 grids: {sample_root}",
        )
        rows = []
        selected_paths = set()
        for code, duration, return_period, extraction_date in (
            SAMPLE_GRID_SPECIFICATIONS
        ):
            matches = [path for path in all_grids if f"__{code}_" in path.name]
            self.assertEqual(
                len(matches),
                1,
                f"explicit TEST catalogue expected one {code} grid",
            )
            grid_path = matches[0]
            selected_paths.add(grid_path)
            rows.append(
                {
                    "grid_path": str(grid_path),
                    "duration_minutes": duration,
                    "return_period_years": return_period,
                    "value_scale_to_mm": 0.1,
                    "crs": TEST_SAMPLE_CRS,
                    "source": "UKCEH FEH DDF sample",
                    "source_version": "TEST_external_sample_v01",
                    "extraction_date": extraction_date,
                    "climate_scenario": "",
                    "notes": "External sample for software validation only",
                }
            )
        self.assertEqual(selected_paths, set(all_grids))
        return rows

    def test_all_sample_grids_and_point_extraction(self) -> None:
        sample_root = self.configured_sample_root()
        rows = self.explicit_sample_rows(sample_root)
        with tempfile.TemporaryDirectory() as directory:
            catalogue_path = Path(directory) / "TEST_sample_catalogue.csv"
            with catalogue_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=GRID_CATALOGUE_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            catalogue = load_gridded_ddf_catalogue(catalogue_path)

        self.assertEqual(len(catalogue.entries), 12)
        self.assertEqual(
            {
                (entry.duration_minutes, entry.return_period_years)
                for entry in catalogue.entries
            },
            {
                (float(duration), float(return_period))
                for _, duration, return_period, _ in SAMPLE_GRID_SPECIFICATIONS
            },
        )
        for entry in catalogue.entries:
            self.assertEqual((entry.grid.nrows, entry.grid.ncols), (12, 12))
            self.assertEqual(entry.grid.cellsize, 1000.0)
            self.assertEqual(entry.value_scale_to_mm, 0.1)

        result = extract_ddf_at_point(
            catalogue,
            x=391000.0,
            y=437000.0,
            crs=TEST_SAMPLE_CRS,
            location_id=TEST_LOCATION_ID,
        )
        self.assertNotIn("stockbridge", result.location_id.lower())
        self.assertEqual(len(result.extractions), 12)
        self.assertEqual(
            (result.selected_grid_x, result.selected_grid_y),
            (391000.0, 437000.0),
        )
        one_hour_fifty_year = next(
            extraction
            for extraction in result.extractions
            if extraction.duration_minutes == 60.0
            and extraction.return_period_years == 50.0
        )
        first_raw_file_value = float(
            one_hour_fifty_year.source_grid_path.read_text(
                encoding="utf-8"
            ).splitlines()[6].split()[0]
        )
        self.assertEqual(first_raw_file_value, 382.0)
        self.assertEqual(one_hour_fifty_year.raw_grid_value, first_raw_file_value)
        self.assertAlmostEqual(one_hour_fifty_year.depth_mm, 38.2)

        for duration in (60.0, 180.0):
            duration_extractions = sorted(
                (
                    extraction
                    for extraction in result.extractions
                    if extraction.duration_minutes == duration
                ),
                key=lambda extraction: extraction.return_period_years,
            )
            depths = [extraction.depth_mm for extraction in duration_extractions]
            self.assertTrue(
                all(lower < upper for lower, upper in zip(depths, depths[1:]))
            )


if __name__ == "__main__":
    unittest.main()
