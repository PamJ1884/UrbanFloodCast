"""Regression against the external, real Phase 2 rainfall dataset."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    RainfallEvent,
    TemporalConfiguration,
    export_citycat_rainfall,
    generate_timeline,
)


PHASE2_ROOT = (
    Path.home()
    / "flood/02_data/CityCAT_Winchester/rainfall/stockbridge_phase2_core_v01"
)
PHASE2_MASTER_CSV = PHASE2_ROOT / "csv_timeseries" / "Rainfall_Data_1.csv"
PHASE2_CITYCAT_TXT = PHASE2_ROOT / "citycat_txt" / "Rainfall_Data_1.txt"


class Phase2RainfallRegressionTests(unittest.TestCase):
    def test_csv_reconstruction_exports_byte_identical_citycat_file(self) -> None:
        missing = [
            str(path)
            for path in (PHASE2_MASTER_CSV, PHASE2_CITYCAT_TXT)
            if not path.is_file()
        ]
        if missing:
            self.skipTest(
                "external Phase 2 regression dataset unavailable: " + ", ".join(missing)
            )

        with PHASE2_MASTER_CSV.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        configuration = TemporalConfiguration(
            rainfall_timestep_seconds=300,
            simulation_horizon_seconds=7200,
            maximum_active_duration_seconds=5400,
            depth_tolerance_mm=1e-6,
            citycat_value_tolerance=1e-15,
        )
        timestamps = np.asarray(
            [int(row["time_seconds"]) for row in rows], dtype=np.int64
        )
        intensities = np.asarray(
            [float(row["intensity_mm_per_hour"]) for row in rows],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(timestamps, generate_timeline(configuration))

        positive_indices = np.flatnonzero(intensities > 0.0)
        self.assertGreater(positive_indices.size, 0)
        endpoint_index = int(positive_indices[-1]) + 1
        self.assertLess(endpoint_index, intensities.size)
        self.assertTrue(np.all(intensities[endpoint_index:] == 0.0))
        active_duration_seconds = int(timestamps[endpoint_index])

        event = RainfallEvent(
            rainfall_index=1,
            phase_event_id="stockbridge_phase2_core_001",
            storm_id="stockbridge_phase2_core_001",
            role="regression_reference",
            profile_family="asymmetric_triangular_pulse",
            target_depth_mm=float(rows[-1]["cumulative_depth_mm"]),
            active_duration_seconds=active_duration_seconds,
            intensities_mm_per_hour=intensities,
            peak_metadata={
                "peak_time_seconds": int(timestamps[int(np.argmax(intensities))])
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            exported_path = export_citycat_rainfall(directory, event, configuration)
            self.assertEqual(
                exported_path.read_bytes(),
                PHASE2_CITYCAT_TXT.read_bytes(),
                "CSV reconstruction must reproduce the complete Phase 2 TXT bytes",
            )


if __name__ == "__main__":
    unittest.main()
