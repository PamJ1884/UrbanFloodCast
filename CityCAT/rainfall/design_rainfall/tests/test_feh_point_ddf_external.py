"""Optional checks of the supplied external licensed FEH22 point export."""

from __future__ import annotations

import csv
import os
import re
import unittest
from pathlib import Path

from CityCAT.rainfall.design_rainfall import load_feh_point_ddf_csv


class ExternalFEHPointDDFTests(unittest.TestCase):
    def test_supplied_feh22_point_export(self):
        configured = os.environ.get('STOCKBRIDGE_FEH_DDF_CSV')
        if not configured:
            self.skipTest('external FEH point CSV unavailable: set STOCKBRIDGE_FEH_DDF_CSV')
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            self.skipTest(f'external FEH point CSV unavailable: {path}')
        original = path.read_bytes()
        result = load_feh_point_ddf_csv(path, location_id='TEST_EXTERNAL_FEH_POINT')
        metadata = result.metadata.parsed
        self.assertEqual(metadata['rainfall_model'], 'FEH22')
        self.assertEqual(metadata['calculation_type'], 'Design rainfall')
        self.assertEqual(metadata['calculation_mode'], 'For a point')
        self.assertEqual(metadata['annual_maximum'], 'yes')
        self.assertEqual(metadata['source_location_type'], 'Point')
        self.assertEqual(metadata['source_location_region'], 'GB')
        self.assertEqual((float(metadata['source_x']), float(metadata['source_y'])),
                         (447157, 129992))
        records = result.table.records
        durations = {record.duration_minutes for record in records}
        self.assertEqual(len(durations), 384)
        self.assertEqual(len({record.return_period_years for record in records}), 13)
        self.assertEqual(len(records), 4992)
        self.assertEqual((min(durations), max(durations)), (15, 5760))
        # Read expected values independently, without using adapter provenance
        # or its parsed column mapping to construct the expected surface.
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle)
            header = next(row for row in reader if 'Duration hours' in row)
            expected = {}
            hours_column = header.index('Duration hours')
            for row in reader:
                duration = float(row[hours_column]) * 60
                for column, label in enumerate(header):
                    match = re.fullmatch(r'([0-9.]+) year rainfall \(mm\)', label)
                    if match:
                        key = (duration, float(match[1]))
                        self.assertNotIn(key, expected)
                        expected[key] = float(row[column])
        actual = {(r.duration_minutes, r.return_period_years): r.depth_mm for r in records}
        self.assertEqual(len(expected), 4992)
        self.assertEqual(actual, expected)
        self.assertEqual(result.table.lookup_exact('TEST_EXTERNAL_FEH_POINT', 60, 10).depth_mm, 25.30)
        self.assertEqual(result.table.lookup_exact('TEST_EXTERNAL_FEH_POINT', 30, 100).depth_mm, 32.78)
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
