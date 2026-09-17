"""Synthetic FEH-style CSV cases; no licensed rainfall data is embedded."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from CityCAT.rainfall.design_rainfall import (
    DDFTable,
    FEHPointDDFMetadata,
    FEHPointDDFResult,
    load_feh_point_ddf_csv,
)


def synthetic_preamble():
    return [
        ['VERSION', ' "TEST FEH, service" ', 'Version', 'TEST_1',
         'exported at', '12:34:56', 'GMT', 'Thu', '01-Jan-99', '', ''],
        ['Parameters', '', '', ''],
        ['Rainfall model=', 'FEH22', '', ''],
        ['Calculation type=', 'Design rainfall', ''],
        ['Calculation mode=', 'For a point', ''],
        ['Calculation location=', 'Point', 'GB', '123456', '234567',
         'TEST GRID REF', '', ''],
        ['Fixed duration=', 'no', ''],
        ['Annual maximum=', 'yes', ''],
    ]


def synthetic_header():
    return ['Duration hours', 'Duration days',
            '2 year rainfall (mm)', '10 year rainfall (mm)']


def synthetic_rows():
    return [['0.25', '0.010416667', '7.123456789', '19.1200'],
            ['1', '0.041666667', '17.5', '41.75']]


class FEHPointDDFTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # An unrelated filename proves that labels and metadata control parsing.
        self.path = Path(directory.name) / 'TEST_unrelated.csv'

    def write_csv(self, *, preamble=None, header=None, rows=None):
        preamble = synthetic_preamble() if preamble is None else preamble
        header = synthetic_header() if header is None else header
        rows = synthetic_rows() if rows is None else rows
        with self.path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\r\n')
            writer.writerows(preamble)
            writer.writerow(header)
            writer.writerows(rows)
        return load_feh_point_ddf_csv(self.path, location_id='TEST_SITE')

    def test_valid_table_units_and_exact_lookup_provenance(self):
        result = self.write_csv()
        self.assertIsInstance(result, FEHPointDDFResult)
        self.assertIsInstance(result.metadata, FEHPointDDFMetadata)
        self.assertIsInstance(result.table, DDFTable)
        self.assertEqual(len(result.table.records), 4)
        self.assertEqual({r.duration_seconds for r in result.table.records}, {900, 3600})
        self.assertEqual({r.duration_minutes for r in result.table.records}, {15, 60})
        self.assertEqual({r.return_period_years for r in result.table.records}, {2, 10})
        self.assertEqual([r.depth_mm for r in result.table.records],
                         [7.123456789, 19.12, 17.5, 41.75])
        record = result.table.lookup_exact('TEST_SITE', 15, 10).source_points[0]
        self.assertEqual(record.depth_mm, 19.12)
        provenance = record.provenance
        self.assertEqual(provenance['source_depth_mm'], '19.1200')
        self.assertEqual(provenance['source_duration_hours'], '0.25')
        self.assertEqual(provenance['source_duration_days'], '0.010416667')
        self.assertEqual(provenance['source_return_period_column'], '10 year rainfall (mm)')
        self.assertEqual(provenance['source_file_path'], str(self.path.resolve()))
        self.assertEqual(provenance['source_export_timestamp'], '12:34:56 GMT Thu 01-Jan-99')
        self.assertEqual(provenance['source_x'], '123456')
        self.assertEqual(provenance['source_y'], '234567')
        self.assertEqual(provenance['source_grid_reference'], 'TEST GRID REF')
        self.assertEqual(provenance['rainfall_model'], 'FEH22')
        self.assertEqual(provenance['annual_maximum'], 'yes')
        self.assertEqual(provenance['calculation_type'], 'Design rainfall')
        self.assertEqual(provenance['calculation_mode'], 'For a point')
        for absent in ('epsg', 'crs', 'climate_scenario', 'uplift_factor',
                       'processing_timestamp', 'extraction_date', 'catchment_centroid'):
            self.assertNotIn(absent, provenance)
        with self.assertRaises(TypeError):
            provenance['source_x'] = 'changed'
        with self.assertRaises(TypeError):
            result.metadata.parsed['source_x'] = 'changed'

    def test_quoted_multiline_metadata_variable_preamble_and_raw_preservation(self):
        for extra in ([], [['Notes=', 'TEST, quoted "note"\nsecond line', '', ''], [],
                           ['Unknown source text', '', '']]):
            with self.subTest(extra=extra):
                preamble = synthetic_preamble() + extra
                result = self.write_csv(preamble=preamble)
                expected = io.StringIO(newline='')
                csv.writer(expected, lineterminator='\r\n').writerows(preamble)
                self.assertEqual(result.metadata.preamble_text, expected.getvalue())
                self.assertEqual(result.metadata.preamble_rows,
                                 tuple(tuple(row) for row in preamble))
                self.assertEqual(result.metadata.parameters['Rainfall model'],
                                 ('FEH22', '', ''))
                self.assertEqual(result.metadata.source_path, self.path.resolve())
                self.assertEqual(result.metadata.header_columns, tuple(synthetic_header()))

    def test_columns_can_be_reordered_and_periods_are_explicit(self):
        order = [3, 1, 2, 0]
        result = self.write_csv(
            header=[synthetic_header()[i] for i in order],
            rows=[[row[i] for i in order] for row in synthetic_rows()],
        )
        self.assertEqual(result.table.lookup_exact('TEST_SITE', 60, 10).depth_mm, 41.75)
        self.assertEqual(result.metadata.duration_hours_column, 3)
        self.assertEqual(result.metadata.duration_days_column, 1)
        self.assertEqual(result.metadata.return_period_columns, ((0, 10.0), (2, 2.0)))

    def test_fractional_minutes_and_rounded_days(self):
        result = self.write_csv(rows=[['0.0025', '0.0001', '1.23', '4.56']])
        record = result.table.lookup_exact('TEST_SITE', 0.15, 2).source_points[0]
        self.assertEqual(record.duration_seconds, 9)
        self.assertEqual(record.provenance['source_duration_days'], '0.0001')

    def test_location_id_is_required_and_validated(self):
        self.write_csv()
        with self.assertRaises(TypeError):
            load_feh_point_ddf_csv(self.path)
        for invalid in (None, '', '  ', 12, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'location_id'):
                load_feh_point_ddf_csv(self.path, location_id=invalid)

    def test_duplicate_duration_keys(self):
        with self.assertRaisesRegex(ValueError, 'duplicate DDF key'):
            self.write_csv(rows=synthetic_rows() + [['0.250', '0.0104', '3', '4']])

    def test_duplicate_duration_columns(self):
        for label in ('Duration hours', 'Duration days'):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'duplicate.*duration column'):
                self.write_csv(header=synthetic_header() + [label])

    def test_duplicate_return_period_columns(self):
        for label in ('2 year rainfall (mm)', '2.0 year rainfall (mm)'):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'duplicate.*return-period column'):
                self.write_csv(header=synthetic_header() + [label])

    def test_invalid_data_numbers(self):
        for column in range(4):
            for invalid in ('NaN', 'inf', '-inf', '0', '-1', 'not a number'):
                row = synthetic_rows()[0]
                row[column] = invalid
                with self.subTest(column=column, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, 'finite and positive|must be numeric'):
                        self.write_csv(rows=[row])

    def test_invalid_return_period_numbers(self):
        for invalid in ('NaN', 'inf', '-inf', '0', '-1', 'unknown'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'return period'):
                self.write_csv(header=synthetic_header()[:2] + [f'{invalid} year rainfall (mm)'])

    def test_invalid_second_representations(self):
        for hours in ('0.0001', '1e-20', '1e308'):
            with self.subTest(hours=hours), self.assertRaisesRegex(ValueError, 'whole|finite'):
                self.write_csv(rows=[[hours, '0.001', '1', '2']])

    def test_missing_values_and_malformed_rows(self):
        invalid_rows = [[], ['1'], synthetic_rows()[0] + ['extra'],
                        synthetic_rows()[0] + ['']]
        for column in range(4):
            row = synthetic_rows()[0]
            row[column] = ' '
            invalid_rows.append(row)
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'malformed or incomplete'):
                self.write_csv(rows=[row])

    def test_empty_table(self):
        with self.assertRaisesRegex(ValueError, 'no data records'):
            self.write_csv(rows=[])

    def test_missing_unknown_or_ambiguous_columns(self):
        headers = [[], synthetic_header()[1:], synthetic_header()[:1] + synthetic_header()[2:],
                   synthetic_header()[:2], synthetic_header() + ['unknown'],
                   synthetic_header() + [''],
                   synthetic_header()[:2] + ['2 year intensity (mm/hr)']]
        for header in headers:
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, 'header|column'):
                self.write_csv(header=header)
        with self.assertRaisesRegex(ValueError, 'header.*ambiguous'):
            self.write_csv(rows=synthetic_rows() + [synthetic_header()] + synthetic_rows())

    def test_missing_duplicate_or_unsupported_settings(self):
        for index, replacement in ((3, 'Rainfall intensity'), (4, 'For a catchment'),
                                   (6, 'yes'), (7, 'no')):
            preamble = synthetic_preamble()
            preamble[index][1] = replacement
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, 'unsupported FEH'):
                self.write_csv(preamble=preamble)
        for index in range(2, 8):
            preamble = synthetic_preamble()
            with self.subTest(missing=index), self.assertRaisesRegex(ValueError, 'missing'):
                self.write_csv(preamble=preamble[:index] + preamble[index + 1:])
            with self.subTest(duplicate=index), self.assertRaisesRegex(ValueError, 'duplicate'):
                self.write_csv(preamble=preamble + [preamble[index]])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            preamble = synthetic_preamble()
            preamble[4][2] = 'another value'
            self.write_csv(preamble=preamble)

    def test_unsupported_location_fields(self):
        for index, value in ((1, 'Catchment'), (2, ''), (3, 'NaN'),
                             (4, 'text'), (5, '')):
            preamble = synthetic_preamble()
            preamble[5][index] = value
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'location|coordinates'):
                self.write_csv(preamble=preamble)

    def test_optional_version_and_invalid_version_metadata(self):
        result = self.write_csv(preamble=synthetic_preamble()[1:])
        self.assertNotIn('source_export_timestamp', result.metadata.parsed)
        with self.assertRaisesRegex(ValueError, 'duplicate.*VERSION'):
            self.write_csv(preamble=synthetic_preamble() + [synthetic_preamble()[0]])
        with self.assertRaisesRegex(ValueError, 'VERSION/export timestamp'):
            self.write_csv(preamble=[['VERSION', 'incomplete']] + synthetic_preamble()[1:])

    def test_unreadable_or_malformed_csv(self):
        with self.assertRaisesRegex(ValueError, 'could not read FEH'):
            load_feh_point_ddf_csv(self.path, location_id='TEST_SITE')
        self.path.write_text('"unterminated field\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'malformed FEH CSV'):
            load_feh_point_ddf_csv(self.path, location_id='TEST_SITE')

    def test_utf8_bom_is_supported_and_source_is_unchanged(self):
        self.write_csv()
        original = b'\xef\xbb\xbf' + self.path.read_bytes()
        self.path.write_bytes(original)
        result = load_feh_point_ddf_csv(self.path, location_id='TEST_SITE')
        self.assertEqual(len(result.table.records), 4)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
