# Reusable Stockbridge design rainfall

This package provides generic, configuration-driven components for rainfall
experiments: temporal grids, rainfall-event validation, CityCAT export, external
experiment configuration, and normalised depth-duration-frequency (DDF) input.

Phase 3 will use a maximum 120-minute horizon so DNO, U-RNN, and SWE-GNN can be
compared fairly. Phase 4 may use longer durations. The generic implementation derives
record counts and timestamps from configuration and imposes no Phase 3 naming,
duration, return-period, profile-mix, or split rules.

## Experiment configuration

Experiments use JSON so loading requires only the Python standard library. Relative
`source_ddf_table_path` and `output_root` values are resolved relative to the JSON
file. Durations may be an explicit `values_seconds` array or an inclusive
`range_seconds` object containing `start`, `stop`, and `step`.

[`examples/phase3_EXAMPLE.json`](examples/phase3_EXAMPLE.json) is deliberately
non-definitive. Its DDF path and output path are placeholders, its return-period list
and evaluation roles are unassigned, and its seed and duration range are examples—not
approved experiment decisions.

## Normalised DDF CSV schema

Each row represents one base DDF depth at one location, duration, and return period.

| Field | Required | Unit/meaning |
|---|---:|---|
| `location_id` | yes | Non-empty location identifier |
| `duration_minutes` | yes | Positive minutes, exactly representable in whole seconds |
| `return_period_years` | yes | Positive years |
| `depth_mm` | yes | Positive base rainfall depth in millimetres |
| `source` | no | Dataset or method name |
| `source_version` | no | Dataset/method version |
| `extraction_date` | no | Source extraction date, preferably ISO 8601 |
| `climate_scenario` | no | Scenario label; does not alter `depth_mm` |
| `uplift_factor` | no | Positive dimensionless provenance value; not applied on load |
| `notes` | no | Free-text provenance |

Additional columns are retained as provenance. Duplicate
`location_id`–`duration_minutes`–`return_period_years` keys are rejected after
duration normalisation to seconds. A multi-location file requires the caller to select
one `location_id` explicitly. Each loaded `DDFTable` represents one location and one
internally coherent DDF surface. For that selected location, at most one distinct
non-empty `climate_scenario` label is permitted. Missing or empty labels remain
unnamed; they are not assigned a default scenario.

The duplicate key remains
`(location_id, duration_seconds, return_period_years)`, so multiple climate scenarios
at the same coordinate cannot coexist in one loaded surface. Store and load each
scenario as a separate coherent table for now; scenario selection and multi-surface
interpolation are not implemented.

## Lookup, interpolation, and uplift

Exact lookup uses location, duration, and return period. Interpolation is explicitly
one-dimensional and the method must be supplied on every call:

- Duration supports `log_log` (linear interpolation of log depth against log
  duration, recommended for FEH-style DDF tables) and `linear` (raw depth against raw
  duration).
- Return period supports `gumbel_log_depth` (linear interpolation of log depth
  against Gumbel reduced variate, recommended for FEH-style DDF tables) and `linear`
  (raw depth against raw return period).

The Gumbel transform requires `return_period_years > 1` for every requested and source
point that it transforms. Raw `linear` interpolation has no Gumbel-specific limit, and
an exact match returns directly without invoking either interpolation transform.

Interpolation returns both source points and the chosen method. Extrapolation is
never implicit: requests outside the available one-dimensional domain raise an error.
Two-dimensional interpolation is intentionally not implemented at this stage.

Climate uplift is a separate explicit calculation:

```text
uplifted_depth_mm = base_depth_mm * uplift_factor
```

The result preserves the base depth, dimensionless factor, and uplifted depth. Loading
a CSV never replaces its source `depth_mm` with an uplifted value and the generic code
contains no built-in UK climate-change allowance. A `climate_scenario` label is only
provenance: it never implies or triggers a climate uplift.

## Gridded DDF representative-point extraction

`gridded_ddf.py` reads Arc/Info ASCII grids using the package's existing NumPy
dependency. Each input must have a strict six-field header containing `NCOLS`,
`NROWS`, `CELLSIZE`, `NODATA_VALUE`, and either the matching
`XLLCENTER`/`YLLCENTER` pair or the matching `XLLCORNER`/`YLLCORNER` pair.
Dimensions and cell size must be positive, coordinates and raster values must be
finite, and the raster must contain exactly the declared rows and columns. Corner
origins are converted to cell-centre coordinates. The original header metadata and
resolved source path remain attached to the parsed grid.

ASCII raster rows are retained in their source order: the first data row is the
northernmost row and the last is the southernmost. `NODATA_VALUE` cells remain
identifiable and cannot be extracted.

Scientific metadata is supplied through an explicit catalogue CSV rather than
inferred from opaque filename codes. The required columns are `grid_path`,
`duration_minutes`, `return_period_years`, `value_scale_to_mm`, `crs`, `source`,
`source_version`, `extraction_date`, `climate_scenario`, and `notes`. Relative grid
paths resolve against the catalogue's directory. Scale and CRS are mandatory because
Arc/Info ASCII files do not reliably carry depth units or coordinate-reference
metadata. A catalogue must have consistent grid geometry and CRS, unique
duration/return-period entries, and at most one non-empty climate scenario.

Extraction requires an explicit location ID, native-grid x/y coordinates, and an
exactly matching CRS identifier. No coordinate transformation is attempted. The
supported point domain extends from the minimum to maximum grid-point centre on each
axis, inclusive; points beyond those coordinates are rejected. Selection uses the
nearest grid point. Exact midpoint ties choose the smaller coordinate—west first and
then south—and the result records the selected coordinates and native-CRS distance.
This is spatial point selection only: it is not bilinear interpolation and is
separate from the existing, explicitly invoked DDF interpolation over duration or
return period. It also performs no catchment averaging, zonal statistic, areal
reduction, climate uplift, or scenario-derived adjustment.

The supplied UKCEH/FEH software-validation sample contains only 1-hour and 3-hour
grids. It cannot provide 30- or 45-minute depths without extrapolation, which this
adapter does not perform. The sample is not Stockbridge data and is used only by the
optional external regression. Set `UKCEH_DDF_SAMPLE_ROOT` to the external sample
directory to run that test; licensed grids are never copied into this repository.
No definitive Phase 3 storms are generated by this adapter.

## FEH point-rainfall CSV input

`load_feh_point_ddf_csv(path, location_id=...)` reads an original FEH point-export
CSV directly into the existing DDF layer, bypassing the gridded adapter. It returns
`FEHPointDDFResult(table: DDFTable, metadata: FEHPointDDFMetadata)` in memory; no
persistent normalised dataset is needed and the source file is never modified.
The caller must supply a non-empty location ID.

The supported source settings are `Calculation type=Design rainfall`,
`Calculation mode=For a point`, `Fixed duration=no` and `Annual maximum=yes`.
These settings and a non-empty `Rainfall model` are required. The model name is
retained as reported; the adapter does not compute or interpret the model itself.
Other calculation settings, including non-annual-maximum exports, are rejected.
`Calculation location` must contain five non-empty fields: `Point`, region,
finite numeric x, finite numeric y, and grid reference. No EPSG code is inferred
from the filename, coordinates, region or grid reference. A catchment-centroid
description would be caller context, not a fact established by this CSV.

CSV parsing supports quoted commas, multiline quoted metadata, trailing empty
metadata fields, UTF-8 (with optional BOM), and variable preamble length. Exactly
one table header must contain one `Duration hours`, one `Duration days`, and one
or more explicit `<T> year rainfall (mm)` columns. Column order is unrestricted;
return periods are parsed from labels. Unknown or empty table columns, repeated
headers, duplicate duration columns, numerically duplicate return periods,
duplicate duration keys, empty tables and incomplete/malformed rows are rejected.
Metadata labels/settings are case-sensitive. Unknown preamble rows are retained;
duplicate `=` parameter labels are rejected as ambiguous. Data rows must match
the header width exactly; blank rows and footers after the header are unsupported.

Hours determine `duration_minutes` and `duration_seconds` through the existing DDF
validation helpers. Durations, return periods and depths must be finite and
positive, and durations must represent positive whole seconds. `Duration days`
is a redundant, potentially rounded field: it must be finite and positive, but
is preserved as original text and is neither used for conversion nor required
to equal hours divided by 24. Depths are parsed directly as floats in millimetres,
with their original numeric text also retained. **The gridded sample's 0.1 scale
does not apply.** Loading performs no depth rounding, interpolation, extrapolation,
uplift, spatial selection or storm generation.

`metadata` preserves the resolved source path, original preamble text (including
quoting and line endings, excluding a UTF-8 BOM), parsed CSV preamble rows, all
`=` parameter fields including trailing empties, named parsed source values,
original header labels, and zero-based duration/return-period column mappings.
If present, the `VERSION` row must follow the inspected FEH layout:
`VERSION,<service>,Version,<version>,exported at,<time>,<zone>,<weekday>,<date>`
with optional trailing empty fields. Its timestamp is preserved as source text
under `source_export_timestamp`, with separate date/time/zone/weekday fields;
it is not parsed into a guessed date or replaced by a processing timestamp.

Each `DDFRecord.provenance` retains the named source metadata, path, source line
number, original duration labels/values (including days), return-period label
and column index, and original depth text. These survive exact lookup through
`DDFEstimate.source_points`. No climate scenario, uplift factor, processing
timestamp or extraction date is invented when absent.

```python
from CityCAT.rainfall.design_rainfall import load_feh_point_ddf_csv

result = load_feh_point_ddf_csv(
    "/path/outside/repository/FEH_point_export.csv",
    location_id="my_point",  # explicit caller identifier
)
estimate = result.table.lookup_exact("my_point", 60, 10)  # minutes, years
depth_mm = estimate.depth_mm
provenance = estimate.source_points[0].provenance
source_settings = result.metadata.parsed
```

Tests use synthetic temporary CSVs only. From the repository root:

```bash
python3 -m unittest CityCAT.rainfall.design_rainfall.tests.test_feh_point_ddf -v
STOCKBRIDGE_FEH_DDF_CSV=/external/path/to/supplied.csv \
  python3 -m unittest CityCAT.rainfall.design_rainfall.tests.test_feh_point_ddf_external -v
python3 -m unittest discover -s CityCAT/rainfall/design_rainfall/tests -v
python3 -m unittest CityCAT.rainfall.design_rainfall.tests.test_phase2_regression -v
```

The optional FEH regression is specific to the supplied FEH22 annual-maximum
GB point export: 384 durations, 13 return periods, 4,992 records, 15–5,760 minutes,
source coordinates 447157/129992, and exact depths 25.30 mm at 60 minutes/10 years
and 32.78 mm at 30 minutes/100 years. It checks every depth against an independent
CSV read. These expectations appear only in tests and documentation. The test
skips clearly if the environment variable is unset or the external file is absent;
licensed data must remain outside Git.

## Single triangular rainfall pulse

`TriangularPulse` describes one reusable pulse using an absolute start time, a
pulse duration, a target depth, and a peak-position ratio strictly between zero
and one. Its endpoint is `start_time_seconds + duration_seconds`. The resulting
`RainfallEvent.active_duration_seconds` is that absolute endpoint measured from
`t=0`, not the amount of time for which the pulse lasts. Consequently, delayed
pulses must finish by both `maximum_active_duration_seconds` and the simulation
horizon.

`generate_triangular_pulse` evaluates the continuous rising and falling limbs at
the existing configured timestamps. A requested peak may fall between samples;
it is neither snapped to a timestep nor inserted into the timeline. Metadata
therefore records the requested peak time separately from the timestamp or
timestamps of the actual sampled maximum.

The dimensionless sampled shape is normalised using the same trapezoidal,
piecewise-linear convention as the core rainfall validator. The reported shape
normalisation factor has units of millimetres per hour and scales that sampled
shape so its discrete integral equals the requested depth. Conservation applies
to this sampled representation, not to a separately integrated ideal continuous
triangle. No intensity cap or clipping is applied.

Start time and duration must align with the rainfall timestep. The endpoint is
zero, as are all samples before or at the start and at or after the endpoint. At
least one positive interior sample is required; in particular, a one-timestep
pulse is rejected rather than being assigned fabricated rainfall.

The caller supplies rainfall and event identifiers, role, and profile-family
metadata. This generator assigns no return period and defines no production
ensemble naming or allocation. FEH/ReFH profiles, multiple-pulse composition,
and event-selection logic are not implemented at this stage.

## Two-pulse triangular composite

`CompositeTriangularPulse` contains exactly two `TriangularPulse` definitions,
ordered in time. The second pulse must start strictly after the first pulse's
absolute endpoint. Touching and overlapping pulses are rejected so the interval
between them is a strictly positive dry gap, including zero intensity at both
gap endpoints.

Each constituent is generated independently by the Stage 3A implementation and
conserves its own discretely integrated target depth. The two sampled intensity
series are added without global rescaling, so the composite depth is the sum of
the two constituent depths. Composite metadata retains the complete Stage 3A
peak metadata for each pulse as separate first- and second-pulse mappings.

The composite `RainfallEvent.active_duration_seconds` is the second pulse's
absolute end time, not either pulse duration or the sum of their durations. The
caller supplies all event identity and role fields. No return period is assigned
automatically to the composite event.

## Rainfall duration and current scope

`active_duration_seconds` denotes the endpoint of active rainfall. Rainfall intensity
must be zero at `t=0`, may be positive strictly before the endpoint, and must be zero
at and after the declared endpoint.

The approved Stockbridge DDF source/version, final Phase 3 durations and return
periods, profile allocation, evaluation split, and climate allowance remain
provisional scientific/experimental decisions. This stage does not generate the
definitive R101–R300 rainfall files and does not change the CityCAT-to-PT converter.
