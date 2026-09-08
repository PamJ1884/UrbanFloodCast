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

## Rainfall duration and current scope

`active_duration_seconds` denotes the endpoint of active rainfall. Rainfall intensity
must be zero at `t=0`, may be positive strictly before the endpoint, and must be zero
at and after the declared endpoint.

The approved Stockbridge DDF source/version, final Phase 3 durations and return
periods, profile allocation, evaluation split, and climate allowance remain
provisional scientific/experimental decisions. This stage does not generate the
definitive R101–R300 rainfall files and does not change the CityCAT-to-PT converter.
