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
ensemble naming or allocation. The FSR/FEH reference-profile experiment below
does not approve a production FEH/ReFH generator or event-selection logic.

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

## Finite-duration Gaussian-family rainfall

`GaussianPulse` and `generate_gaussian_pulse` provide a tapered, finite-duration
skew-normal-based temporal shape. All seven pulse parameters are required:
`start_time_seconds`, `duration_seconds`, `target_depth_mm`,
`peak_position_ratio`, `scale_ratio`, `skewness_parameter` and `edge_taper_ratio`.
There are no scientific parameter defaults. Event identity arguments match the
triangular generator: `rainfall_index`, `phase_event_id`, `storm_id`, `role` and
`profile_family`. Start and duration must align with the configured timestep;
the absolute endpoint `start + duration` must be within both the maximum active
endpoint and simulation horizon. This absolute endpoint becomes
`RainfallEvent.active_duration_seconds`.

DDF depth and temporal shape are separate inputs. This generator receives a
positive target depth and does no DDF lookup, return-period assignment, climate
uplift or ensemble selection. Assigning a DDF depth does not establish the
frequency of occurrence of the exact hyetograph. The approved dataset target is
100 existing Phase 2 events, 100 new FEH/ReFH events, 100 new individual
DDF-based triangular events and 50 new Gaussian-family events; the illustrative
parameters below do not select those events. Two-pulse experiments are deferred
to longer horizons, and their existing implementation remains available.

### Shape, mode and finite support

The underlying standard skew-normal shape is
`f(z; alpha) = 2 * phi(z) * Phi(alpha*z)`, where `phi` and `Phi` are the standard
normal density and CDF. Alpha zero gives the normal (Gaussian) shape. Nonzero
alpha controls asymmetry; it is a distribution parameter, **not a measured
skewness coefficient**. Set

```text
scale_seconds = scale_ratio * duration_seconds
requested_peak_time = start + peak_position_ratio * duration
location_seconds = requested_peak_time - scale_seconds * m(alpha)
z = (t - location_seconds) / scale_seconds
```

`m(alpha)` is the standardized mode, not the location parameter. Its calculation
uses `m(0)=0` and `m(-a)=-m(a)`. For positive `a`, differentiation gives
`m*Phi(a*m) = a*phi(a*m)`. The implementation solves an increasing equation by
sign-bracketed bisection, with relative bracket tolerance `1e-13` and at most
128 iterations. For `a<=1` it solves for `q=m/a` in `[0,sqrt(2/pi)]` using
`q=phi(a*a*q)/Phi(a*a*q)`. For `a>1` it solves for `v=a*m` in log form:
`log(v)+log(Phi(v))+v*v/2+log(2*pi)/2-2*log(a)=0`, with a bounded bracket search.
This avoids squaring a very large alpha or bisecting a very small mode directly.
Invalid brackets, non-finite residuals and non-convergence raise an error.
There is no fallback that equates location with the requested peak.

The requested peak is never snapped to the rainfall timestep. Shape evaluation
uses the equivalent expression `z=(t-requested_peak_time)/scale_seconds+m(alpha)`
to reduce cancellation involving the location. The CDF uses `math.erfc` to avoid
negative-tail cancellation. Only NumPy and Python's standard library are needed.
SciPy was available in the inspected `dno` environment, but was not a declared
repository dependency and is not required by this generator or its tests.

For `u=(t-start)/duration` and `e=edge_taper_ratio`, the dimensionless sampled
shape is `f(z; alpha)*w(u)`, with exactly the following taper:

```text
w(u) = sin((pi/2) * min(u/e, (1-u)/e, 1))**2,  0 < u < 1
w(u) = 0,                                     otherwise
```

The taper ratio must be strictly between zero and one half. Rainfall is exactly
zero at/before start and at/after the endpoint. This is a tapered finite-duration
profile, **not an unchanged normal or skew-normal probability distribution**.
Tapering can shift the continuous maximum, and sampling can shift the observed
maximum further. The final peak is not guaranteed to equal the requested peak.
Metadata distinguishes the requested unmodified mode, distribution location and
all timeline samples attaining the numerical maximum. A continuous maximum of
the tapered profile is not calculated.

### Depth conservation and diagnostics

The shape is sampled on the full configured timeline, then multiplied by
`target_depth_mm / integral(shape, time_hours)` using the same discrete
trapezoidal convention as the core. The pre-normalization integral must be finite
and positive. The normalization factor has units mm/hour. Final intensities must
be finite and non-negative, and integrated depth must match the target within
`configuration.depth_tolerance_mm`. The existing core validates the final event.
There is no clipping, intensity cap or normalization to a prescribed maximum.

At least two strictly positive interior samples are required; otherwise loading
the shape raises a temporal-resolution error. This is only a structural minimum,
not proof of adequate resolution. In particular, narrow profiles can pass while
their peaks and widths remain poorly represented. Floating-point tails can
underflow to zero; unrepresentable scale/location or normalization values are
rejected. The mode tolerance is numerical, not a physical peak-time guarantee
when multiplied by an extreme scale. Review timestep, positive-sample count,
sampled maximum and effective width before later ensemble selection.

`peak_metadata` preserves every input parameter, absolute start/endpoint,
scale in seconds, standardized mode and solver method/tolerance, distribution
location, requested peak, maximum intensity and all maximum times, raw shape
integral, normalization factor/units, integrated/target depths, timestep,
integration method, positive-interior-sample count and the exact taper definition.
It also reports:

- `temporal_centroid_seconds`: mean rainfall time from simulation time zero.
- `effective_temporal_standard_deviation_seconds`: measured rainfall width in
  seconds about that centroid; `scale_ratio` is not this effective width.
- `depth_fraction_up_to_requested_peak`: fraction of the sampled rainfall depth
  accumulated by the unsnapped requested peak, including partial intervals.

Diagnostics use rainfall depth as the weight of the **piecewise-linear intensity
curve**, not nodal point masses. Segment integrals are analytic. For a segment
`[a,b]` of length `h` with endpoint weights `y0,y1`, its area is
`h*(y0+y1)/2` and first moment is
`h*((2*a+b)*y0+(a+2*b)*y1)/6`. After centering the endpoints on the calculated
mean, its second moment is
`h*((3*a*a+2*a*b+b*b)*y0+(a*a+2*a*b+3*b*b)*y1)/12`.
Calculations use dimensionless pulse time and intensity divided by its maximum
for numerical stability; these factors cancel in the ratios. Seconds are
restored in the reported centroid and standard deviation. For a partial segment,
the intensity at the cutoff is linearly interpolated and its partial trapezoid
is integrated exactly. Tests independently check these diagnostics using
two-node Gauss-Legendre integration (exact for the degree-three integrands) and
an analytically known triangular curve.

### Illustrative TEST configuration

These explicit values are software-test parameters, not approved ensemble defaults.
The example creates only in-memory events; it writes no production files.

```python
from CityCAT.rainfall.design_rainfall import (
    GaussianPulse, TemporalConfiguration, generate_gaussian_pulse,
)

configuration = TemporalConfiguration(
    rainfall_timestep_seconds=300,
    simulation_horizon_seconds=7200,
    maximum_active_duration_seconds=5400,
)
test_events = []
for index, (alpha, scale_ratio) in enumerate(
    [(-2.0, 0.18), (0.0, 0.15), (2.0, 0.18)], start=901
):
    pulse = GaussianPulse(
        start_time_seconds=0,
        duration_seconds=3600,
        target_depth_mm=20.0,
        peak_position_ratio=0.5,
        scale_ratio=scale_ratio,
        skewness_parameter=alpha,
        edge_taper_ratio=1.0 / 12.0,
    )
    test_events.append(generate_gaussian_pulse(
        configuration, pulse,
        rainfall_index=index,
        phase_event_id=f"TEST-GAUSSIAN-{index}",
        storm_id=f"TEST-STORM-{index}",
        role="TEST_ROLE",
        profile_family="TEST_GAUSSIAN_FAMILY",
    ))
```

Run focused tests from the repository root with
`python -m unittest CityCAT.rainfall.design_rainfall.tests.test_gaussian_pulse -v`
in `dno`. Synthetic file exports are confined to temporary test directories.

## Reference FSR/FEH temporal profiles and experimental conversion

This stage separates four things: the published FSR/FEH profile approximation,
interval depths calculated from that approximation, our experimental conversion
to nodal intensities, and full ReFH2 design-package behaviour. These are not
interchangeable. No definitive ensemble is generated.

The [ReFH technical guide's profile page](https://refhdocs.hydrosolutions.co.uk/Design-DDF-Rainfall-Hyetographs/Design-Storm-Profiles/)
gives `Y(x)=(1-a**(x**b))/(1-a)`, with summer `(a,b)=(0.100,0.815)` and winter
`(0.060,1.026)`. Y describes the fraction of depth within the fraction x of
duration **centred on the peak**. It is not cumulative rainfall from storm start.
By symmetry, the rainfall before the left boundary of that centred duration is
half of `1-Y(x)`. With normalized chronological time `u=(t-start)/duration`, this
gives the derived transformation:

```text
C(u) = [1-Y(1-2*u)]/2,  0 <= u <= 0.5
C(u) = [1+Y(2*u-1)]/2,  0.5 <= u <= 1
C(u) = 0 before the event; 1 after it
```

`central_depth_fraction` and `chronological_cumulative_fraction` implement these
scalar functions. Y uses `expm1`; C also uses `log1p` and `expm1` in its left
branch to avoid subtracting nearly equal numbers near the start. Boundary
values are exact: Y(0)=0, Y(1)=1, C(0)=0, C(0.5)=0.5, C(1)=1. The central
instantaneous derivative is never evaluated.

`build_fsr_feh_reference` requires an explicit profile name (`summer` or `winter`),
start, duration, finite positive target depth and `TemporalConfiguration`.
Start/duration must be non-boolean integers aligned with the timestep, with the
absolute endpoint within the simulation horizon and maximum active endpoint.
`FSRFEHReferenceProfile` contains the full configured interval-edge timeline in
seconds, cumulative depths in mm, interval depths in mm, interval-average
intensities in mm/hour, parameters, source link and unit labels. For edges u_i,
each interval receives `P*(C(u_(i+1))-C(u_i))`. This is exact discretization of the
approximation within floating-point precision, without derivative sampling or
normalization. Depth before/after the event is constant and external intervals
have zero rainfall. `reference.cumulative_depth_at(t)` evaluates the original
chronological curve at any finite time in absolute seconds.

```python
from CityCAT.rainfall.design_rainfall import (
    TemporalConfiguration, build_fsr_feh_reference,
)

# TEST only; 7200 as maximum active endpoint is not an ensemble decision.
configuration = TemporalConfiguration(300, 7200, 7200)
reference = build_fsr_feh_reference(
    configuration, profile_name="summer", start_time_seconds=0,
    duration_seconds=3600, target_depth_mm=20.0,
)
```

`experimental_fsr_feh.adapt_fsr_feh_reference_experimental` is intentionally
**not exported from the package public API**. It requires explicit event identity
arguments, as in the existing generators. Its result retains the unchanged
reference and an experimental `RainfallEvent`: start/endpoint nodes are zero,
each interior node is the arithmetic mean of its two adjacent reference
interval-average intensities, and a global dimensionless normalization factor
makes the trapezoidal integral equal P. At least two event intervals are needed
for a positive interior node; invalid integrals/intensities are rejected. The
core validates the event and its absolute active endpoint. This is a proposed
numerical adaptation, not an official ReFH method or approved production generator.

The adaptation reports pre-normalization depth, factor, final depth, adapted
depth for every interval alongside the reference, maximum absolute cumulative
error at **reference interval edges** in mm and as a fraction of P, reference
maximum interval-average intensity, and adapted maximum nodal intensity and its
sample times. These are different peak measures. Peak times use exact numerical
equality; nominally symmetric ties may differ by floating-point roundoff.
`cumulative_depth_at(t)` on the
adaptation integrates each partial linear-intensity interval analytically:
for elapsed time s within a timestep h, its added depth is
`[I_left*s + (I_right-I_left)*s*s/(2*h)]/3600`. Cumulative plots therefore use
quadratic segments rather than linear interpolation between cumulative nodes.
Preserving total depth does not preserve each reference interval depth.

Run the reproducible synthetic review in `dno` from the repository root:

```bash
python -m unittest CityCAT.rainfall.design_rainfall.tests.test_fsr_feh_profiles -v
python -m CityCAT.rainfall.design_rainfall.examples.review_fsr_feh_profiles
```

The review script uses the already available Matplotlib installation only for
figures; it is not a dependency of the reference API. It creates a new temporary
`TEST_fsr_feh_review_*` directory and prints its path. Outputs include Markdown
and HTML reviews, CSV/JSON comparison metrics, all interval depths, sampled
cumulative curves, three PNG figures and a three-page PDF. Every comparison is
labelled TEST / experimental. Both profiles use P=20 mm, timestep=300 seconds,
horizon/TEST maximum active endpoint=7200 seconds and durations 15, 30, 45, 60,
90 and 120 minutes. Additional delayed cases use duration/start pairs of
15/15 and 60/30 minutes. Odd interval counts retain their off-grid centres.

The review additionally estimates the maximum continuous-curve cumulative
difference on a 300/1024-second grid, including the centre and all event edges.
It records the estimate's change from a 300/512-second grid. This is a dense-grid
estimate, not a proven continuous supremum or an error bound. The exact
edge-only metric and estimated curve metric are named separately. No smoothing,
peak snapping, clipping or limiter is used. Very small interval depths can lose
relative precision when differencing cumulative depths; comparisons use
floating-point tolerances on the target-depth scale.

The primary guide mentions peak limiting in ReFH2.3 for these profiles but does
not specify its algorithm on the profile page. Limiter compatibility remains
unresolved; no limiter is invented and software equivalence is not claimed.
The [design-depth page](https://refhdocs.hydrosolutions.co.uk/Design-DDF-Rainfall-Hyetographs/Design-Rainfall-Depth/)
describes separate DDF, areal-reduction and seasonal-correction components.
This stage applies none of those operations, nor uplift. Summer/winter names
select temporal shapes only; no seasonal or catchment-average frequency meaning
is inferred from an externally supplied depth.

## Rainfall duration and current scope

`active_duration_seconds` denotes the endpoint of active rainfall. Rainfall intensity
must be zero at `t=0`, may be positive strictly before the endpoint, and must be zero
at and after the declared endpoint.

The approved Stockbridge DDF source/version, final Phase 3 durations and return
periods, profile allocation, evaluation split, and climate allowance remain
provisional scientific/experimental decisions. This stage does not generate the
definitive R101–R300 rainfall files and does not change the CityCAT-to-PT converter.
