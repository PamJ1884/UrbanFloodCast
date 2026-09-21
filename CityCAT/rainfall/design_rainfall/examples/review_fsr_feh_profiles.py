"""Reproducible TEST / experimental comparison; outputs only to a new temp dir.

Run as a module from the repository root. Matplotlib is a review dependency
already available in dno, not an import dependency of the reference API.
"""

from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..core import TemporalConfiguration
from ..fsr_feh_profiles import build_fsr_feh_reference, PROFILE_SOURCE_REFERENCE
from ..experimental_fsr_feh import adapt_fsr_feh_reference_experimental


STATUS = "TEST / experimental — no production or ReFH2 equivalence approval"


def _curve_error(result, subdivisions):
    source = result.reference
    intervals = source.duration_seconds // source.configuration.rainfall_timestep_seconds
    times = np.linspace(source.start_time_seconds, source.end_time_seconds,
                        intervals * subdivisions + 1)
    errors = np.array([abs(result.cumulative_depth_at(float(t))
                          - source.cumulative_depth_at(float(t))) for t in times])
    index = int(np.argmax(errors))
    return float(errors[index]), float(times[index])


def main() -> None:
    output = Path(tempfile.mkdtemp(prefix="TEST_fsr_feh_review_"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    configuration = TemporalConfiguration(300, 7200, 7200, depth_tolerance_mm=1e-9)
    cases = [(profile, minutes * 60, 0) for profile in ("summer", "winter")
             for minutes in (15, 30, 45, 60, 90, 120)]
    cases += [(profile, duration, start) for profile in ("summer", "winter")
              for duration, start in ((900, 900), (3600, 1800))]
    summaries, results = [], []
    interval_rows, curve_rows = [], []
    for index, (profile, duration, start) in enumerate(cases, 901):
        case_id = f"TEST_{profile}_{duration//60}min_start{start//60}min"
        source = build_fsr_feh_reference(
            configuration, profile_name=profile, start_time_seconds=start,
            duration_seconds=duration, target_depth_mm=20,
        )
        result = adapt_fsr_feh_reference_experimental(
            source, rainfall_index=index, phase_event_id=case_id,
            storm_id=case_id, role="TEST_REVIEW",
            profile_family="TEST_EXPERIMENTAL_FSR_FEH",
        )
        results.append((case_id, result))
        coarse_error, _ = _curve_error(result, 512)
        curve_error, curve_time = _curve_error(result, 1024)
        summaries.append({
            "status": STATUS, "case_id": case_id, "profile": profile,
            "duration_minutes": duration//60, "start_minutes": start//60,
            "rainfall_intervals": duration//300,
            "depth_before_normalization_mm": result.depth_before_normalization_mm,
            "normalization_factor": result.normalization_factor,
            "final_total_depth_mm": result.final_total_depth_mm,
            "maximum_cumulative_error_at_edges_mm": result.maximum_absolute_cumulative_depth_error_mm,
            "maximum_cumulative_error_at_edges_fraction": result.maximum_absolute_cumulative_depth_error_fraction,
            "estimated_maximum_curve_error_mm": curve_error,
            "estimated_maximum_curve_error_fraction": curve_error/20,
            "estimated_maximum_curve_error_time_seconds": curve_time,
            "curve_error_refinement_change_mm": abs(curve_error-coarse_error),
            "curve_error_grid_spacing_seconds": 300/1024,
            "reference_maximum_interval_average_mm_per_hour": result.reference_maximum_interval_average_intensity_mm_per_hour,
            "adapted_maximum_nodal_mm_per_hour": result.adapted_maximum_nodal_intensity_mm_per_hour,
            "adapted_maximum_nodal_times_seconds": result.adapted_maximum_nodal_times_seconds,
        })
        for j, (left, right) in enumerate(zip(source.interval_edges_seconds,
                                             source.interval_edges_seconds[1:])):
            interval_rows.append({
                "status": STATUS, "case_id": case_id,
                "left_seconds": left, "right_seconds": right,
                "reference_interval_depth_mm": source.interval_depths_mm[j],
                "adapted_interval_depth_mm": result.adapted_interval_depths_mm[j],
                "reference_average_mm_per_hour": source.interval_average_intensities_mm_per_hour[j],
                "adapted_left_nodal_mm_per_hour": float(result.event.intensities_mm_per_hour[j]),
                "adapted_right_nodal_mm_per_hour": float(result.event.intensities_mm_per_hour[j+1]),
                "reference_cumulative_at_right_mm": source.cumulative_depths_mm[j+1],
                "adapted_cumulative_at_right_mm": result.adapted_cumulative_depths_mm[j+1],
            })
        for time in np.linspace(0, 7200, 24*128+1):
            curve_rows.append({
                "status": STATUS, "case_id": case_id, "time_seconds": float(time),
                "reference_cumulative_mm": source.cumulative_depth_at(float(time)),
                "adapted_cumulative_mm": result.cumulative_depth_at(float(time)),
            })

    for filename, rows in (("TEST_comparison.csv", summaries),
                           ("TEST_all_intervals.csv", interval_rows),
                           ("TEST_cumulative_curves.csv", curve_rows)):
        with (output/filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/"TEST_comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")

    groups = [("summer", results[:6]), ("winter", results[6:12]), ("delayed", results[12:])]
    with PdfPages(output/"TEST_profile_comparisons.pdf") as pdf:
        for group, selected in groups:
            fig, axes = plt.subplots(len(selected), 2, figsize=(12, 2.45*len(selected)),
                                     squeeze=False, layout="constrained")
            fig.suptitle(f"TEST / experimental · FSR/FEH {group} comparison\n"
                         "20 mm · 5-minute timestep · 120-minute horizon", fontsize=14)
            for row, (case_id, result) in enumerate(selected):
                source = result.reference
                times = np.array(source.interval_edges_seconds)
                left = max(0, source.start_time_seconds-300)
                right = min(7200, source.end_time_seconds+300)
                plot_times = np.linspace(left, right, int((right-left)/300)*128+1)
                curve_ref = [source.cumulative_depth_at(float(t)) for t in plot_times]
                curve_adapt = [result.cumulative_depth_at(float(t)) for t in plot_times]
                ax, cumulative_ax = axes[row]
                ax.stairs(source.interval_average_intensities_mm_per_hour, times/60,
                          label="Reference interval average", color="#245c8a", linewidth=1.5)
                ax.plot(times/60, result.event.intensities_mm_per_hour, "o-",
                        color="#ce6327", markersize=3, linewidth=1.3,
                        label="Experimental linear nodal intensity")
                ax.set_ylabel("Intensity (mm/hour)")
                cumulative_ax.plot(plot_times/60, curve_ref, color="#245c8a",
                                   label="Reference curve P·C(u)", linewidth=1.5)
                cumulative_ax.plot(plot_times/60, curve_adapt, color="#ce6327",
                                   label="Integral of experimental linear intensity", linewidth=1.5)
                cumulative_ax.plot(times/60, source.cumulative_depths_mm, ".", color="#245c8a", markersize=3)
                cumulative_ax.set_ylabel("Cumulative depth (mm)")
                cumulative_ax.set_ylim(0, 21)
                for axis in (ax, cumulative_ax):
                    axis.set_xlim(left/60, right/60)
                    axis.set_xlabel("Time since simulation start (minutes)")
                    axis.grid(alpha=0.18)
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.axvline((source.start_time_seconds+source.duration_seconds/2)/60,
                                 color="0.5", linewidth=0.8, linestyle=":")
                ax.set_ylim(bottom=0)
                ax.set_title(f"{source.profile_name.title()} · D={source.duration_seconds//60} min · "
                             f"start={source.start_time_seconds//60} min", fontsize=10)
                cumulative_ax.set_title(
                    f"Max edge error ≈ {result.maximum_absolute_cumulative_depth_error_mm:.3f} mm; "
                    "between-edge errors reported separately", fontsize=9,
                )
                if row == 0:
                    ax.legend(fontsize=7, loc="upper right")
                    cumulative_ax.legend(fontsize=7, loc="lower right")
            fig.savefig(output/f"TEST_{group}_comparison.png", dpi=150)
            pdf.savefig(fig)
            plt.close(fig)

    heading = (
        "# TEST / experimental FSR/FEH temporal-profile review\n\n"
        "20 mm; 300-second timestep; 7200-second simulation horizon and TEST-only "
        "maximum active endpoint. No production storms or ReFH2 equivalence approval.\n\n"
        "Reference interval depths are exact differences of the published approximation "
        "within floating-point precision. Nodal adaptation uses adjacent interval means, "
        "zero endpoints and global depth normalization. It does not preserve individual "
        "interval depths. Peak measures below describe different representations. "
        "Dotted vertical lines mark the unsnapped reference centre; markers show actual nodes. "
        "Peak times use exact numerical equality to the maximum: nominally symmetric "
        "ties may differ by floating-point roundoff.\n\n"
        "Edge error is the exact maximum over configured interval edges. Curve error is "
        "an estimate over a 0.29296875-second grid including the centre and edges, not a "
        "proven continuous supremum. The JSON/CSV records the change from a half-density "
        "grid. Cumulative plots integrate each partial linear-intensity interval analytically.\n\n"
        f"[Profile source]({PROFILE_SOURCE_REFERENCE}); "
        "[depth context](https://refhdocs.hydrosolutions.co.uk/Design-DDF-Rainfall-Hyetographs/Design-Rainfall-Depth/). "
        "ReFH2.3 peak limiting remains an unresolved compatibility question; no limiter, "
        "ARF, seasonal correction, DDF lookup or uplift is applied.\n\n"
    )
    table = ["| Profile | Start min | D min | Intervals | Before mm | Factor | Final mm | Edge error mm (%) | Curve error ≈ mm | Ref peak mm/h | Nodal peak mm/h | Peak times s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        table.append(
            f"| {row['profile']} | {row['start_minutes']} | {row['duration_minutes']} | "
            f"{row['rainfall_intervals']} | {row['depth_before_normalization_mm']:.6f} | "
            f"{row['normalization_factor']:.6f} | {row['final_total_depth_mm']:.6f} | "
            f"{row['maximum_cumulative_error_at_edges_mm']:.6f} "
            f"({100*row['maximum_cumulative_error_at_edges_fraction']:.3f}%) | "
            f"{row['estimated_maximum_curve_error_mm']:.6f} | "
            f"{row['reference_maximum_interval_average_mm_per_hour']:.3f} | "
            f"{row['adapted_maximum_nodal_mm_per_hour']:.3f} | "
            f"{row['adapted_maximum_nodal_times_seconds']} |"
        )
    links = "\n\n[All interval depths](TEST_all_intervals.csv) · [Full metrics](TEST_comparison.csv) · [PDF figures](TEST_profile_comparisons.pdf)\n\n"
    links += "\n\n".join(f"![TEST / experimental {group}](TEST_{group}_comparison.png)" for group, _ in groups)
    (output/"TEST_review.md").write_text(heading + "\n".join(table) + links + "\n", encoding="utf-8")
    html_rows = "".join("<tr>" + "".join(f"<td>{html.escape(cell.strip())}</td>" for cell in line.strip("|").split("|")) + "</tr>"
                        for line in table[:1]+table[2:])
    page = '<!doctype html><meta charset="utf-8"><title>TEST / experimental FSR/FEH review</title>'
    page += '<style>body{font:15px system-ui;margin:2rem;max-width:1500px}td{padding:.4rem;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:12px}img{width:100%;max-width:1200px}pre{white-space:pre-wrap}</style>'
    page += '<h1>TEST / experimental FSR/FEH comparison</h1><pre>' + html.escape(heading) + '</pre>'
    page += '<table>' + html_rows + '</table>'
    page += ''.join(f'<h2>TEST / experimental {group}</h2><img src="TEST_{group}_comparison.png" alt="TEST {group} comparison">' for group, _ in groups)
    (output/"TEST_review.html").write_text(page, encoding="utf-8")
    print(output)
    print("\n".join(table))
    print("Largest curve-grid refinement change (mm):",
          max(row["curve_error_refinement_change_mm"] for row in summaries))


if __name__ == "__main__":
    main()
