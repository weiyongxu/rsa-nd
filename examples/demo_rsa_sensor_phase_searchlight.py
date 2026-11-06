r"""Sensor-level spatiotemporal spectral RSA diagnostic
===================================================

This mirrors ``demo_rsa_source_roi_searchlight`` but keeps the data at sensor
level. For each selected sensor type it runs per-sensor x time searchlights
for log-power and phase-only TFR features, then compares the phase maximum
against a condition-label permutation max-null.

Run with:

    uv run python examples/demo_rsa_sensor_phase_searchlight.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_rsa
import numpy as np
from _support.rsa_outputs import clean_generated_outputs, examples_output_dir
from _support.rsa_sample import (
    RSA_TMAX,
    SENSOR_LABELS,
    SENSOR_TYPES,
    TEMPORAL_RADIUS,
    TFR_FREQS,
    load_balanced_sensor_epochs,
    load_sample_raw_events,
    sensitivity_metric,
    validate_sensor_types,
)

from rsa_nd import prepare_rsa_data, rsa_nd

OUT_DIR = examples_output_dir(__file__)
COMPOSITE_PNG = OUT_DIR / "sensor_spatiotemporal_phase_rsa_composite.png"
SUMMARY_CSV = OUT_DIR / "sensor_spatiotemporal_phase_rsa_summary.csv"
GENERATED_OUTPUT_PATTERNS = (
    "sensor_spatiotemporal_phase_rsa_*.png",
    "sensor_spatiotemporal_phase_rsa_*.csv",
)

DEFAULT_N_PER_CONDITION = 10
DEFAULT_N_PERMUTATIONS = 20
DEFAULT_RANDOM_SEED = 20260430
TOP_K_SENSORS = 6


def _prepare_rsa_data_only(*args, **kwargs):
    return prepare_rsa_data(*args, **kwargs).rsa_data


def clean_outputs(output_dir=OUT_DIR):
    return clean_generated_outputs(output_dir, GENERATED_OUTPUT_PATTERNS)


def peak_info(rsa_vals, times, sensors):
    """Return the global peak RSA value with sensor and time coordinates."""

    peak_sensor_idx, peak_time_idx = np.unravel_index(
        np.nanargmax(rsa_vals), np.asarray(rsa_vals).shape
    )
    peak_sensor_idx = int(peak_sensor_idx)
    peak_time_idx = int(peak_time_idx)
    return {
        "sensor_idx": peak_sensor_idx,
        "time_idx": peak_time_idx,
        "sensor": str(sensors[peak_sensor_idx]),
        "time": float(times[peak_time_idx]),
        "value": float(rsa_vals[peak_sensor_idx, peak_time_idx]),
    }


def permutation_null_summary(observed_max, permutation_maxima):
    """Summarize a max-statistic permutation null distribution."""

    maxima = np.asarray(permutation_maxima, dtype=float)
    if maxima.ndim != 1 or maxima.size == 0:
        raise ValueError("permutation_maxima must be a non-empty 1-D array.")
    if not np.isfinite(maxima).all():
        raise ValueError("permutation_maxima must contain only finite values.")
    observed = float(observed_max)
    return {
        "n_permutations": int(maxima.size),
        "median": float(np.median(maxima)),
        "q95": float(np.quantile(maxima, 0.95)),
        "max": float(np.max(maxima)),
        "p_value": float((1 + np.sum(maxima >= observed)) / (maxima.size + 1)),
    }


def _compute_feature_rsa(tfr, model_rdm, labels, *, tfr_feature, log):
    prepared = _prepare_rsa_data_only(
        tfr,
        time=(float(tfr.times[0]), RSA_TMAX),
        tfr_feature=tfr_feature,
        log=log,
        standardize="features",
    )
    return rsa_nd(
        prepared,
        model_rdm,
        search_axes=("space", "time"),
        radii={"space": 0, "time": TEMPORAL_RADIUS},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )


def _phase_permutation_maxima(tfr, condition_codes, labels, *, n_permutations, seed):
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(n_permutations):
        permuted_codes = condition_codes[rng.permutation(len(condition_codes))]
        permuted_model = mne_rsa.compute_rdm(
            permuted_codes,
            metric=sensitivity_metric,
        )
        permuted = _compute_feature_rsa(
            tfr,
            permuted_model,
            labels,
            tfr_feature="phase",
            log=False,
        )
        maxima.append(float(np.nanmax(permuted.rsa)))
    return np.asarray(maxima, dtype=float)


def analyze_sensor_type(
    sensor_type,
    raw,
    events,
    *,
    n_per_condition=DEFAULT_N_PER_CONDITION,
    n_permutations=DEFAULT_N_PERMUTATIONS,
    seed=DEFAULT_RANDOM_SEED,
):
    """Run per-sensor x time power/phase RSA and the phase max-null."""

    epochs = load_balanced_sensor_epochs(
        raw,
        events,
        sensor_type=sensor_type,
        n_per_condition=n_per_condition,
    )
    model_rdm = mne_rsa.compute_rdm(epochs.events[:, 2], metric=sensitivity_metric)
    labels = np.arange(len(epochs))

    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=TFR_FREQS,
        average=False,
        output="complex",
        n_cycles=1.0,
        use_fft=True,
        n_jobs=1,
        verbose=False,
    )

    power = _compute_feature_rsa(
        tfr,
        model_rdm,
        labels,
        tfr_feature="power",
        log=True,
    )
    phase = _compute_feature_rsa(
        tfr,
        model_rdm,
        labels,
        tfr_feature="phase",
        log=False,
    )
    sensors = np.asarray(phase.search_coords["space"], dtype=object)
    times = np.asarray(phase.search_coords["time"], dtype=float)
    power_peak = peak_info(power.rsa, times, sensors)
    phase_peak = peak_info(phase.rsa, times, sensors)

    null = None
    if n_permutations:
        maxima = _phase_permutation_maxima(
            tfr,
            epochs.events[:, 2],
            labels,
            n_permutations=n_permutations,
            seed=seed,
        )
        null = permutation_null_summary(phase_peak["value"], maxima)

    return {
        "sensor_type": sensor_type,
        "n_channels": len(sensors),
        "info": epochs.info.copy(),
        "sensors": sensors,
        "times": times,
        "power_rsa": power.rsa,
        "phase_rsa": phase.rsa,
        "power_peak": power_peak,
        "phase_peak": phase_peak,
        "phase_null": null,
    }


def summary_rows(results):
    """Return CSV-ready summary rows for power and phase results."""

    rows = []
    for result in results:
        base = {
            "sensor_type": result["sensor_type"],
            "n_channels": result["n_channels"],
        }
        for feature, peak_key in (("power", "power_peak"), ("phase", "phase_peak")):
            peak = result[peak_key]
            row = {
                **base,
                "feature": feature,
                "peak_sensor": peak["sensor"],
                "peak_time": peak["time"],
                "peak_rsa": peak["value"],
                "phase_null_n": "",
                "phase_null_median": "",
                "phase_null_q95": "",
                "phase_null_max": "",
                "phase_null_p": "",
            }
            if feature == "phase" and result.get("phase_null") is not None:
                null = result["phase_null"]
                row.update(
                    {
                        "phase_null_n": null["n_permutations"],
                        "phase_null_median": null["median"],
                        "phase_null_q95": null["q95"],
                        "phase_null_max": null["max"],
                        "phase_null_p": null["p_value"],
                    }
                )
            rows.append(row)
    return rows


def save_summary_csv(results, path=SUMMARY_CSV):
    rows = summary_rows(results)
    fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sensor_layout_xy(info, sensors, sensor_type):
    layout = mne.channels.find_layout(info, ch_type=sensor_type)
    name_to_xy = {name: pos[:2] for name, pos in zip(layout.names, layout.pos)}
    xy = []
    missing = []
    for sensor in sensors:
        sensor_name = str(sensor)
        if sensor_name in name_to_xy:
            xy.append(name_to_xy[sensor_name])
        else:
            missing.append(sensor_name)
    if missing:
        raise RuntimeError(f"Layout has no coordinates for sensors: {missing[:5]}")
    return np.asarray(xy, dtype=float)


def _plot_sensor_map(ax, result, values, *, title):
    xy = _sensor_layout_xy(result["info"], result["sensors"], result["sensor_type"])
    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=30,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="0.35",
        linewidths=0.25,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return sc


def _plot_top_sensor_timecourses(ax, result, rsa_vals, peak, *, title):
    times = result["times"]
    sensors = result["sensors"]
    sensor_peaks = np.nanmax(rsa_vals, axis=1)
    top_order = list(np.argsort(sensor_peaks)[::-1][:TOP_K_SENSORS])

    for sensor_idx in range(len(sensors)):
        if sensor_idx in top_order:
            continue
        ax.plot(times, rsa_vals[sensor_idx], color="0.82", linewidth=0.45, zorder=1)

    cmap = plt.get_cmap("tab10")
    for rank, sensor_idx in enumerate(top_order):
        is_peak = sensor_idx == peak["sensor_idx"]
        ax.plot(
            times,
            rsa_vals[sensor_idx],
            color="tab:red" if is_peak else cmap(rank % 10),
            linewidth=2.2 if is_peak else 1.2,
            linestyle="-" if is_peak else "--",
            label=f"{sensors[sensor_idx]}" + (" (peak)" if is_peak else ""),
            zorder=4 if is_peak else 3,
        )
    ax.axvline(0, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8)
    ax.axvline(peak["time"], color="tab:red", linestyle=":", linewidth=0.8)
    ax.set(title=title, xlabel="Time (s)", ylabel="Spearman RSA")
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, loc="best")


def plot_composite(results, *, path=COMPOSITE_PNG):
    n_rows = len(results)
    fig = plt.figure(figsize=(14, 4.2 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(n_rows, 3, width_ratios=[0.85, 0.85, 1.2])

    for row, result in enumerate(results):
        power_peak = result["power_peak"]
        phase_peak = result["phase_peak"]
        sensor_label = SENSOR_LABELS[result["sensor_type"]]

        ax_power = fig.add_subplot(grid[row, 0])
        power_values = result["power_rsa"][:, power_peak["time_idx"]]
        power_sc = _plot_sensor_map(
            ax_power,
            result,
            power_values,
            title=(
                f"{sensor_label} power\n"
                f"{power_peak['sensor']} at {power_peak['time']:+.3f} s"
            ),
        )
        fig.colorbar(power_sc, ax=ax_power, fraction=0.046, pad=0.02)

        ax_phase = fig.add_subplot(grid[row, 1])
        phase_values = result["phase_rsa"][:, phase_peak["time_idx"]]
        null = result.get("phase_null")
        null_text = f", p={null['p_value']:.3f}" if null is not None else ""
        phase_sc = _plot_sensor_map(
            ax_phase,
            result,
            phase_values,
            title=(
                f"{sensor_label} phase\n"
                f"{phase_peak['sensor']} at {phase_peak['time']:+.3f} s{null_text}"
            ),
        )
        fig.colorbar(phase_sc, ax=ax_phase, fraction=0.046, pad=0.02)

        ax_tc = fig.add_subplot(grid[row, 2])
        _plot_top_sensor_timecourses(
            ax_tc,
            result,
            result["phase_rsa"],
            phase_peak,
            title=f"{sensor_label} phase per-sensor searchlights",
        )

    fig.suptitle(
        "Sensor-level spatiotemporal spectral RSA "
        "(auditory-left model): power vs phase",
        fontsize=14,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_result(result):
    sensor_type = result["sensor_type"]
    print(
        f"\n[{sensor_type}] {result['n_channels']} {SENSOR_LABELS[sensor_type]} channels"
    )
    for feature, peak_key in (("power", "power_peak"), ("phase", "phase_peak")):
        peak = result[peak_key]
        print(
            f"  {feature:5s} peak={peak['value']:+.3f} "
            f"in {peak['sensor']} at {peak['time']:+.3f} s"
        )
    null = result.get("phase_null")
    if null is not None:
        print(
            "  phase max-null: "
            f"median={null['median']:+.3f}, "
            f"95%={null['q95']:+.3f}, "
            f"max={null['max']:+.3f}, "
            f"p={null['p_value']:.3f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        choices=SENSOR_TYPES,
        default=list(SENSOR_TYPES),
        help="Sensor types to run. Default: eeg mag grad.",
    )
    parser.add_argument(
        "--n-per-condition",
        type=int,
        default=DEFAULT_N_PER_CONDITION,
        help=f"Trials per condition. Default: {DEFAULT_N_PER_CONDITION}.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f"Phase max-null permutations. Default: {DEFAULT_N_PERMUTATIONS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Permutation seed. Default: {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Keep previous generated sensor diagnostic outputs.",
    )
    return parser.parse_args(argv)


def main(
    sensor_types=None,
    *,
    n_per_condition=DEFAULT_N_PER_CONDITION,
    n_permutations=DEFAULT_N_PERMUTATIONS,
    seed=DEFAULT_RANDOM_SEED,
    clean_output=True,
):
    mne.set_log_level("WARNING")
    OUT_DIR.mkdir(exist_ok=True)
    if clean_output:
        removed = clean_outputs()
        if removed:
            print(f"Cleaned {len(removed)} previous sensor diagnostic output files.")

    selected = validate_sensor_types(
        SENSOR_TYPES if sensor_types is None else sensor_types
    )
    raw, events = load_sample_raw_events()
    results = []
    for sensor_type in selected:
        result = analyze_sensor_type(
            sensor_type,
            raw,
            events,
            n_per_condition=n_per_condition,
            n_permutations=n_permutations,
            seed=seed,
        )
        print_result(result)
        results.append(result)

    save_summary_csv(results, SUMMARY_CSV)
    plot_composite(results, path=COMPOSITE_PNG)
    print(f"\nSaved summary CSV: {SUMMARY_CSV}")
    print(f"Saved composite figure: {COMPOSITE_PNG}")
    return results


if __name__ == "__main__":
    args = parse_args()
    main(
        args.sensor_types,
        n_per_condition=args.n_per_condition,
        n_permutations=args.n_permutations,
        seed=args.seed,
        clean_output=not args.no_clean_output,
    )
