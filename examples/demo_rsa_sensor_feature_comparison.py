r"""Run direct spectral RSA on the MNE sample dataset
=================================================

This mirrors the dataset setup from the MNE-RSA whitening example:
https://mne.tools/mne-rsa/stable/auto_examples/sensor_level/plot_whitening.html

The difference is that this script computes spectral features and runs the
package API directly:

    prepare_rsa_data(...).rsa_data + rsa_nd(...)

Run with:

    uv run python examples/demo_rsa_sensor_feature_comparison.py

Choose sensor types with:

    uv run python examples/demo_rsa_sensor_feature_comparison.py --sensor-types eeg mag grad
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import mne
import mne_rsa
import numpy as np
from _support.rsa_outputs import clean_generated_outputs, examples_output_dir
from _support.rsa_sample import (
    SENSOR_LABELS,
    SENSOR_TYPES,
    condition_model_rdm,
    load_audio_visual_epochs,
    load_mixed_audio_visual_epochs,
    load_sample_raw_events,
    validate_sensor_types,
)

from rsa_nd import prepare_rsa_data, rsa_nd

OUT_DIR = examples_output_dir(__file__)
TFR_FEATURE_MODES = (
    ("power", "power", True),
    ("amplitude", "amplitude", False),
    ("real+imag", "real_imag", False),
    ("amp+phase", "amp_phase", False),
    ("phase", "phase", False),
)
GENERATED_OUTPUT_PATTERNS = ("mne_sample_*.png", "mne_sample_*.csv")
COMPOSITE_OUTPUT = OUT_DIR / "mne_sample_all_outputs_composite.png"


def _prepare_rsa_data_only(*args, **kwargs):
    return prepare_rsa_data(*args, **kwargs).rsa_data


def clean_output_dir(output_dir=OUT_DIR):
    """Remove generated demo PNG/CSV outputs from one directory."""

    return clean_generated_outputs(output_dir, GENERATED_OUTPUT_PATTERNS)


def make_conditionwise_induced_epochs(epochs):
    """Subtract the evoked response separately within each condition."""

    induced = epochs.copy()
    induced_data = induced.get_data(copy=True)
    selection_to_position = {
        selection: position for position, selection in enumerate(epochs.selection)
    }

    for condition in epochs.event_id:
        condition_epochs = epochs[condition].copy()
        condition_epochs.subtract_evoked()
        condition_data = condition_epochs.get_data(copy=True)
        for condition_position, selection in enumerate(condition_epochs.selection):
            induced_position = selection_to_position[selection]
            induced_data[induced_position] = condition_data[condition_position]

    induced._data = induced_data
    return induced


def plot_output_composite(sensor_results, *, mixed_result=None, path=COMPOSITE_OUTPUT):
    """Build one composite image directly from RSA result arrays."""

    n_rows = len(sensor_results)
    n_cols = 5

    fig = plt.figure(figsize=(22, 3.3 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(
        n_rows,
        n_cols,
    )

    for row, summary in enumerate(sensor_results):
        sensor_label = SENSOR_LABELS[summary["sensor_type"]]
        _plot_all_band_with_time_domain_ax(
            fig.add_subplot(grid[row, 0]),
            summary["time_centers"],
            summary["all_bands_rsa"],
            summary["time_domain_times"],
            summary["time_domain_rsa"],
            title=f"{sensor_label}: total all-band TFR",
            mixed_result=mixed_result,
        )
        _plot_band_curves_ax(
            fig.add_subplot(grid[row, 1]),
            summary["time_centers"],
            {"all bands": summary["all_bands_rsa"], **summary["band_results"]},
            title=f"{sensor_label}: bands",
        )
        _plot_band_curves_ax(
            fig.add_subplot(grid[row, 2]),
            summary["feature_mode_times"],
            summary["feature_mode_results"],
            title=f"{sensor_label}: TFR features",
        )
        _plot_tf_map_ax(
            fig,
            fig.add_subplot(grid[row, 3]),
            summary["tf_times"],
            summary["tf_freqs"],
            summary["tf_map_rsa"],
            title=f"{sensor_label}: frequency x time",
        )
        _plot_time_curve_ax(
            fig.add_subplot(grid[row, 4]),
            summary["psd_freqs"],
            summary["psd_rsa"],
            title=f"{sensor_label}: PSD",
            xlabel="Frequency (Hz)",
            ylabel="Spearman RSA",
        )

    fig.suptitle("MNE sample spectral RSA outputs", fontsize=16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_all_band_with_time_domain_ax(
    ax,
    tfr_times,
    tfr_rsa,
    time_domain_times,
    time_domain_rsa,
    *,
    title,
    mixed_result=None,
):
    ax.plot(
        tfr_times,
        tfr_rsa,
        color="black",
        linewidth=1.7,
        label="all-band TFR",
    )
    ax.plot(
        time_domain_times,
        time_domain_rsa,
        color="tab:blue",
        linewidth=1.4,
        linestyle="--",
        label="mne_rsa.rsa_epochs",
    )
    if mixed_result is not None:
        ax.plot(
            mixed_result["times"],
            mixed_result["rsa"],
            color="tab:purple",
            linewidth=1.4,
            linestyle=":",
            label="mixed MEG+EEG, whitened",
        )
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="tab:red", linewidth=0.8, linestyle=":")
    ax.set(title=title, xlabel="Time (s)", ylabel="Spearman RSA")
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)


def _plot_time_curve_ax(ax, x, y, *, title, xlabel, ylabel, vline=None):
    ax.plot(x, y, color="black", linewidth=1.5)
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    if vline is not None:
        ax.axvline(vline, color="tab:red", linewidth=0.8, linestyle=":")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, color="0.9", linewidth=0.7)


def _plot_band_curves_ax(ax, times, band_results, *, title):
    for band_name, values in band_results.items():
        linewidth = 2.0 if band_name == "all bands" else 1.3
        linestyle = "--" if band_name == "all bands" else "-"
        ax.plot(
            times, values, linewidth=linewidth, linestyle=linestyle, label=band_name
        )
    ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="tab:red", linewidth=0.8, linestyle=":")
    ax.set(title=title, xlabel="Time (s)", ylabel="Spearman RSA")
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, ncols=2)


def _plot_tf_map_ax(fig, ax, times, freqs, rsa_map, *, title):
    vmax = np.nanmax(np.abs(rsa_map))
    image = ax.imshow(
        rsa_map,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0, color="black", linewidth=0.9, linestyle=":")
    ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman RSA")


def run_sensor_analysis(sensor_type, raw, events, *, n_per_condition=20):
    """Run the full spectral RSA demo for one sensor type."""

    sensor_label = SENSOR_LABELS[sensor_type]
    epochs = load_audio_visual_epochs(
        raw,
        events,
        sensor_type=sensor_type,
        n_per_condition=n_per_condition,
    )
    model_rdm = condition_model_rdm(epochs)
    item_labels = np.arange(len(epochs))

    print(
        f"\n[{sensor_type}] Epochs: {len(epochs)} trials, "
        f"{len(epochs.ch_names)} {sensor_label} channels"
    )
    print(f"Time range: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f} s")

    bands = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }

    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=np.arange(4.0, 31.0, 2.0),
        average=False,
        output="complex",
        n_cycles=1.0,
        use_fft=True,
        n_jobs=1,
    )

    tfr_band_data = _prepare_rsa_data_only(
        tfr,
        freq=bands,
        tfr_feature="power",
        log=True,
        standardize="features",
    )
    tfr_result = rsa_nd(
        tfr_band_data,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0.02},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )

    time_centers = tfr_result.search_coords["time"]

    separate_band_results = {}
    for band_name, band_range in bands.items():
        band_data = _prepare_rsa_data_only(
            tfr,
            freq={band_name: band_range},
            tfr_feature="power",
            log=True,
            standardize="features",
        )
        band_result = rsa_nd(
            band_data,
            model_rdm,
            search_axes=("time",),
            radii={"time": 0.02},
            labels=item_labels,
            labels_rdm_model=item_labels,
            data_rdm_metric="correlation",
            rsa_metric="spearman",
            n_jobs=1,
        )
        separate_band_results[band_name] = band_result.rsa

    feature_mode_results = {}
    feature_mode_times = None
    for tfr_label, tfr_feature, tfr_log in TFR_FEATURE_MODES:
        feature_data = _prepare_rsa_data_only(
            tfr,
            freq=(4.0, 30.0),
            tfr_feature=tfr_feature,
            log=tfr_log,
            standardize="features",
        )
        feature_result = rsa_nd(
            feature_data,
            model_rdm,
            search_axes=("time",),
            radii={"time": 0.02},
            labels=item_labels,
            labels_rdm_model=item_labels,
            data_rdm_metric="correlation",
            rsa_metric="spearman",
            n_jobs=1,
        )
        feature_mode_results[tfr_label] = feature_result.rsa
        if feature_mode_times is None:
            feature_mode_times = feature_result.search_coords["time"]
        elif not np.array_equal(
            feature_mode_times, feature_result.search_coords["time"]
        ):
            raise RuntimeError("Feature-mode RSA results have mismatched time grids.")

    tf_map_data = _prepare_rsa_data_only(
        tfr,
        tfr_feature="power",
        log=True,
        standardize="features",
    )
    tf_map_result = rsa_nd(
        tf_map_data,
        model_rdm,
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.02},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=4.0,
        fmax=30.0,
        n_fft=64,
        n_per_seg=64,
    )
    psd_data = _prepare_rsa_data_only(
        spectrum,
        log=True,
        standardize="features",
    )
    psd_result = rsa_nd(
        psd_data,
        model_rdm,
        search_axes=("freq",),
        radii={"freq": 0.0},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )

    time_domain_result = mne_rsa.rsa_epochs(
        epochs,
        model_rdm,
        spatial_radius=None,
        temporal_radius=0.02,
        epochs_rdm_metric="correlation",
        rsa_metric="spearman",
        labels_epochs=item_labels,
        labels_rdm_model=item_labels,
        n_jobs=1,
        verbose=False,
    )
    time_domain_rsa = np.ravel(time_domain_result.data)
    time_domain_times = time_domain_result.times

    full_tfr_data = _prepare_rsa_data_only(
        tfr,
        tfr_feature="power",
        log=True,
        standardize="features",
    )
    full_tfr_result = rsa_nd(
        full_tfr_data,
        model_rdm,
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )

    print("TFR RSA shape:", tfr_result.rsa.shape)
    print("TFR freq x time RSA map shape:", tf_map_result.rsa.shape)
    print("TFR mode: total power (raw epochs; evoked response not subtracted)")
    all_peak_idx = int(np.nanargmax(tfr_result.rsa))
    all_peak_rsa = float(tfr_result.rsa[all_peak_idx])
    all_peak_time = float(time_centers[all_peak_idx])
    print(f"all bands peak RSA: {all_peak_rsa:.3f} at {all_peak_time:.3f} s")
    for band_name, band_values in separate_band_results.items():
        peak_idx = int(np.nanargmax(band_values))
        print(
            f"{band_name} band peak RSA: {band_values[peak_idx]:.3f} "
            f"at {time_centers[peak_idx]:.3f} s"
        )
    for mode_name, mode_values in feature_mode_results.items():
        peak_idx = int(np.nanargmax(mode_values))
        print(
            f"{mode_name} feature peak RSA: {mode_values[peak_idx]:.3f} "
            f"at {feature_mode_times[peak_idx]:.3f} s"
        )
    print("PSD RSA shape:", psd_result.rsa.shape)
    print("mne_rsa.rsa_epochs temporal RSA shape:", time_domain_rsa.shape)
    print("Full TFR feature RSA shape:", full_tfr_result.rsa.shape)
    print("Full TFR feature RSA value:", float(full_tfr_result.rsa))

    return {
        "sensor_type": sensor_type,
        "n_channels": len(epochs.ch_names),
        "time_centers": time_centers,
        "all_bands_rsa": tfr_result.rsa,
        "band_results": separate_band_results,
        "feature_mode_times": feature_mode_times,
        "feature_mode_results": feature_mode_results,
        "tf_times": tf_map_result.search_coords["time"],
        "tf_freqs": tf_map_result.search_coords["freq"],
        "tf_map_rsa": tf_map_result.rsa,
        "psd_freqs": psd_result.search_coords["freq"],
        "psd_rsa": psd_result.rsa,
        "time_domain_times": time_domain_times,
        "time_domain_rsa": time_domain_rsa,
        "all_peak_rsa": all_peak_rsa,
        "all_peak_time": all_peak_time,
        "full_tfr_rsa": float(full_tfr_result.rsa),
    }


def run_mixed_sensor_whitened_rsa(raw, events, *, n_per_condition=20):
    """Replicate the official mixed-sensor whitening example more closely."""

    epochs = load_mixed_audio_visual_epochs(
        raw,
        events,
        n_per_condition=n_per_condition,
    )
    model_rdm = condition_model_rdm(epochs)
    item_labels = np.arange(len(epochs))

    noise_cov = mne.compute_covariance(
        epochs,
        tmin=-0.2,
        tmax=0.0,
        method="shrunk",
        rank="info",
        verbose=False,
    )
    rsa_result = mne_rsa.rsa_epochs(
        epochs,
        model_rdm,
        noise_cov=noise_cov,
        spatial_radius=None,
        temporal_radius=0.02,
        epochs_rdm_metric="correlation",
        rsa_metric="spearman",
        labels_epochs=item_labels,
        labels_rdm_model=item_labels,
        n_jobs=1,
        verbose=False,
    )

    rsa_values = np.ravel(rsa_result.data)
    times = rsa_result.times
    peak_idx = int(np.nanargmax(rsa_values))
    print(
        "\n[mixed] mne_rsa.rsa_epochs with baseline noise covariance whitening: "
        f"shape={rsa_values.shape}, peak={rsa_values[peak_idx]:.3f} "
        f"at {times[peak_idx]:.3f} s"
    )
    return {
        "times": times,
        "rsa": rsa_values,
        "peak_rsa": float(rsa_values[peak_idx]),
        "peak_time": float(times[peak_idx]),
    }


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
        default=20,
        help="Number of trials per condition to keep. Default: 20.",
    )
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        default=True,
        help="Remove previous generated demo outputs before running. Default: on.",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Keep previous generated demo outputs.",
    )
    return parser.parse_args(argv)


def main(sensor_types=None, *, n_per_condition=20, clean_output=True):
    mne.set_log_level("WARNING")
    OUT_DIR.mkdir(exist_ok=True)
    if clean_output:
        removed = clean_output_dir(OUT_DIR)
        if removed:
            print(f"Cleaned {len(removed)} generated output files from {OUT_DIR}")

    selected = validate_sensor_types(
        SENSOR_TYPES if sensor_types is None else sensor_types
    )
    raw, events = load_sample_raw_events()
    mixed_result = run_mixed_sensor_whitened_rsa(
        raw,
        events,
        n_per_condition=n_per_condition,
    )
    sensor_results = [
        run_sensor_analysis(
            sensor_type,
            raw,
            events,
            n_per_condition=n_per_condition,
        )
        for sensor_type in selected
    ]

    composite_png = COMPOSITE_OUTPUT
    plot_output_composite(sensor_results, mixed_result=mixed_result, path=composite_png)
    print("\nComposite figure saved:")
    print(f"  {composite_png}")

    return sensor_results


if __name__ == "__main__":
    args = parse_args()
    main(
        args.sensor_types,
        n_per_condition=args.n_per_condition,
        clean_output=args.clean_output,
    )
