r"""Compare plain Epochs RSA in mne-rsa and rsa-nd
==============================================

This demo uses the MNE sample audio/visual epochs directly, without any
frequency-domain transformation. It compares the dedicated
``mne_rsa.rsa_epochs`` implementation with the generic ``rsa_nd`` path:

    prepare_rsa_data(epochs) -> rsa_nd(...) -> result.to_mne()

Run with:

    uv run python examples/demo_rsa_epochs_implementation_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import mne_rsa
import numpy as np
from _support.rsa_outputs import examples_output_dir
from _support.rsa_sample import (
    SENSOR_LABELS,
    SENSOR_TYPES,
    condition_model_rdm,
    load_audio_visual_epochs,
    load_sample_raw_events,
    validate_sensor_types,
)

from rsa_nd import prepare_rsa_data, rsa_nd

OUT_DIR = examples_output_dir(__file__)
OUTPUT_PNG = OUT_DIR / "mne_sample_plain_epochs_rsa_comparison.png"

DEFAULT_SENSOR_TYPE = "eeg"
DEFAULT_N_PER_CONDITION = 20
MNE_RSA_SPATIAL_RADIUS = 1e-9
MNE_RSA_TEMPORAL_RADIUS = 0.02
RSA_ND_TEMPORAL_RADIUS = 0.020001


def align_to_reference_times(
    data: np.ndarray,
    source_times: np.ndarray,
    reference_times: np.ndarray,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Select columns from ``data`` whose source times match reference times."""

    arr = np.asarray(data)
    source = np.asarray(source_times, dtype=float)
    reference = np.asarray(reference_times, dtype=float)
    if source.ndim != 1 or reference.ndim != 1:
        raise ValueError("source_times and reference_times must be 1-D arrays.")
    if arr.shape[-1] != len(source):
        raise ValueError(
            "The last data dimension must match source_times "
            f"({arr.shape[-1]} vs {len(source)})."
        )

    indices = []
    for time in reference:
        matches = np.flatnonzero(np.isclose(source, time, rtol=0.0, atol=atol))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one source time matching {time}, "
                f"found {len(matches)}."
            )
        indices.append(int(matches[0]))
    return np.take(arr, indices, axis=-1)


def rsa_difference_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Summarize numeric agreement between two RSA result arrays."""

    reference_arr = np.asarray(reference, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)
    if reference_arr.shape != candidate_arr.shape:
        raise ValueError(
            "RSA arrays must have the same shape "
            f"({reference_arr.shape} vs {candidate_arr.shape})."
        )
    difference = candidate_arr - reference_arr
    abs_diff = np.abs(difference)
    finite = abs_diff[np.isfinite(abs_diff)]
    max_abs_diff = float(np.max(finite)) if finite.size else float("nan")
    return {
        "max_abs_diff": max_abs_diff,
        "allclose": bool(
            np.allclose(
                reference_arr,
                candidate_arr,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        ),
        "rtol": rtol,
        "atol": atol,
    }


def run_mne_rsa_epochs(epochs, model_rdm, item_labels):
    """Run the dedicated mne-rsa Epochs implementation."""

    return mne_rsa.rsa_epochs(
        epochs,
        model_rdm,
        spatial_radius=MNE_RSA_SPATIAL_RADIUS,
        temporal_radius=MNE_RSA_TEMPORAL_RADIUS,
        epochs_rdm_metric="correlation",
        rsa_metric="spearman",
        labels_epochs=item_labels,
        labels_rdm_model=item_labels,
        picks="data",
        n_jobs=1,
        verbose=False,
    )


def run_rsa_nd_epochs(epochs, model_rdm, item_labels):
    """Run the generic rsa-nd prepared-data RSA path."""

    prepared = prepare_rsa_data(epochs, picks="data")
    result = rsa_nd(
        prepared,
        model_rdm,
        search_axes=("space", "time"),
        radii={
            "space": MNE_RSA_SPATIAL_RADIUS,
            "time": RSA_ND_TEMPORAL_RADIUS,
        },
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )
    return result.to_mne(), result


def compare_epoch_rsa(epochs, *, sensor_type: str) -> dict[str, Any]:
    """Run both implementations and return aligned arrays for plotting."""

    item_labels = np.arange(len(epochs))
    model_rdm = condition_model_rdm(epochs)

    mne_rsa_evoked = run_mne_rsa_epochs(epochs, model_rdm, item_labels)
    rsa_nd_evoked, rsa_nd_result = run_rsa_nd_epochs(epochs, model_rdm, item_labels)

    mne_rsa_data = np.asarray(mne_rsa_evoked.data, dtype=float)
    rsa_nd_data = align_to_reference_times(
        np.asarray(rsa_nd_evoked.data, dtype=float),
        np.asarray(rsa_nd_result.search_coords["time"], dtype=float),
        np.asarray(mne_rsa_evoked.times, dtype=float),
    )
    difference = rsa_nd_data - mne_rsa_data
    summary = rsa_difference_summary(mne_rsa_data, rsa_nd_data)

    return {
        "sensor_type": sensor_type,
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "sensors": np.asarray(epochs.ch_names, dtype=object),
        "times": np.asarray(mne_rsa_evoked.times, dtype=float),
        "mne_rsa": mne_rsa_data,
        "rsa_nd": rsa_nd_data,
        "difference": difference,
        "summary": summary,
    }


def _symmetric_vmax(*arrays: np.ndarray, floor: float = 1e-12) -> float:
    values = np.concatenate([np.ravel(np.asarray(arr, dtype=float)) for arr in arrays])
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return floor
    return max(float(np.max(finite)), floor)


def _sensor_tick_positions(n_sensors: int, max_ticks: int = 8) -> np.ndarray:
    if n_sensors <= max_ticks:
        return np.arange(n_sensors)
    return np.unique(np.linspace(0, n_sensors - 1, max_ticks, dtype=int))


def _plot_heatmap(
    fig,
    ax,
    values: np.ndarray,
    *,
    times: np.ndarray,
    sensors: np.ndarray,
    title: str,
    vmax: float,
):
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], -0.5, len(sensors) - 0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    tick_pos = _sensor_tick_positions(len(sensors))
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(sensors[idx]) for idx in tick_pos], fontsize=7)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.set(title=title, xlabel="Time (s)", ylabel="Sensor")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def plot_comparison(result: dict[str, Any], *, path: Path = OUTPUT_PNG) -> None:
    """Save the one-row implementation comparison figure."""

    path = Path(path)
    path.parent.mkdir(exist_ok=True)

    times = np.asarray(result["times"], dtype=float)
    sensors = np.asarray(result["sensors"], dtype=object)
    mne_rsa_data = np.asarray(result["mne_rsa"], dtype=float)
    rsa_nd_data = np.asarray(result["rsa_nd"], dtype=float)
    difference = np.asarray(result["difference"], dtype=float)
    summary = result["summary"]

    rsa_vmax = _symmetric_vmax(mne_rsa_data, rsa_nd_data)
    diff_vmax = _symmetric_vmax(difference)

    fig = None
    try:
        fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
        axes = fig.subplots(1, 3)
        _plot_heatmap(
            fig,
            axes[0],
            mne_rsa_data,
            times=times,
            sensors=sensors,
            title="mne-rsa rsa_epochs",
            vmax=rsa_vmax,
        )
        _plot_heatmap(
            fig,
            axes[1],
            rsa_nd_data,
            times=times,
            sensors=sensors,
            title="rsa-nd",
            vmax=rsa_vmax,
        )
        _plot_heatmap(
            fig,
            axes[2],
            difference,
            times=times,
            sensors=sensors,
            title="rsa-nd - mne-rsa",
            vmax=diff_vmax,
        )
        sensor_label = SENSOR_LABELS.get(result["sensor_type"], result["sensor_type"])
        fig.suptitle(
            "Plain Epochs sensor-level RSA comparison "
            f"({sensor_label}; max_abs_diff={summary['max_abs_diff']:.3g}; "
            f"allclose={summary['allclose']})",
            fontsize=13,
        )
        fig.savefig(path, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close("all")


def print_summary(result: dict[str, Any], *, path: Path) -> None:
    """Print a compact command-line summary."""

    sensor_label = SENSOR_LABELS.get(result["sensor_type"], result["sensor_type"])
    summary = result["summary"]
    print(
        f"\n[{result['sensor_type']}] {result['n_epochs']} epochs, "
        f"{result['n_channels']} {sensor_label} channels"
    )
    print(f"RSA map shape: {result['mne_rsa'].shape}")
    print(
        "Agreement: "
        f"max_abs_diff={summary['max_abs_diff']:.6g}, "
        f"allclose={summary['allclose']} "
        f"(rtol={summary['rtol']}, atol={summary['atol']})"
    )
    print(f"Saved comparison figure: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor-type",
        choices=SENSOR_TYPES,
        default=DEFAULT_SENSOR_TYPE,
        help=f"Sensor type to compare. Default: {DEFAULT_SENSOR_TYPE}.",
    )
    parser.add_argument(
        "--n-per-condition",
        type=int,
        default=DEFAULT_N_PER_CONDITION,
        help=f"Trials per condition. Default: {DEFAULT_N_PER_CONDITION}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help=f"Output PNG path. Default: {OUTPUT_PNG}.",
    )
    return parser.parse_args(argv)


def main(
    *,
    sensor_type: str = DEFAULT_SENSOR_TYPE,
    n_per_condition: int = DEFAULT_N_PER_CONDITION,
    output: Path = OUTPUT_PNG,
) -> dict[str, Any]:
    mne.set_log_level("WARNING")
    validate_sensor_types([sensor_type])
    raw, events = load_sample_raw_events()
    epochs = load_audio_visual_epochs(
        raw,
        events,
        sensor_type=sensor_type,
        n_per_condition=n_per_condition,
    )

    result = compare_epoch_rsa(epochs, sensor_type=sensor_type)
    plot_comparison(result, path=output)
    print_summary(result, path=output)
    return result


if __name__ == "__main__":
    args = parse_args()
    main(
        sensor_type=args.sensor_type,
        n_per_condition=args.n_per_condition,
        output=args.output,
    )
