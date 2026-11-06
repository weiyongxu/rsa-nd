r"""Source-level spatiotemporal spectral RSA demo
=============================================

Mirrors the dataset and ROI setup from the MNE-RSA source-level ROI tutorial:
https://mne.tools/mne-rsa/stable/auto_examples/source_level/plot_rsa_roi.html

Pipeline:

  1. Estimate source activity for each epoch.
  2. For each anatomical aparc parcel, compute spectral RSA from all source
     vertices inside the parcel, matching ``mne_rsa.rsa_stcs_rois`` semantics.
  3. Run a temporal searchlight inside each ROI vertex patch (radius =
     TEMPORAL_RADIUS), with vertices, frequencies, and spectral components
     flattened into the feature pattern.
  4. Render one composite figure: top-ROI RSA time courses on the left,
     ``mne_rsa.plot_roi_map`` 3-D brain rendering at peak time on the right.

Run with:

    uv run python examples/demo_rsa_source_roi_searchlight.py
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import mne_rsa
import numpy as np
from _support.rsa_outputs import clean_generated_outputs, examples_output_dir
from _support.rsa_sample import (
    EVENT_ID,
    RSA_TMAX,
    TEMPORAL_RADIUS,
    TFR_FREQS,
    load_balanced_audvis_epochs,
    sensitivity_metric,
)
from mne.viz import Brain

from rsa_nd import compute_rsa

OUT_DIR = examples_output_dir(__file__)
COMPOSITE_PNG = OUT_DIR / "source_roi_spatiotemporal_composite.png"
GENERATED_OUTPUT_PATTERNS = (
    "source_roi_spectral_*.png",
    "source_roi_spatiotemporal_*.png",
)

TOP_K_TIMECOURSES = 6
PHASE_NULL_PERMUTATIONS = 30
PHASE_NULL_RANDOM_SEED = 20260430


def clean_outputs(output_dir=OUT_DIR):
    """Remove previously generated demo outputs from one directory."""

    return clean_generated_outputs(output_dir, GENERATED_OUTPUT_PATTERNS)


def peak_info(rsa_vals, times, roi_names):
    """Return the global peak RSA value with its ROI and time coordinates."""

    peak_roi_idx, peak_time_idx = np.unravel_index(
        np.nanargmax(rsa_vals), np.asarray(rsa_vals).shape
    )
    peak_roi_idx = int(peak_roi_idx)
    peak_time_idx = int(peak_time_idx)
    return {
        "roi_idx": peak_roi_idx,
        "time_idx": peak_time_idx,
        "roi_name": roi_names[peak_roi_idx],
        "time": float(times[peak_time_idx]),
        "value": float(rsa_vals[peak_roi_idx, peak_time_idx]),
    }


def permutation_null_summary(observed_max, permutation_maxima):
    """Summarize a max-statistic permutation null distribution."""

    maxima = np.asarray(permutation_maxima, dtype=float)
    if maxima.ndim != 1 or maxima.size == 0:
        raise ValueError("permutation_maxima must be a non-empty 1-D array.")
    finite = maxima[np.isfinite(maxima)]
    if finite.size != maxima.size:
        raise ValueError("permutation_maxima must contain only finite values.")
    observed = float(observed_max)
    return {
        "n_permutations": int(maxima.size),
        "median": float(np.median(maxima)),
        "q95": float(np.quantile(maxima, 0.95)),
        "max": float(np.max(maxima)),
        "p_value": float((1 + np.sum(maxima >= observed)) / (maxima.size + 1)),
    }


def phase_permutation_maxima(
    stcs,
    events,
    rois,
    *,
    n_permutations=PHASE_NULL_PERMUTATIONS,
    seed=PHASE_NULL_RANDOM_SEED,
):
    """Compute max phase-RSA values after permuting condition labels."""

    rng = np.random.default_rng(seed)
    labels = np.arange(len(events))
    permuted_models = []
    for _ in range(n_permutations):
        permuted_events = events[rng.permutation(len(events))]
        permuted_models.append(
            mne_rsa.compute_rdm(
                permuted_events,
                metric=sensitivity_metric,
            )
        )
    permuted = compute_rsa(
        stcs,
        permuted_models,
        space=rois,
        tfr={"freqs": TFR_FREQS, "n_jobs": 1, "verbose": False},
        time=(float(stcs[0].times[0]), RSA_TMAX),
        tfr_feature="phase",
        log=False,
        standardize="features",
        search_axes=("space", "time"),
        radii={"time": TEMPORAL_RADIUS},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )
    if permuted.rsa.ndim != 3:
        raise RuntimeError(
            "Phase permutation RSA expected model x ROI x time output; "
            f"got shape {permuted.rsa.shape}."
        )
    return np.nanmax(permuted.rsa, axis=(1, 2))


def estimate_source_epochs(epochs, sample_root):
    """Apply the truncated inverse operator and keep vertex-level STCs."""

    testing_root = mne.datasets.testing.data_path(download=True, verbose=False)
    inv_path = (
        testing_root
        / "MEG"
        / "sample"
        / "sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif"
    )
    inv = mne.minimum_norm.read_inverse_operator(inv_path, verbose=False)

    subjects_dir = sample_root / "subjects"
    rois = mne.read_labels_from_annot(
        "sample",
        parc="aparc",
        subjects_dir=subjects_dir,
        verbose=False,
    )

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inv,
        lambda2=1.0 / 9.0,
        method="dSPM",
        pick_ori="normal",
        return_generator=False,
        verbose=False,
    )
    return stcs, rois, subjects_dir


def _plot_roi_map_split(values, rois, *, subject, subjects_dir, cmap="plasma"):
    """``mne_rsa.plot_roi_map`` reimplemented with ``hemi='split'`` so that
    lateral and medial views of both hemispheres render in a 2x2 grid.
    """

    cmap_fn = plt.get_cmap(cmap)
    max_val = float(np.nanmax(values))
    if not np.isfinite(max_val) or max_val == 0:
        max_val = 1.0

    brain = Brain(
        subject=subject,
        subjects_dir=subjects_dir,
        surf="inflated",
        hemi="split",
        views=["lateral", "medial"],
        size=(1200, 900),
        background="white",
    )
    labels_lh = np.zeros(len(brain.geo["lh"].coords), dtype=int)
    labels_rh = np.zeros(len(brain.geo["rh"].coords), dtype=int)
    ctab_lh: list[list[int]] = []
    ctab_rh: list[list[int]] = []
    for idx, (roi, value) in enumerate(zip(rois, values), 1):
        if roi.hemi == "lh":
            labels = labels_lh
            ctab = ctab_lh
        else:
            labels = labels_rh
            ctab = ctab_rh
        labels[roi.vertices] = idx
        ctab.append([int(x * 255) for x in cmap_fn(value / max_val)[:4]] + [idx])
    brain.add_annotation(
        [(labels_lh, np.array(ctab_lh)), (labels_rh, np.array(ctab_rh))],
        borders=False,
        alpha=1.0,
    )
    return brain


@contextmanager
def _offscreen_3d():
    """Render 3-D brains off-screen with MNE's ``notebook`` PyVista backend."""
    import pyvista

    previous_show = Brain.show
    pyvista.OFF_SCREEN = True
    mne.viz.set_3d_backend("notebook")
    # MNE calls Brain.show() while plotting, even for off-screen use.
    Brain.show = lambda self: None
    try:
        yield
    finally:
        Brain.show = previous_show
        pyvista.close_all()


def render_brain_at_peak(rsa_at_peak, rois, *, subjects_dir, dest):
    """Render a multi-view 3-D brain map and save it as a PNG at ``dest``.

    Rendering is off-screen so the demo also runs on machines without a
    working desktop OpenGL context.
    """

    with _offscreen_3d():
        brain = _plot_roi_map_split(
            rsa_at_peak,
            rois,
            subject="sample",
            subjects_dir=subjects_dir,
        )
        brain.save_image(str(dest))
        brain.close()


def _draw_timecourse_panel(
    ax, times, rsa_vals, roi_names, peak_roi_idx, peak_time, *, top_k, title
):
    roi_peaks = np.nanmax(rsa_vals, axis=1)
    top_order = list(np.argsort(roi_peaks)[::-1][:top_k])
    top_set = set(top_order)

    # All parcels in faint gray so the overall envelope is visible.
    for roi_idx in range(len(roi_names)):
        if roi_idx in top_set:
            continue
        ax.plot(times, rsa_vals[roi_idx], color="0.78", linewidth=0.6, zorder=1)

    # Top-K ROIs in color; peak ROI in bold red on top.
    cmap = plt.get_cmap("tab10")
    for rank, roi_idx in enumerate(top_order):
        is_peak = roi_idx == peak_roi_idx
        ax.plot(
            times,
            rsa_vals[roi_idx],
            linewidth=2.4 if is_peak else 1.4,
            linestyle="-" if is_peak else "--",
            color="tab:red" if is_peak else cmap(rank % 10),
            label=f"{roi_names[roi_idx]}" + (" (peak)" if is_peak else ""),
            zorder=4 if is_peak else 3,
        )
    ax.axvline(0, color="0.55", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=0.9)
    ax.axvline(peak_time, color="tab:red", linestyle=":", linewidth=0.9)
    ax.set(title=title, xlabel="Time (s)", ylabel="Spearman RSA")
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")


def _draw_brain_panel(fig, ax, brain_png_path, rsa_at_peak, peak_time, *, panel_title):
    ax.imshow(plt.imread(str(brain_png_path)))
    ax.set_axis_off()
    ax.set_title(panel_title)

    vmax = float(np.nanmax(rsa_at_peak))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax),
        cmap="plasma",
    )
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        label="Spearman RSA at peak time",
    )


def plot_feature_comparison_composite(
    times,
    feature_results,
    rois,
    subjects_dir,
    *,
    path=COMPOSITE_PNG,
    top_k=TOP_K_TIMECOURSES,
):
    """One composite figure with one row per feature mode (time course + brain)."""

    roi_names = [roi.name for roi in rois]
    n_modes = len(feature_results)

    # Render all brain PNGs first; matplotlib handles the layout afterwards.
    brain_tmp_paths = []
    for mode in feature_results:
        peak_time_idx = mode["peak_time_idx"]
        tmp = path.with_name(f"{path.stem}_brain_tmp_{mode['key']}.png")
        render_brain_at_peak(
            mode["rsa_vals"][:, peak_time_idx],
            rois,
            subjects_dir=subjects_dir,
            dest=tmp,
        )
        brain_tmp_paths.append(tmp)

    fig = plt.figure(figsize=(16, 6.5 * n_modes), constrained_layout=True)
    grid = fig.add_gridspec(n_modes, 2, width_ratios=[1.0, 1.1])

    for row, (mode, brain_png) in enumerate(zip(feature_results, brain_tmp_paths)):
        rsa_vals = mode["rsa_vals"]
        peak_roi_idx = mode["peak_roi_idx"]
        peak_time_idx = mode["peak_time_idx"]
        peak_time = float(times[peak_time_idx])
        peak_roi_name = roi_names[peak_roi_idx]
        null_summary = mode.get("null_summary")
        null_text = (
            f"; max-null p={null_summary['p_value']:.3f}"
            if null_summary is not None
            else ""
        )

        ax_tc = fig.add_subplot(grid[row, 0])
        _draw_timecourse_panel(
            ax_tc,
            times,
            rsa_vals,
            roi_names,
            peak_roi_idx,
            peak_time,
            top_k=top_k,
            title=(f"{mode['label']}: ROI time courses (top {top_k}{null_text})"),
        )

        ax_brain = fig.add_subplot(grid[row, 1])
        _draw_brain_panel(
            fig,
            ax_brain,
            brain_png,
            rsa_vals[:, peak_time_idx],
            peak_time,
            panel_title=(
                f"{mode['label']}: peak {peak_roi_name} at {peak_time:+.3f} s"
            ),
        )

    fig.suptitle(
        "Source-level spatiotemporal spectral RSA on MNE-sample "
        "(aparc, audio-left model): power vs phase features",
        fontsize=14,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    for tmp in brain_tmp_paths:
        tmp.unlink(missing_ok=True)


def main(*, n_per_condition=10):
    mne.set_log_level("WARNING")
    OUT_DIR.mkdir(exist_ok=True)
    removed = clean_outputs()
    if removed:
        print(f"Cleaned {len(removed)} previous demo output files.")

    epochs, sample_root = load_balanced_audvis_epochs(
        n_per_condition=n_per_condition,
    )
    print(
        f"Loaded {len(epochs)} epochs over {', '.join(EVENT_ID)} "
        f"at {epochs.info['sfreq']} Hz, "
        f"window [{epochs.tmin:.2f}, {epochs.tmax:.2f}] s."
    )

    stcs, rois, subjects_dir = estimate_source_epochs(epochs, sample_root)
    n_vertices, n_times = stcs[0].data.shape
    print(
        f"Source estimates: {len(stcs)} epochs, {n_vertices} vertices, "
        f"{n_times} time points. Running ROI vertex-patch RSA over "
        f"{len(rois)} aparc ROIs."
    )

    model_rdm = mne_rsa.compute_rdm(epochs.events[:, 2], metric=sensitivity_metric)
    item_labels = np.arange(len(epochs))
    print(f"Model RDM shape: {np.asarray(model_rdm).shape}")

    feature_modes = [
        {
            "key": "power",
            "label": "log10 power",
            "tfr_feature": "power",
            "log": True,
        },
        {
            "key": "phase",
            "label": "sin/cos phase",
            "tfr_feature": "phase",
            "log": False,
        },
    ]

    feature_results = []
    times = None
    analysis_rois = None
    roi_names = None
    for mode in feature_modes:
        print(
            f"\nRunning ROI vertex-patch spectral RSA with "
            f"tfr_feature={mode['tfr_feature']!r} (log={mode['log']})..."
        )
        result = compute_rsa(
            stcs,
            model_rdm,
            space=rois,
            tfr={"freqs": TFR_FREQS, "n_jobs": 1, "verbose": False},
            time=(float(epochs.tmin), RSA_TMAX),
            tfr_feature=mode["tfr_feature"],
            log=mode["log"],
            standardize="features",
            search_axes=("space", "time"),
            radii={"time": TEMPORAL_RADIUS},
            labels=item_labels,
            labels_rdm_model=item_labels,
            data_rdm_metric="correlation",
            rsa_metric="spearman",
            n_jobs=1,
        )
        rsa_vals = result.rsa  # (n_rois, n_times)
        result_times = np.asarray(result.search_coords["time"], dtype=float)
        result_roi_names = [str(name) for name in result.search_coords["space"]]
        if times is None:
            times = result_times
            roi_names = result_roi_names
            roi_by_name = {str(roi.name): roi for roi in rois}
            analysis_rois = [roi_by_name[name] for name in roi_names]
        elif not np.array_equal(times, result_times):
            raise RuntimeError("Feature-mode source RSA results have mismatched times.")
        elif roi_names != result_roi_names:
            raise RuntimeError("Feature-mode source RSA results have mismatched ROIs.")
        peak = peak_info(rsa_vals, times, roi_names)
        peak_roi_idx = peak["roi_idx"]
        peak_time_idx = peak["time_idx"]
        peak_rsa = peak["value"]
        peak_time = peak["time"]
        print(
            f"  RSA shape (n_rois x n_times): {rsa_vals.shape}; "
            f"peak={peak_rsa:+.3f} in {peak['roi_name']} at {peak_time:+.3f} s"
        )

        roi_peaks = np.nanmax(rsa_vals, axis=1)
        order = np.argsort(roi_peaks)[::-1][:TOP_K_TIMECOURSES]
        print(f"  Top {TOP_K_TIMECOURSES} ROIs by peak RSA:")
        for rank, roi_idx in enumerate(order, 1):
            peak_t_idx = int(np.nanargmax(rsa_vals[roi_idx]))
            print(
                f"    {rank:2d}. {roi_names[roi_idx]:>30s}  "
                f"peak={roi_peaks[roi_idx]:+.3f}  at {times[peak_t_idx]:+.3f} s"
            )

        null_summary = None
        if mode["key"] == "phase" and PHASE_NULL_PERMUTATIONS:
            print(
                f"  Computing phase max-statistic null "
                f"({PHASE_NULL_PERMUTATIONS} condition-label permutations)..."
            )
            permutation_maxima = phase_permutation_maxima(
                stcs,
                epochs.events[:, 2],
                analysis_rois,
            )
            null_summary = permutation_null_summary(peak_rsa, permutation_maxima)
            print(
                "  Phase max-null: "
                f"median={null_summary['median']:+.3f}, "
                f"95%={null_summary['q95']:+.3f}, "
                f"max={null_summary['max']:+.3f}, "
                f"p={null_summary['p_value']:.3f}. "
                "Interpret phase ROI peaks cautiously unless they exceed this null."
            )

        feature_results.append(
            {
                "key": mode["key"],
                "label": mode["label"],
                "rsa_vals": rsa_vals,
                "peak_roi_idx": peak_roi_idx,
                "peak_time_idx": peak_time_idx,
                "null_summary": null_summary,
            }
        )

    plot_feature_comparison_composite(
        times,
        feature_results,
        analysis_rois,
        subjects_dir,
    )
    print(f"\nSaved composite figure: {COMPOSITE_PNG}")


if __name__ == "__main__":
    main()
