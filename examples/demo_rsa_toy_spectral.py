r"""Toy examples for the three-layer RSA pipeline
=============================================

Walks through the three layers of ``rsa_nd``:

  1. **Input preparation** via :func:`prepare_rsa_data` — convert MNE
     spectral inputs (or arrays) into a typed
     :class:`rsa_nd.PreparedRsaData`.
  2. **RSA core** via :func:`rsa_nd` — accepts the prepared object as a
     single positional and runs the within-subject RSA against a model RDM.
  3. **Output conversion** via ``result.to_mne()`` — turn the
     ``RsaResult`` into an MNE-shaped container when the geometry permits
     (raising for non-MNE axes such as grouped windows / bands).

Run with:

    uv run python examples/demo_rsa_toy_spectral.py
"""

from __future__ import annotations

import mne
import numpy as np

from rsa_nd import (
    prepare_rsa_data,
    rsa_nd,
)


def make_toy_epochs(n_epochs=8, n_channels=4, sfreq=100.0, duration=2.0):
    """Create small synthetic epochs with category-specific oscillations."""

    rng = np.random.default_rng(42)
    n_times = int(sfreq * duration)
    times = np.arange(n_times) / sfreq
    ch_names = [f"EEG {idx:03d}" for idx in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")

    labels = np.array(["negative"] * (n_epochs // 2) + ["positive"] * (n_epochs // 2))
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 0.2e-6

    for epoch_idx, label in enumerate(labels):
        if label == "negative":
            data[epoch_idx, 0] += 1.0e-6 * np.sin(2 * np.pi * 6.0 * times)
            data[epoch_idx, 1] += 0.8e-6 * np.sin(2 * np.pi * 10.0 * times)
        else:
            data[epoch_idx, 2] += 1.0e-6 * np.sin(2 * np.pi * 20.0 * times)
            data[epoch_idx, 3] += 0.8e-6 * np.sin(2 * np.pi * 14.0 * times)

    events = np.column_stack(
        [
            np.arange(n_epochs),
            np.zeros(n_epochs, dtype=int),
            np.arange(1, n_epochs + 1),
        ]
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={f"clip_{idx:02d}": idx + 1 for idx in range(n_epochs)},
        tmin=0.0,
        baseline=None,
    )
    return epochs, labels


def categorical_rdm(labels):
    """Simple model RDM: 0 within category, 1 across category."""

    labels = np.asarray(labels)
    return (labels[:, None] != labels[None, :]).astype(float)


def main():
    epochs, labels = make_toy_epochs()
    model_rdm = categorical_rdm(labels)
    item_labels = [f"clip_{idx:02d}" for idx in range(len(epochs))]

    bands = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    tfr_kwargs = {
        "method": "morlet",
        "freqs": np.arange(4.0, 31.0, 2.0),
        "average": False,
        "n_cycles": 2.0,
        "use_fft": True,
    }

    print("Toy epochs:", epochs.get_data().shape)

    tfr = epochs.compute_tfr(**tfr_kwargs)

    # ---- Example 1: TFR RSA over time, channel x band features ------------
    # Layer 1: typed input preparation.
    prepared_band = prepare_rsa_data(tfr, freq=bands, log=True)
    print(
        f"Layer 1: prepared.input_kind={prepared_band.input_kind!r}, "
        f"axes={prepared_band.rsa_data.axes}, "
        f"is_grouped={prepared_band.space.is_grouped}, "
        f"freq_axis={prepared_band.freq_axis!r} "
        f"(freq_is_grouped={prepared_band.freq_is_grouped})."
    )
    # Layer 2: RSA core. Reads prepared.space and prepared.item_labels.
    rsa_time = rsa_nd(
        prepared_band,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )
    # Layer 3: assemble. With ("time",) search and no space search, the
    # honest MNE container is a one-misc-channel `EvokedArray` carrying
    # the RSA-over-time curve.
    rsa_time_evoked = rsa_time.to_mne()
    print(
        f"Layer 2: rsa_time.rsa.shape={rsa_time.rsa.shape}; "
        f"Layer 3: assembled type={type(rsa_time_evoked).__name__}, "
        f"channels={getattr(rsa_time_evoked, 'ch_names', None)}."
    )
    # ---- Example 2: windowed TFR RSA -------------------------------------
    prepared_window = prepare_rsa_data(
        tfr,
        freq=bands,
        time={
            "early": (0.0, 0.7),
            "middle": (0.7, 1.4),
            "late": (1.4, 2.0),
        },
        log=True,
    )
    print(
        f"Layer 1 (windowed): time_axis={prepared_window.time_axis!r} "
        f"(time_is_grouped={prepared_window.time_is_grouped}), "
        f"freq_axis={prepared_window.freq_axis!r} "
        f"(freq_is_grouped={prepared_window.freq_is_grouped})."
    )
    rsa_windows = rsa_nd(
        prepared_window,
        model_rdm,
        search_axes=("window",),
        radii={"window": 0},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )
    # Grouped time -> "window" axis is not a native MNE time grid, so
    # to_mne() cannot represent it and raises; the RsaResult stays the
    # canonical object for non-MNE geometry.
    try:
        rsa_windows.to_mne()
    except ValueError as exc:
        print(
            f"Layer 2: rsa_windows.rsa.shape={rsa_windows.rsa.shape}; "
            f"Layer 3: to_mne() unsupported for grouped windows ({exc})."
        )
    # ---- Example 3: PSD frequency RSA ------------------------------------
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=4.0,
        fmax=30.0,
        n_fft=128,
        n_per_seg=128,
    )
    prepared_psd = prepare_rsa_data(spectrum, log=True)
    rsa_freq = rsa_nd(
        prepared_psd,
        model_rdm,
        search_axes=("freq",),
        radii={"freq": 0},
        labels=item_labels,
        labels_rdm_model=item_labels,
        data_rdm_metric="correlation",
        rsa_metric="spearman",
        n_jobs=1,
    )
    rsa_freq_assembled = rsa_freq.to_mne()
    print(
        f"Layer 2: rsa_freq.rsa.shape={rsa_freq.rsa.shape}; "
        f"Layer 3: assembled type={type(rsa_freq_assembled).__name__}."
    )
    print("PSD sample freqs:", rsa_freq.search_coords["freq"][:5], "...")

    # ---- Example 4: reuse the prepared object for another rsa_nd call ----
    reused = rsa_nd(
        prepared_band,
        model_rdm,
        search_axes=("time",),
        radii={"time": 0},
        labels=item_labels,
        labels_rdm_model=item_labels,
    )
    print(f"Reused PreparedRsaData -> rsa_nd result shape: {reused.rsa.shape}")


if __name__ == "__main__":
    main()
