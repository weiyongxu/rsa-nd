"""Simulation: choosing between rsa_dynamic and rsa_dynamic_windows
================================================================

This script shows the two main dynamic-RSA workflows:

1. rsa_dynamic:
   Use this when the full model-time x neural-time matrix is small enough
   to compute directly, and when you already have one model RDM per time
   point. This is the natural choice for short event-locked clips or trial
   epochs.

2. rsa_dynamic_windows:
   Use this when the recording is long and a full temporal-generalization
   matrix would be expensive. The helper draws matched random windows from
   neural data and a scalar model signal, builds per-window model RDMs, runs
   rsa_dynamic internally, and averages the relative-time result.

Ground truth in both examples: neural data lag the model by TRUE_LAG seconds.
The generalization matrix should show a ridge parallel to the diagonal, offset
by TRUE_LAG on the neural-time axis.

Run with:
    uv run python examples/demo_rsa_dynamic_simulation.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import mne
import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.distance import pdist, squareform

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsa_nd import RsaData, rsa_dynamic, rsa_dynamic_windows


def dynamic_rdms_from_scalar_signal(
    signal: np.ndarray,
    *,
    metric: str = "cityblock",
) -> np.ndarray:
    """Build one model RDM per time point from an item-by-time signal."""

    signal = np.asarray(signal, dtype=float)
    n_items, n_times = signal.shape
    rdms = np.empty((n_times, n_items, n_items), dtype=float)
    for time_idx in range(n_times):
        rdms[time_idx] = squareform(pdist(signal[:, [time_idx]], metric=metric))
    return rdms


def finite_row_mean(values: np.ndarray) -> np.ndarray:
    """Mean over columns, leaving rows with no finite values as nan."""

    means = np.full(values.shape[0], np.nan)
    for row_idx, row in enumerate(values):
        finite = np.isfinite(row)
        if finite.any():
            means[row_idx] = row[finite].mean()
    return means


def symmetric_vmax(values: np.ndarray, percentile: float = 98.0) -> float:
    """Robust symmetric color limit for RSA matrices."""

    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    return vmax if vmax > 0 else 1.0


# -- Parameters ----------------------------------------------------------------
N_CLIPS = 25
TRUE_LAG = 2.0
RNG_SEED = 42
OUTPUT_PATH = (
    Path(__file__).with_name("output") / "sim_dynamic_rsa_direct_vs_windows.png"
)

# Short direct-rsa_dynamic example.
SHORT_DURATION = 8.0
SHORT_MODEL_SFREQ = 10.0
SHORT_NEURAL_SFREQ = 50.0
SHORT_CHANNELS = 16
SHORT_NOISE = 0.30
LAG_GRID = np.arange(0.0, 4.01, 0.25)

# Long windowed-rsa_dynamic_windows example.
CLIP_DURATION = 60.0
EEG_SFREQ = 250.0
EDA_SFREQ = 1000.0
RSA_EEG_SFREQ = 50.0
RSA_EDA_SFREQ = 10.0
N_CHANNELS = 32
WINDOW_S = 10.0
N_WINDOWS = 100
SNR = 1.0


rng = np.random.default_rng(RNG_SEED)
mne.set_log_level("WARNING")
t0 = time.time()

# -- Step 1: Direct dynamic RSA on a short event-like epoch --------------------
print("Step 1/5  Running direct rsa_dynamic on a short epoch...")

short_rng = np.random.default_rng(RNG_SEED + 1)
short_model_times = np.arange(int(SHORT_DURATION * SHORT_MODEL_SFREQ))
short_model_times = short_model_times / SHORT_MODEL_SFREQ
short_neural_times = np.arange(int(SHORT_DURATION * SHORT_NEURAL_SFREQ))
short_neural_times = short_neural_times / SHORT_NEURAL_SFREQ

# Smooth random item trajectories make each model-time RDM distinctive.
kernel_x = np.linspace(-3.0, 3.0, 15)
kernel = np.exp(-0.5 * kernel_x**2)
kernel /= kernel.sum()
raw = short_rng.standard_normal((N_CLIPS, short_model_times.size))
short_model_signal = np.vstack([np.convolve(row, kernel, mode="same") for row in raw])
short_model_signal -= short_model_signal.mean(axis=0, keepdims=True)
short_model_signal /= short_model_signal.std(axis=0, keepdims=True) + 1e-12

short_driver = np.vstack(
    [
        np.interp(
            short_neural_times - TRUE_LAG,
            short_model_times,
            short_model_signal[item_idx],
            left=0.0,
            right=0.0,
        )
        for item_idx in range(N_CLIPS)
    ]
)

short_spatial = short_rng.standard_normal(SHORT_CHANNELS)
short_spatial /= np.linalg.norm(short_spatial)
short_neural_data = short_spatial[None, :, None] * short_driver[:, None, :]
short_neural_data += SHORT_NOISE * short_rng.standard_normal(short_neural_data.shape)

short_neural = RsaData(
    data=short_neural_data,
    axes=("item", "space", "time"),
    coords={
        "space": np.arange(SHORT_CHANNELS),
        "time": short_neural_times,
    },
)
short_model_rdms = dynamic_rdms_from_scalar_signal(short_model_signal)

direct = rsa_dynamic(
    short_neural,
    model_rdms=short_model_rdms,
    model_times=short_model_times,
    mode="generalization",
    dynamic_axis="time",
    search_axes=("time",),
    radii={"time": 0.0},
    data_rdm_metric="euclidean",
    rsa_metric="spearman",
)
direct_lagged = rsa_dynamic(
    short_neural,
    model_rdms=short_model_rdms,
    model_times=short_model_times,
    mode="lagged",
    dynamic_axis="time",
    lags=LAG_GRID,
    search_axes=("time",),
    radii={"time": 0.0},
    data_rdm_metric="euclidean",
    rsa_metric="spearman",
)
direct_lag_scores = finite_row_mean(direct_lagged.rsa)
best_direct_lag = LAG_GRID[int(np.nanargmax(direct_lag_scores))]
print(
    f"  rsa_dynamic matrix: {direct.rsa.shape} | "
    f"best lag from mode='lagged': {best_direct_lag:.2f} s"
)

# -- Step 2: EDA simulation (NeuroKit2) ----------------------------------------
print("Step 2/5  Simulating EDA (NeuroKit2)...")
n_eda_full = int(CLIP_DURATION * EDA_SFREQ)
eda_full = np.zeros((N_CLIPS, n_eda_full))
for clip_idx in range(N_CLIPS):
    sig = np.asarray(
        nk.eda_simulate(
            duration=int(CLIP_DURATION),
            sampling_rate=int(EDA_SFREQ),
            scr_number=int(rng.integers(3, 12)),
            noise=0.02,
            random_state=clip_idx * 17 + 3,
        )
    )
    eda_full[clip_idx] = sig[:n_eda_full]

eda_ds = resample_poly(eda_full, 1, int(EDA_SFREQ / RSA_EDA_SFREQ), axis=1)
eda_eeg = resample_poly(eda_full, 1, int(EDA_SFREQ / EEG_SFREQ), axis=1)
n_eda_t = eda_ds.shape[1]
n_eeg_full = eda_eeg.shape[1]
print(
    f"  EDA: {eda_ds.shape} at {RSA_EDA_SFREQ} Hz | "
    f"driver: {eda_eeg.shape} at {EEG_SFREQ} Hz"
)

# -- Step 3: EEG simulation (MNE) ---------------------------------------------
print("Step 3/5  Simulating EEG (MNE)...")
lag_samp = int(TRUE_LAG * EEG_SFREQ)

# Lagged EDA drives EEG; global normalization preserves inter-clip ranks.
all_lagged = np.zeros((N_CLIPS, n_eeg_full))
for clip_idx in range(N_CLIPS):
    all_lagged[clip_idx, lag_samp:] = eda_eeg[clip_idx, : n_eeg_full - lag_samp]
all_lagged /= all_lagged.std() + 1e-9

# Shared spatial pattern so Euclidean distance between clips reflects EDA.
spatial = rng.standard_normal(N_CHANNELS)
spatial /= np.linalg.norm(spatial)

eeg_clean = np.zeros((N_CLIPS, N_CHANNELS, n_eeg_full))
for clip_idx in range(N_CLIPS):
    eeg_clean[clip_idx] = spatial[:, None] * all_lagged[clip_idx, None, :] * 2e-6

ch_names = [f"EEG{i:03d}" for i in range(N_CHANNELS)]
info = mne.create_info(ch_names, sfreq=EEG_SFREQ, ch_types="eeg")
epochs = mne.EpochsArray(eeg_clean.copy(), info, tmin=0.0, baseline=None)
mne.set_eeg_reference(epochs, ref_channels="average", projection=True, verbose=False)
epochs.apply_proj()

noise_cov = mne.make_ad_hoc_cov(info)
noise_std_mne = float(np.sqrt(np.diag(noise_cov.data).mean()))
epochs._data *= (SNR * noise_std_mne) / (eeg_clean.std() + 1e-12)
mne.simulation.add_noise(epochs, noise_cov, random_state=RNG_SEED)

epochs_ds = epochs.copy().resample(RSA_EEG_SFREQ)
eeg_ds = epochs_ds.get_data()
n_eeg_t = eeg_ds.shape[2]
print(f"  EEG: {eeg_ds.shape} at {RSA_EEG_SFREQ} Hz")

# -- Step 4: Random-window dynamic RSA via rsa_dynamic_windows -----------------
print(f"Step 4/5  Running {N_WINDOWS} random windows (rsa_dynamic_windows)...")

win_eeg = int(WINDOW_S * RSA_EEG_SFREQ)
win_eda = int(WINDOW_S * RSA_EDA_SFREQ)
eeg_rel_t = np.arange(win_eeg) / RSA_EEG_SFREQ
eda_rel_t = np.arange(win_eda) / RSA_EDA_SFREQ

neural = RsaData(
    data=eeg_ds,
    axes=("item", "space", "time"),
    coords={"time": np.arange(n_eeg_t) / RSA_EEG_SFREQ},
)
eda_times = np.arange(n_eda_t) / RSA_EDA_SFREQ

# The wrapper draws windows, builds EDA model RDMs, runs RSA, and averages.
windowed = rsa_dynamic_windows(
    neural,
    model_signal=eda_ds,
    model_times=eda_times,
    window_s=WINDOW_S,
    n_windows=N_WINDOWS,
    mode="generalization",
    model_metric="cityblock",
    data_rdm_metric="euclidean",
    rsa_metric="spearman",
    random_state=RNG_SEED,
)

window_gen = windowed.rsa
print(f"  RSA range: [{window_gen.min():.3f}, {window_gen.max():.3f}]")

# -- Step 5: Plot --------------------------------------------------------------
print("Step 5/5  Plotting...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: rsa_dynamic full temporal generalization.
ax = axes[0, 0]
vmax = symmetric_vmax(direct.rsa)
im = ax.imshow(
    direct.rsa,
    origin="lower",
    aspect="auto",
    extent=[
        short_neural_times[0],
        short_neural_times[-1],
        short_model_times[0],
        short_model_times[-1],
    ],
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax, label="Spearman r")
ax.plot(
    [TRUE_LAG, SHORT_DURATION],
    [0, SHORT_DURATION - TRUE_LAG],
    "k--",
    lw=1.5,
    label=f"Expected ridge (lag = {TRUE_LAG} s)",
)
ax.set_xlabel("Neural time (s)")
ax.set_ylabel("Model time (s)")
ax.set_title("rsa_dynamic: full short-epoch generalization")
ax.legend(fontsize=9)

# Panel B: rsa_dynamic mode='lagged' is the compact latency scan.
ax = axes[0, 1]
ax.plot(LAG_GRID, direct_lag_scores, marker="o", lw=1.5)
ax.axvline(TRUE_LAG, color="red", ls="--", lw=1.5, label="True lag")
ax.axvline(best_direct_lag, color="black", ls=":", lw=1.5, label="Best lag")
ax.set_xlabel("Candidate lag (s)")
ax.set_ylabel("Mean RSA over valid neural times")
ax.set_title("rsa_dynamic: lagged mode for a small latency grid")
ax.legend(fontsize=9)

# Panel C: rsa_dynamic_windows averaged relative-time generalization.
ax = axes[1, 0]
vmax = symmetric_vmax(window_gen)
im = ax.imshow(
    window_gen,
    origin="lower",
    aspect="auto",
    extent=[eeg_rel_t[0], eeg_rel_t[-1], eda_rel_t[0], eda_rel_t[-1]],
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax, label="Spearman r")
ax.plot(
    [TRUE_LAG, WINDOW_S],
    [0, WINDOW_S - TRUE_LAG],
    "k--",
    lw=1.5,
    label=f"Expected ridge (lag = {TRUE_LAG} s)",
)
ax.set_xlabel("EEG relative time (s)")
ax.set_ylabel("EDA relative time (s)")
ax.set_title(
    "rsa_dynamic_windows: averaged random windows\n"
    f"(N = {N_CLIPS} clips, {N_WINDOWS} x {WINDOW_S:.0f}-s windows, SNR = {SNR})"
)
ax.legend(fontsize=9)

# Panel D: lag recovery from the windowed generalization matrix.
ax = axes[1, 1]
est_lags = eeg_rel_t[np.argmax(window_gen, axis=1)] - eda_rel_t
inner = slice(int(eda_rel_t.size * 0.05), int(eda_rel_t.size * 0.70))
ax.plot(eda_rel_t[inner], est_lags[inner], lw=1.5, label="Estimated lag")
ax.axhline(TRUE_LAG, color="red", ls="--", lw=1.5, label="True lag")
ax.set_xlabel("EDA relative time (s)")
ax.set_ylabel("Estimated EEG lag (s)")
ax.set_title("Windowed lag recovery")
ax.legend(fontsize=9)
ax.set_ylim(-1, 5)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {OUTPUT_PATH}")
print(f"Total: {time.time() - t0:.1f} s")
