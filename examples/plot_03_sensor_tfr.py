"""
=========================
Time-frequency RSA on MEG
=========================

Which sensors, times and frequencies tell auditory and visual trials apart? This
example runs RSA on the time-frequency representation of single trials from the
MNE sample dataset.
"""

# %%
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.spatial.distance import pdist

from rsa_nd import compute_rsa, prepare_rsa_data, rsa_nd

# %%
# Load 30 trials per condition.
data_path = mne.datasets.sample.data_path() / "MEG" / "sample"
raw = mne.io.read_raw_fif(data_path / "sample_audvis_filt-0-40_raw.fif")
events = mne.find_events(raw)
event_id = {"aud/left": 1, "aud/right": 2, "vis/left": 3, "vis/right": 4}
epochs = mne.Epochs(
    raw, events, event_id, tmin=-0.2, tmax=0.5, picks="mag", preload=True
)
epochs = mne.concatenate_epochs([epochs[name][:30] for name in event_id])

# %%
# The model RDM says that trials of the same modality are similar. Each trial is
# its own item, so we give every trial a unique label.
is_visual = np.isin(epochs.events[:, 2], [3, 4]).astype(float)
model_rdm = pdist(is_visual[:, np.newaxis])
labels = np.arange(len(epochs))

tfr = epochs.compute_tfr(
    "morlet",
    freqs=np.arange(6.0, 31.0, 2),
    n_cycles=2.0,
    decim=2,
    return_itc=False,
    average=False,
)

# %%
# Search frequency bands and time, using all sensors as features. All three bands
# tell the modalities apart in the first 300 ms, theta most strongly.
bands = {"theta": (6, 8), "alpha": (8, 13), "beta": (13, 30)}
band_colors = {"theta": "#6250d6", "alpha": "#1baf7a", "beta": "#eb6834"}
band_result = compute_rsa(
    tfr,
    model_rdm,
    freq=bands,
    log=True,
    standardize="features",
    search_axes=("band", "time"),
    radii={"time": 0.04},
    labels=labels,
)
times = band_result.coords["time"]

fig, ax = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
ax.axhline(0, color="0.85", linewidth=1, zorder=0)
ax.axvline(0, color="0.85", linewidth=1, zorder=0)
for idx, (band, scores) in enumerate(zip(band_result.coords["band"], band_result.rsa)):
    ax.plot(times, scores, color=band_colors[band], linewidth=2)
    low, high = bands[band]
    ax.text(0.98, 0.95 - idx * 0.11, f"{band}  {low}–{high} Hz", ha="right",
            va="top", color=band_colors[band], fontweight="bold",
            transform=ax.transAxes)  # fmt: skip
ax.spines[["top", "right"]].set_visible(False)
ax.set(xlabel="Time (s)", ylabel="Spearman r")
ax.set_title("All sensors, three bands", loc="left", fontweight="bold")

# %%
# A spatial searchlight over the alpha band returns an MNE container, so all the
# usual plotting works. At 100 ms, the scores are highest over the back of the
# head.
prepared = prepare_rsa_data(tfr, freq=(8, 13), log=True)
result = rsa_nd(
    prepared,
    model_rdm,
    search_axes=("space", "time"),
    radii={"space": 0.05, "time": 0.04},
    labels=labels,
)
evoked = result.to_mne()
limit = np.abs(evoked.data).max()
fig = evoked.plot_topomap(
    times=[0.1, 0.2, 0.3],
    ch_type="mag",
    scalings=1,
    units="r",
    cmap="RdBu_r",
    vlim=(-limit, limit),
    sensors=False,
    contours=0,
    extrapolate="head",
    time_unit="ms",
)
fig.suptitle("Alpha band, sensor searchlight", x=0.02, ha="left", fontweight="bold")

# %%
# Phase carries information too. Complex coefficients can be turned into sine and
# cosine features, which is not possible with power alone. Alpha phase tells the
# modalities apart too, less strongly than power but for longer.
complex_tfr = epochs.compute_tfr(
    "morlet",
    freqs=[8.0, 10.0, 12.0],
    n_cycles=2.0,
    decim=2,
    output="complex",
    average=False,
)
result = compute_rsa(
    complex_tfr,
    model_rdm,
    tfr_feature="phase",
    search_axes=("time",),
    radii={"time": 0.04},
    labels=labels,
)

fig, ax = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
ax.axhline(0, color="0.85", linewidth=1, zorder=0)
ax.axvline(0, color="0.85", linewidth=1, zorder=0)
alpha_power = band_result.rsa[list(bands).index("alpha")]
for idx, (name, scores, color) in enumerate(
    [("power", alpha_power, band_colors["alpha"]), ("phase", result.rsa, "#0b0b0b")]
):
    ax.plot(times, scores, color=color, linewidth=2)
    ax.text(0.98, 0.95 - idx * 0.11, name, ha="right", va="top", color=color,
            fontweight="bold", transform=ax.transAxes)  # fmt: skip
ax.spines[["top", "right"]].set_visible(False)
ax.set(xlabel="Time (s)", ylabel="Spearman r")
ax.set_title("Alpha band: power and phase", loc="left", fontweight="bold")

plt.show()
