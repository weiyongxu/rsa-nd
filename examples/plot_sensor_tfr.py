"""
=========================
Time-frequency RSA on MEG
=========================

Which sensors, times and frequencies tell auditory and visual trials apart? This
example runs RSA on the time-frequency representation of single trials from the
MNE sample dataset.
"""

# %%
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
    raw, events, event_id, tmin=-0.2, tmax=0.5, picks="mag", decim=2, preload=True
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
    return_itc=False,
    average=False,
)

# %%
# Search frequency bands and time, using all sensors as features.
result = compute_rsa(
    tfr,
    model_rdm,
    freq={"theta": (6, 8), "alpha": (8, 13), "beta": (13, 30)},
    log=True,
    standardize="features",
    search_axes=("band", "time"),
    radii={"time": 0.04},
    labels=labels,
)
fig, ax = plt.subplots()
for band, scores in zip(result.coords["band"], result.rsa):
    ax.plot(result.coords["time"], scores, label=band)
ax.set(xlabel="Time (s)", ylabel="Spearman RSA")
ax.legend()

# %%
# A spatial searchlight over the alpha band returns an MNE container, so all the
# usual plotting works.
prepared = prepare_rsa_data(tfr, freq=(8, 13), log=True)
result = rsa_nd(
    prepared,
    model_rdm,
    search_axes=("space", "time"),
    radii={"space": 0.05, "time": 0.04},
    labels=labels,
)
evoked = result.to_mne()
evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type="mag", scalings=1, units="RSA")

# %%
# Phase carries information too. Complex coefficients can be turned into sine and
# cosine features, which is not possible with power alone.
complex_tfr = epochs.compute_tfr(
    "morlet", freqs=[8.0, 10.0, 12.0], n_cycles=2.0, output="complex", average=False
)
result = compute_rsa(
    complex_tfr,
    model_rdm,
    tfr_feature="phase",
    search_axes=("time",),
    radii={"time": 0.04},
    labels=labels,
)
fig, ax = plt.subplots()
ax.plot(result.coords["time"], result.rsa)
ax.set(xlabel="Time (s)", ylabel="Spearman RSA", title="Alpha phase")

plt.show()
