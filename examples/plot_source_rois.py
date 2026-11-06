"""
===================================
Time-frequency RSA in brain regions
===================================

This example runs RSA on source estimates, using anatomical labels as searchlight
patches: each label gives one score per time point.
"""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from scipy.spatial.distance import pdist

from rsa_nd import compute_rsa

# %%
# Compute single-trial source estimates for 20 trials per condition.
data_path = mne.datasets.sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw = mne.io.read_raw_fif(meg_path / "sample_audvis_filt-0-40_raw.fif")
events = mne.find_events(raw)
epochs = mne.Epochs(
    raw,
    events,
    {"al": 1, "ar": 2, "vl": 3, "vr": 4},
    tmin=-0.2,
    tmax=0.5,
    picks="meg",
    decim=2,
    preload=True,
)
epochs = mne.concatenate_epochs([epochs[name][:20] for name in ("al", "vl")])

inverse = read_inverse_operator(meg_path / "sample_audvis-meg-oct-6-meg-inv.fif")
stcs = apply_inverse_epochs(epochs, inverse, lambda2=1 / 9, pick_ori="normal")

# %%
# Use auditory and visual cortex labels. The model RDM again compares modalities.
labels = mne.read_labels_from_annot(
    "sample",
    "aparc",
    regexp="transversetemporal|superiortemporal|pericalcarine|lateraloccipital",
    subjects_dir=data_path / "subjects",
)
is_visual = (epochs.events[:, 2] == 3).astype(float)
model_rdm = pdist(is_visual[:, np.newaxis])

# %%
# Each label is one searchlight patch: all its vertices, at all frequencies in the
# time window, are the features.
result = compute_rsa(
    stcs,
    model_rdm,
    space=labels,
    tfr={"freqs": [8.0, 10.0, 12.0], "n_cycles": 2.0},
    log=True,
    search_axes=("space", "time"),
    radii={"time": 0.04},
    labels=np.arange(len(stcs)),
)
print(result)

fig, ax = plt.subplots()
for name, scores in zip(result.coords["space"], result.rsa):
    ax.plot(result.coords["time"], scores, label=name)
ax.set(xlabel="Time (s)", ylabel="Spearman RSA")
ax.legend(fontsize="small")

plt.show()
