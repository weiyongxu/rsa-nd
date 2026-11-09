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
# Compute single-trial source estimates for 20 trials per condition: tones in the
# left ear and checkerboards in the left visual field.
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
    tfr={"freqs": [8.0, 10.0, 12.0], "n_cycles": 2.0, "decim": 2},
    log=True,
    search_axes=("space", "time"),
    radii={"time": 0.04},
    labels=np.arange(len(stcs)),
)
print(result)

# %%
# Plot each region, in each hemisphere. The checkerboards were in the left visual
# field, which projects to the right hemisphere, so the right visual regions tell
# the modalities apart best.
scores = dict(zip(result.coords["space"], result.rsa))
times = result.coords["time"]
regions = {
    "Auditory cortex": {
        "transversetemporal": "Transverse temporal",
        "superiortemporal": "Superior temporal",
    },
    "Visual cortex": {
        "pericalcarine": "Pericalcarine",
        "lateraloccipital": "Lateral occipital",
    },
}
hemis = {"lh": ("left", "#2a78d6"), "rh": ("right", "#eb6834")}

fig = plt.figure(figsize=(7.5, 2.9), layout="constrained")
axs = []
for subfig, (group, names) in zip(fig.subfigures(1, 2), regions.items()):
    subfig.suptitle(group, fontweight="bold")
    for ax, (name, title) in zip(subfig.subplots(1, 2), names.items()):
        if axs:
            ax.sharey(axs[0])
        axs.append(ax)
        ax.axhline(0, color="0.85", linewidth=1, zorder=0)
        ax.axvline(0, color="0.85", linewidth=1, zorder=0)
        for hemi, (_, color) in hemis.items():
            ax.plot(times, scores[f"{name}-{hemi}"], color=color, linewidth=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize="medium")
fig.supxlabel("Time (s)", fontsize="medium")
axs[0].set_ylabel("Spearman r")
for idx, (side, color) in enumerate(hemis.values()):
    axs[1].text(0.03, 0.97 - idx * 0.12, f"{side} hemisphere", color=color, size=9,
                fontweight="bold", va="top", transform=axs[1].transAxes)  # fmt: skip

plt.show()
