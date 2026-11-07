"""
====================================
Group statistics across participants
====================================

In the LIMO EEG dataset, each participant saw two faces, A and B, about 1000
times. Each image was shown at one of 18 levels of phase coherence, from noise to
a clear face. This example runs RSA for each participant, with a channel
searchlight over time, for three models. Then it tests the scores across
participants with a cluster permutation test from MNE-Python.

To keep the documentation build short, it uses 10 of the 18 participants.
"""

# %%
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy.spatial.distance import squareform

from rsa_nd import compute_rsa


# %%
# **Items.** Trials with the same face and the same phase coherence are repeats of
# one item, so each participant has 2 × 18 = 36 items. Items 0 to 17 are face A,
# from the lowest to the highest coherence, and items 18 to 35 are face B. Given
# these labels, ``mne-rsa`` averages the trials of each item before it compares
# the items.
def item_labels(epochs):
    """Give each trial the number of its item."""
    is_face_b = (epochs.metadata["face"] == "B").to_numpy()
    _, level = np.unique(epochs.metadata["phase-coherence"], return_inverse=True)
    return 18 * is_face_b + level


# %%
# **Models.** Each model RDM says how much each pair of items should differ.
#
# - *Face identity*: items that show different faces differ, but only as far as
#   the noisier image of the pair allows. At the lowest coherence, both faces are
#   the same kind of noise, so they should not differ at all.
# - *Phase coherence*: items differ more when their coherence differs more.
# - *Clarity*: a control. Two clear images may differ more than two noisy ones,
#   whichever faces they show. Without this control, the face model could pick
#   that up instead.
face = np.repeat([0, 1], 18)
coherence = np.tile(np.arange(18), 2)
i, j = np.triu_indices(36, k=1)  # every pair of items, in the order of an RDM
clarity = np.minimum(coherence[i], coherence[j])  # the noisier image of each pair
models = {
    "Face identity": (face[i] != face[j]) * clarity,
    "Phase coherence": np.abs(coherence[i] - coherence[j]),
    "Clarity": clarity,
}

fig, axs = plt.subplots(1, 3, figsize=(8, 2.9), layout="constrained")
for ax, (name, rdm) in zip(axs, models.items()):
    ax.imshow(squareform(rdm) / rdm.max(), cmap="RdBu_r", vmin=-0.5, vmax=1.5)
    ax.axhline(17.5, color="white", linewidth=1)
    ax.axvline(17.5, color="white", linewidth=1)
    ax.set_xticks([8.5, 26.5], ["Face A", "Face B"])
    ax.set_yticks([8.5, 26.5], ["Face A", "Face B"] if ax is axs[0] else [])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(name, loc="left", fontweight="bold")

# %%
# **Each participant.** Interpolating the bad channels gives every participant
# the same 128 EEG channels. The searchlight covers 40 mm and 20 ms around each
# channel and time. ``search_axes=("time", "space")`` returns the scores of each
# model as times × channels, the layout that MNE's cluster test expects.
#
# The models overlap: for example, pairs of clear images score high on both the
# face and the clarity model. A partial Spearman correlation scores each model
# while controlling for the others, which works because all models are scored in
# the same call.
per_participant = []
for subject in range(1, 11):
    epochs = mne.datasets.limo.load_data(subject=subject, update_path=True)
    epochs.pick("eeg").interpolate_bads()
    epochs.crop(-0.1, 0.4)
    result = compute_rsa(
        epochs,
        list(models.values()),
        model_names=list(models),
        search_axes=("time", "space"),
        radii={"time": 0.02, "space": 0.04},
        labels=item_labels(epochs),
        rsa_metric="partial-spearman",
    )
    per_participant.append(result.rsa)  # models × times × channels
print(result)

# One array per model, of participants × times × channels
scores = dict(zip(models, np.stack(per_participant, axis=1)))

# %%
# **Statistics.** For each model, a one-sample cluster permutation test asks where
# the scores are above zero across participants. Neighbouring channels and
# neighbouring times can join one cluster. The test is one-sided, because only a
# positive score means that the data look like the model.
adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, "eeg")
significant = {}
for name, score in scores.items():
    _, clusters, p_values, _ = permutation_cluster_1samp_test(
        score, adjacency=adjacency, tail=1, rng=0
    )
    significant[name] = np.zeros(score.shape[1:], dtype=bool)
    for cluster, p_value in zip(clusters, p_values):
        if p_value < 0.05:
            significant[name][cluster] = True
    print(f"{name}: clusters with p < 0.05: {p_values[p_values < 0.05]}")

# %%
# **When.** The scores averaged over the channels in each model's cluster, for
# each participant (thin lines) and for the group (thick line). The bar marks the
# times in the cluster. Note the different scales: face identity explains much
# less of the data than phase coherence. The clarity control is left out of the
# figures.
#
# Both clusters start too early: the face identity cluster even starts before
# the images appear. Before the images, the scores are no higher than chance, but
# the cluster takes in neighbouring points that pass the threshold by chance. A
# cluster test shows that an effect exists, but not exactly when or where it
# starts (Sassenhagen & Draschkow, 2019, *Psychophysiology*).
times = result.coords["time"]
colors = {"Face identity": "#0b0b0b", "Phase coherence": "#2a78d6"}
fig, axs = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
axs[0].set_ylabel("Spearman r")
for ax, name in zip(axs, colors):
    channels = significant[name].any(axis=0)
    cluster_mean = scores[name][:, :, channels].mean(axis=2)  # participants × times
    ax.axhline(0, color="0.85", linewidth=1, zorder=0)
    ax.axvline(0, color="0.85", linewidth=1, zorder=0)
    ax.plot(times, cluster_mean.T, color=colors[name], linewidth=0.7, alpha=0.4)
    ax.plot(times, cluster_mean.mean(axis=0), color=colors[name], linewidth=2.5)
    in_cluster = significant[name].any(axis=1)
    bar = ax.get_xaxis_transform()  # x in seconds, y as a fraction of the height
    ax.plot(times, np.where(in_cluster, 0.03, np.nan), color=colors[name],
            linewidth=4, solid_capstyle="butt", transform=bar)  # fmt: skip
    ax.text(times[in_cluster][0] - 0.01, 0.03, "cluster", ha="right", va="center",
            color=colors[name], transform=bar)  # fmt: skip
    ax.margins(y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_title(name, loc="left", fontweight="bold")

# %%
# **Where.** The group mean scores at five times. Each row has its own colour
# scale. Both clusters cover nearly every channel, so these maps show where the
# scores are strongest, not where they are significant. Both models are strongest
# over the back of the head, where earlier studies of these data also found the
# strongest effects of phase coherence (Rousselet et al., 2008, *BMC
# Neuroscience*). Face identity here means which of the two images was shown, so
# any difference between the two images can drive it.
show = [0.1, 0.15, 0.2, 0.25, 0.3]
fig = plt.figure(figsize=(8, 4), layout="constrained")
for subfig, name in zip(fig.subfigures(2, 1), colors):
    axs = subfig.subplots(1, len(show) + 1, width_ratios=[1] * len(show) + [0.08])
    group_mean = scores[name].mean(axis=0)  # times × channels
    limit = np.abs(group_mean[(times >= show[0]) & (times <= show[-1])]).max()
    evoked = mne.EvokedArray(group_mean.T, epochs.info, tmin=times[0])
    evoked.plot_topomap(
        times=show,
        axes=axs,
        ch_type="eeg",
        scalings=1,
        units="r",
        cbar_fmt="%g",
        cmap="RdBu_r",
        vlim=(-limit, limit),
        sensors=False,
        contours=0,
        time_unit="ms",
    )
    subfig.suptitle(name, x=0.01, ha="left", fontweight="bold")

plt.show()
