"""
================================
RSA over any combination of axes
================================

This example uses simulated data to show the central idea of ``rsa-nd``: name
the axes of your data, then choose which ones to search over. Every other axis
provides the features of each searchlight patch.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from rsa_nd import prepare_rsa_data, rsa_nd

# %%
# Simulate data with shape items x channels x frequencies x times. From 8 to 13 Hz
# and from 0.2 to 0.4 s, the channels follow one value per item.
rng = np.random.default_rng(0)
n_items, n_channels = 30, 12
freqs = np.arange(4.0, 32.0, 2.0)
times = np.arange(0, 0.6, 0.02)
item_values = rng.standard_normal(n_items)

data = rng.standard_normal((n_items, n_channels, len(freqs), len(times)))
pattern = rng.standard_normal(n_channels)
in_band = (freqs >= 8) & (freqs < 14)
in_window = (times >= 0.2) & (times < 0.4)
effect = 2 * item_values[:, np.newaxis] * pattern
data[:, :, in_band[:, np.newaxis] & in_window] += effect[:, :, np.newaxis]

axes = ("item", "space", "freq", "time")
coords = {"freq": freqs, "time": times}
model_rdm = pdist(item_values[:, np.newaxis])

# %%
# Search over frequency and time. The channels are not searched, so each position
# compares the items using the pattern across all channels.
result = rsa_nd(
    data,
    model_rdm,
    axes=axes,
    coords=coords,
    search_axes=("freq", "time"),
    radii={"freq": 2, "time": 0.04},
)
print(result)

# %%
# The scores are high where the effect was simulated (dashed box). The radii
# spread each score over 2 Hz and 40 ms, which blurs the edges of the box.
fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
limit = np.abs(result.rsa).max()
image = ax.pcolormesh(times, freqs, result.rsa, cmap="RdBu_r", vmin=-limit, vmax=limit)
step_t, step_f = times[1] - times[0], freqs[1] - freqs[0]
t, f = times[in_window], freqs[in_band]
box = plt.Rectangle(
    (t[0] - step_t / 2, f[0] - step_f / 2),
    t[-1] - t[0] + step_t,
    f[-1] - f[0] + step_f,
    fill=False,
    edgecolor="white",
    linestyle="--",
    linewidth=1.5,
)
ax.add_patch(box)
ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)")
ax.set_title("Searched: frequency and time", loc="left", fontweight="bold")
fig.colorbar(image, label="Spearman r", shrink=0.8)

# %%
# Named groups give each band and window a single score. With the default
# ``"aggregate"`` mode, each group is averaged into one value per channel.
prepared = prepare_rsa_data(
    data,
    axes=axes,
    coords=coords,
    freq={"theta": (4, 8), "alpha": (8, 14), "beta": (14, 30)},
    time={"early": (0, 0.2), "middle": (0.2, 0.4), "late": (0.4, 0.6)},
)
result = rsa_nd(prepared, model_rdm, search_axes=("band", "window"))
print(result)

# %%
# Several models add a leading ``"model"`` axis. Prepared data can be reused for
# each of them. The true model scores high only in the alpha band, in the middle
# window. A random model scores near zero everywhere.
models = [model_rdm, pdist(rng.standard_normal((n_items, 1)))]
result = rsa_nd(
    prepared,
    models,
    model_names=["True model", "Random model"],
    search_axes=("band", "window"),
)
print(result)

fig, axs = plt.subplots(1, 2, figsize=(6, 2.8), layout="constrained")
for ax, name, scores in zip(axs, result.coords["model"], result.rsa):
    ax.imshow(scores, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for (row, col), score in np.ndenumerate(scores.round(2) + 0.0):  # No "-0.00"
        color = "white" if abs(score) > limit / 2 else "black"
        ax.text(col, row, f"{score:.2f}", ha="center", va="center", color=color)
    ax.set_xticks(range(3), result.coords["window"])
    ax.set_yticks(range(3), result.coords["band"] if ax is axs[0] else [])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(name, loc="left", fontweight="bold")

plt.show()
