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
# Simulate data with shape items x channels x frequencies x times. Between 8 and
# 13 Hz and from 0.2 to 0.4 s, the channels follow one value per item.
rng = np.random.default_rng(0)
n_items, n_channels = 30, 12
freqs = np.arange(4.0, 32.0, 2.0)
times = np.arange(0, 0.6, 0.02)
item_values = rng.standard_normal(n_items)

data = rng.standard_normal((n_items, n_channels, len(freqs), len(times)))
pattern = rng.standard_normal(n_channels)
in_band = (freqs >= 8) & (freqs <= 13)
in_window = (times >= 0.2) & (times <= 0.4)
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

fig, ax = plt.subplots()
extent = [times[0], times[-1], freqs[0], freqs[-1]]
image = ax.imshow(result.rsa, origin="lower", aspect="auto", extent=extent)
ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)")
fig.colorbar(image, label="Spearman RSA")

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
print(result.coords["band"], result.coords["window"])
print(np.round(result.rsa, 2))

# %%
# Several models add a leading ``"model"`` axis. Prepared data can be reused for
# each of them.
models = [model_rdm, pdist(rng.standard_normal((n_items, 1)))]
result = rsa_nd(
    prepared,
    models,
    model_names=["true", "random"],
    search_axes=("band", "window"),
)
print(result)
print(np.round(result.rsa[:, 1, 1], 2))

plt.show()
