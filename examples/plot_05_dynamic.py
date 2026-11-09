"""
===========
Dynamic RSA
===========

When a stimulus changes over time, so does the model RDM. This example simulates
a neural response that follows a stimulus feature 100 ms later, and uses
:func:`rsa_nd.rsa_dynamic` to recover that lag.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist

from rsa_nd import RsaData, rsa_dynamic

# %%
# Simulate a slowly changing feature for each item, and ten channels that follow
# it with a delay of 100 ms. The feature starts 100 ms early, so that the delayed
# copy covers the whole recording.
rng = np.random.default_rng(0)
n_items, n_channels, sfreq, lag = 30, 10, 100.0, 0.1
times = np.arange(0, 2, 1 / sfreq)
shift = round(lag * sfreq)

feature = rng.standard_normal((n_items, len(times) + shift))
feature = gaussian_filter1d(feature, 5, axis=1)
feature /= feature.std()
delayed, feature = feature[:, :-shift], feature[:, shift:]
weights = rng.standard_normal(n_channels)
neural = weights[:, np.newaxis] * delayed[:, np.newaxis, :]
neural += rng.standard_normal(neural.shape)

data = RsaData(neural, ("item", "space", "time"), {"time": times})
model_rdms = [pdist(feature[:, [idx]]) for idx in range(len(times))]

# %%
# ``"generalization"`` mode compares each model time with each data time. The
# scores form a ridge 100 ms to the right of the diagonal (dotted line), because
# the data follow the model 100 ms later.
result = rsa_dynamic(
    data, model_rdms, times, search_axes=("time",), radii={"time": 0.02}
)
fig, ax = plt.subplots(figsize=(4.6, 3.8), layout="constrained")
limit = np.abs(result.rsa).max()
image = ax.pcolormesh(times, times, result.rsa, cmap="RdBu_r", vmin=-limit, vmax=limit)
ax.plot(times[[0, -1]], times[[0, -1]], color="black", linestyle=":", linewidth=1)
ax.set(xlabel="Data time (s)", ylabel="Model time (s)", aspect="equal")
ax.set_title("Every model time × data time", loc="left", fontweight="bold")
fig.colorbar(image, label="Spearman r", shrink=0.8)

# %%
# ``"lagged"`` mode keeps the model times a fixed lag before each data time.
# Averaging over time gives one score per lag, which peaks at the true delay.
lags = np.arange(-0.1, 0.31, 0.02)
result = rsa_dynamic(
    data,
    model_rdms,
    times,
    mode="lagged",
    lags=lags,
    search_axes=("time",),
    radii={"time": 0.02},
)
mean_scores = np.nanmean(result.rsa, axis=1)
best = np.argmax(mean_scores)
print(f"Best lag: {lags[best]:.2f} s")

fig, ax = plt.subplots(figsize=(4.6, 3), layout="constrained")
ax.axhline(0, color="0.85", linewidth=1, zorder=0)
ax.axvline(lag, color="black", linestyle=":", linewidth=1)
ax.plot(lags, mean_scores, color="#2a78d6", linewidth=2)
ax.plot(lags[best], mean_scores[best], "o", color="#2a78d6")
ax.set_ylim(top=1.2 * mean_scores[best])
ax.text(lag + 0.01, 1.18 * mean_scores[best], "true lag", va="top")
ax.spines[["top", "right"]].set_visible(False)
ax.set(xlabel="Lag (s)", ylabel="Mean Spearman r", xticks=np.arange(-0.1, 0.31, 0.1))
ax.set_title("One score per lag", loc="left", fontweight="bold")

plt.show()
