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
# it with a delay of 100 ms.
rng = np.random.default_rng(0)
n_items, n_channels, sfreq, lag = 30, 10, 100.0, 0.1
times = np.arange(0, 2, 1 / sfreq)

feature = gaussian_filter1d(rng.standard_normal((n_items, len(times))), 5, axis=1)
feature /= feature.std()
delayed = np.roll(feature, round(lag * sfreq), axis=1)
weights = rng.standard_normal(n_channels)
neural = weights[:, np.newaxis] * delayed[:, np.newaxis, :]
neural += rng.standard_normal(neural.shape)

data = RsaData(neural, ("item", "space", "time"), {"time": times})
model_rdms = [pdist(feature[:, [idx]]) for idx in range(len(times))]

# %%
# ``"generalization"`` mode compares each model time with each data time.
result = rsa_dynamic(
    data, model_rdms, times, search_axes=("time",), radii={"time": 0.02}
)
fig, ax = plt.subplots()
extent = [times[0], times[-1], times[0], times[-1]]
image = ax.imshow(result.rsa, origin="lower", extent=extent)
ax.set(xlabel="Data time (s)", ylabel="Model time (s)")
fig.colorbar(image, label="Spearman RSA")

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
print(f"Best lag: {lags[np.argmax(mean_scores)]:.2f} s")

fig, ax = plt.subplots()
ax.plot(lags, mean_scores)
ax.axvline(lag, color="k", linestyle="--", label="True lag")
ax.set(xlabel="Lag (s)", ylabel="Mean Spearman RSA")
ax.legend()

plt.show()
