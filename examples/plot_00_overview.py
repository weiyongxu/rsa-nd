"""
=======================================
RSA by brain region, frequency and time
=======================================

In the MNE sample dataset, one participant hears tones and sees checkerboards, on
the left or on the right. This example asks which brain regions, frequency bands
and time points tell these trials apart, for two models at once, in one call.

``mne-rsa`` can search brain regions and time. ``rsa-nd`` adds frequency bands as
another searched dimension, and several models as one more.

These are the scores of one participant, shown without statistics.
"""

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyvista
from matplotlib.colors import Normalize
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from scipy.spatial.distance import pdist, squareform

from rsa_nd import compute_rsa

# %%
# **Data.** Single-trial source estimates for 30 trials of each condition.
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
raw = mne.io.read_raw_fif(meg_path / "sample_audvis_filt-0-40_raw.fif")
events = mne.find_events(raw)
conditions = {"aud/left": 1, "aud/right": 2, "vis/left": 3, "vis/right": 4}
epochs = mne.Epochs(
    raw, events, conditions, tmin=-0.3, tmax=0.6, picks="meg", preload=True
)
epochs = mne.concatenate_epochs([epochs[name][:30] for name in conditions])
inverse = read_inverse_operator(meg_path / "sample_audvis-meg-oct-6-meg-inv.fif")
stcs = apply_inverse_epochs(epochs, inverse, lambda2=1 / 9, pick_ori="normal")

# %%
# **Models.** Two model RDMs say which trials should differ: trials of different
# modalities, or trials on different sides.
codes = epochs.events[:, 2]
models = {
    "Auditory vs visual": pdist(np.isin(codes, [3, 4]).astype(float)[:, np.newaxis]),
    "Left vs right": pdist(np.isin(codes, [2, 4]).astype(float)[:, np.newaxis]),
}

# %%
# **One call.** Search 68 cortical parcels, 3 frequency bands and time. Each
# parcel's vertices are the features, so each position compares trials using the
# pattern of band power across one parcel. Each trial is its own item.
parcels = mne.read_labels_from_annot("sample", "aparc", subjects_dir=subjects_dir)
parcels = [label for label in parcels if not label.name.startswith("unknown")]
bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
result = compute_rsa(
    stcs,
    list(models.values()),
    model_names=list(models),
    space=parcels,
    tfr={"freqs": [5.0, 7.0, 9.0, 12.0, 16.0, 24.0], "n_cycles": 2.0, "decim": 6},
    freq=bands,
    time=(-0.1, 0.5),
    log=True,
    standardize="features",
    search_axes=("space", "band", "time"),
    radii={"time": 0.04},
    labels=np.arange(len(stcs)),
)
print(result)

# %%
# **Plot.** Color each parcel by its score on the inflated cortex, at the time of
# each model's highest score. The parcel with that score is outlined, and its
# scores are shown over time.
surf_dir = subjects_dir / "sample" / "surf"
surfaces = {
    hemi: mne.read_surface(surf_dir / f"{hemi}.inflated") for hemi in ("lh", "rh")
}
parcel_by_name = {label.name: label for label in parcels}
times = result.coords["time"]
band_colors = ["#6250d6", "#1baf7a", "#eb6834"]
ink, muted = "#0b0b0b", "#52514e"


def brain_image(parcel_colors, hemi, view, outline=None):
    """Render one inflated hemisphere, with a color for each parcel."""
    rr, tris = surfaces[hemi]
    sulci = mne.surface.read_curvature(surf_dir / f"{hemi}.curv", binary=True)
    colors = np.where(sulci[:, np.newaxis], 0.62, 0.8) * np.ones(3)
    for name, color in parcel_colors.items():
        if name.endswith(hemi):
            colors[parcel_by_name[name].vertices] = color
    if outline is not None and outline.endswith(hemi):
        inside = np.zeros(len(rr), bool)
        inside[parcel_by_name[outline].vertices] = True
        edge = np.unique(tris[inside[tris].any(axis=1) & ~inside[tris].all(axis=1)])
        for _ in range(2):  # Widen the outline so that it shows when small
            edge = np.unique(tris[np.isin(tris, edge).any(axis=1)])
        colors[edge] = 0.05
    faces = np.column_stack([np.full(len(tris), 3), tris])
    plotter = pyvista.Plotter(off_screen=True, window_size=(700, 500))
    mesh = pyvista.PolyData(rr, faces)
    plotter.add_mesh(mesh, scalars=colors, rgb=True, smooth_shading=True)
    side = 1 if (hemi == "rh") == (view == "lateral") else -1
    plotter.view_vector((side, 0, 0), (0, 0, 1))
    plotter.enable_parallel_projection()
    image = plotter.screenshot(transparent_background=True)
    plotter.close()
    rows, cols = np.nonzero(image[..., 3])
    return image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def parcel_view(name):
    """Return the view that shows a parcel, and where it is in that image."""
    hemi = name[-2:]
    rr = surfaces[hemi][0]
    x, y, z = rr[parcel_by_name[name].vertices].mean(axis=0)
    outer = (x - rr[:, 0].mean()) * (1 if hemi == "rh" else -1) > 0
    view = "lateral" if outer else "medial"
    (_, y_min, z_min), (_, y_max, z_max) = rr.min(axis=0), rr.max(axis=0)
    across = (y - y_min) / (y_max - y_min)
    if (hemi == "rh") != outer:  # The front of the brain is on the left
        across = 1 - across
    return view, across, (z_max - z) / (z_max - z_min)


width, height = 13.0, 7.5
fig = plt.figure(figsize=(width, height))


def place(left, bottom, w, h):
    """Add axes at a position given in inches."""
    return fig.add_axes([left / width, bottom / height, w / width, h / height])


def write(x, y, text, **kwargs):
    """Write text at a position given in inches."""
    fig.text(x / width, y / height, text, **kwargs)


heading = dict(fontsize=13, fontweight="bold", va="top")

# Data: one icon per axis
write(0.3, 7.3, "Data", **heading)
ax = place(0.25, 5.0, 1.95, 1.75)
atlas = {name: parcel_by_name[name].color[:3] for name in result.coords["space"]}
ax.imshow(brain_image(atlas, "lh", "lateral"))
ax.axis("off")
icons = place(2.35, 5.0, 0.4, 1.75)
icons.set(xlim=(0, 1), ylim=(0, 4))
icons.axis("off")
for idx, color in enumerate(band_colors):
    icons.add_patch(plt.Rectangle((0.05 + idx * 0.32, 2.3), 0.26, 0.4, color=color))
icons.annotate("", (1, 1.5), (0, 1.5), arrowprops=dict(arrowstyle="-|>", color=muted))
for idx in range(3):
    icons.add_patch(plt.Rectangle((0.1 + idx * 0.12, 0.2 + idx * 0.12), 0.6, 0.45,
                                  facecolor="#ffffff", edgecolor=muted))  # fmt: skip
start, stop = (f"{t:.1f}".replace("-", "−") for t in times[[0, -1]])
labels = [
    f"{len(result.coords['space'])} parcels",
    f"{len(bands)} bands",
    f"{start} to {stop} s",
    f"{len(stcs)} trials",
]
for idx, text in enumerate(labels):
    write(2.85, 6.6 - idx * 0.44, text, fontsize=11, va="center")

# Models: which trials should differ
write(4.25, 7.3, "Models", **heading)
edges = np.arange(0, len(stcs) + 1, 30)
names = ["Aud L", "Aud R", "Vis L", "Vis R"]
for idx, (name, rdm) in enumerate(models.items()):
    ax = place(4.95 + idx * 2.0, 5.15, 1.5, 1.5)
    ax.imshow(squareform(rdm), cmap="RdBu_r", vmin=-0.5, vmax=1.5)
    for edge in edges[1:-1] - 0.5:
        ax.axhline(edge, color="#ffffff", linewidth=0.8)
        ax.axvline(edge, color="#ffffff", linewidth=0.8)
    ticks = edges[:-1] + 14.5
    ax.set_xticks(ticks, names, fontsize=8.5, rotation=90)
    ax.set_yticks(ticks, names if idx == 0 else [], fontsize=8.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(name, fontsize=11)

# One call: the code, and the shape of its result
write(9.0, 7.3, "One call", **heading)
shape = " × ".join(str(n) for n in result.rsa.shape)
code = (
    "result = compute_rsa(\n"
    "    stcs,\n"
    "    [modality_rdm, side_rdm],\n"
    "    space=parcels,\n"
    '    tfr={"freqs": freqs, ...},\n'
    '    freq={"theta": (4, 8), ...},\n'
    '    search_axes=("space", "band", "time"),\n'
    ")\n"
    f"# → {shape} scores"
)
box = dict(boxstyle="round,pad=0.6", facecolor="#f3f2ee", edgecolor="none")
write(9.1, 6.82, code, family="monospace", fontsize=9.2, va="top", linespacing=1.45,
      bbox=box)  # fmt: skip

# Results: one row per model and one column per band
for col, (band, (low, high)) in enumerate(bands.items()):
    write(0.85 + col * 2.45 + 1.17, 4.55, f"{band}  {low}–{high} Hz",
          ha="center", fontsize=11.5, fontweight="bold", color=band_colors[col])  # fmt: skip
for row, name in enumerate(result.coords["model"]):
    scores = result.rsa[row]
    limit = np.abs(scores).max()
    norm = Normalize(-limit, limit)
    parcel_idx, band_idx, time_idx = np.unravel_index(scores.argmax(), scores.shape)
    peak = result.coords["space"][parcel_idx]
    peak_view, across, down = parcel_view(peak)
    bottom = 2.45 - row * 2.05
    when = f"{times[time_idx] * 1000:.0f} ms"
    write(0.42, bottom + 0.9, f"{name}\n{when}", rotation=90, ha="center",
          va="center", fontsize=11.5, fontweight="bold")  # fmt: skip
    for col in range(len(bands)):
        map_colors = {
            name: plt.cm.RdBu_r(norm(score))[:3]
            for name, score in zip(result.coords["space"], scores[:, col, time_idx])
        }
        for view_row, view in enumerate(("lateral", "medial")):
            for hemi_col, hemi in enumerate(("lh", "rh")):
                ax = place(0.85 + col * 2.45 + hemi_col * 1.19,
                           bottom + (1 - view_row) * 0.9, 1.15, 0.88)  # fmt: skip
                image = brain_image(map_colors, hemi, view, outline=peak)
                ax.imshow(image)
                ax.axis("off")
                if col == band_idx and view == peak_view and peak.endswith(hemi):
                    xy = (across * image.shape[1], down * image.shape[0])
                    ax.annotate(peak[:-3], xy, xytext=(0.5, -0.12),
                                textcoords="axes fraction", ha="center", va="top",
                                fontsize=9.5, fontweight="bold",
                                arrowprops=dict(arrowstyle="-", color=ink, lw=1),
                                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff",
                                          ec="none", alpha=0.85))  # fmt: skip
                if row == col == view_row == 0:
                    ax.set_title(("left", "right")[hemi_col], fontsize=9, color=muted,
                                 pad=1)  # fmt: skip
    cax = place(8.2, bottom + 0.2, 0.1, 1.4)
    fig.colorbar(plt.cm.ScalarMappable(norm, "RdBu_r"), cax=cax)
    cax.set_title("r", fontsize=10)

    ax = place(9.5, bottom + 0.3, 3.25, 1.35)
    for col, color in enumerate(band_colors):
        ax.plot(times, scores[parcel_idx, col], color=color, linewidth=2)
    ax.axhline(0, color="#d9d8d3", linewidth=1, zorder=0)
    ax.axvline(times[time_idx], color=muted, linewidth=1, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    parcel = peak.replace("-lh", ", left").replace("-rh", ", right")
    ax.set_title(parcel, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylim(min(scores[parcel_idx].min(), 0) - 0.1 * limit, 1.1 * limit)
    ax.set_ylabel("r", rotation=0, labelpad=8)
    if row == 1:
        ax.set_xlabel("Time (s)")

plt.show()
