"""Draw the schematic of how rsa-nd works, using simulated values.

Run from the repository root:

    uv run python docs/make_concept_figure.py

The explanation is in ``source/_static/concept.caption.md``.
"""

from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

OUTPUT = Path(__file__).parent / "source" / "_static" / "concept.png"
WIDTH, HEIGHT = 13.0, 9.0

INK, MUTED, GRID, GHOST = "#0b0b0b", "#52514e", "#c9c8c2", "#f3f2ee"
ORANGE = "#d9531e"
PATCH = {"front": "#5d5d68", "top": "#8e8e9a", "side": "#45454e"}
CMAP = plt.get_cmap("RdBu_r")

N_TIME, N_SPACE, N_FREQ = 8, 5, 6
FREQS = ("4", "6", "9", "12", "20", "30")
BANDS = ("theta", "alpha", "beta")  # Two frequencies each
DEPTH_X, DEPTH_Y = 0.48, 0.40  # How far one space cell shifts the drawing
LAYER = count(10)  # Later shapes are drawn on top of earlier ones
ARROW = dict(color=MUTED, linewidth=1.1, arrowstyle="-|>", mutation_scale=11)


# Simulated data ---------------------------------------------------------------
def simulate(n_items=6, seed=4):
    """Return data, model RDM and the RSA score at every time and frequency."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([0, 1], n_items // 2)
    pattern = np.array([1.0, -0.8, 0.6, -1.0, 0.8])
    patterns = np.stack([pattern, -pattern])
    t, f = np.meshgrid(np.arange(N_TIME), np.arange(N_FREQ), indexing="ij")
    blob = np.exp(-((t - 4) ** 2) / 3 - (f - 5) ** 2 / 4)
    data = np.empty((n_items, N_TIME, N_SPACE, N_FREQ))
    for item, group in enumerate(groups):
        noise = gaussian_filter(rng.standard_normal(data.shape[1:]), 0.8)
        effect = 2.5 * blob[:, np.newaxis, :] * patterns[group][:, np.newaxis]
        data[item] = noise / noise.std() * 0.6 + effect
    model = pdist(groups[:, np.newaxis].astype(float))
    scores = np.array(
        [
            [
                spearmanr(pdist(data[:, x, :, z], "correlation"), model)[0]
                for z in range(N_FREQ)
            ]
            for x in range(N_TIME)
        ]
    )
    return data, squareform(model), scores


DATA, MODEL, SCORES = simulate()
TIME_IDX, FREQ_IDX = 4, 5  # The patch of the worked example
ROWS = DATA[:, TIME_IDX, :, FREQ_IDX]
DATA_RDM = squareform(pdist(ROWS, "correlation"))
VALUE_NORM = Normalize(-np.abs(DATA).max(), np.abs(DATA).max())


# Drawing helpers --------------------------------------------------------------
def project(origin, x, y, z):
    """Return the drawing position of a cube point given in cell units."""
    left, bottom, cell = origin
    return left + (x + y * DEPTH_X) * cell, bottom + (z + y * DEPTH_Y) * cell


def shade(color, amount):
    """Mix a color with white (positive ``amount``) or black (negative)."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + ((amount > 0) - rgb) * abs(amount))


def add_face(ax, origin, corners, **kwargs):
    points = [project(origin, *corner) for corner in corners]
    ax.add_patch(Polygon(points, closed=True, zorder=next(LAYER), **kwargs))


def add_arrow(ax, start, stop, rad=0.0, **kwargs):
    style = {**ARROW, **kwargs}
    arrow = FancyArrowPatch(
        start, stop, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, **style
    )
    arrow.set_zorder(next(LAYER) + 1000)
    ax.add_patch(arrow)


def add_box(ax, origin, box, colors, line, z_step=1):
    """Draw the three visible faces of a box, with grid lines between cells."""
    x0, x1, y0, y1, z0, z1 = box
    style = dict(edgecolor=line, linewidth=0.8)
    faces = {
        "top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        "side": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        "front": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
    }
    for name, corners in faces.items():
        add_face(ax, origin, corners, facecolor=colors[name], **style)
    lines = []
    for x in range(x0 + 1, x1):
        lines += [((x, y0, z0), (x, y0, z1)), ((x, y0, z1), (x, y1, z1))]
    for y in range(y0 + 1, y1):
        lines += [((x0, y, z1), (x1, y, z1)), ((x1, y, z0), (x1, y, z1))]
    for z in np.arange(z0 + z_step, z1, z_step):
        lines += [((x0, y0, z), (x1, y0, z)), ((x1, y0, z), (x1, y1, z))]
    for a, b in lines:
        xy = [project(origin, *a), project(origin, *b)]
        ax.plot(*zip(*xy), color=line, linewidth=0.5, zorder=next(LAYER))


def add_data_cube(ax, origin, values):
    """Draw a cube whose visible cells are colored by their values."""
    for x in range(N_TIME):
        for y in range(N_SPACE):
            color = shade(CMAP(VALUE_NORM(values[x, y, -1])), 0.12)
            corners = [(x, y, N_FREQ), (x + 1, y, N_FREQ), (x + 1, y + 1, N_FREQ),
                       (x, y + 1, N_FREQ)]  # fmt: skip
            add_face(ax, origin, corners, facecolor=color, edgecolor="#ffffff", lw=0.3)
    for y in range(N_SPACE):
        for z in range(N_FREQ):
            color = shade(CMAP(VALUE_NORM(values[-1, y, z])), -0.15)
            corners = [(N_TIME, y, z), (N_TIME, y + 1, z), (N_TIME, y + 1, z + 1),
                       (N_TIME, y, z + 1)]  # fmt: skip
            add_face(ax, origin, corners, facecolor=color, edgecolor="#ffffff", lw=0.3)
    for x in range(N_TIME):
        for z in range(N_FREQ):
            color = CMAP(VALUE_NORM(values[x, 0, z]))
            corners = [(x, 0, z), (x + 1, 0, z), (x + 1, 0, z + 1), (x, 0, z + 1)]
            add_face(ax, origin, corners, facecolor=color, edgecolor="#ffffff", lw=0.3)
    # Outline the patch: one time, one frequency, all of space
    x, z = TIME_IDX, FREQ_IDX
    outline = dict(facecolor="none", edgecolor=INK, linewidth=1.8)
    top = [
        (x, 0, z + 1),
        (x + 1, 0, z + 1),
        (x + 1, N_SPACE, z + 1),
        (x, N_SPACE, z + 1),
    ]
    add_face(ax, origin, top, **outline)
    front = [(x, 0, z), (x + 1, 0, z), (x + 1, 0, z + 1), (x, 0, z + 1)]
    add_face(ax, origin, front, **outline)


def add_matrix(ax, left, bottom, values, cell, norm, numbers=True):
    """Draw a matrix with its first row on top, and return its geometry."""
    n_rows, n_cols = values.shape
    for row in range(n_rows):
        for col in range(n_cols):
            corner = (left + col * cell, bottom + (n_rows - row - 1) * cell)
            color = CMAP(norm(values[row, col]))
            ax.add_patch(Rectangle(corner, cell, cell, facecolor=color,
                                   edgecolor="#ffffff", linewidth=0.8))  # fmt: skip
    if numbers:
        style = dict(ha="center", va="center", color=MUTED, fontsize=9)
        for row in range(n_rows):
            y = bottom + (n_rows - row - 0.5) * cell
            ax.text(left - 0.14, y, row + 1, **style)

    def at(row, col=0, where="center"):
        x = left + (col + {"left": 0, "center": 0.5, "right": 1}[where]) * cell
        return x, bottom + (n_rows - row - 0.5) * cell

    return at


def add_outline(ax, left, bottom, width, height):
    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, edgecolor=INK,
                           linewidth=1.8, zorder=900))  # fmt: skip


def title(ax, x, y, text, size=12.5):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=INK, va="top")


# How it works -----------------------------------------------------------------
def add_mechanism(ax, top):
    title(ax, 0.2, top, "How it works", size=16)
    head = top - 0.5
    middle = top - 2.7

    # 1: the same patch in every item
    title(ax, 0.2, head, "1  Patch")
    cell, left = 0.14, 1.05
    cubes = {}
    items = ((0, middle + 0.6), (1, middle - 0.75), (5, middle - 2.35))
    for item, bottom in items:
        origin = (left, bottom, cell)
        add_data_cube(ax, origin, DATA[item])
        ax.text(0.2, bottom + 0.5, f"Item {item + 1}", va="center", fontsize=10.5)
        cubes[item] = project(origin, TIME_IDX + 0.5, N_SPACE, N_FREQ)
    ax.text(
        1.75, middle - 0.97, "⋮", ha="center", va="center", fontsize=16, color=MUTED
    )
    first = (left, middle + 0.6, cell)
    label = dict(color=MUTED, fontsize=8.5)
    ax.text(
        *project(first, N_TIME / 2, 0, -0.4), "time", ha="center", va="top", **label
    )
    ax.text(*project(first, -0.4, 0, N_FREQ / 2), "freq", ha="right", va="center",
            rotation=90, **label)  # fmt: skip
    ax.text(*project(first, N_TIME + 0.3, N_SPACE / 2, 0), "space", ha="left",
            va="top", **label)  # fmt: skip

    # 2: each patch becomes one row
    title(ax, 3.35, head, "2  Flatten")
    cell, left = 0.32, 3.55
    bottom = middle - 3 * cell
    row_at = add_matrix(ax, left, bottom, ROWS, cell, VALUE_NORM)
    ax.text(left + 2.5 * cell, bottom - 0.08, "space", ha="center", va="top", **label)
    for item, start in cubes.items():
        stop = row_at(item, 0, "left")
        add_arrow(ax, start, (stop[0] - 0.27, stop[1]), rad=-0.15)
    for row in (0, 5):
        add_outline(ax, left, bottom + (5 - row) * cell, 5 * cell, cell)

    # 3: the data RDM, with the model RDM below it
    title(ax, 5.95, head, "3  RDMs")
    cell, left = 0.25, 6.25
    rdm_norm = Normalize(-1, 3)  # Distances of 0 to 2, in softer colours
    data_bottom = middle + 0.12
    data_at = add_matrix(ax, left, data_bottom, DATA_RDM, cell, rdm_norm)
    model_bottom = middle - 1.85
    add_matrix(ax, left, model_bottom, MODEL * 2, cell, rdm_norm)
    ax.text(left + 0.75, data_bottom - 0.06, "data", ha="center", va="top", **label)
    ax.text(left + 0.75, model_bottom - 0.06, "model", ha="center", va="top", **label)
    add_outline(ax, left + 5 * cell, data_bottom + 5 * cell, cell, cell)
    start = row_at(2.5, 4, "right")
    add_arrow(ax, (start[0] + 0.08, start[1]), (left - 0.3, data_at(2.5)[1]), rad=-0.2)

    # 4: compare the two RDMs
    title(ax, 8.35, head, "4  RSA score")
    score = SCORES[TIME_IDX, FREQ_IDX]
    score_norm = Normalize(-1.6, 1.6)  # Scores of -1 to 1, in softer colours
    center = (8.95, middle - 0.35)
    ax.add_patch(Circle(center, 0.36, facecolor=CMAP(score_norm(score)),
                        edgecolor=INK, linewidth=1.8, zorder=900))  # fmt: skip
    ax.text(*center, f"{score:.2f}", ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold", zorder=901)  # fmt: skip
    ax.text(center[0], center[1] - 0.45, "Spearman r", ha="center", va="top", **label)
    add_arrow(ax, (left + 6 * cell + 0.05, data_bottom + 3 * cell),
              (center[0] - 0.36, center[1] + 0.12), rad=-0.2)  # fmt: skip
    add_arrow(ax, (left + 6 * cell + 0.05, model_bottom + 3 * cell),
              (center[0] - 0.36, center[1] - 0.12), rad=0.2)  # fmt: skip

    # 5: the score goes to the patch's time and frequency
    title(ax, 10.2, head, "5  Result")
    cell, left = 0.3, 10.45
    bottom = middle - 3 * cell - 0.35
    result = SCORES.T[::-1]  # Frequency up, time to the right
    add_matrix(ax, left, bottom, result, cell, score_norm, numbers=False)
    corner = (left + TIME_IDX * cell, bottom + FREQ_IDX * cell)
    add_outline(ax, *corner, cell, cell)
    add_arrow(ax, (center[0] + 0.36, center[1]), (corner[0], corner[1] + cell / 2),
              rad=-0.25)  # fmt: skip
    ax.text(left + 4 * cell, bottom - 0.08, "time", ha="center", va="top", **label)
    ax.text(left - 0.1, bottom + 3 * cell, "freq", ha="right", va="center",
            rotation=90, **label)  # fmt: skip


# Choose what to search --------------------------------------------------------
def add_axis_labels(ax, origin, searched, ticks=None, groups=None):
    """Label the cube axes, with an orange arrow along each searched axis."""
    cell = origin[2]

    def style(axis):
        on = axis in searched
        return dict(color=ORANGE if on else MUTED, fontweight="bold" if on else "normal",
                    fontsize=10.5)  # fmt: skip

    x0, y0 = project(origin, 0, 0, 0)
    x1, _ = project(origin, N_TIME, 0, 0)
    if "time" in searched:
        add_arrow(ax, (x0, y0 - 0.1), (x1, y0 - 0.1), color=ORANGE, linewidth=1.8)
    ax.text((x0 + x1) / 2, y0 - 0.17, "time", ha="center", va="top", **style("time"))

    _, top = project(origin, 0, 0, N_FREQ)
    offset = 0.1
    if ticks:
        for z, text in enumerate(ticks):
            ax.text(x0 - 0.06, y0 + (z + 0.5) * cell, text, ha="right", va="center",
                    fontsize=8, color=MUTED)  # fmt: skip
        offset = 0.33
    if groups:
        for idx, text in enumerate(groups):
            ax.text(x0 - 0.06, y0 + (2 * idx + 1) * cell, text, ha="right", va="center",
                    fontsize=9, color=ORANGE, fontweight="bold")  # fmt: skip
        offset = 0.5
    axis = "band" if groups else "freq"
    if axis in searched:
        add_arrow(
            ax, (x0 - offset, y0), (x0 - offset, top), color=ORANGE, linewidth=1.8
        )
    name = "freq (Hz)" if ticks else axis
    ax.text(x0 - offset - 0.09, (y0 + top) / 2, name, ha="right", va="center",
            rotation=90, **style(axis))  # fmt: skip

    start = np.add(project(origin, N_TIME, 0, 0), (0.1, -0.03))
    stop = np.add(project(origin, N_TIME, N_SPACE, 0), (0.1, -0.03))
    if "space" in searched:
        add_arrow(ax, start, stop, color=ORANGE, linewidth=1.8)
    ax.text(*(stop + (0.08, -0.12)), "space", ha="left", va="center", **style("space"))


def add_tile(ax, left, top, code, result, patch, searched, z_step=1, **labels):
    """Draw one way of searching: the code, the patch, and what comes out."""
    ax.text(left, top, code, family="monospace", fontsize=8.8, color=INK, va="top",
            linespacing=1.35)  # fmt: skip
    origin = (left + 0.85, top - 2.05, 0.17)
    ghost = dict(front=GHOST, top=GHOST, side=GHOST)
    add_box(ax, origin, (0, N_TIME, 0, N_SPACE, 0, N_FREQ), ghost, GRID, z_step)
    add_box(ax, origin, patch, PATCH, "#ffffff", z_step)
    add_axis_labels(ax, origin, searched, **labels)
    ax.text(left + 0.85, top - 2.5, f"→ {result}", fontsize=10.5, color=INK, va="top")


fig = plt.figure(figsize=(WIDTH, HEIGHT))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
ax.axis("off")

add_mechanism(ax, HEIGHT - 0.2)

top = HEIGHT - 5.55
title(ax, 0.2, top, "Choose what to search", size=16)
legend_x = 4.6
ax.add_patch(Rectangle((legend_x, top - 0.27), 0.2, 0.2, facecolor=PATCH["front"]))
ax.text(legend_x + 0.3, top - 0.17, "one patch", va="center", fontsize=10, color=MUTED)
add_arrow(ax, (legend_x + 1.5, top - 0.17), (legend_x + 1.95, top - 0.17),
          color=ORANGE, linewidth=1.8)  # fmt: skip
ax.text(legend_x + 2.05, top - 0.17, "searched axis: one score per position",
        va="center", fontsize=10, color=MUTED)  # fmt: skip

row = top - 0.6
tiles = (
    ('search_axes=("time",),\nradii={"time": 0.05}', "time course",
     (3, 6, 0, N_SPACE, 0, N_FREQ), ("time",), {}),
    ('search_axes=("freq", "time")', "frequency × time map",
     (4, 5, 0, N_SPACE, 5, 6), ("freq", "time"), {"ticks": FREQS}),
    ('search_axes=("space", "freq", "time"),\nradii={"space": 0.04, ...}',
     "space × freq × time map",
     (3, 6, 1, 4, 3, 6), ("space", "freq", "time"), {}),
    ('freq={"alpha": (8, 13), ...},\nsearch_axes=("band", "time")', "band × time map",
     (4, 5, 0, N_SPACE, 2, 4), ("band", "time"), {"groups": BANDS, "z_step": 2}),
)  # fmt: skip
for idx, (code, result, patch, searched, labels) in enumerate(tiles):
    add_tile(
        ax, (0.2, 3.25, 6.3, 9.55)[idx], row, code, result, patch, searched, **labels
    )

fig.savefig(OUTPUT, dpi=200, facecolor="white")
