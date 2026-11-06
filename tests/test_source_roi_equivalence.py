"""Correctness tests for source ROI / Label RSA on the dense path.

``compute_rsa(stcs, space=rois, tfr={...})`` materialises the full source TFR
tensor and uses ROI / Label patches as Searchlight neighbourhoods. These tests
check that dense source RSA is correct -- including that the geodesic spatial
searchlight matches ``mne_rsa.rsa_stcs(spatial_radius=...)`` exactly.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from rsa_nd import (
    PreparedRsaData,
    RsaData,
    SourceSpace,
    compute_rsa,
    prepare_rsa_data,
    rsa_nd,
)


def _ordinal_rdm(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def _toy_source_estimates_and_rois():
    times = np.linspace(0.0, 0.49, 50)
    vertices = [np.arange(40, dtype=int), np.arange(20, 40, dtype=int)]
    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    rng = np.random.default_rng(0)
    vertex_weights = rng.standard_normal(len(vertices[0]) + len(vertices[1]))

    stcs = []
    for item_value in item_values:
        data = (
            np.sin(2 * np.pi * 10.0 * times)[np.newaxis, :]
            + item_value * vertex_weights[:, np.newaxis]
            + 0.01 * np.arange(len(vertices[0]) + len(vertices[1]))[:, np.newaxis]
        )
        stcs.append(
            mne.SourceEstimate(
                data,
                vertices=vertices,
                tmin=0.0,
                tstep=times[1] - times[0],
                subject="sample",
            )
        )

    rois = [
        mne.Label(np.array([1, 5, 10]), hemi="lh", name="roi-a-lh"),
        mne.Label(np.array([22, 25]), hemi="rh", name="roi-b-rh"),
    ]
    return stcs, rois, item_values


def test_source_space_aggregate_mode_matches_vertex_mean():
    """space_mode='aggregate' averages ROI vertices into a per-ROI signal."""
    stcs, rois, item_values = _toy_source_estimates_and_rois()
    model = _ordinal_rdm(item_values)

    prepared = prepare_rsa_data(stcs, space=rois, space_mode="aggregate")
    assert prepared.rsa_data.axes == ("item", "roi", "time")
    assert list(prepared.rsa_data.coords["roi"]) == ["roi-a-lh", "roi-b-rh"]
    assert prepared.space is None
    res = rsa_nd(prepared, model, search_axes=("roi",))
    assert np.asarray(res.rsa).shape == (2,)

    # equals RSA over manually vertex-averaged ROI signals
    patches = prepare_rsa_data(stcs, space=rois).space.patches
    man = np.stack(
        [
            np.stack([np.asarray(s.data)[patches[name]].mean(0) for s in stcs])
            for name in patches
        ],
        axis=1,
    )
    man_data = RsaData(
        man,
        axes=("item", "roi", "time"),
        coords={"roi": np.array(list(patches), dtype=object)},
    )
    ref = rsa_nd(man_data, model, search_axes=("roi",))
    np.testing.assert_allclose(np.asarray(res.rsa), np.asarray(ref.rsa))


def test_prepare_rsa_data_builds_grouped_source_space():
    stcs, rois, _ = _toy_source_estimates_and_rois()
    prepared = prepare_rsa_data(
        stcs,
        space=rois,
        tfr={"freqs": np.array([10.0]), "n_cycles": 1.0},
    )
    assert isinstance(prepared, PreparedRsaData)
    assert prepared.input_kind == "source_estimates"
    assert isinstance(prepared.space, SourceSpace)
    assert prepared.space.is_grouped is True
    assert prepared.space.assembly_role() == "per_group"
    assert set(prepared.space.patches.keys()) == {"roi-a-lh", "roi-b-rh"}


def test_native_source_space_radius_requires_src():
    """A geodesic (meters) radius search needs the source-space geometry."""
    space = SourceSpace(vertices=(np.arange(3),), src=None)

    # radius=0 still works without geometry (one patch per vertex).
    patches, centers = space.searchlight_patches(0)
    assert len(patches) == 3
    assert all(p.tolist() == [i] for i, p in enumerate(patches))
    assert centers.tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="requires the source-space"):
        space.searchlight_patches(0.02)


def test_native_source_space_rejects_negative_radius():
    space = SourceSpace(vertices=(np.arange(3),), src=object())
    with pytest.raises(ValueError, match="must be non-negative"):
        space.searchlight_patches(-0.01)


def test_compute_rsa_source_roi_tfr_auto_assembles_roi_time_scores():
    stcs, rois, item_values = _toy_source_estimates_and_rois()
    freqs = np.array([10.0])
    model_rdm = _ordinal_rdm(item_values)

    result = compute_rsa(
        stcs,
        model_rdm,
        space=rois,
        tfr={"freqs": freqs, "n_cycles": 1.0},
        tfr_feature="power",
        standardize="features",
        search_axes=("space", "time"),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
    )
    out = result.to_mne()

    assert isinstance(out, mne.EvokedArray)
    assert out.ch_names == ["roi-a-lh", "roi-b-rh"]
    assert out.data.shape == (2, len(stcs[0].times))
    assert np.isfinite(out.data).any()


def test_generic_grouped_source_path_rejects_radius_for_grouped_space():
    """Grouped space + non-default radii['space'] raises."""
    stcs, rois, _ = _toy_source_estimates_and_rois()
    freqs = np.array([10.0])
    prepared = prepare_rsa_data(
        stcs,
        space=rois,
        tfr={"freqs": freqs, "n_cycles": 1.0},
    )
    model_rdm = _ordinal_rdm([0.0, 1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="grouped"):
        rsa_nd(
            prepared,
            model_rdm,
            search_axes=("space",),
            radii={"space": 1.0},
        )


@pytest.fixture(scope="module")
def sample_oct6_src():
    """Sample surface source space (oct-6) with precomputed distances."""
    data_path = mne.datasets.sample.data_path()
    src_path = data_path / "subjects" / "sample" / "bem" / "sample-oct-6-src.fif"
    return mne.read_source_spaces(src_path, verbose=False)


def _stcs_on_src(src, *, n_items, n_times, seed=0):
    verts = [src[0]["vertno"], src[1]["vertno"]]
    n_total = sum(len(v) for v in verts)
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((n_items, n_total))
    base = rng.standard_normal((n_total, n_times))
    return [
        mne.SourceEstimate(
            base + weights[i][:, None] + 0.1 * np.arange(n_times)[None, :],
            verts,
            tmin=0.0,
            tstep=0.01,
            subject="sample",
        )
        for i in range(n_items)
    ]


@pytest.mark.slowtest
@pytest.mark.parametrize("radius", [0.015, 0.025])
def test_geodesic_source_searchlight_matches_mne_rsa(sample_oct6_src, radius):
    """The native (non-grouped) source searchlight scores match
    ``mne_rsa.rsa_stcs(spatial_radius=...)`` exactly, since both measure
    geodesic distance in meters off the same source-space geometry.

    (``radius=0`` is intentionally not compared: mne-rsa's ``spatial_radius``
    has no per-vertex setting -- ``None`` pools every vertex into one patch --
    whereas ``radii={"space": 0}`` here means one patch per vertex.)
    """
    mne_rsa = pytest.importorskip("mne_rsa")
    src = sample_oct6_src
    n_items, n_times = 6, 4
    stcs = _stcs_on_src(src, n_items=n_items, n_times=n_times)
    model = np.abs(np.arange(n_items)[:, None] - np.arange(n_items)[None, :]).astype(
        float
    )

    ref = mne_rsa.rsa_stcs(
        stcs,
        model,
        src=src,
        spatial_radius=radius,
        temporal_radius=None,
        stc_rdm_metric="correlation",
        rsa_metric="spearman",
        verbose=False,
    )
    ref_vals = np.asarray(ref.data).ravel()

    prepared = prepare_rsa_data(stcs, src=src)
    result = rsa_nd(
        prepared,
        model,
        search_axes=("space",),
        radii={"space": radius},
        data_rdm_metric="correlation",
        rsa_metric="spearman",
    )
    ez_vals = np.asarray(result.rsa).ravel()

    assert ez_vals.shape == ref_vals.shape
    np.testing.assert_allclose(ez_vals, ref_vals, rtol=0, atol=1e-6, equal_nan=True)


def test_volume_source_estimates_input_through_adapter():
    """A list of ``mne.VolSourceEstimate`` feeds the source adapter with a
    volume `SourceSpace` (single vertex array)."""
    n_items, n_vert, n_t = 4, 6, 10
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n_vert, n_t))
    delta = rng.standard_normal((n_vert, n_t))
    stcs = [
        mne.VolSourceEstimate(
            base + i * delta,
            vertices=[np.arange(n_vert)],
            tmin=0.0,
            tstep=0.01,
            subject="sample",
        )
        for i in range(n_items)
    ]

    prepared = prepare_rsa_data(stcs)
    assert prepared.input_kind == "source_estimates"
    assert prepared.rsa_data.axes == ("item", "space", "time")
    assert isinstance(prepared.space, SourceSpace)
    assert prepared.space.estimate_kind == "volume"

    model = _ordinal_rdm(np.arange(n_items))
    result = rsa_nd(
        prepared,
        model,
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
    )
    assert np.asarray(result.rsa).shape == (n_t,)


@pytest.mark.slowtest
def test_sample_source_roi_tfr_compute_rsa_uses_real_sample_geometry():
    """Smoke-test source ROI TFR on MNE sample source geometry and labels."""
    try:
        sample_root = mne.datasets.sample.data_path(download=False, verbose=False)
    except Exception as exc:  # pragma: no cover - depends on local data cache
        pytest.skip(f"MNE sample dataset is not available: {exc}")

    subjects_dir = sample_root / "subjects"
    src_path = subjects_dir / "sample" / "bem" / "sample-oct-6-src.fif"
    src = mne.read_source_spaces(src_path, verbose=False)
    vertices = [src[0]["vertno"], src[1]["vertno"]]
    labels = mne.read_labels_from_annot(
        "sample", parc="aparc", subjects_dir=subjects_dir, verbose=False
    )
    roi_by_name = {label.name: label for label in labels}
    rois = [roi_by_name["transversetemporal-lh"], roi_by_name["transversetemporal-rh"]]

    n_items, n_times, sfreq = 4, 30, 100.0
    times = np.arange(n_times) / sfreq
    n_vertices = sum(len(v) for v in vertices)
    rng = np.random.default_rng(42)
    spatial = rng.normal(size=n_vertices)
    temporal = np.sin(2 * np.pi * 10.0 * times)
    item_values = np.arange(n_items, dtype=float)
    stcs = []
    for value in item_values:
        data = (
            value * spatial[:, None]
            + temporal[None, :]
            + 0.01 * rng.normal(size=(n_vertices, n_times))
        )
        stcs.append(
            mne.SourceEstimate(
                data,
                vertices=vertices,
                tmin=0.0,
                tstep=1.0 / sfreq,
                subject="sample",
            )
        )

    result = compute_rsa(
        stcs,
        _ordinal_rdm(item_values),
        space=rois,
        tfr={"freqs": np.array([8.0, 12.0]), "n_cycles": 1.0, "use_fft": True},
        tfr_feature="power",
        standardize="features",
        search_axes=("space", "time"),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="spearman",
    )

    assert result.rsa.shape == (2, n_times)
    assert result.search_coords["space"].tolist() == [
        "transversetemporal-lh",
        "transversetemporal-rh",
    ]
    assert np.isfinite(result.rsa).any()
