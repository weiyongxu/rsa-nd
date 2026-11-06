import inspect
import typing
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import rsa_nd


def _ordinal_rdm(values):
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, np.newaxis] - values[np.newaxis, :])


def _prepare_rsa_data_only(*args, **kwargs):
    return rsa_nd.prepare_rsa_data(*args, **kwargs).rsa_data


def test_rsa_nd_public_api_imports():
    expected = {
        "__version__",
        "RsaData",
        "RsaResult",
        "DynamicInfo",
        "PreparedRsaData",
        "compute_rsa",
        "rsa_nd",
        "rsa_dynamic",
        "rsa_dynamic_windows",
        "prepare_rsa_data",
        "source_roi_patches",
    }

    assert set(rsa_nd.__all__) == expected
    for name in expected:
        assert hasattr(rsa_nd, name)

    for old_name in (
        "NamedSpectralData",
        "SpectralRsaResult",
        "DynamicSpectralRsaResult",
        "SourceSpectralRsaResult",
        "to_named_spectral_data",
        "rsa_spectral",
    ):
        assert not hasattr(rsa_nd, old_name)

    for compat_name in (
        "SpaceDescription",
        "SensorSpace",
        "SourceSpace",
        "RsaInputKind",
        "RsaTemplate",
        "safe_log10_power",
    ):
        assert hasattr(rsa_nd, compat_name)
        assert compat_name not in rsa_nd.__all__
    for removed_name in (
        "SourceRoiRsaResult",
        "rsa_stcs_rois_spectral",
        "to_rsa_array",
        "DynamicRsaResult",
        "assemble_rsa_result",
    ):
        assert not hasattr(rsa_nd, removed_name)


def test_public_type_hints_resolve_for_exported_callables():
    for name in (
        "compute_rsa",
        "rsa_nd",
        "rsa_dynamic",
        "rsa_dynamic_windows",
        "prepare_rsa_data",
    ):
        hints = typing.get_type_hints(getattr(rsa_nd, name))
        assert isinstance(hints, dict)


def test_core_container_names_are_generic():
    assert rsa_nd.RsaData.__name__ == "RsaData"
    assert rsa_nd.RsaResult.__name__ == "RsaResult"


def test_rsa_data_has_no_mne_spectral_metadata_field():
    field_names = {field.name for field in fields(rsa_nd.RsaData)}
    assert field_names == {"data", "axes", "coords"}
    assert not hasattr(
        rsa_nd.RsaData(np.ones((1, 1)), axes=("item", "space"), coords={}),
        "spectral",
    )


def test_prepared_rsa_data_space_contract_distinguishes_geometry_from_axis_name():
    abstract = rsa_nd.RsaData(
        np.ones((2, 3)),
        axes=("item", "space"),
        coords={"space": np.array(["a", "b", "c"])},
    )
    prepared = rsa_nd.PreparedRsaData(
        rsa_data=abstract,
        input_kind="rsa_data",
        space=None,
    )
    assert prepared.space is None

    with pytest.raises(ValueError, match="requires PreparedRsaData.space"):
        rsa_nd.PreparedRsaData(
            rsa_data=abstract,
            input_kind="epochs",
            space=None,
        )

    no_space_axis = rsa_nd.RsaData(
        np.ones((2, 3)),
        axes=("item", "time"),
        coords={"time": np.arange(3)},
    )
    with pytest.raises(ValueError, match="has no 'space' axis"):
        rsa_nd.PreparedRsaData(
            rsa_data=no_space_axis,
            input_kind="rsa_data",
            space=rsa_nd.SensorSpace(channel_names=("a",), info=None),
        )


def test_result_uses_rsa_axes_coords_contract():
    field_names = {field.name for field in fields(rsa_nd.RsaResult)}
    assert {"rsa", "axes", "coords"}.issubset(field_names)
    # The prepared input tensor is no longer carried on the result.
    assert "rsa_data" not in field_names
    assert "spectral_data" not in field_names


def test_generic_core_has_no_mne_coupling():
    import rsa_nd._core as rsa_core

    root = Path(__file__).resolve().parents[1]
    rsa_source = (root / "src" / "rsa_nd" / "_core.py").read_text(encoding="utf-8")

    assert "spatial_radius" not in inspect.signature(rsa_core.rsa_nd).parameters
    assert "EpochsTFR" not in rsa_source
    assert "EpochsSpectrum" not in rsa_source
    assert 'info["chs"]' not in rsa_source


def test_rsa_nd_with_named_array_matches_two_step_pattern():
    from rsa_nd import rsa_nd

    labels = np.array(["a", "b", "c", "d"])
    model_rdm = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    freq_values = np.array([6.0, 10.0, 20.0])
    time_values = np.array([0.0, 0.1, 0.2])
    data = (
        item_values[:, None, None]
        + freq_values[None, :, None]
        + time_values[None, None, :]
    )

    direct = rsa_nd(
        data,
        model_rdm,
        axes=("item", "freq", "time"),
        coords={"freq": freq_values, "time": time_values},
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    via_prep = rsa_nd(
        _prepare_rsa_data_only(
            data,
            axes=("item", "freq", "time"),
            coords={"freq": freq_values, "time": time_values},
        ),
        model_rdm,
        search_axes=("freq", "time"),
        radii={"freq": 0.0, "time": 0.0},
        labels=labels,
        labels_rdm_model=labels,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert direct.rsa.shape == (3, 3)
    assert direct.search_axes == ("freq", "time")
    assert np.allclose(direct.rsa, via_prep.rsa, equal_nan=True)
    assert np.allclose(direct.search_coords["freq"], freq_values)
    assert np.allclose(direct.search_coords["time"], time_values)


def test_rsa_nd_item_space_matches_direct_mne_rsa_array():
    import mne_rsa

    from rsa_nd import rsa_nd

    item_values = np.array([0.0, 1.0, 3.0, 6.0])
    data = np.column_stack([item_values, item_values**2])
    model_rdm = _ordinal_rdm(item_values)

    result = rsa_nd(
        data,
        model_rdm,
        axes=("item", "space"),
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    expected = mne_rsa.rsa_array(
        data[:, :, np.newaxis],
        model_rdm,
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert result.rsa.shape == ()
    assert result.rsa == np.asarray(expected)


def test_rsa_nd_item_space_time_searchlight():
    from rsa_nd import rsa_nd

    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    space_values = np.array(["s0", "s1"])
    time_values = np.array([0.0, 0.1, 0.2, 0.3])
    data = (
        item_values[:, np.newaxis, np.newaxis]
        + np.arange(len(space_values))[np.newaxis, :, np.newaxis]
        + time_values[np.newaxis, np.newaxis, :]
    )

    result = rsa_nd(
        data,
        _ordinal_rdm(item_values),
        axes=("item", "space", "time"),
        coords={"space": space_values, "time": time_values},
        search_axes=("time",),
        radii={"time": 0.1},
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert result.rsa.shape == (len(time_values),)
    assert result.search_axes == ("time",)
    assert np.allclose(result.search_coords["time"], time_values)


@pytest.mark.parametrize(
    "feature_axes",
    [
        ("space",),
        ("space", "time"),
    ],
)
def test_rsa_nd_searchlight_accepts_explicit_feature_axes(feature_axes):
    from rsa_nd import rsa_nd

    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    space_values = np.array(["s0", "s1"])
    time_values = np.array([0.0, 0.1, 0.2])
    data = (
        item_values[:, np.newaxis, np.newaxis]
        + np.arange(len(space_values))[np.newaxis, :, np.newaxis]
        + time_values[np.newaxis, np.newaxis, :]
    )

    explicit = rsa_nd(
        data,
        _ordinal_rdm(item_values),
        axes=("item", "space", "time"),
        coords={"space": space_values, "time": time_values},
        search_axes=("time",),
        feature_axes=feature_axes,
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )
    default = rsa_nd(
        data,
        _ordinal_rdm(item_values),
        axes=("item", "space", "time"),
        coords={"space": space_values, "time": time_values},
        search_axes=("time",),
        radii={"time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert explicit.rsa.shape == (len(time_values),)
    assert np.allclose(explicit.rsa, default.rsa, equal_nan=True)


def test_rsa_nd_space_freq_time_component_searchlight_with_multiple_models():
    from rsa_nd import rsa_nd

    item_values = np.array([0.0, 1.0, 2.0, 4.0])
    space_values = np.array(["s0", "s1", "s2"])
    freq_values = np.array([8.0, 12.0])
    time_values = np.array([0.0, 0.1, 0.2])
    component_values = np.array(["real", "imag"])
    data = (
        item_values[:, None, None, None, None]
        + np.arange(len(space_values))[None, :, None, None, None]
        + 0.01 * freq_values[None, None, :, None, None]
        + time_values[None, None, None, :, None]
        + np.arange(len(component_values))[None, None, None, None, :]
    )
    model_rdms = [
        _ordinal_rdm(item_values),
        _ordinal_rdm(item_values**2),
    ]

    result = rsa_nd(
        data,
        model_rdms,
        axes=("item", "space", "freq", "time", "component"),
        coords={
            "space": space_values,
            "freq": freq_values,
            "time": time_values,
            "component": component_values,
        },
        search_axes=("space", "time"),
        radii={"space": 0.0, "time": 0.0},
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
    )

    assert result.rsa.shape == (2, len(space_values), len(time_values))
    assert result.search_axes == ("space", "time")
    assert list(result.search_coords["space"]) == list(space_values)
    assert np.allclose(result.search_coords["time"], time_values)


def test_rsa_nd_repeated_item_labels_and_folds():
    from rsa_nd import rsa_nd

    labels = np.array(["a", "a", "b", "b", "c", "c"])
    label_values = {"a": 0.0, "b": 1.0, "c": 3.0}
    item_values = np.array([label_values[label] for label in labels])
    data = np.column_stack([item_values, item_values + 0.1])
    model_values = np.array([0.0, 1.0, 3.0])

    result = rsa_nd(
        data,
        _ordinal_rdm(model_values),
        axes=("item", "space"),
        labels=labels,
        labels_rdm_model=np.array(["a", "b", "c"]),
        data_rdm_metric="euclidean",
        rsa_metric="pearson",
        n_folds=2,
    )

    assert result.rsa.shape == ()
    assert float(result.rsa) == pytest.approx(1.0)


def test_rsa_nd_rejects_complex_data_without_spectral_preparation():
    from rsa_nd import rsa_nd

    complex_data = np.ones((3, 2, 4), dtype=complex)

    with np.testing.assert_raises_regex(ValueError, "real-valued data only"):
        rsa_nd(
            complex_data,
            np.eye(3),
            axes=("item", "space", "time"),
        )

    prepared = _prepare_rsa_data_only(
        complex_data,
        axes=("item", "space", "time"),
        coords={"time": np.arange(4)},
        tfr_feature="phase",
    )

    assert not np.iscomplexobj(prepared.data)
    assert prepared.axes == ("item", "space", "time", "component")


def test_dynamic_prepare_kwargs_come_from_prepare_signature():
    """Dynamic RSA should not hand-maintain a parallel prep-key grammar."""
    import rsa_nd._dynamic as dynamic

    expected = frozenset(inspect.signature(rsa_nd.prepare_rsa_data).parameters) - {
        "inst_or_data"
    }
    assert dynamic.prepare_rsa_data_kwarg_names() == expected
    assert dynamic._PREP_KEYS_RSA == expected


def test_renamed_demo_files_exist():
    root = Path(__file__).resolve().parents[1]

    assert (root / "examples" / "demo_rsa_toy_spectral.py").exists()
    assert (root / "examples" / "demo_rsa_dynamic_simulation.py").exists()
    assert (root / "examples" / "demo_rsa_sensor_feature_comparison.py").exists()
    assert (root / "examples" / "demo_rsa_sensor_phase_searchlight.py").exists()
    assert (root / "examples" / "demo_rsa_source_roi_searchlight.py").exists()


def test_implementation_is_split_into_focused_modules():
    import rsa_nd._blocks as blocks
    import rsa_nd._core as core
    import rsa_nd._dynamic as dynamic
    import rsa_nd._features as features
    import rsa_nd._prepare as prepare

    # Layer 1 (preparation) lives in _prepare; the spectral engine in _features.
    assert features.safe_log10_power.__module__ == "rsa_nd._features"
    assert prepare.prepare_rsa_data.__module__ == "rsa_nd._prepare"
    # Layer 2 frontend (runner + one-call) in _core.
    assert core.compute_rsa.__module__ == "rsa_nd._core"
    assert core.rsa_nd.__module__ == "rsa_nd._core"
    # The searchlight execution engine lives in _blocks.
    assert blocks.build_searchlight_plan.__module__ == "rsa_nd._blocks"
    assert blocks.FlatPatchGenerator.__module__ == "rsa_nd._blocks"
    assert blocks.run_searchlight_rsa.__module__ == "rsa_nd._blocks"
    # Dynamic RSA (frontend + its Model-RDM machinery) in _dynamic.
    assert dynamic.normalize_dynamic_model_rdms.__module__ == "rsa_nd._dynamic"
    assert dynamic.rsa_dynamic.__module__ == "rsa_nd._dynamic"


@pytest.mark.parametrize(
    "module_name",
    [
        "rsa_nd._core",
        "rsa_nd._prepare",
        "rsa_nd._dynamic",
        "rsa_nd._source",
        "rsa_nd._assemble",
        "rsa_nd._features",
        "rsa_nd._types",
    ],
)
def test_module_all_does_not_advertise_private_helpers(module_name):
    module = __import__(module_name, fromlist=["__all__"])
    private_exports = [
        name for name in getattr(module, "__all__", ()) if name.startswith("_")
    ]

    assert private_exports == []


def test_sensor_searchlight_radius_boundary_is_exclusive():
    """SensorSpace spatial radius uses a strict ``dist < radius`` boundary,
    matching the source searchlight and mne-rsa's own sensor searchlight
    (delegated to ``mne_rsa.searchlight._get_in_radius``).
    """
    import mne

    info = mne.create_info(["a", "b", "c"], sfreq=1.0, ch_types="eeg")
    locs = np.array([[0.0, 0, 0], [0.04, 0, 0], [0.08, 0, 0]])
    for ch, loc in zip(info["chs"], locs):
        ch["loc"][:3] = loc

    space = rsa_nd.SensorSpace(channel_names=("a", "b", "c"), info=info)

    # Channel "b" sits exactly 0.04 m from "a"; a strict boundary excludes it.
    patches, centers = space.searchlight_patches(0.04)
    assert patches[0].tolist() == [0]
    assert list(centers) == ["a", "b", "c"]

    # Just past the boundary, "b" is included.
    patches_wider, _ = space.searchlight_patches(0.0401)
    assert patches_wider[0].tolist() == [0, 1]
