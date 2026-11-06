import ast
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "demo_rsa_sensor_feature_comparison.py"
)
SUPPORT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "_support" / "rsa_sample.py"
)


def test_sample_sensor_feature_comparison_exists_and_uses_direct_spectral_rsa():
    demo_path = DEMO_PATH

    assert demo_path.exists()
    source = demo_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "main" in functions
    assert "prepare_rsa_data" in source
    assert "rsa_nd" in source
    assert "compute_tfr" in source
    assert "compute_psd" in source
    assert "mne_rsa.rsa_epochs" in source
    assert "compute_covariance" in source
    assert "noise_cov=" in source
    assert "run_mixed_sensor_whitened_rsa" in source
    assert "rsa_epochs_spectral" not in source
    assert "spectrum_to_frequency_epochs" not in source
    assert "mne_sample_all_outputs_composite.png" in source
    assert "MNE sample: all-band TFR RSA by sensor type" not in source
    assert "plot_sensor_comparison_ax" not in source
    assert "_plot_all_band_with_time_domain_ax" in source
    assert "save_csv(" not in source
    assert "save_band_csv(" not in source
    assert "save_tf_csv(" not in source
    assert "separate_band_results" in source
    assert "all bands" in source
    assert "time_domain_rsa" in source
    assert "make_conditionwise_induced_epochs" in source
    assert "subtract_evoked" in source


def test_sample_sensor_feature_comparison_exposes_sensor_type_options():
    import demo_rsa_sensor_feature_comparison as demo

    assert demo.SENSOR_TYPES == ("eeg", "mag", "grad")
    assert demo.validate_sensor_types(["eeg", "grad"]) == ("eeg", "grad")
    assert demo.COMPOSITE_OUTPUT.name == "mne_sample_all_outputs_composite.png"
    assert demo.OUT_DIR == DEMO_PATH.parent / "output"
    assert demo.COMPOSITE_OUTPUT.parent == DEMO_PATH.parent / "output"


def test_sample_sensor_feature_comparison_uses_standard_event_related_epoch_window():
    source = SUPPORT_PATH.read_text(encoding="utf-8")

    assert source.count("tmax=0.5") >= 2
    assert "tmax=1.0" not in source


def test_sample_sensor_feature_comparison_uses_total_tfr_without_evoked_subtraction():
    demo_path = DEMO_PATH
    source = demo_path.read_text(encoding="utf-8")

    assert "induced_epochs = make_conditionwise_induced_epochs(epochs)" not in source
    assert "tfr = epochs.compute_tfr(" in source
    assert "TFR mode: total power" in source


def test_sample_sensor_feature_comparison_compares_complex_tfr_feature_modes():
    demo_path = DEMO_PATH
    source = demo_path.read_text(encoding="utf-8")

    assert 'output="complex"' in source
    assert "TFR_FEATURE_MODES" in source
    for mode_name in ["power", "amplitude", "real_imag", "amp_phase", "phase"]:
        assert f'"{mode_name}"' in source
    assert "tfr_feature=tfr_feature" in source
    assert "log=tfr_log" in source
    assert "feature_mode_results" in source


def test_sample_sensor_feature_comparison_cleans_only_generated_output_files(tmp_path):
    import demo_rsa_sensor_feature_comparison as demo

    keep = tmp_path / "notes.txt"
    generated = tmp_path / "mne_sample_old.csv"
    keep.write_text("keep", encoding="utf-8")
    generated.write_text("remove", encoding="utf-8")

    removed = demo.clean_output_dir(tmp_path)

    assert removed == [generated]
    assert keep.exists()
    assert not generated.exists()


def test_make_conditionwise_induced_epochs_subtracts_each_condition_evoked():
    import demo_rsa_sensor_feature_comparison as demo

    info = mne.create_info(["EEG 001"], sfreq=100.0, ch_types="eeg")
    data = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[3.0, 4.0, 5.0]],
            [[10.0, 20.0, 30.0]],
            [[14.0, 24.0, 34.0]],
        ]
    )
    events = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 3],
            [3, 0, 3],
        ]
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=0.0,
        event_id={"audio/left": 1, "visual/left": 3},
        verbose=False,
    )

    induced = demo.make_conditionwise_induced_epochs(epochs)

    assert np.allclose(induced["audio/left"].get_data().mean(axis=0), 0.0)
    assert np.allclose(induced["visual/left"].get_data().mean(axis=0), 0.0)
    assert np.array_equal(induced.events, epochs.events)


def test_make_conditionwise_induced_epochs_matches_manual_interleaved_residuals():
    import demo_rsa_sensor_feature_comparison as demo

    info = mne.create_info(["EEG 001", "EEG 002"], sfreq=100.0, ch_types="eeg")
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]],
            [[16.0, 26.0, 36.0], [17.0, 27.0, 37.0]],
        ]
    )
    events = np.array(
        [
            [0, 0, 1],
            [1, 0, 3],
            [2, 0, 1],
            [3, 0, 3],
        ]
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=0.0,
        event_id={"audio/left": 1, "visual/left": 3},
        verbose=False,
    )

    induced = demo.make_conditionwise_induced_epochs(epochs)

    expected = data.copy()
    for event_code in (1, 3):
        mask = events[:, 2] == event_code
        expected[mask] = data[mask] - data[mask].mean(axis=0)

    assert np.allclose(induced.get_data(), expected)
    assert np.array_equal(induced.events, epochs.events)
    assert np.array_equal(induced.selection, epochs.selection)


def test_tf_map_panel_adds_colorbar():
    import demo_rsa_sensor_feature_comparison as demo

    fig, ax = plt.subplots()
    demo._plot_tf_map_ax(
        fig,
        ax,
        np.array([0.0, 0.1]),
        np.array([8.0, 10.0]),
        np.array([[0.1, -0.2], [0.3, 0.0]]),
        title="test",
    )

    assert len(fig.axes) == 2
    plt.close(fig)


def test_all_band_panel_overlays_time_domain_curve():
    import demo_rsa_sensor_feature_comparison as demo

    fig, ax = plt.subplots()
    demo._plot_all_band_with_time_domain_ax(
        ax,
        np.array([0.0, 0.1]),
        np.array([0.2, 0.3]),
        np.array([0.0, 0.1]),
        np.array([0.1, 0.4]),
        title="test",
    )

    assert [line.get_label() for line in ax.lines[:2]] == [
        "all-band TFR",
        "mne_rsa.rsa_epochs",
    ]
    plt.close(fig)


def test_composite_plotter_writes_single_figure(tmp_path):
    import demo_rsa_sensor_feature_comparison as demo

    summary = {
        "sensor_type": "eeg",
        "time_centers": np.array([0.0, 0.1]),
        "all_bands_rsa": np.array([0.2, 0.3]),
        "band_results": {
            "theta": np.array([0.1, 0.2]),
            "alpha": np.array([0.3, 0.1]),
        },
        "feature_mode_times": np.array([0.0, 0.1]),
        "feature_mode_results": {
            "power": np.array([0.2, 0.3]),
            "real+imag": np.array([0.1, 0.2]),
        },
        "tf_times": np.array([0.0, 0.1]),
        "tf_freqs": np.array([8.0, 10.0]),
        "tf_map_rsa": np.array([[0.1, -0.2], [0.3, 0.0]]),
        "psd_freqs": np.array([8.0, 10.0]),
        "psd_rsa": np.array([0.4, 0.1]),
        "time_domain_times": np.array([0.0, 0.1]),
        "time_domain_rsa": np.array([0.1, 0.4]),
    }
    out = tmp_path / "composite.png"

    mixed = {
        "times": np.array([0.0, 0.1]),
        "rsa": np.array([0.5, 0.2]),
        "peak_rsa": 0.5,
        "peak_time": 0.0,
    }

    demo.plot_output_composite([summary], mixed_result=mixed, path=out)

    assert out.exists()


def test_all_band_panel_can_overlay_mixed_whitened_curve():
    import demo_rsa_sensor_feature_comparison as demo

    fig, ax = plt.subplots()
    demo._plot_all_band_with_time_domain_ax(
        ax,
        np.array([0.0, 0.1]),
        np.array([0.2, 0.3]),
        np.array([0.0, 0.1]),
        np.array([0.1, 0.4]),
        title="test",
        mixed_result={
            "times": np.array([0.0, 0.1]),
            "rsa": np.array([0.2, 0.5]),
        },
    )

    assert [line.get_label() for line in ax.lines[:3]] == [
        "all-band TFR",
        "mne_rsa.rsa_epochs",
        "mixed MEG+EEG, whitened",
    ]
    plt.close(fig)
