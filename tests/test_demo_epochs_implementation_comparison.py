from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "demo_rsa_epochs_implementation_comparison.py"
)


def test_plain_epochs_comparison_demo_exists_and_uses_plain_epoch_rsa():
    assert DEMO_PATH.exists()
    source = DEMO_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "main" in functions
    assert "mne_rsa.rsa_epochs" in source
    assert "prepare_rsa_data" in source
    assert "rsa_nd" in source
    assert ".to_mne()" in source
    assert "load_audio_visual_epochs" in source
    assert "condition_model_rdm" in source
    assert "compute_tfr" not in source
    assert "compute_psd" not in source
    assert "prepare_spectral_data" not in source
    assert "mne_sample_plain_epochs_rsa_comparison.png" in source


def test_plain_epochs_comparison_demo_outputs_to_examples_output():
    import demo_rsa_epochs_implementation_comparison as demo

    expected = DEMO_PATH.parent / "output"

    assert demo.OUT_DIR == expected
    assert demo.OUTPUT_PNG.parent == expected


def test_align_to_reference_times_selects_matching_time_columns():
    import demo_rsa_epochs_implementation_comparison as demo

    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )
    source_times = np.array([-0.1, 0.0, 0.1, 0.2])
    reference_times = np.array([0.0, 0.2])

    aligned = demo.align_to_reference_times(data, source_times, reference_times)

    assert np.array_equal(aligned, np.array([[2.0, 4.0], [6.0, 8.0]]))


def test_rsa_difference_summary_reports_allclose_and_max_abs_diff():
    import demo_rsa_epochs_implementation_comparison as demo

    left = np.array([[0.0, 1.0], [2.0, 3.0]])
    right = np.array([[0.0, 1.0 + 1e-10], [2.0, 3.1]])

    summary = demo.rsa_difference_summary(left, right, rtol=1e-7, atol=1e-9)

    assert summary["allclose"] is False
    assert np.isclose(summary["max_abs_diff"], 0.1)


def test_plot_comparison_writes_three_panel_figure(tmp_path):
    import demo_rsa_epochs_implementation_comparison as demo

    result = {
        "times": np.array([0.0, 0.1, 0.2]),
        "sensors": np.array(["EEG 001", "EEG 002"], dtype=object),
        "mne_rsa": np.array([[0.1, 0.2, 0.0], [0.2, 0.1, -0.1]]),
        "rsa_nd": np.array([[0.1, 0.2, 0.0], [0.2, 0.1, -0.1]]),
        "difference": np.zeros((2, 3)),
        "summary": {"max_abs_diff": 0.0, "allclose": True},
        "sensor_type": "eeg",
        "n_epochs": 4,
        "n_channels": 2,
    }
    out = tmp_path / "comparison.png"

    for _ in range(4):
        plt.figure()
    assert plt.get_fignums()

    demo.plot_comparison(result, path=out)

    assert out.exists()
    assert plt.get_fignums() == []
