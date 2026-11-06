from pathlib import Path

import demo_rsa_source_roi_searchlight as demo
import numpy as np

DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "demo_rsa_source_roi_searchlight.py"
)


def test_peak_info_reports_roi_time_and_value():
    rsa_vals = np.array([[0.1, -0.2, 0.3], [0.05, 0.4, 0.2]])
    times = np.array([-0.1, 0.0, 0.1])
    roi_names = ["a-lh", "b-rh"]

    peak = demo.peak_info(rsa_vals, times, roi_names)

    assert peak == {
        "roi_idx": 1,
        "time_idx": 1,
        "roi_name": "b-rh",
        "time": 0.0,
        "value": 0.4,
    }


def test_permutation_null_summary_uses_max_statistic():
    summary = demo.permutation_null_summary(
        observed_max=0.6,
        permutation_maxima=np.array([0.2, 0.4, 0.7]),
    )

    assert summary["n_permutations"] == 3
    assert summary["median"] == 0.4
    assert summary["q95"] == np.quantile([0.2, 0.4, 0.7], 0.95)
    assert summary["p_value"] == 0.5


def test_source_demo_uses_general_compute_rsa_source_roi_path():
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert "compute_rsa" in source
    assert "rsa_stcs_rois_spectral" not in source
    assert "extract_label_time_course" not in source
    assert "ROI vertex-patch" in source
    assert "Phase max-null" in source


def test_source_demo_uses_normal_orientation_for_phase_features():
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert 'pick_ori="normal"' in source


def test_source_demo_outputs_to_examples_output():
    expected = DEMO_PATH.parent / "output"

    assert demo.OUT_DIR == expected
    assert demo.COMPOSITE_PNG.parent == expected
