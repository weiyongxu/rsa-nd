import demo_rsa_sensor_phase_searchlight as demo
import numpy as np


def test_peak_info_reports_sensor_time_and_value():
    rsa_vals = np.array([[0.1, 0.2, -0.1], [0.05, 0.4, 0.3]])
    times = np.array([-0.1, 0.0, 0.1])
    sensors = np.array(["A", "B"], dtype=object)

    peak = demo.peak_info(rsa_vals, times, sensors)

    assert peak == {
        "sensor_idx": 1,
        "time_idx": 1,
        "sensor": "B",
        "time": 0.0,
        "value": 0.4,
    }


def test_permutation_null_summary_uses_observed_max():
    summary = demo.permutation_null_summary(0.5, [0.1, 0.6, 0.3])

    assert summary["n_permutations"] == 3
    assert summary["median"] == 0.3
    assert summary["q95"] == np.quantile([0.1, 0.6, 0.3], 0.95)
    assert summary["p_value"] == 0.5


def test_summary_rows_include_power_and_phase_null_fields():
    result = {
        "sensor_type": "mag",
        "n_channels": 2,
        "power_peak": {"sensor": "MEG 0011", "time": 0.04, "value": 0.2},
        "phase_peak": {"sensor": "MEG 0021", "time": 0.09, "value": 0.5},
        "phase_null": {
            "n_permutations": 10,
            "median": 0.12,
            "q95": 0.18,
            "max": 0.2,
            "p_value": 0.09,
        },
    }

    assert demo.summary_rows([result]) == [
        {
            "sensor_type": "mag",
            "n_channels": 2,
            "feature": "power",
            "peak_sensor": "MEG 0011",
            "peak_time": 0.04,
            "peak_rsa": 0.2,
            "phase_null_n": "",
            "phase_null_median": "",
            "phase_null_q95": "",
            "phase_null_max": "",
            "phase_null_p": "",
        },
        {
            "sensor_type": "mag",
            "n_channels": 2,
            "feature": "phase",
            "peak_sensor": "MEG 0021",
            "peak_time": 0.09,
            "peak_rsa": 0.5,
            "phase_null_n": 10,
            "phase_null_median": 0.12,
            "phase_null_q95": 0.18,
            "phase_null_max": 0.2,
            "phase_null_p": 0.09,
        },
    ]


def test_sensor_spatiotemporal_demo_outputs_to_examples_output():
    expected = demo.Path(demo.__file__).resolve().parent / "output"

    assert demo.OUT_DIR == expected
    assert demo.COMPOSITE_PNG.parent == expected
    assert demo.SUMMARY_CSV.parent == expected
