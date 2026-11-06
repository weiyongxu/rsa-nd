"""MNE sample-data helpers shared by RSA example scripts."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np

SENSOR_TYPES = ("eeg", "mag", "grad")
SENSOR_LABELS = {
    "eeg": "EEG",
    "mag": "magnetometer",
    "grad": "gradiometer",
}
EVENT_ID = {
    "audio/left": 1,
    "audio/right": 2,
    "visual/left": 3,
    "visual/right": 4,
}
TFR_FREQS = np.arange(4.0, 31.0, 2.0)
TEMPORAL_RADIUS = 0.05
RSA_TMAX = 0.5


def validate_sensor_types(sensor_types: Any) -> tuple[str, ...]:
    """Return sensor types as a tuple after validating supported choices.

    Parameters
    ----------
    sensor_types : iterable of str
        Sensor types requested by the demo.

    Returns
    -------
    tuple of str
        Validated sensor-type names.
    """

    selected = tuple(sensor_types)
    unknown = [
        sensor_type for sensor_type in selected if sensor_type not in SENSOR_TYPES
    ]
    if unknown:
        raise ValueError(
            f"Unknown sensor types: {unknown}. Valid choices are {SENSOR_TYPES}."
        )
    return selected


def sensitivity_metric(event_id_1: Any, event_id_2: Any) -> float:
    """Return the MNE-RSA tutorial auditory-left sensitivity RDM entry."""

    a = int(np.asarray(event_id_1).ravel()[0])
    b = int(np.asarray(event_id_2).ravel()[0])
    if a == 1 and b == 1:
        return 0.0
    if a == 2 and b == 2:
        return 0.5
    if (a, b) in ((1, 2), (2, 1)):
        return 0.5
    return 1.0


def load_sample_raw_events(
    *, verbose: bool = False
) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """Load the raw file and events from the MNE sample dataset.

    Parameters
    ----------
    verbose : bool
        Whether to let the sample-data path lookup print progress messages.

    Returns
    -------
    raw : mne.io.BaseRaw
        Sample auditory/visual raw data.
    events : ndarray, shape (n_events, 3)
        MNE events array.
    """

    sample_root = mne.datasets.sample.data_path(download=True, verbose=verbose)
    sample_path = sample_root / "MEG" / "sample"
    raw = mne.io.read_raw_fif(
        sample_path / "sample_audvis_filt-0-40_raw.fif",
        preload=False,
        verbose=False,
    )
    events = mne.read_events(sample_path / "sample_audvis_filt-0-40_raw-eve.fif")
    return raw, events


def load_audio_visual_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    sensor_type: str = "eeg",
    n_per_condition: int = 20,
) -> mne.Epochs:
    """Load a small audio/visual subset for one sensor type."""

    validate_sensor_types([sensor_type])
    event_id = {"audio/left": 1, "visual/left": 3}
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-0.2,
        tmax=0.5,
        baseline=(-0.2, 0.0),
        picks=sensor_type,
        preload=True,
        proj=True,
        verbose=False,
    )
    epochs.resample(100, verbose=False)

    audio = epochs["audio/left"][:n_per_condition]
    visual = epochs["visual/left"][:n_per_condition]
    return mne.concatenate_epochs([audio, visual], verbose=False)


def load_mixed_audio_visual_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    n_per_condition: int = 20,
) -> mne.Epochs:
    """Load mixed MEG+EEG audio/visual epochs for whitening comparison."""

    event_id = {"audio/left": 1, "visual/left": 3}
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=-0.2,
        tmax=0.5,
        baseline=(-0.2, 0.0),
        picks=("meg", "eeg"),
        preload=True,
        proj=True,
        verbose=False,
    )
    epochs.resample(100, verbose=False)

    audio = epochs["audio/left"][:n_per_condition]
    visual = epochs["visual/left"][:n_per_condition]
    return mne.concatenate_epochs([audio, visual], verbose=False)


def load_balanced_sensor_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    sensor_type: str,
    n_per_condition: int,
) -> mne.Epochs:
    """Load balanced four-condition sample epochs for one sensor type."""

    validate_sensor_types([sensor_type])
    epochs = mne.Epochs(
        raw,
        events,
        EVENT_ID,
        tmin=-0.3,
        tmax=0.7,
        baseline=(-0.2, 0.0),
        picks=sensor_type,
        preload=True,
        proj=True,
        verbose=False,
    )
    epochs.resample(100, verbose=False)
    pieces = [epochs[name][:n_per_condition] for name in EVENT_ID]
    return mne.concatenate_epochs(pieces, verbose=False)


def load_balanced_audvis_epochs(
    *,
    n_per_condition: int = 10,
) -> tuple[mne.Epochs, Any]:
    """Load balanced four-condition MEG sample epochs and sample root path."""

    sample_root = mne.datasets.sample.data_path(download=True, verbose=False)
    sample_path = sample_root / "MEG" / "sample"
    raw = mne.io.read_raw_fif(
        sample_path / "sample_audvis_filt-0-40_raw.fif",
        preload=False,
        verbose=False,
    )
    events = mne.read_events(sample_path / "sample_audvis_filt-0-40_raw-eve.fif")

    # The default Epochs window (-0.2 to 0.5 s) is too short for a 4 Hz Morlet
    # wavelet at sfreq=100; widen it slightly so the lowest theta wavelet fits.
    epochs = mne.Epochs(
        raw,
        events,
        EVENT_ID,
        tmin=-0.3,
        tmax=0.7,
        baseline=(-0.2, 0.0),
        picks="meg",
        preload=True,
        proj=True,
        verbose=False,
    )
    epochs.resample(100, verbose=False)

    pieces = [epochs[name][:n_per_condition] for name in EVENT_ID]
    return mne.concatenate_epochs(pieces, verbose=False), sample_root


def condition_model_rdm(epochs: mne.Epochs) -> np.ndarray:
    """Return a model RDM with 0 within modality and 1 across modality."""

    condition_codes = epochs.events[:, 2]
    return (condition_codes[:, None] != condition_codes[None, :]).astype(float)
