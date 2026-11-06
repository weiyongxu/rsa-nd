"""Helpers shared by the tests."""

import mne

POSITIONS = {
    "Fz": (0.0, 0.06, 0.07),
    "Cz": (0.0, 0.0, 0.09),
    "Pz": (0.0, -0.06, 0.07),
    "Oz": (0.0, -0.09, 0.0),
    "C3": (-0.06, 0.0, 0.06),
    "C4": (0.06, 0.0, 0.06),
}


def make_info(ch_names, sfreq):
    """Make EEG info with fixed channel locations."""
    info = mne.create_info(list(ch_names), sfreq, "eeg")
    ch_pos = {name: POSITIONS[name] for name in ch_names}
    info.set_montage(mne.channels.make_dig_montage(ch_pos, coord_frame="head"))
    return info
