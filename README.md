# rsa-nd

**Representational similarity analysis (RSA) over any EEG/MEG features and
dimensions.**

`rsa-nd` extends [MNE-Python](https://mne.tools/) and
[mne-rsa](https://mne.tools/mne-rsa/). Give it an MNE object or a named N-D
array and say which axes to search over. It builds the feature tensor, runs the
searchlight through `mne-rsa`, and returns a result that keeps its axis names
and coordinates.

```python
from rsa_nd import compute_rsa

result = compute_rsa(
    tfr,  # an mne EpochsTFR
    model_rdm,
    freq={"alpha": (8, 13), "beta": (13, 30)},
    search_axes=("space", "time"),
    radii={"space": 0.04, "time": 0.02},
)
print(result.axes, result.rsa.shape)
```

As in `mne-rsa`, each MNE event code marks one item, so epochs that share a code
are treated as repeats of the same item. If that doesn't match your model RDM,
pass `labels=` and `labels_rdm_model=` to say which row belongs to which item.

## What it adds to mne-rsa

`mne-rsa` provides the RSA engine and ready-made workflows for epochs, evokeds
and source estimates. `rsa-nd` builds on that engine and adds:

- **More input types:** `Epochs`, `EpochsTFR`, `EpochsSpectrum`, and lists of
  `Evoked`, `AverageTFR`, `Spectrum` or source estimates, plus any named N-D
  array.
- **One vocabulary for every dimension:** `time`, `freq` and `space` are all
  searchable axes. Each can be searched with a radius, kept as a single value,
  grouped into named bands or windows, or pooled into the features.
- **Time-frequency features:** power, amplitude, phase, real/imaginary and
  amplitude/phase projections of complex TFRs.
- **Source ROI time-frequency RSA:** pass source estimates and `mne.Label`s
  directly.
- **Dynamic RSA:** `rsa_dynamic` and `rsa_dynamic_windows` for time-lagged and
  windowed model comparisons.
- **Self-describing results:** `RsaResult` keeps axes and coordinates, converts
  to an MNE container with `.to_mne()`, and saves to HDF5 with `.save()`.

## Install

```bash
pip install git+https://github.com/weiyongxu/rsa-nd.git
```

`rsa-nd` needs Python 3.10+, MNE-Python 1.7+ and mne-rsa 1.x.

## Quick start

### Any named array

```python
import numpy as np
from rsa_nd import rsa_nd

rng = np.random.default_rng(0)
data = rng.standard_normal((8, 5, 20))  # items x freq x time
model_rdm = np.abs(np.subtract.outer(np.arange(8.0), np.arange(8.0)))

result = rsa_nd(
    data,
    model_rdm,
    axes=("item", "freq", "time"),
    coords={"freq": [4, 8, 13, 20, 30], "time": np.linspace(0, 0.95, 20)},
    search_axes=("time",),
    radii={"time": 0.1},
)
print(result.axes, result.rsa.shape)  # ('time',) (20,)
```

### Prepare once, test many models

`compute_rsa()` is the one-call shortcut. To score several models against the
same features, prepare the data once:

```python
from rsa_nd import prepare_rsa_data, rsa_nd

prepared = prepare_rsa_data(tfr, freq={"alpha": (8, 13)}, log=True)
semantic = rsa_nd(
    prepared,
    semantic_rdm,
    search_axes=("space", "time"),
    radii={"space": 0.04, "time": 0.02},
)
visual = rsa_nd(
    prepared,
    visual_rdm,
    search_axes=("space", "time"),
    radii={"space": 0.04, "time": 0.02},
)
```

### Source ROI time-frequency RSA

```python
result = compute_rsa(
    stcs,
    model_rdm,
    space=labels,
    tfr={"freqs": np.arange(8, 31), "n_cycles": 2.0},
    search_axes=("space", "time"),
    radii={"time": 0.02},
)
```

### Save and reload

```python
from rsa_nd import RsaResult

result.save("result.h5")
result = RsaResult.load("result.h5")
evoked_like = result.to_mne()
```

## Examples

Runnable demos are in `examples/`. Those that use real data download the MNE
sample dataset on first use.

```bash
uv run python examples/demo_rsa_toy_spectral.py
uv run python examples/demo_rsa_sensor_feature_comparison.py
uv run python examples/demo_rsa_source_roi_searchlight.py
```

`examples/demo_rsa_epochs_implementation_comparison.py` compares `rsa-nd`
directly with `mne_rsa.rsa_epochs` on the same data.

## Development

```bash
uv sync
uv run pytest tests/ -m "not slowtest"
uv run ruff check .
uv run sphinx-build -b html docs/source docs/build/html
```

## Citing

See [CITATION.cff](CITATION.cff). Please also cite
[MNE-Python](https://mne.tools/stable/documentation/cite.html) and
[mne-rsa](https://mne.tools/mne-rsa/).

## License

MIT License. See [LICENSE.txt](LICENSE.txt).

## Acknowledgments

`rsa-nd` builds on [MNE-Python](https://mne.tools/) and
[mne-rsa](https://mne.tools/mne-rsa/) by Marijn van Vliet and contributors.
