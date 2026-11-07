Quick start
===========

Use :func:`rsa_nd.compute_rsa` for the one-call workflow:

.. code-block:: python

   from rsa_nd import compute_rsa

   result = compute_rsa(
       tfr,
       model_rdm,
       freq={"alpha": (8, 13), "beta": (13, 30)},
       search_axes=("space", "time"),
       radii={"space": 0.04, "time": 0.02},
   )
   print(result.axes, result.rsa.shape)

Each MNE event code marks one item, so epochs that share a code are treated as
repeats of the same item. Pass ``labels=`` and ``labels_rdm_model=`` to set the
mapping between epochs and model RDM rows yourself.

To score several models against the same features, call
:func:`rsa_nd.prepare_rsa_data` once and pass the prepared data to
:func:`rsa_nd.rsa_nd` for each model.

An :class:`rsa_nd.RsaResult` converts to an MNE container with
``result.to_mne()`` and saves with ``result.save(...)``. Load it again with
``RsaResult.load(...)``.

A real analysis
---------------

``rsa-nd`` is one step in a normal MNE pipeline, not a replacement for it:

.. list-table::
   :header-rows: 1

   * - Step
     - Tool
   * - Clean, epoch, compute source estimates
     - MNE-Python
   * - Build the model RDMs
     - :func:`mne_rsa.compute_rdm`, or your own
   * - Build features, search, and label the results
     - ``rsa-nd``
   * - Compute the data RDMs and compare them with the model
     - ``mne-rsa``, called by ``rsa-nd``
   * - Plot and run group statistics
     - MNE-Python, through ``result.to_mne()`` or ``result.rsa``

A typical study runs RSA per participant, then tests the scores across the group:

.. code-block:: python

   import mne
   import mne_rsa
   import numpy as np
   from mne.stats import permutation_cluster_1samp_test

   from rsa_nd import compute_rsa

   # Model: one row per stimulus, in the order of the sorted event codes
   model_rdm = mne_rsa.compute_rdm(stimulus_features, metric="euclidean")

   freqs = np.arange(4.0, 31.0)
   scores = []
   for subject in subjects:
       # Features: your usual MNE pipeline
       epochs = mne.read_epochs(f"{subject}-epo.fif")
       tfr = epochs.compute_tfr(
           "morlet", freqs=freqs, n_cycles=freqs / 2, return_itc=False, average=False
       )

       # RSA: the question is set by the axes you search
       result = compute_rsa(
           tfr,
           model_rdm,
           freq={"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)},
           log=True,
           search_axes=("band", "time"),
           radii={"time": 0.02},
       )
       result.save(f"{subject}-rsa.h5")
       scores.append(result.rsa)

   # Group statistics: scores have shape (subjects, bands, times)
   t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
       np.stack(scores), tail=1
   )

To ask a different question, change only the :func:`~rsa_nd.compute_rsa`
arguments. For example, ``search_axes=("space", "time")`` gives a sensor map for
each band.
