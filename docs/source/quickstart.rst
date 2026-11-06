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
