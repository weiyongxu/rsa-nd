API reference
=============

Everything below is imported from ``rsa_nd``. An ``RsaResult`` converts to an
MNE container with ``result.to_mne()`` and persists with ``result.save(...)`` /
``RsaResult.load(...)``.

Data Containers
---------------

.. autoclass:: rsa_nd.RsaData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rsa_nd.PreparedRsaData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rsa_nd.RsaResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rsa_nd.DynamicInfo
   :members:
   :undoc-members:
   :show-inheritance:

Space Descriptions
------------------

.. autoclass:: rsa_nd.SpaceDescription
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rsa_nd.SensorSpace
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rsa_nd.SourceSpace
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: rsa_nd.RsaTemplate

Preparation
-----------

.. autodata:: rsa_nd.RsaInputKind

.. autofunction:: rsa_nd.prepare_rsa_data

RSA Workflows
-------------

.. autofunction:: rsa_nd.compute_rsa

.. autofunction:: rsa_nd.rsa_nd

.. autofunction:: rsa_nd.rsa_dynamic

.. autofunction:: rsa_nd.rsa_dynamic_windows

Source-space Helpers
--------------------

.. autofunction:: rsa_nd.source_roi_patches

Feature Helpers
---------------

.. autofunction:: rsa_nd.safe_log10_power
