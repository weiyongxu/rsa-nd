rsa-nd
======

Representational similarity analysis (RSA) over any EEG/MEG features and
dimensions, built on `MNE-Python <https://mne.tools/>`_ and
`mne-rsa <https://mne.tools/mne-rsa/>`_.

Give ``rsa-nd`` an MNE object or a named N-D array and say which axes to search
over. It builds the feature tensor, runs the searchlight through ``mne-rsa``,
and returns a result that keeps its axis names and coordinates.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   auto_examples/index
   api
