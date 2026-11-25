.. _api_documentation:

API Documentation
-----------------

.. currentmodule:: mne_rsa

Main MNE-RSA functions
^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: function.rst

    rsa_epochs
    rsa_evokeds
    rsa_stcs
    rsa_nifti

Constructing RDMs
^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: function.rst

    compute_rdm
    compute_rdm_cv
    rdm_epochs
    rdm_stcs
    rdm_nifti

.. autosummary::
    :toctree: functions
    :template: class.rst

    rdm_array

Performing RSA
^^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: function.rst

    rsa
    rsa_array
    rsa_gen

Making searchlight patches
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: class.rst

    searchlight

Visualization
^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: function.rst

    plot_rdms
    plot_rdms_topo
    plot_roi_map


Utility functions
^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: functions
    :template: function.rst

    create_folds
    squareform
