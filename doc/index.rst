Representational Similarity Analysis
------------------------------------

|unit_tests| |build_docs| |doi|

.. |unit_tests| image:: https://github.com/wmvanvliet/mne-rsa/workflows/unit%20tests/badge.svg
   :target: https://github.com/wmvanvliet/mne-rsa/actions?query=workflow%3A%22unit+tests%22

.. |build_docs| image:: https://github.com/wmvanvliet/mne-rsa/workflows/build-docs/badge.svg
   :target: https://github.com/wmvanvliet/mne-rsa/actions?query=workflow%3Abuild-docs

.. |doi| image:: https://zenodo.org/badge/194268560.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.11242874

This is a Python package for performing representational similarity
analysis (RSA) using
`MNE-Python <https://martinos.org/mne/stable/index.html>`__ data
structures. The RSA is computed using a “searchlight” approach.

Read more on RSA in the paper that introduced the technique:

Nikolaus Kriegeskorte, Marieke Mur and Peter Bandettini (2008).
Representational similarity analysis - connecting the branches of
systems neuroscience. Frontiers in Systems Neuroscience, 2(4).
https://doi.org/10.3389/neuro.06.004.2008

.. image:: https://raw.githubusercontent.com/wmvanvliet/mne-rsa/main/doc/rsa.png
   :width: 600
   :class: only-light

.. image:: https://raw.githubusercontent.com/wmvanvliet/mne-rsa/main/doc/rsa_dark.png
   :width: 600
   :class: only-dark


Installation
------------

The package can be installed either through PIP:  
``pip install mne-rsa``  
or through conda using the conda-forge channel:  
``conda install -c conda-forge mne-rsa``


Use cases
---------

This is what the package can do for you:

-  Compute RDMs on arbitrary data
-  Compute RDMs in a searchlight across:

   -  vertices/voxels and samples (source level)
   -  sensors and samples (sensor level)
   -  vertices/voxels only (source level)
   -  sensors only (sensor level)
   -  samples only (source and sensor level)

-  Use cross-validated distance metrics when computing RDMs
-  And of course: compute RSA between RDMs

Supported metrics for comparing RDMs:

-  Spearman correlation (the default)
-  Pearson correlation
-  Kendall’s Tau-A
-  Linear regression (when comparing multiple RDMs at once)
-  Partial correlation (when comparing multiple RDMs at once)

Juicy bits of the API
---------------------

.. code:: python

   compute_rdm(data, metric='correlation', **kwargs)

   rsa_stcs(stcs, rdm_model, src, spatial_radius=0.04, temporal_radius=0.1,
            stc_rdm_metric='correlation', stc_rdm_params=dict(),
            rsa_metric='spearman', y=None, n_folds=1, sel_vertices=None,
            tmin=None, tmax=None, n_jobs=1, verbose=False)

   rsa_evokeds(evokeds, rdm_model, noise_cov=None, spatial_radius=0.04,
               temporal_radius=0.1, evoked_rdm_metric='correlation',
               evoked_rdm_params=dict(), rsa_metric='spearman', y=None,
               n_folds=1, picks=None, tmin=None, tmax=None, n_jobs=1,
               verbose=False)

   rsa_epochs(epochs, rdm_model, noise_cov=None, spatial_radius=0.04,
              temporal_radius=0.1, epochs_rdm_metric='correlation',
              epochs_rdm_params=dict(), rsa_metric='spearman', y=None,
              n_folds=1, picks=None, tmin=None, tmax=None, n_jobs=1,
              verbose=False)

   rsa_nifti(image, rdm_model, spatial_radius=0.01,
             image_rdm_metric='correlation', image_rdm_params=dict(),
             rsa_metric='spearman', y=None, n_folds=1, roi_mask=None,
             brain_mask=None, n_jobs=1, verbose=False)

Example usage
-------------

Basic example on the EEG “kiloword” data:

.. code:: python

   import mne
   import rsa
   data_path = mne.datasets.kiloword.data_path(verbose=True)
   epochs = mne.read_epochs(data_path + '/kword_metadata-epo.fif')
   # Compute the model RDM using all word properties
   rdm_model = rsa.compute_rdm(epochs.metadata.iloc[:, 1:].values)
   evoked_rsa = rsa.rsa_epochs(epochs, rdm_model,
                               spatial_radius=0.04, temporal_radius=0.01,
                               verbose=True)

Documentation
-------------

For quick guides on how to do specific things, see the
`examples <https://users.aalto.fi/~vanvlm1/mne-rsa/auto_examples/index.html>`__.

Finally, there is the
`API reference <https://users.aalto.fi/~vanvlm1/mne-rsa/api.html>`__
documentation.

Integration with other packages
-------------------------------

I mainly wrote this package to perform RSA analysis on MEG data. Hence,
integration functions with `MNE-Python <https://mne.tools>`__ are
provided. There is also some integration with `nipy <https://nipy.org>`__ for
fMRI.

Performance
-----------

This package aims to be fast and memory efficient. An important design
feature is that under the hood, everything operates on generators. The
searchlight routines produce a generator of RDMs which are consumed by a
generator of RSA values. Parallel processing is also supported, so you
can use all of your CPU cores.

Development
-----------

Here is how to set up the package as a developer:

.. code:: bash

   git clone git@github.com:wmvanvliet/mne-rsa.git
   cd mne-rsa
   python setup.py develop --user

.. toctree::
   :hidden:

   Tutorial <auto_examples/tutorial>
   Examples <auto_examples/index>
   API Reference <api>
