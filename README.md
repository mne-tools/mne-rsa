# MNE-RSA

[![Unit tests](https://github.com/mne-tools/mne-rsa/workflows/unit%20tests/badge.svg)](https://github.com/mne-tools/mne-rsa/actions?query=workflow%3A%22unit+tests%22)
[![docs](https://github.com/mne-tools/mne-rsa/workflows/build-docs/badge.svg)](https://github.com/mne-tools/mne-rsa/actions?query=workflow%3Abuild-docs)
[![doi](https://zenodo.org/badge/194268560.svg)](https://zenodo.org/doi/10.5281/zenodo.11242874)

This is a Python package for performing representational similarity analysis (RSA) using [MNE-Python](https://martinos.org/mne/stable/index.html>) data structures.
The main use-case is to perform RSA using a “searchlight” approach through time and/or a
volumetric or surface source space.

Read more on RSA in the paper that introduced the technique:

Nikolaus Kriegeskorte, Marieke Mur and Peter Bandettini (2008).
Representational similarity analysis - connecting the branches of systems neuroscience.
Frontiers in Systems Neuroscience, 2(4).
[https://doi.org/10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)

<picture>
<<<<<<< HEAD
  <source media="(prefers-color-scheme: light)" srcset="doc/rsa.png">
  <source media="(prefers-color-scheme: dark)" srcset="doc/rsa_dark.png">
=======
  <source media="(prefers-color-scheme: dark)" srcset="doc/rsa_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="doc/rsa.png">
>>>>>>> main
  <img src="doc/rsa.png" width="600">
</picture>


## Use cases

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

## Installation

The package can be installed either through PIP: `pip install mne-rsa`  
or through conda using the conda-forge channel: `conda install -c conda-forge mne-rsa`


## Example usage

Basic example on the EEG “kiloword” data:

```python
import mne
import mne_rsa
data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path / "kword_metadata-epo.fif")
# Create model RDMs based on each stimulus property
columns = epochs.metadata.columns[1:]  # Skip the first column: WORD
model_rdms = [mne_rsa.compute_rdm(epochs.metadata[col], metric="euclidean") for col in columns]
# Perform RSA in a sliding window across time
rsa_results = mne_rsa.rsa_epochs(epochs, model_rdms, temporal_radius=0.01)
# Plot the result
mne.viz.plot_compare_evokeds({column: result for column, result in zip(columns, rsa_results)},
                            picks="rsa", legend="lower center", title="RSA result")
```
<picture>
  <source media="(prefers-color-scheme: light)" srcset="doc/rsa_result.png">
  <source media="(prefers-color-scheme: dark)" srcset="doc/rsa_result_dark.png">
  <img src="rsa_result.png" width="600">
</picture>

## Documentation
For a detailed guide on RSA analyis from start to finish on an example dataset, see the [tutorials](auto_examples/tutorials/index.html).

For quick guides on how to do specific things, see the [examples](auto_examples/index.html).

Finally, there is the [API reference](api.html) documentation.

## Integration with other packages

The main purpose of this package is to perform RSA analysis on MEG data.
Hence, integration functions with [MNE-Python](https://mne.tools) are provided.
However, there is also some integration with [nipy](https://nipy.org) for fMRI that should well in a [nilearn](https://nilearn.github.io) setup.


## Support

This free software comes without any form of official support.
However, if you think you have encountered a bug or have a particularly great idea for improvements, please open an [issue on Github](https://github.com/mne-tools/mne-rsa/issues).
For questions and help with your analysis, you are welcome to post on the [MNE forum](https://mne.discourse.group/).

## Contributing

Development of the package happens on [Github](https://github.com/mne-tools/mne-rsa).
Everyone is welcome to raise [issues](https://github.com/mne-tools/mne-rsa/issues) or contribute [pull requests](https://github.com/mne-tools/mne-rsa/pulls).

Here is how to install the additional required packages for developing MNE-RSA and set up the package in development mode:

```bash
git clone git@github.com:mne-tools/mne-rsa.git
cd mne-rsa
pip install -r requirements-dev.txt
pip install -e .
```

To run the test suite, execute `pytest` in the main `mne-rsa` folder.  
To build the documentation, execute `make html` in the `mne-rsa/doc` folder (or on
Windows: `sphinx-build . _build/html`).

## Citation
If you end up using this package for the data analysis that is part of a scientific
article, please cite:

Marijn van Vliet, Takao Shimizu, Stefan Appelhoff, Yuan-Fang Zhao, & Richard
Höchenberger. (2024). mne-tools/mne-rsa: Version 0.9.1 (0.9.1). Zenodo.
https://doi.org/10.5281/zenodo.11258133
