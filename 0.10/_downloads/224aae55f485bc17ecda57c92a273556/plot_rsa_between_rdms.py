#!/usr/bin/env python
# coding: utf-8

"""
Compute RSA between RDMs
------------------------

This example showcases the most basic version of RSA: computing the similarity between
two RDMs. Then we continue with computing RSA between many RDMs efficiently.

| Authors:
| Marijn van Vliet <marijn.vanvliet@aalto.fi>
"""
# sphinx_gallery_thumbnail_number=2

import mne
import mne_rsa

# Import required packages
import pandas as pd
from matplotlib import pyplot as plt

########################################################################################
# MNE-Python contains a built-in data loader for the kiloword dataset, which is used
# here as an example dataset. Since we only need the words shown during the experiment,
# which are in the metadata, we can pass ``preload=False`` to prevent MNE-Python from
# loading the EEG data, which is a nice speed gain.

data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path / "kword_metadata-epo.fif")

# Show the metadata of 10 random epochs
epochs.metadata.sample(10)

########################################################################################
# Compute RDMs based on word length and visual complexity.

metadata = epochs.metadata
rdm1 = mne_rsa.compute_rdm(metadata.NumberOfLetters, metric="euclidean")
rdm2 = mne_rsa.compute_rdm(metadata.VisualComplexity, metric="euclidean")

# Plot the RDMs
mne_rsa.plot_rdms([rdm1, rdm2], names=["Word length", "Vis. complexity"])

########################################################################################
# Perform RSA between the two RDMs using Spearman correlation

rsa_result = mne_rsa.rsa(rdm1, rdm2, metric="spearman")
print("RSA score:", rsa_result)

########################################################################################
# We can compute RSA between multiple RDMs by passing lists to the :func:`mne_rsa.rsa`
# function.

# Create RDMs for each stimulus property
columns = metadata.columns[1:]  # Skip the first column: WORD
rdms = [mne_rsa.compute_rdm(metadata[col], metric="euclidean") for col in columns]

# Plot the RDMs
fig = mne_rsa.plot_rdms(rdms, names=columns, n_rows=2)
fig.set_size_inches(12, 4)

# Compute RSA between the first two RDMs (Concreteness and WordFrequency) and the
# others.
rsa_results = mne_rsa.rsa(rdms[:2], rdms[2:], metric="spearman")

# Pack the result into a Pandas DataFrame for easy viewing
print(pd.DataFrame(rsa_results, index=columns[:2], columns=columns[2:]))

########################################################################################
# What if we have many RDMs? The :func:`mne_rsa.rsa` function is optimized for the case
# where the first parameter (the "data" RDMs) is a large list of RDMs and the second
# parameter (the "model" RDMs) is a smaller list. To save memory, you can also pass
# generators instead of lists.
#
# Let's create a generator that creates RDMs for each time-point in the EEG data and
# compute the RSA between those RDMs and all the "model" RDMs we computed above. This is
# a basic example of using a "searchlight" and in other examples, you can learn how to
# use the :class:`mne_rsa.searchlight` generator to build more advanced searchlights.
# However, since this is such a simple case, it is educational to construct the
# generator manually.
#
# The RSA computation will take some time. Therefore, we pass a few extra parameters to
# :func:`mne_rsa.rsa` to enable some improvements. First, the ``verbose=True`` enables a
# progress bar. However, since we are using a generator, the progress bar cannot
# automatically infer how many RDMs there will be. Hence, we provide this information
# explicitly using the ``n_data_rdms`` parameter. Finally, depending on how many CPUs
# you have on your system, consider increasing the ``n_jobs`` parameter to parallelize
# the computation over multiple CPUs.

epochs.resample(100)  # Downsample to speed things up for this example
eeg_data = epochs.get_data()
n_trials, n_sensors, n_times = eeg_data.shape


def generate_eeg_rdms():
    """Generate RDMs for each time sample."""
    for i in range(n_times):
        yield mne_rsa.compute_rdm(eeg_data[:, :, i], metric="correlation")


rsa_results = mne_rsa.rsa(
    generate_eeg_rdms(),
    rdms,
    metric="spearman",
    verbose=True,
    n_data_rdms=n_times,
    n_jobs=1,
)

# Plot the RSA values over time using standard matplotlib commands
plt.figure(figsize=(8, 4))
plt.plot(epochs.times, rsa_results)
plt.xlabel("time (s)")
plt.ylabel("RSA value")
plt.legend(columns)
