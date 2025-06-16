---
title: 'MNE-RSA: Representational Similarity Analysis on EEG, MEG and fMRI data'
tags:
  - Python
  - neuroscience
  - represenational similarity analysis
  - EEG
  - MEG
  - fMRI
authors:
  - given name: Marijn
    dropping-particle: van
    surname: Vliet
    orcid: 0000-0002-6537-6899
    affiliation: "1"
affiliations:
 - name: Department of Neuroscience and Biomedical Engineering, Aalto University, Finland
   index: 1
date: 14 June 2025
bibliography: paper.bib

---

# Summary

MNE-RSA is a software package for performing representational similarity analysis (RSA) on non-invasive measurements of brain activity, namely electroencephalography (EEG), magnetoencephalography (MEG) and functional magnetic resonance imaging (fMRI).
It serves as an extension to MNE-Python [@Gramfort2013] to provide a straightforward way to incorporate RSA in a bigger analysis pipeline that otherwise encompasses the many preprocessing steps required for this type of analysis.

## About RSA
RSA is a technique to compare information flows within complex systems [@Kriegeskorte2008].
In the context of this software package, this mostly means comparing different representations of input stimuli to neural representations at different locations and times in the brain.
Example representations of a stimulus would be the pixels of an image, or the semantic features of the object depicted in the image ("has a tail", "barks", "good boy"), or an embedding vector obtained with a convolutional neural network (CNN) or large language network (LLM) [@Diedrichsen2017].
Example neural representations include the pattern of electric potentials across EEG sensors, or the magnetic field pattern across MEG sensors, or the pattern of source localized activity across the cortex, or the pattern of beta values across fMRI voxels.
Whenever one can create multiple representations of the same stimuli, one can compare these representations using RSA to judge their "representational similarity" (\autoref{fig:rsa}).
The key to this is the creation of a representational dissimilarity matrix (RDM) which is an all-to-all distance matrix between the representations of a set of stimuli, usually obtained by correlating the representation vectors of each pair of stimuli.
Once an RDM is obtained for the different representation schemes (typically you have one obtained through some model and one obtained from brain activity) they can be compared (again using correlation) to yield an RSA score.
When one does this in a "searchlight" pattern across the brain, the result is a map of RSA scores indicating where and when in the brain the neural representation corresponds to the model.

![Schematic overview of representational similarity analysis (RSA).\label{fig:rsa}](rsa.pdf){width="10cm"}


## Features
MNE-RSA current supports the following use cases:

- Compute RDMs on arbitrary data
- Compute RDMs in a searchlight across:

   - vertices/voxels and samples (source level)
   - sensors and samples (sensor level)
   - vertices/voxels only (source level)
   - sensors only (sensor level)
   - samples only (source and sensor level)

- Use cross-validated distance metrics when computing RDMs
- And of course: compute RSA between RDMs

MNE-RSA currently spports the following metrics for comparing RDMs:

-  Spearman correlation (the default)
-  Pearson correlation
-  Kendall’s Tau-A
-  Linear regression (when comparing multiple RDMs at once)
-  Partial correlation (when comparing multiple RDMs at once)

## Performance

Performing RSA in a searchlight pattern will produce tens of thousands of RDMs that can take up multiple gigabytes of space.
For memory efficiency, RDMs are never kept in memory longer than they need to be, hence the useage of python generators.
It is almost always easier to re-compute RDMs than it is to write them to disk and later read them back in.
The computation of RDMs is parallelized across CPU cores.


# Statement of need
While the core computations behind RSA ae simple, getting the details right is hard.
Creating a "searchlight" patches across the cortex means using geodesic rather than Euclidean distance (\autoref{fig:distances}), combining MEG gradiometers and magnetometers requires signal whitening, creating proper evoked responses requires averaging across stimulus repetitions, and creating reliable brain RDMs requires cross-validated distance metrics [@Guggenmos2018].
MNE-RSA provides turn-key solutions for all of these details by interfacing with the metadata available in MNE-Python objects.

![Depiction of geodesic versus Euclidean distance between points along the cortex.\label{fig:distances}](distances.pdf){width="6cm"}

At the time of writing, MNE-RSA has been used in five studies, two of which involve the author [@Hulten2021; @Xu2024; @Messi2025; @Ghazaryan2023; @Klimovich-Gray2021].

## Software ecosystem

The original RSA-toolbox was implemented in MATLAB (https://github.com/rsagroup/rsatoolbox_matlab), with the third iteration now implemented in python [@Bosch2025].
While its focus is mostly on fMRI analysis, the RSA-toolbox aims for a broad implementation of everything related to RSA and its documentation includes an MEG demo.
Another python package worth mentioning is PyMVPA [@Hanke2009], which implements a wide array of machine learning methods, including an RSA variant where RDMs are created using decoding performance as distance metric.
While it is possible to use it for EEG and MEG analysis, it mostly focuses on fMRI.
In contrast to these packages, the scope of MNE-RSA is more narrow, aiming to be an extention of MNE-Python.
Hence its focus is mostly on MEG and EEG analysis, providing a streamlined user experience for the most common use cases in this domain.


# Acknowledgements

At the time of writing, MNE-RSA has received contributions from Stefan
Appelhoff, Egor Eremin, Richard Höchenberger, Ossi Lehtonen, Takao Shimizu, and
Yuan-Fang Zhao. Annika Hultén helped write the first RSA script that eventually
led to the creation of this package.

# References
