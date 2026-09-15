"""
.. _tut-source-level:

Tutorial part 2: RSA on source-level MEG data
---------------------------------------------

In this tutorial, we will perform source-level RSA analysis on MEG data.

If you haven't done so, I recommend first completing :ref:`tut-sensor-level`.

While sensor-level RSA is useful to get a first impression of how neural representations
unfold over time, it is not really suited to study differences between brain regions.
For this, you want to so RSA in a searchlight pattern across the cortex.

The knowledge you have gained from your sensor-level analysis will serve you well for
this part, as the API of MNE-RSA is mostly the same across sensor- and source-level
analysis. However, performing a searchlight analysis is a heavy computation that can
take a lot of time. Hence, we will also learn about the API regarding restricting the
analysis to parts of the data in several ways.

In the cell below, update the ``data_path`` variable to point to where you have
extracted the `rsa-data.zip <https://github.com/wmvanvliet/neuroscience_tutorials/releases/download/2/rsa-data.zip>`__
file to.
"""
# ruff: noqa: E402
# sphinx_gallery_thumbnail_number=3

# Set this to where you've extracted `data.zip` to
data_path = "data"

########################################################################################
# We’ll start by loading the ``epochs`` again, but this time, we will restrict them to
# only two experimental conditions: the first presentations of famous faces versus
# scrambled faces. This will reduce the number of rows/columns in our RDMs and hence
# speed up computing and comparing them.
import mne

epochs = mne.read_epochs(f"{data_path}/sub-02/sub-02-epo.fif")
epochs = epochs[["face/famous/first", "scrambled/first"]]
epochs

########################################################################################
# When we select a subset of epochs, the ``epochs.metadata`` field is likewise updated
# to match the new selection. This feature is one of the main reasons to use the
# ``.metadata`` field instead of keeping a separate :class:`pandas.DataFrame` manually.

epochs.metadata.info()

########################################################################################
# This means we can use the ``epochs.metadata["file"]`` column to restrict the pixel and
# FaceNet RDMs to just those images still present in the MEG data.
#
# In the cell below, we read the images and FaceNet embeddings and select the proper
# rows from the data matrices and use ``compute_dsm`` to compute the appropriate RDMs.

from glob import glob

import numpy as np
from PIL import Image

files = sorted(glob(f"{data_path}/stimuli/*.bmp"))
pixels = np.array([np.array(Image.open(f)) for f in files])

store = np.load(f"{data_path}/stimuli/facenet_embeddings.npz")
filenames = store["filenames"]
embeddings = store["embeddings"]

# Select the proper filenames
epochs_filenames = set(epochs.metadata["file"])
selection = [f in epochs_filenames for f in filenames]
filenames = filenames[selection]

# Select the proper rows from `pixels` and `embeddings` and compute the RDMs.
from mne_rsa import compute_rdm

pixel_rdm = compute_rdm(pixels[selection])
facenet_rdm = compute_rdm(embeddings[selection])

########################################################################################
# Executing the cell below will test whether the RDMs have been properly constructed and
# plot them.

from mne_rsa import plot_rdms
from scipy.spatial.distance import squareform

if len(pixel_rdm) != len(facenet_rdm):
    print("The pixel and FaceNet RDMs are of difference sizes, that can't be right. 🤔")
elif len(pixel_rdm) != 43956:
    print("Looks like the RDMs do not have the correct rows. 🤔")
elif (
    squareform(pixel_rdm)[:150, :150].mean() >= squareform(pixel_rdm)[150:, 150:].mean()
):
    print(
        "The pixels RDM doesn't look quite right. Make sure the rows are in "
        "alphabetical filename order. 🤔"
    )
elif (
    squareform(facenet_rdm)[:150, :150].mean()
    <= squareform(facenet_rdm)[150:, 150:].mean()
):
    print(
        "The FaceNet RDM doesn't look quite right. Make sure the rows are in "
        "alphabetical filename order. 🤔"
    )
else:
    print("The RDMs look just right! 😊")
    plot_rdms([pixel_rdm, facenet_rdm], names=["pixels", "facenet"])

########################################################################################
# To source space!
# ----------------
#
# In order to perform RSA in source space, we must create source estimates for the
# epochs. There’s many different ways to do this, for example you’ll learn about
# beamformers during this workshop, but here we’re going to use the one that is fastest.
# If we use MNE, we can use a pre-computed inverse operator and apply it to the epochs
# to quickly get source estimates.

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

inv = read_inverse_operator(f"{data_path}/sub-02/sub-02-inv.fif")
stcs = apply_inverse_epochs(epochs, inv, lambda2=1 / 9)

########################################################################################
# The result is a list of 297 ``SourceEstimate`` objects. Here are the
# first 5 of them:

stcs[:5]

########################################################################################
# The plan is to perform RSA in a searchlight pattern, not only as a sliding window
# through time, but also sliding across different locations across the cortex. To this
# end, we’ll define spatial patches with a certain radius, and only source points that
# fall within a patch are taken into account when computing the RDM for that patch. The
# cortex is heavily folded and ideally we define distances between source point as the
# shortest path along the cortex, what is known as the geodesic distance, rather than
# straight euclidean distance between the XYZ coordinates. MNE-Python is here to help us
# out in this regard, as it contains a function to compute such distances and store them
# within the :class:`mne.SourceSpaces` object (through the
# :func:`mne.add_source_space_distances` function).
#
# Let’s load the file containing the proper source space with pre-computed geodesic
# distances between source points:

from mne import read_source_spaces

src = read_source_spaces(f"{data_path}/freesurfer/sub-02/bem/sub-02-oct5-src.fif")

########################################################################################
# To speed things up, let’s restrict the analysis to only the occipital, parietal and
# temporal areas on the left hemisphere. There are several ways to tell MNE-RSA which
# source points to use, and one of the most convenient ones is to use :class:`mne.Label`
# objects. This allows us to define the desired areas using the “aparc” atlas that
# FreeSurfer has constructed for us:

rois = mne.read_labels_from_annot(
    "sub-02", parc="aparc", subjects_dir=f"{data_path}/freesurfer", hemi="lh"
)

# These are the regions we're interested in
roi_sel = [
    "inferiortemporal-lh",
    "middletemporal-lh",
    "fusiform-lh",
    "bankssts-lh",
    "inferiorparietal-lh",
    "lateraloccipital-lh",
    "lingual-lh",
    "pericalcarine-lh",
    "cuneus-lh",
    "supramarginal-lh",
    "superiorparietal-lh",
]
rois = [r for r in rois if r.name in roi_sel]


########################################################################################
# Source-space RSA
# ----------------
#
# Time to actually perform the RSA in source space. The function you need is
# :func:`mne_rsa.rsa_stcs` Take a look at the documentation of that function, which
# should look familiar is it is very similar to :func:`mne_rsa.rsa_epochs` that you have
# used before.
#
# We will perform RSA on the source estimates, using the pixel and FaceNet RDMs as model
# RDMs. As was the case with the sensor-level RSA, we will need specify labels to
# indicate which image was shown during which epoch and which image corresponds to each
# row/column of the ``pixel_rdm`` and ``facenet_rdm``. We will use the filenames for
# this.
#
# Searchlight patches swill have a spatial radius of 2cm (=0.02 meters) and a
# temporal radius of 50 ms (=0.05 seconds). We will restrict the analysis to 0.0 to 0.5
# seconds after stimulus onset and to the cortical regions (``rois``) we’ve selected
# above. We can optionally set ``n_jobs=-1`` to use all CPU cores and ``verbose=True``
# to show a progress bar.
#
# Depending on the speed of your computer, this may take anywhere from a few seconds to
# a few minutes to complete.

from mne_rsa import rsa_stcs

stc_rsa = rsa_stcs(
    stcs,
    [pixel_rdm, facenet_rdm],
    src=src,
    labels_stcs=epochs.metadata.file,
    labels_rdm_model=filenames,
    tmin=0,
    tmax=0.5,
    sel_vertices=rois,
    spatial_radius=0.02,
    temporal_radius=0.05,
    verbose=True,
    n_jobs=-1,
)

########################################################################################
# If everything went as planned, executing the cell below will plot the result.

# For clarity, only show positive RSA scores
stc_rsa[0].data[stc_rsa[0].data < 0] = 0
stc_rsa[1].data[stc_rsa[1].data < 0] = 0

# Show the RSA maps for both the pixel and FaceNet RDMs
brain_pixels = stc_rsa[0].plot(
    "sub-02",
    subjects_dir=f"{data_path}/freesurfer",
    hemi="both",
    initial_time=0.081,
    views="ventral",
    title="pixels",
)
brain_facenet = stc_rsa[1].plot(
    "sub-02",
    subjects_dir=f"{data_path}/freesurfer",
    hemi="both",
    initial_time=0.180,
    views="parietal",
    title="FaceNet",
)

########################################################################################
# If you’ve made it this far, you have successfully completed your first sensor-level
# RSA! 🎉 This is the end of this tutorial. In the next tutorial, we will discuss
# group-level analysis and statistics: :ref:`tut-statistics`.
