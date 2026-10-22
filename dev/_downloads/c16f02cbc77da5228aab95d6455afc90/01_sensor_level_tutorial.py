"""
.. _tut-sensor-level:

Tutorial part 1: RSA on sensor-level MEG data
---------------------------------------------
In this tutorial, we will perform sensor-level RSA analysis on MEG data.

We will explore how representational similarity analysis (RSA) can be used to study the
neural representational code within visual cortex. We will start with performing RSA on
the sensor level data, followed by source level and finally we will perform group level
statistical analysis. Along the way, we will encounter many of the functions and classes
offered by MNE-RSA, which will always be presented in the form of links to the
:ref:`api_documentation` which you are encouraged to explore.

The dataset we will be working with today is the `Wakeman & Nelson (2015) “faces”
dataset <https://www.nature.com/articles/sdata20151>`__. During this experiment,
participants were presented with a series of images, containing:

- Faces of famous people that the participants likely knew
- Faces of people that the participants likely did not know
- Scrambled faces: the images were cut-up and randomly put together again

As a first step, you need to download and extract the dataset: `rsa-data.zip <https://github.com/wmvanvliet/neuroscience_tutorials/releases/download/2/rsa-data.zip>`__.
You can either do this by executing the cell below, or you can do so manually. In any
case, make sure that the ``data_path`` variable points to where you have extracted the
`rsa-data.zip <https://github.com/wmvanvliet/neuroscience_tutorials/releases/download/2/rsa-data.zip>`__
file to.
"""
# ruff: noqa: E402
# sphinx_gallery_thumbnail_number=8

import os

import pooch

# Download and unzip the data
pooch.retrieve(
    url="https://github.com/wmvanvliet/neuroscience_tutorials/releases/download/2/rsa-data.zip",
    known_hash="md5:859c0684dd25f8b82d011840725cbef6",
    progressbar=True,
    processor=pooch.Unzip(members=["data"], extract_dir=os.getcwd()),
)
# Set this to where you've extracted `rsa-data.zip` to
data_path = "data"

########################################################################################
# A representational code for the stimuli
# ---------------------------------------
#
# Let’s start by taking a look at the stimuli that were presented during the experiment.
# They reside in the ``stimuli`` folder for you as ``.bmp`` image files. The Python
# Imaging Library (PIL) can open them and we can use matplotlib to display them.

import matplotlib.pyplot as plt
from PIL import Image

# Show the first "famous" face and the first "scrambled" face
img_famous = Image.open(f"{data_path}/stimuli/f001.bmp")
img_scrambled = Image.open(f"{data_path}/stimuli/s001.bmp")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_famous, cmap="gray")
axes[0].set_title(f"Famous face: {img_famous.width} x {img_famous.height} pixels")
axes[1].imshow(img_scrambled, cmap="gray")
axes[1].set_title(
    f"Scrambled face: {img_scrambled.width} x {img_scrambled.height} pixels"
)
axes[0].axis("off")
axes[1].axis("off")

########################################################################################
# Loaded like this, the stimuli are in a representational space defined by their pixels.
# Each image is represented by 128 x 162 = 20736 values between 0 (black) and 255
# (white). Let's create a Representational Dissimilarity Matrix (RDM) where images are
# compared based on the difference between their pixels. To get the pixels of an image,
# you can convert it to a NumPy array like this:

import numpy as np

pixels_famous = np.array(img_famous)
pixels_scrambled = np.array(img_scrambled)

print("Shape of the pixel array for the famous face:", pixels_famous.shape)
print("Shape of the pixel array for the scrambled face:", pixels_scrambled.shape)

########################################################################################
# We can now compute the “dissimilarity” between the two images, based on their pixels.
# For this, we need to decide on a metric to use. The default metric used in the
# original publication (`Kiegeskorte et al. 2008 <https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full>`__)
# was Pearson Correlation, so let’s use that. Of course, correlation is a
# metric of similarity and we want a metric of *dis*\ similarity. Let’s make it easy on
# ourselves and just do :math:`1 - r`.

from scipy.stats import pearsonr

similarity, _ = pearsonr(pixels_famous.flatten(), pixels_scrambled.flatten())
dissimilarity = 1 - similarity
print(
    "The dissimilarity between the pixels of the famous and scrambled faces is:",
    dissimilarity,
)

########################################################################################
# To construct the full RDM, we need to do this for all pairs of images. In the cell
# below, we make a list of all image files and load all of them (there are 450), convert
# them to NumPy arrays and concatenate them all together in a single big array called
# ``pixels`` of shape ``n_images x width x height``.

from glob import glob

files = sorted(glob(f"{data_path}/stimuli/*.bmp"))
print(f"There are {len(files)} images to read.")

pixels = np.array([np.array(Image.open(f)) for f in files])
print("The dimensions of the `pixel` array are:", pixels.shape)

########################################################################################
# Your first RDM
# --------------
#
# Now that you have all the images loaded in, computing the pairwise dissimilarities is
# a matter of looping over them and computing correlations. We could do this manually,
# but we can make our life a lot easier by using MNE-RSA’s :func:`mne_rsa.compute_rdm`
# function. It wants the big matrix as input and also takes a ``metric`` parameter to
# select which dissimilarity metric to use. Setting it to ``metric="correlation"``,
# which is also the default by the way, will make it use (1 - Pearson correlation) as a
# metric like we did manually above.

from mne_rsa import compute_rdm, plot_rdms

pixel_rdm = compute_rdm(pixels)
plot_rdms(pixel_rdm, names="pixels")

########################################################################################
# Staring deeply into this RDM will reveal to you which images belonged to the
# “scrambled faces” class, as those pixels are quite different from the actual faces and
# each other. We also see that for some reason, the famous faces are a little more alike
# than the unknown faces.
#
# The RDM is symmetric along the diagonal, which is all zeros. Take a moment to ponder
# why that would be.
#
# .. note::
#    The :func:`mne_rsa.compute_rdm` function is a wrapper around
#    :func:`scipy.spatial.distance.pdist`. This means that all the metrics supported by
#    :func:`~scipy.spatial.distance.pdist` are also valid for
#    :func:`mne_rsa.compute_rdm`. This also means that in MNE-RSA, the native format for
#    an RDM is the so-called "condensed" form. Since RDMs are symmetric, only the upper
#    triangle is stored. The :func:`scipy.spatial.distance.squareform` function can be
#    used to go from a square matrix to its condensed form and back.

########################################################################################
# Your second RDM
# ---------------
#
# There are many sensible representations possible for images. One intriguing one is to
# create them using convolutional neural networks (CNNs). For example, there is the
# `FaceNet <https://github.com/davidsandberg/facenet>`__ model by `Schroff et al. (2015)
# <http://arxiv.org/abs/1503.03832>`__ that can generate high-level representations,
# such that different photos of the same face have similar representations. I have run
# the stimulus images through FaceNet and recorded the generated embeddings for you to
# use:

store = np.load(f"{data_path}/stimuli/facenet_embeddings.npz")
filenames = store["filenames"]
embeddings = store["embeddings"]
print(
    "For each of the 450 images, the embedding is a vector of length 512:",
    embeddings.shape,
)

facenet_rdm = compute_rdm(embeddings)

########################################################################################
# Lets plot both RDMs side-by-side:
plot_rdms([pixel_rdm, facenet_rdm], names=["pixels", "facenet"])

########################################################################################
# A look at the brain data
# ------------------------
#
# We’ve seen how we can create RDMs using properties of the images or embeddings
# generated by a model. Now it’s time to see how we create RDMs based on the MEG data.
# For that, we first load the epochs from a single participant.

import mne

epochs = mne.read_epochs(f"{data_path}/sub-02/sub-02-epo.fif")
epochs

########################################################################################
# Each epoch corresponds to the presentation of an image, and the signal across the
# sensors over time can be used as the neural representation of that image. Hence, one
# could make a neural RDM of, for example the gradiometers in the time window 100 to
# 200 ms after stimulus onset, like this:

neural_rdm = compute_rdm(epochs.get_data("grad", tmin=0.1, tmax=0.2))
plot_rdms(neural_rdm)

########################################################################################
# To compute RSA scores, we want to compare the resulting neural RDM with the RDMs we’ve
# created earlier. However, if we inspect the neural RDM closely, we see that its rows
# and column don’t line up with those of the previous RDMs. There are too many (879
# vs. 450) and they are in the wrong order. Making sure that the RDMs match is an
# important and sometimes tricky part of RSA.
#
# To help us out, a useful feature of MNE-Python is that epochs have an associated
# :attr:`mne.Epochs.metadata` field. This metadata is a :class:`pandas.DataFrame` where
# each row contains information about the corresponding epoch. The epochs in this
# tutorial come with some useful ``.metadata`` already:
epochs.metadata

########################################################################################
# While the trigger codes only indicate what type of stimulus was shown, the ``file``
# column of the metadata tells us the exact image. Couple of challenges here: the
# stimuli where shown in a random order, stimuli were repeated twice during the
# experiment, and some epochs were dropped during preprocessing so not every image is
# necessarily present twice in the ``epochs`` data. 😩
#
# Luckily, MNE-RSA has a way to make our lives easier. Let’s take a look at the
# :func:`mne_rsa.rdm_epochs` function, the Swiss army knife for computing RDMs from an
# MNE-Python :class:`mne.Epochs` object.
#
# In MNE-Python tradition, the function has a lot of parameters, but
# all-but-one have a default so you only have to specify the ones that are
# relevant to you. For example, to redo the neural RDM we created above,
# we could do something like:

from mne_rsa import rdm_epochs

neural_rdm_gen = rdm_epochs(epochs, tmin=0.1, tmax=0.2)

# `rdm_epochs` returns a generator of RDMs
# unpacking the first (and only) RDM from the generator
neural_rdm = next(neural_rdm_gen)
plot_rdms(neural_rdm)

########################################################################################
# Take note that :func:`mne_rsa.rdm_epochs` returns a `generator
# <https://wiki.python.org/moin/Generators>`__ of RDMs. This is because one of the main
# use-cases for MNE-RSA is to produce RDMs using sliding windows (in time and also in
# space), which can produce a large amount of RDMs that can take up a lot of memory of
# you’re not careful.
#
# Alignment between model and data RDM ordering
# ---------------------------------------------
#
# Looking at the neural RDM above, something is clearly different from the
# one we made before. This one has 9 rows and columns. Closely inspecting
# the docstring of :class:`mne_rsa.rdm_epochs` reveals that it is the ``labels``
# parameter that is responsible for this:
#
# ::
#
#   labels : list | None
#       For each epoch, a label that identifies the item to which it corresponds.
#       Multiple epochs may correspond to the same item, in which case they should have
#       the same label and will either be averaged when computing the data RDM
#       (``n_folds=1``) or used for cross-validation (``n_folds>1``). Labels may be of
#       any python type that can be compared with ``==`` (int, float, string, tuple,
#       etc). By default (``None``), the epochs event codes are used as labels.
#
# Instead of producing one row per epoch, :func:`mne_rsa.rdm_epochs` produced one row
# per event type, averaging across epochs of the same type before computing
# dissimilarity. This is not quite what we want though. If we want to match
# ``pixel_rdm`` and ``facenet_rdm``, we want every single one of the 450 images to be
# its own stimulus type. We can achieve this by setting the ``labels`` parameter of
# :func:`mne_rsa.rdm_epochs` to a list that assigns each of the 879 epochs to a label
# that indicates which image was shown. An image is identified by its filename, and the
# ``epochs.metadata.file`` column contains the filenames corresponding to the epochs,
# so let's use that.

neural_rdm = next(rdm_epochs(epochs, labels=epochs.metadata.file, tmin=0.1, tmax=0.2))

# This plots your RDM
plot_rdms(neural_rdm)
########################################################################################
# The cell below will compure RSA between the neural RDM and the pixel and FaceNet RDMs
# we created earlier. The RSA score will be the Spearman correlation between the RDMs,
# which is the default metric used in the `original RSA paper <https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full>`__.

from mne_rsa import rsa

rsa_pixel = rsa(neural_rdm, pixel_rdm, metric="spearman")
rsa_facenet = rsa(neural_rdm, facenet_rdm, metric="spearman")

print("RSA score between neural RDM and pixel RDM:", rsa_pixel)
print("RSA score between neural RDM and FaceNet RDM:", rsa_facenet)

########################################################################################
# Slippin’ and slidin’ across time
# --------------------------------
#
# The neural representation of a stimulus is different across brain
# regions and evolves over time. For example, we would expect that the
# pixel RDM would be more similar to a neural RDM that we computed across
# the visual cortex at an early time point, and that the FaceNET RDM might
# be more similar to a neural RDM that we computed at a later time point.
#
# For the remainder of this tutorial, we’ll restrict the ``epochs`` to
# only contain the sensors over the left occipital cortex.
#
# .. warning::
#    Just because we select sensors over a certain brain region, does not mean the
#    magnetic fields originate from that region. This is especially true for
#    magnetometers. To make it a bit more accurate, we only select gradiometers.

picks = mne.channels.read_vectorview_selection("Left-occipital")
picks = ["".join(p.split(" ")) for p in picks]
epochs.pick(picks).pick("grad").crop(-0.1, 1)

########################################################################################
# In the cell below, we use :func:`mne_rsa.rdm_epochs` to compute RDMs using a sliding
# window by setting the ``temporal_radius`` parameter to ``0.1`` seconds. We use the
# entire time range (``tmin=None`` and ``tmax=None``) and leave the result as a
# generator (so no ``next()`` calls).

neural_rdms_gen = rdm_epochs(epochs, labels=epochs.metadata.file, temporal_radius=0.1)

########################################################################################
# And now we can consume the generator (with a nice progress bar) and plot
# a few of the generated RDMs:

from tqdm import tqdm

times = epochs.times[(epochs.times >= 0) & (epochs.times <= 0.9)]
neural_rdms_list = list(tqdm(neural_rdms_gen, total=len(times)))
plot_rdms(neural_rdms_list[::10], names=[f"t={t:.2f}" for t in times[::10]])

########################################################################################
# Putting it altogether for sensor-level RSA
# ------------------------------------------
#
# Now all that is left to do is compute RSA scores between the neural RDMs you’ve just
# created and the pixel and FaceNet RDMs. We could do this using the
# :func:`mne_rsa.rsa_gen` function, but I’d rather directly show you the
# :func:`mne_rsa.rsa_epochs` function that combines computing the neural RDMs with
# computing the RSA scores.
#
# The signature of :func:`mne_rsa.rsa_epochs` is very similar to that of
# :func:`mne_rsa.rdm_epochs` The main difference is that we also give it the “model”
# RDMs, in our case the pixel and FaceNet RDMs. We can also specify ``labels_rdm_model``
# to indicate which rows of the model RDMs correspond to which images to make sure the
# ordering is the same. :func:`mne_rsa.rsa_epochs` will return the RSA scores as a list
# of :class:`mne.Evoked` objects: one for each model RDM we gave it.
#
# We compute the RSA scores for ``epochs`` against ``[pixel_rdm, facenet_rdm]`` and do
# this in a sliding windows across time, with a temporal radius of 0.1 seconds. Setting
# ``verbose=True`` will activate a progress bar. We can optionally set ``n_jobs=-1`` to
# use multiple CPU cores to speed things up.

from mne_rsa import rsa_epochs

ev_rsa = rsa_epochs(
    epochs,
    [pixel_rdm, facenet_rdm],
    labels_epochs=epochs.metadata.file,
    labels_rdm_model=filenames,
    temporal_radius=0.1,
    verbose=True,
    n_jobs=-1,
)

# Create a nice plot of the result
ev_rsa[0].comment = "pixels"
ev_rsa[1].comment = "facenet"
mne.viz.plot_compare_evokeds(
    ev_rsa, picks=[0], ylim=dict(misc=[-0.02, 0.2]), show_sensors=False
)

########################################################################################
# We see that first, the “pixels” representation is the better match to the
# representation in the brain, but after around 150 ms the representation produced by
# the FaceNet model matches better. The best match between the brain and FaceNet is
# found at around 250 ms.
#
# If you’ve made it this far, you have successfully completed your first sensor-level
# RSA! 🎉 This is the end of this tutorial. I invite you to join me in the next
# tutorial where we will do source level RSA: :ref:`tut-source-level`
