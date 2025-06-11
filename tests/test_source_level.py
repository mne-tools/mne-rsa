"""Unit tests for source-level analysis."""

from copy import deepcopy

import mne
import nibabel as nib
import numpy as np
import pytest
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_rsa import rdm_nifti, rdm_stcs, squareform
from numpy.testing import assert_equal


def make_stcs():
    """Create a list of SourceEstimates from some of the MNE-Sample epochs."""
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw = mne.io.read_raw_fif(path / "sample_audvis_filt-0-40_raw.fif", preload=False)
    raw.pick("meg")  # only use the MEG data for these unit tests
    events = mne.read_events(path / "sample_audvis_filt-0-40_raw-eve.fif")
    events = events[:50]  # only use the first 50 events
    epochs = mne.Epochs(raw, events, event_id=[1, 2, 3, 4], preload=True)
    epochs.resample(100)  # nice round number

    # Project to source space using the inverse operator
    inv = read_inverse_operator(path / "sample_audvis-meg-oct-6-meg-inv.fif")
    stcs = apply_inverse_epochs(epochs, inv, lambda2=1, verbose=False)

    # Also return the source space
    return stcs, inv["src"], epochs.events[:, 2]


def make_nifty():
    """Create a 4D nifty image containing 10x10x10 voxels with 4 instances."""
    rng = np.random.RandomState(0)
    data = rng.randn(10, 10, 10, 4)
    affine = np.eye(4) * 5  # each voxel is 5 mm
    affine[3, 3] = 1.0
    mask = np.zeros_like(data[:, :, :, 0])  # create a mask as well
    mask[2:4, 2:4, 2:4] = 1
    return nib.Nifti1Image(data, affine), nib.Nifti2Image(mask, affine)


class TestStcRDMs:
    """Test making RDMs from source-level MEG data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        stcs, src, y = make_stcs()
        rdms = list(rdm_stcs(stcs, src, y=y))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)

    def test_rdm_temporal(self):
        """Test making RDMs with a sliding temporal window."""
        stcs, src, y = make_stcs()

        rdms = list(rdm_stcs(stcs, src, y=y, temporal_radius=0.1))  # 10 samples
        assert len(rdms) == len(stcs[0].times) - 2 * 10
        assert squareform(rdms[0]).shape == (4, 4)

        # Restrict in time
        rdms = list(rdm_stcs(stcs, src, y=y, temporal_radius=0.1, tmin=0, tmax=0.3))
        assert len(rdms) == 30

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            next(rdm_stcs(stcs, src, y=y, temporal_radius=0.1, tmin=-5))
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            next(rdm_stcs(stcs, src, y=y, temporal_radius=0.1, tmax=5))
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            next(rdm_stcs(stcs, src, y=y, temporal_radius=0.1, tmax=0.1, tmin=0.2))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across sensors."""
        stcs, src, y = make_stcs()
        rdms = list(rdm_stcs(stcs, src, y=y, spatial_radius=0.05))  # 5 cm
        assert len(rdms) == len(stcs[0].data)
        assert squareform(rdms[0]).shape == (4, 4)

        # Restrict vertices to a label
        label = mne.read_labels_from_annot(
            "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
        )[0]
        rdms = list(rdm_stcs(stcs, src, y=y, spatial_radius=0.05, sel_vertices=label))
        assert len(rdms) == len(stcs[0].in_label(label).vertices[0])

        # Restrict vertices to 2 selected indices.
        rdms = list(
            rdm_stcs(stcs, src, y=y, spatial_radius=0.05, sel_vertices_by_index=[0, 1])
        )
        assert len(rdms) == 2

        # Pick non-existing vertices
        with pytest.raises(ValueError, match="not present in the data"):
            next(
                rdm_stcs(
                    stcs,
                    src,
                    y=y,
                    spatial_radius=0.05,
                    sel_vertices=[[-1, 109209], [-304, 120930904]],
                )
            )

    def test_rdm_spatio_temporal(self):
        """Test making RDMs with a searchlight across both sensors and time."""
        stcs, src, y = make_stcs()
        rdms = list(rdm_stcs(stcs, src, y=y, spatial_radius=0.05, temporal_radius=0.1))
        assert len(rdms) == len(stcs[0].data) * (len(stcs[0].times) - 2 * 10)
        assert squareform(rdms[0]).shape == (4, 4)


class TestNiftyRDMs:
    """Test making RDMs from fMRI data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        bold, mask = make_nifty()
        rdms = list(rdm_nifti(bold))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)

        bold_nans = deepcopy(bold)
        bold_nans.get_fdata()[~mask.get_fdata().astype("bool")] = np.nan

        # An ROI mask limits the result to only the mask
        rdms = list(rdm_nifti(bold_nans, roi_mask=mask))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)
        assert not np.any(np.isnan(rdms))

        # A brain mask limits the input voxels to only the mask and is in the case of a
        # single patch the same thing as specifying roi_mask
        rdms = list(rdm_nifti(bold_nans, brain_mask=mask))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)
        assert not np.any(np.isnan(rdms))

        # Specify `y`
        y = [1, 1, 2, 2]
        rdms = list(rdm_nifti(bold, y=y))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (2, 2)

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across voxels."""
        bold, mask = make_nifty()
        rdms = list(rdm_nifti(bold, spatial_radius=0.01))
        assert len(rdms) == 10 * 10 * 10
        assert squareform(rdms[0]).shape == (4, 4)

        # Restrict voxels to a mask.
        rdms = list(rdm_nifti(bold, spatial_radius=0.01, roi_mask=mask))
        assert len(rdms) == 2 * 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)

        rdms = list(rdm_nifti(bold, spatial_radius=0.01, brain_mask=mask))
        assert len(rdms) == 2 * 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)
