"""Unit tests for source-level analysis."""

from copy import deepcopy

import mne
import nibabel as nib
import numpy as np
import pytest
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_rsa import rdm_nifti, rdm_stcs, rsa_nifti, rsa_stcs, rsa_stcs_rois, squareform
from mne_rsa.source_level import (
    _check_src_compatibility,
    _check_stcs_compatibility,
    _get_distance_matrix,
    _restrict_src_to_vertices,
    vertex_selection_to_indices,
)
from numpy.testing import assert_equal


def load_epochs():
    """Load some of the MNE-Sample data epochs."""
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw = mne.io.read_raw_fif(path / "sample_audvis_filt-0-40_raw.fif", preload=False)
    raw.pick("meg")  # only use the MEG sensors
    events = mne.read_events(path / "sample_audvis_filt-0-40_raw-eve.fif")
    events = events[:50]  # only use the first 50 events
    epochs = mne.Epochs(raw, events, event_id=[1, 2, 3, 4], preload=True).crop(0.1, 0.2)
    epochs.resample(100)  # nice round number
    return epochs


def make_stcs():
    """Create a list of SourceEstimates from some of the MNE-Sample epochs."""
    epochs = load_epochs()

    # Project to source space using the inverse operator
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    inv = read_inverse_operator(path / "sample_audvis-meg-oct-6-meg-inv.fif")
    stcs = apply_inverse_epochs(epochs, inv, lambda2=1, verbose=False)

    # Also return the source space
    return stcs, inv["src"], epochs.events[:, 2]


def make_vol_stcs():
    """Create a list of VolSourceEstimates from some of the MNE-Sample epochs."""
    epochs = load_epochs()

    # Project to source space using the inverse operator
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    inv = read_inverse_operator(path / "sample_audvis-meg-vol-7-meg-inv.fif")
    stcs = apply_inverse_epochs(epochs, inv, lambda2=1, verbose=False)

    # Also return the source space
    return stcs, inv["src"], epochs.events[:, 2]


def make_nifti():
    """Create a 4D nifti image containing 10x10x10 voxels with 4 instances."""
    rng = np.random.RandomState(0)
    data = rng.randn(10, 10, 10, 4)
    affine = np.eye(4) * 5  # each voxel is 5 mm
    affine[3, 3] = 1.0
    mask = np.zeros_like(data[:, :, :, 0])  # create a mask as well
    mask[2:4, 2:4, 2:4] = 1
    return nib.Nifti1Image(data, affine), nib.Nifti2Image(mask, affine)


def apply_mask(img, mask, invert=False):
    """Apply a mask to an MRI image."""
    mask = mask.get_fdata().astype("bool")
    if invert:
        mask = ~mask
    return img.get_fdata()[mask]


class TestStcRDMs:
    """Test making RDMs from source-level MEG data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        stcs, _, labels = make_stcs()
        rdms = list(rdm_stcs(stcs, labels=labels))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)

        rdms = list(rdm_stcs(stcs, y=labels))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)

        # Invalid inputs.
        with pytest.raises(ValueError, match="the number of items"):
            next(rdm_stcs(stcs, labels=[1, 2, 3]))

    def test_rdm_temporal(self):
        """Test making RDMs with a sliding temporal window."""
        stcs, _, labels = make_stcs()

        rdms = list(rdm_stcs(stcs, labels=labels, temporal_radius=0.02))  # 2 samples
        assert len(rdms) == len(stcs[0].times) - 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)

        # Restrict in time
        rdms = list(
            rdm_stcs(stcs, labels=labels, temporal_radius=0.02, tmin=0.139, tmax=0.161)
        )
        assert len(rdms) == 3

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            next(rdm_stcs(stcs, labels=labels, temporal_radius=0.02, tmin=-5))
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            next(rdm_stcs(stcs, labels=labels, temporal_radius=0.02, tmax=5))
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            next(
                rdm_stcs(stcs, labels=labels, temporal_radius=0.02, tmax=0.1, tmin=0.2)
            )

        # Too small to or large temporal radius
        with pytest.raises(ValueError, match="less than one sample"):
            next(rdm_stcs(stcs, labels=labels, temporal_radius=0))
        with pytest.raises(ValueError, match="too large"):
            next(rdm_stcs(stcs, labels=labels, temporal_radius=100))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across sensors."""
        stcs, src, labels = make_stcs()
        rdms = list(rdm_stcs(stcs, src=src, labels=labels, spatial_radius=0.05))  # 5 cm
        assert len(rdms) == len(stcs[0].data)
        assert squareform(rdms[0]).shape == (4, 4)

        # Restrict vertices to a label
        label = mne.read_labels_from_annot(
            "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
        )[0]
        rdms = list(
            rdm_stcs(
                stcs, src=src, labels=labels, spatial_radius=0.05, sel_vertices=label
            )
        )
        assert len(rdms) == len(stcs[0].in_label(label).vertices[0])

        # Restrict vertices to 2 selected indices.
        rdms = list(
            rdm_stcs(
                stcs,
                src=src,
                labels=labels,
                spatial_radius=0.05,
                sel_vertices_by_index=[0, 1],
            )
        )
        assert len(rdms) == 2

        # Pick non-existing vertices
        with pytest.raises(ValueError, match="not present in the data"):
            next(
                rdm_stcs(
                    stcs,
                    src,
                    labels=labels,
                    spatial_radius=0.05,
                    sel_vertices=[[-1, 109209], [-304, 120930904]],
                )
            )

        # Pick duplicate vertices
        with pytest.raises(ValueError, match="vertices are not unique"):
            next(
                rdm_stcs(
                    stcs,
                    src,
                    labels=labels,
                    spatial_radius=0.05,
                    sel_vertices_by_index=[1, 1],
                )
            )

        # Invalid inputs.
        with pytest.raises(ValueError, match="you also need to set `src`"):
            next(rdm_stcs(stcs, labels=labels, spatial_radius=0.05))

    def test_rdm_spatio_temporal(self):
        """Test making RDMs with a searchlight across both sensors and time."""
        stcs, src, labels = make_stcs()
        rdms = list(
            rdm_stcs(
                stcs, src=src, labels=labels, spatial_radius=0.05, temporal_radius=0.02
            )
        )
        assert len(rdms) == len(stcs[0].data) * (len(stcs[0].times) - 2 * 2)
        assert squareform(rdms[0]).shape == (4, 4)


class TestNiftiRDMs:
    """Test making RDMs from fMRI data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        bold, mask = make_nifti()
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

        # Specify `labels`
        labels = [1, 1, 2, 2]
        rdms = list(rdm_nifti(bold, labels=labels))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (2, 2)

        rdms = list(rdm_nifti(bold, y=labels))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (2, 2)

        # Invalid data.
        with pytest.raises(ValueError, match="data must be 4-dimensional Nifti-like"):
            next(rdm_nifti(bold.get_fdata()))
        with pytest.raises(ValueError, match="The number of labels in `labels`"):
            next(rdm_nifti(bold, labels=[1, 2, 3]))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across voxels."""
        bold, mask = make_nifti()
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

        rdms = list(
            rdm_nifti(bold, spatial_radius=0.01, roi_mask=mask, brain_mask=mask)
        )
        assert len(rdms) == 2 * 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)

        # Invalid data.
        with pytest.raises(ValueError, match="ROI mask"):
            next(rdm_nifti(bold, roi_mask=mask.slicer[:5, :5, :5]))
        with pytest.raises(ValueError, match="Brain mask"):
            next(rdm_nifti(bold, brain_mask=mask.slicer[:5, :5, :5]))


class TestStcRSA:
    """Test performing RSA on source estimates."""

    def test_rsa_single_searchlight_patch(self):
        """Test performing RSA with a single searchlight patch."""
        stcs, _, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_stcs(stcs, model_rdm, labels_stcs=labels_stcs)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()

        rsa_result = rsa_stcs(stcs, model_rdm, y=labels_stcs)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()

        # Try using different metrics
        rsa_euc = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, stc_rdm_metric="euclidean"
        )
        assert not np.allclose(rsa_result, rsa_euc)
        rsa_tau = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, rsa_metric="kendall-tau-a"
        )
        assert not np.allclose(rsa_result, rsa_tau)

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_stcs(stcs, [model_rdm, model_rdm2], labels_stcs=labels_stcs)
        assert rsa_result.shape == (2,)

        # Invalid inputs.
        with pytest.raises(ValueError, match="The number of source estimates"):
            rsa_stcs(stcs, model_rdm)
        with pytest.raises(ValueError, match="items encoded in the `labels_stcs` list"):
            rsa_stcs(stcs, model_rdm, labels_stcs=[1, 2, 3])

    def test_rsa_temporal(self):
        """Test performing RSA with a sliding temporal window."""
        stcs, _, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=0.02
        )  # 2 samples
        assert rsa_result.data.shape == (1, len(stcs[0].times) - 2 * 2)
        assert rsa_result.data.max().round(2) == 0.62
        assert rsa_result.times[0].round(2) == 0.12
        assert rsa_result.times[-1].round(2) == 0.18

        # Restrict in time
        rsa_result = rsa_stcs(
            stcs,
            model_rdm,
            labels_stcs=labels_stcs,
            temporal_radius=0.02,
            tmin=0.139,
            tmax=0.161,
        )
        print(stcs[0].times)
        print(rsa_result.times)
        assert rsa_result.data.shape == (1, 3)
        assert rsa_result.times[0].round(2) == 0.14
        assert rsa_result.times[-1].round(2) == 0.16

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_stcs(
            stcs,
            [model_rdm, model_rdm2],
            labels_stcs=labels_stcs,
            temporal_radius=0.02,
        )
        assert len(rsa_result) == 2
        assert not np.array_equal(rsa_result[0].data, rsa_result[1].data)
        assert rsa_result[0].data.shape == (1, len(stcs[0].times) - 2 * 2)
        assert rsa_result[0].times[0].round(2) == 0.12
        assert rsa_result[0].times[-1].round(2) == 0.18

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            rsa_stcs(
                stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=0.02, tmin=-5
            )
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            rsa_stcs(
                stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=0.02, tmax=5
            )
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            rsa_stcs(
                stcs,
                model_rdm,
                labels_stcs=labels_stcs,
                temporal_radius=0.02,
                tmax=0.1,
                tmin=0.2,
            )

        # Too small to or large temporal radius
        with pytest.raises(ValueError, match="less than one sample"):
            next(rsa_stcs(stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=0))
        with pytest.raises(ValueError, match="too large"):
            next(
                rsa_stcs(stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=100)
            )

    def test_rsa_spatial(self):
        """Test performing RSA with a searchlight across vertices."""
        stcs, src, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, src=src, spatial_radius=0.05
        )
        assert rsa_result.data.shape == (stcs[0].shape[0], 1)
        assert len(rsa_result.times) == 1
        assert_equal(rsa_result.times, stcs[0].times[len(stcs[0].times) // 2])

        # Restrict vertices to a label
        label = mne.read_labels_from_annot(
            "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
        )[0]
        rsa_result = rsa_stcs(
            stcs,
            model_rdm,
            src=src,
            labels_stcs=labels_stcs,
            spatial_radius=0.05,
            sel_vertices=label,
        )
        assert_equal(rsa_result.vertices[0], stcs[0].in_label(label).vertices[0])
        assert_equal(rsa_result.vertices[1], [])

        # Restrict vertices to 2 selected indices.
        rsa_result = rsa_stcs(
            stcs,
            model_rdm,
            src=src,
            labels_stcs=labels_stcs,
            spatial_radius=0.05,
            sel_vertices_by_index=[0, 1],
        )
        assert_equal(rsa_result.vertices[0], stcs[0].vertices[0][:2])
        assert_equal(rsa_result.vertices[1], [])

        # Restrict vertices by restricting the source estimates themselves.
        stcs_restricted = [stc.in_label(label) for stc in stcs]
        rsa_result = rsa_stcs(
            stcs_restricted,
            model_rdm,
            src=src,
            labels_stcs=labels_stcs,
            spatial_radius=0.05,
            sel_vertices_by_index=[0, 1],
        )
        assert_equal(rsa_result.vertices[0], stcs_restricted[0].vertices[0][:2])
        assert_equal(rsa_result.vertices[1], [])

        # Pick non-existing vertices
        with pytest.raises(ValueError, match="not present in the data"):
            rsa_stcs(
                stcs,
                model_rdm,
                src=src,
                labels_stcs=labels_stcs,
                spatial_radius=0.05,
                sel_vertices=[[-1, 109209], [-304, 120930904]],
            )

        # Pick duplicate vertices
        with pytest.raises(ValueError, match="vertices are not unique"):
            rsa_stcs(
                stcs,
                model_rdm,
                src=src,
                labels_stcs=labels_stcs,
                spatial_radius=0.05,
                sel_vertices_by_index=[1, 1],
            )

        # Invalid inputs.
        with pytest.raises(ValueError, match="you also need to set `src`"):
            rsa_stcs(stcs, model_rdm, labels_stcs=labels_stcs, spatial_radius=0.05)

    def test_rsa_spatio_temporal(self):
        """Test performing RSA with a searchlight across both vertices and time."""
        stcs, src, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_stcs(
            stcs,
            model_rdm,
            labels_stcs=labels_stcs,
            src=src,
            temporal_radius=0.02,
            spatial_radius=0.05,
        )
        assert rsa_result.data.shape == (
            stcs[0].data.shape[0],
            len(stcs[0].times) - 2 * 2,
        )

    def test_rsa_vol(self):
        """Test performing RSA on volumetric source estimates."""
        stcs, src, labels_stcs = make_vol_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])

        # Single patch
        rsa_result = rsa_stcs(stcs, model_rdm, labels_stcs=labels_stcs)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_stcs(stcs, [model_rdm, model_rdm2], labels_stcs=labels_stcs)
        assert rsa_result.shape == (2,)

        # Temporal
        rsa_result = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, temporal_radius=0.02
        )  # 2 samples
        assert rsa_result.data.shape == (1, len(stcs[0].times) - 2 * 2)
        assert rsa_result.times[0].round(2) == 0.12
        assert rsa_result.times[-1].round(2) == 0.18

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_stcs(
            stcs,
            [model_rdm, model_rdm2],
            labels_stcs=labels_stcs,
            temporal_radius=0.02,
        )
        assert len(rsa_result) == 2
        assert not np.array_equal(rsa_result[0].data, rsa_result[1].data)
        assert rsa_result[0].data.shape == (1, len(stcs[0].times) - 2 * 2)
        assert rsa_result[0].times[0].round(2) == 0.12
        assert rsa_result[0].times[-1].round(2) == 0.18

        # Spatial
        rsa_result = rsa_stcs(
            stcs, model_rdm, labels_stcs=labels_stcs, src=src, spatial_radius=0.05
        )
        assert rsa_result.data.shape == (stcs[0].shape[0], 1)
        assert len(rsa_result.times) == 1
        assert_equal(rsa_result.times, stcs[0].times[len(stcs[0].times) // 2])

        # Spatio-temporal
        rsa_result = rsa_stcs(
            stcs,
            model_rdm,
            labels_stcs=labels_stcs,
            src=src,
            temporal_radius=0.02,
            spatial_radius=0.05,
        )
        assert rsa_result.data.shape == (
            stcs[0].data.shape[0],
            len(stcs[0].times) - 2 * 2,
        )


class TestRoiRSA:
    """Test performing RSA on ROIs."""

    def test_rsa_rois(self):
        """Test performing RSA with a searchlight across ROIs."""
        stcs, src, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rois = mne.read_labels_from_annot(
            "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
        )

        rsa_result, rsa_stc = rsa_stcs_rois(
            stcs, model_rdm, src, rois, labels_stcs=labels_stcs
        )
        assert rsa_result.shape == (len(rois),)
        assert rsa_stc.data.shape == (stcs[0].shape[0], 1)
        assert len(rsa_stc.times) == 1
        assert_equal(rsa_stc.times, stcs[0].times[len(stcs[0].times) // 2])

        rsa_result, rsa_stc = rsa_stcs_rois(stcs, model_rdm, src, rois, y=labels_stcs)
        assert rsa_result.shape == (len(rois),)
        assert rsa_stc.data.shape == (stcs[0].shape[0], 1)
        assert len(rsa_stc.times) == 1
        assert_equal(rsa_stc.times, stcs[0].times[len(stcs[0].times) // 2])

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result, rsa_stc = rsa_stcs_rois(
            stcs, [model_rdm, model_rdm2], src, rois, labels_stcs=labels_stcs
        )
        assert rsa_result.shape == (2, len(rois))
        assert not np.array_equal(rsa_result[0], rsa_result[1])
        assert len(rsa_stc) == 2
        assert rsa_stc[0].data.shape == (stcs[0].shape[0], 1)
        assert len(rsa_stc[0].times) == 1
        assert_equal(rsa_stc[0].times, stcs[0].times[len(stcs[0].times) // 2])

        # Invalid inputs.
        with pytest.raises(ValueError, match="The number of source estimates"):
            rsa_stcs_rois(stcs, model_rdm, src, rois)
        with pytest.raises(ValueError, match="items encoded in the `labels_stcs`"):
            rsa_stcs_rois(stcs, model_rdm, src, rois, labels_stcs=[1, 2, 3])

    def test_rsa_rois_temporal(self):
        """Test performing temporal RSA with a searchlight across ROIs."""
        stcs, src, labels_stcs = make_stcs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rois = mne.read_labels_from_annot(
            "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
        )

        rsa_result, rsa_stc = rsa_stcs_rois(
            stcs, model_rdm, src, rois, labels_stcs=labels_stcs, temporal_radius=0.02
        )
        assert rsa_result.shape == (len(rois), len(stcs[0].times) - 2 * 2)
        assert rsa_stc.data.shape == (stcs[0].shape[0], len(stcs[0].times) - 2 * 2)
        assert rsa_stc.times[0].round(2) == 0.12
        assert rsa_stc.times[-1].round(2) == 0.18

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            rsa_stcs_rois(
                stcs,
                model_rdm,
                src,
                rois,
                labels_stcs=labels_stcs,
                temporal_radius=0.02,
                tmin=-5,
            )
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            rsa_stcs_rois(
                stcs,
                model_rdm,
                src,
                rois,
                labels_stcs=labels_stcs,
                temporal_radius=0.02,
                tmax=5,
            )
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            rsa_stcs_rois(
                stcs,
                model_rdm,
                src,
                rois,
                labels_stcs=labels_stcs,
                temporal_radius=0.02,
                tmax=0.1,
                tmin=0.2,
            )

        # Too small to or large temporal radius
        with pytest.raises(ValueError, match="less than one sample"):
            next(
                rsa_stcs_rois(
                    stcs,
                    model_rdm,
                    src,
                    rois,
                    labels_stcs=labels_stcs,
                    temporal_radius=0,
                )
            )
        with pytest.raises(ValueError, match="too large"):
            next(
                rsa_stcs_rois(
                    stcs,
                    model_rdm,
                    src,
                    rois,
                    labels_stcs=labels_stcs,
                    temporal_radius=100,
                )
            )


class TestNiftiRSA:
    """Test performing RSA on fMRI data."""

    def test_rsa_single_searchlight_patch(self):
        """Test performing RSA with a single searchlight patch."""
        bold, mask = make_nifti()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_nifti(bold, model_rdm)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()

        # Specify labels
        rsa_result = rsa_nifti(bold, model_rdm, labels_image=[1, 0, 2, 3])
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()
        rsa_result = rsa_nifti(bold, model_rdm, y=[1, 0, 2, 3])
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_nifti(bold, [model_rdm, model_rdm2])
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == (2,)

        # Invalid data.
        with pytest.raises(ValueError, match="data must be 4-dimensional Nifti-like"):
            rsa_result = rsa_nifti(bold.get_fdata(), model_rdm)
        with pytest.raises(ValueError, match="The number of images"):
            rsa_result = rsa_nifti(bold.slicer[:, :, :, :3], model_rdm)
        with pytest.raises(ValueError, match="items encoded in the `labels_image`"):
            rsa_result = rsa_nifti(bold, model_rdm, labels_image=[1, 2, 3])

    def test_rsa_spatial(self):
        """Test performing RSA with a searchlight across vertices."""
        bold, mask = make_nifti()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_nifti(bold, model_rdm, spatial_radius=0.01)
        assert rsa_result.shape == (10, 10, 10)

        # Restrict voxels to a mask.
        rsa_result = rsa_nifti(bold, model_rdm, spatial_radius=0.01, roi_mask=mask)
        assert rsa_result.shape == (10, 10, 10)
        assert np.any(apply_mask(rsa_result, mask) != 0)
        assert apply_mask(rsa_result, mask)[2] == 0
        assert_equal(apply_mask(rsa_result, mask, invert=True), 0)

        rsa_result = rsa_nifti(bold, model_rdm, spatial_radius=0.01, brain_mask=mask)
        assert rsa_result.shape == (10, 10, 10)
        assert np.all(apply_mask(rsa_result, mask) != 0)
        assert_equal(apply_mask(rsa_result, mask, invert=True), 0)

        rsa_result = rsa_nifti(
            bold, model_rdm, spatial_radius=0.01, brain_mask=mask, roi_mask=mask
        )
        assert rsa_result.shape == (10, 10, 10)
        assert np.all(apply_mask(rsa_result, mask) != 0)
        assert_equal(apply_mask(rsa_result, mask, invert=True), 0)

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_nifti(bold, [model_rdm, model_rdm2], spatial_radius=0.01)
        assert len(rsa_result) == 2
        assert rsa_result[0].shape == (10, 10, 10)

        # Invalid data.
        with pytest.raises(ValueError, match="ROI mask"):
            rsa_result = rsa_nifti(bold, model_rdm, roi_mask=mask.slicer[:5, :5, :5])
        with pytest.raises(ValueError, match="Brain mask"):
            rsa_result = rsa_nifti(bold, model_rdm, brain_mask=mask.slicer[:5, :5, :5])


def test_restrict_src_to_vertices():
    """Test restricting a source space."""
    stcs, src, _ = make_stcs()
    vertices = [stcs[0].vertices[0][:5], stcs[0].vertices[1][:5]]
    src_restricted = _restrict_src_to_vertices(src, vertices)
    assert_equal(src_restricted[0]["vertno"], vertices[0])
    assert_equal(src_restricted[1]["vertno"], vertices[1])
    assert src_restricted[0]["nuse"] == len(vertices[0])
    assert src_restricted[1]["nuse"] == len(vertices[1])
    assert src_restricted[0]["inuse"].sum() == len(vertices[0])
    assert src_restricted[1]["inuse"].sum() == len(vertices[1])
    assert src_restricted[0]["nuse_tri"] == 0
    assert src_restricted[1]["nuse_tri"] == 0

    with pytest.raises(ValueError, match="One or more vertices were not present"):
        vertices = [[50923892], []]
        _restrict_src_to_vertices(src, vertices)

    stcs_vol, src_vol, _ = make_vol_stcs()
    vertices = [
        stcs_vol[0].vertices[0][:5],
    ]
    src_vol_restricted = _restrict_src_to_vertices(src_vol, vertices)
    assert_equal(src_vol_restricted[0]["vertno"], vertices[0])
    assert src_vol_restricted[0]["nuse"] == len(vertices[0])
    assert src_vol_restricted[0]["inuse"].sum() == len(vertices[0])
    assert src_vol_restricted[0]["nuse_tri"] == 0

    with pytest.raises(ValueError, match="One or more vertices were not present"):
        vertices = [[50923892], []]
        _restrict_src_to_vertices(src_vol, vertices)


def test_check_stcs_compatibility():
    """Test checking SourceEstimate objects for compatibility."""
    stcs, _, _ = make_stcs()
    _check_stcs_compatibility(stcs)

    # Inconsistent number of vertices.
    stcs_bad = [stc.copy() for stc in stcs]
    stcs_bad[0].vertices[0] = stcs_bad[0].vertices[0][:3]
    with pytest.raises(ValueError, match="same vertices"):
        _check_stcs_compatibility(stcs_bad)

    # Inconsistent vertex numbers
    stcs_bad = [stc.copy() for stc in stcs]
    stcs_bad[0].vertices[0][0] += 1
    with pytest.raises(ValueError, match="same vertices"):
        _check_stcs_compatibility(stcs_bad)

    # Inconsistent number of time samples.
    stcs_bad = [stc.copy() for stc in stcs]
    stcs_bad[0].crop(0.1, 0.15)
    with pytest.raises(ValueError, match="same time points"):
        _check_stcs_compatibility(stcs_bad)

    # Inconsistent time samples.
    stcs_bad = [stc.copy() for stc in stcs]
    stcs_bad[0].tmin += 1
    with pytest.raises(ValueError, match="same time points"):
        _check_stcs_compatibility(stcs_bad)


def test_check_src_compatibility():
    """Test checking compatibility between a source space and a source estimtate."""
    stcs, src, _ = make_stcs()
    _check_src_compatibility(src, stcs[0])

    stcs_vol, src_vol, _ = make_vol_stcs()
    _check_src_compatibility(src_vol, stcs_vol[0])

    with pytest.raises(ValueError, match="Volume source space"):
        _check_src_compatibility(src_vol, stcs[0])
    with pytest.raises(ValueError, match="Volume source estimate"):
        _check_src_compatibility(src, stcs_vol[0])

    # Remove a vertex and check that the source space is updated accordingly.
    stc = stcs[0].copy()
    stc.vertices[0] = stc.vertices[0][1:]
    stc.data = stc.data[1:]
    src2 = _check_src_compatibility(src, stc)
    assert src2[0]["nuse"] == src[0]["nuse"] - 1

    stc = stcs_vol[0].copy()
    stc.vertices[0] = stc.vertices[0][1:]
    stc.data = stc.data[1:]
    src_vol2 = _check_src_compatibility(src_vol, stc)
    assert src_vol2[0]["nuse"] == src_vol[0]["nuse"] - 1


def test_get_distance_matrix():
    """Test computing distances in source space."""
    _, src, _ = make_stcs()
    _, src_vol, _ = make_vol_stcs()

    # src already has distances computed
    dist = _get_distance_matrix(src)
    nuse = src[0]["nuse"] + src[1]["nuse"]
    assert dist.shape == (nuse, nuse)

    # Cross-hemisphere distances should be infinity.
    assert np.all(dist[: src[0]["nuse"] :, src[0]["nuse"] :] == np.inf)
    assert np.all(dist[src[0]["nuse"] :, : src[0]["nuse"]] == np.inf)

    # Remove precomputed distances from src and re-compute them.
    src[0]["dist"] = None
    src[1]["dist"] = None
    dist = _get_distance_matrix(src, dist_lim=0.01)
    assert dist.shape == (nuse, nuse)
    assert src[0]["dist_limit"][0] == 0.01
    assert src[1]["dist_limit"][0] == 0.01

    # If we want smaller distances, no need to re-compute.
    dist = _get_distance_matrix(src, dist_lim=0.005)
    assert dist.shape == (nuse, nuse)
    assert src[0]["dist_limit"][0] == 0.01

    # But if we want greater distances, we will need to re-compute.
    with pytest.warns(UserWarning, match="distances are smaller than the searchlight"):
        dist = _get_distance_matrix(src, dist_lim=0.02)
        assert dist.shape == (nuse, nuse)
        assert src[0]["dist_limit"][0] == 0.02

    # src_vol does not have distances pre-computed
    dist = _get_distance_matrix(src_vol, dist_lim=0.01)
    nuse = src_vol[0]["nuse"]
    assert dist.shape == (nuse, nuse)
    assert src_vol[0]["dist_limit"][0] == 0.01


def test_vertex_selection_to_indices():
    """Test selecting vertices."""
    stcs, src, _ = make_stcs()
    stc = stcs[0]
    n_vert_lh = len(stc.vertices[0])
    n_vert_rh = len(stc.vertices[1])

    assert_equal(
        vertex_selection_to_indices(stc.vertices, [[841, 155323], [1492, 156838]]),
        [0, n_vert_lh - 2, n_vert_lh + 0, n_vert_lh + n_vert_rh - 1],
    )
    assert_equal(vertex_selection_to_indices(stc.vertices, [[841, 841], []]), [0])

    labels = mne.read_labels_from_annot(
        "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
    )
    assert_equal(
        vertex_selection_to_indices(stc.vertices, labels[0]),
        np.searchsorted(
            stc.vertices[0], np.intersect1d(stc.vertices[0], labels[0].vertices)
        ),
    )

    # Invalid input.
    with pytest.raises(TypeError, match="Invalid type for sel_vertices"):
        vertex_selection_to_indices(stc.vertices, "bla")
    with pytest.raises(ValueError, match="Empty list"):
        vertex_selection_to_indices(stc.vertices, [])
    with pytest.raises(ValueError, match="one for each hemisphere"):
        vertex_selection_to_indices(stc.vertices, [1, 2, 3])
    with pytest.raises(ValueError, match="not present in the data"):
        vertex_selection_to_indices(stc.vertices, [[900], []])
    with pytest.raises(ValueError, match="right hemisphere"):
        vertex_selection_to_indices([stc.vertices[0]], labels[-1])
