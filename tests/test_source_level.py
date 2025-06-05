"""Unit tests for source-level analysis."""

import mne
import numpy as np
import pytest
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_rsa import rdm_stcs, squareform
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
