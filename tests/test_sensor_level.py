"""Unit tests for sensor-level analysis."""

import mne
import numpy as np
import pytest
from mne_rsa import rdm_epochs, rdm_evokeds, rsa_epochs, rsa_evokeds, squareform
from numpy.testing import assert_equal


def load_epochs():
    """Load some of the MNE-Sample data epochs."""
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw = mne.io.read_raw_fif(path / "sample_audvis_filt-0-40_raw.fif", preload=False)
    raw.pick("eeg")  # only use the 60 EEG sensors for these unit tests
    events = mne.read_events(path / "sample_audvis_filt-0-40_raw-eve.fif")
    events = events[:50]  # only use the first 50 events
    epochs = mne.Epochs(raw, events, event_id=[1, 2, 3, 4], preload=True)
    epochs.resample(100)  # nice round number
    return epochs


class TestSensorLevelRDMs:
    """Test making RDMs from sensor-level data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        epochs = load_epochs()
        rdms = list(rdm_epochs(epochs))
        assert len(rdms) == 1
        assert squareform(rdms[0]).shape == (4, 4)

    def test_nans(self):
        """Test inserting NaNs at dropped epochs."""
        # Start with a clean drop log
        epochs = load_epochs()
        y = epochs.events[:, 2]
        epochs.drop_log = tuple(len(epochs) * [tuple()])
        epochs.selection = np.arange(len(epochs))

        # If we still have some epochs for each event type, the RDMs should not have any
        # NaNs in them.
        epochs.drop([4])
        rdms = list(rdm_epochs(epochs, y=y, dropped_as_nan=True))
        assert not np.any(np.isnan(rdms))
        assert squareform(rdms[0]).shape == (4, 4)

        # When we drop all the epochs of some event type, the RDM should have NaNs in
        # those places.
        epochs.drop(np.flatnonzero(epochs.events[:, 2] == 4))
        rdms = list(rdm_epochs(epochs, y=y, dropped_as_nan=True))
        assert np.all(np.isnan(squareform(rdms[0])[:3, 3]))
        assert np.all(np.isnan(squareform(rdms[0])[3, :3]))

        # For this to work, a proper `y` must be supplied
        with pytest.raises(ValueError, match="you must specify a list/array `y`"):
            next(rdm_epochs(epochs, dropped_as_nan=True))

    def test_rdm_temp(self):
        """Test making RDMs with a sliding temporal window."""
        epochs = load_epochs()

        rdms = list(rdm_epochs(epochs, temporal_radius=0.1))  # 10 samples
        assert len(rdms) == len(epochs.times) - 2 * 10
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_epochs(epochs, temporal_radius=0.1, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict in time
        rdms = list(rdm_epochs(epochs, temporal_radius=0.1, tmin=0, tmax=0.3))
        assert len(rdms) == 30

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            rdms = list(rdm_epochs(epochs, temporal_radius=0.1, tmin=-5))
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            rdms = list(rdm_epochs(epochs, temporal_radius=0.1, tmax=5))
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            rdms = list(rdm_epochs(epochs, temporal_radius=0.1, tmax=0.1, tmin=0.2))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across sensors."""
        epochs = load_epochs()
        rdms = list(rdm_epochs(epochs, spatial_radius=0.1))  # 10 cm
        assert len(rdms) == epochs.info["nchan"] - len(epochs.info["bads"])
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_epochs(epochs, spatial_radius=0.1, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict channels
        rdms = list(
            rdm_epochs(epochs, spatial_radius=0.1, picks=["EEG 020", "EEG 051"])
        )
        assert len(rdms) == 2

        # Pick non-existing channels
        with pytest.raises(ValueError, match=r"\['foo'\] could not be picked"):
            rdms = list(
                rdm_epochs(epochs, spatial_radius=0.1, picks=["EEG 020", "foo"])
            )

        # Pick duplicate channels
        with pytest.raises(ValueError, match="`picks` are not unique"):
            rdms = list(
                rdm_epochs(epochs, spatial_radius=0.1, picks=["EEG 020", "EEG 020"])
            )

    def test_rdm_spatio_temporal(self):
        """Test making RDMs with a searchlight across both sensors and time."""
        epochs = load_epochs()
        rdms = list(rdm_epochs(epochs, spatial_radius=0.1, temporal_radius=0.1))
        assert len(rdms) == (epochs.info["nchan"] - len(epochs.info["bads"])) * (
            len(epochs.times) - 2 * 10
        )
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(
            rdm_epochs(epochs, spatial_radius=0.1, temporal_radius=0.1, noise_cov=cov)
        )
        assert not np.allclose(rdms, rdms_whitened)
