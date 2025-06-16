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
    epochs = mne.Epochs(raw, events, event_id=[1, 2, 3, 4], preload=True).crop(0.1, 0.2)
    epochs.resample(100)  # nice round number
    return epochs


def load_evokeds():
    """Load the MNE-Sample data evokeds."""
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    evokeds = mne.read_evokeds(path / "sample_audvis-ave.fif")
    evokeds = [evoked.pick("eeg").resample(100).crop(0.1, 0.2) for evoked in evokeds]
    return evokeds


rng = np.random.RandomState(0)


def random_dist(a, b):
    """Produce a random distance between a and b."""
    return rng.rand(1)[0]


class TestEpochRDMs:
    """Test making RDMs from sensor-level Epoch data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        epochs = load_epochs()
        rdms_epochs = list(rdm_epochs(epochs))
        assert len(rdms_epochs) == 1
        assert squareform(rdms_epochs[0]).shape == (4, 4)

    def test_nans(self):
        """Test inserting NaNs at dropped epochs."""
        # Start with a clean drop log
        epochs = load_epochs()
        labels = epochs.events[:, 2]
        epochs.drop_log = tuple(len(epochs) * [tuple()])
        epochs.selection = np.arange(len(epochs))

        # If we still have some epochs for each event type, the RDMs should not have any
        # NaNs in them.
        epochs.drop([4])
        rdms = list(rdm_epochs(epochs, labels=labels, dropped_as_nan=True))
        assert not np.any(np.isnan(rdms))
        assert squareform(rdms[0]).shape == (4, 4)

        # When we drop all the epochs of some event type, the RDM should have NaNs in
        # those places.
        epochs.drop(np.flatnonzero(epochs.events[:, 2] == 4))
        rdms = list(rdm_epochs(epochs, labels=labels, dropped_as_nan=True))
        assert np.all(np.isnan(squareform(rdms[0])[:3, 3]))
        assert np.all(np.isnan(squareform(rdms[0])[3, :3]))

        # This should also work with the legacy `y` parameter.
        epochs.drop(np.flatnonzero(epochs.events[:, 2] == 4))
        rdms = list(rdm_epochs(epochs, y=labels, dropped_as_nan=True))
        assert np.all(np.isnan(squareform(rdms[0])[:3, 3]))
        assert np.all(np.isnan(squareform(rdms[0])[3, :3]))

        # For this to work, a proper `labels` must be supplied
        with pytest.raises(ValueError, match="you must specify a list/array `labels`"):
            next(rdm_epochs(epochs, dropped_as_nan=True))

    def test_rdm_temporal(self):
        """Test making RDMs with a sliding temporal window."""
        epochs = load_epochs()

        rdms = list(rdm_epochs(epochs, temporal_radius=0.02))  # 2 samples
        assert len(rdms) == len(epochs.times) - 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_epochs(epochs, temporal_radius=0.02, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict in time
        rdms = list(rdm_epochs(epochs, temporal_radius=0.02, tmin=0.139, tmax=0.161))
        assert len(rdms) == 3

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            next(rdm_epochs(epochs, temporal_radius=0.02, tmin=-5))
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            next(rdm_epochs(epochs, temporal_radius=0.02, tmax=5))
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            next(rdm_epochs(epochs, temporal_radius=0.02, tmax=0.1, tmin=0.2))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across sensors."""
        epochs = load_epochs()
        rdms = list(rdm_epochs(epochs, spatial_radius=0.05))  # 5 cm
        assert len(rdms) == epochs.info["nchan"] - len(epochs.info["bads"])
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_epochs(epochs, spatial_radius=0.05, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict channels
        rdms = list(
            rdm_epochs(epochs, spatial_radius=0.05, picks=["EEG 020", "EEG 051"])
        )
        assert len(rdms) == 2

        # Pick non-existing channels
        with pytest.raises(ValueError, match="could not be picked"):
            next(rdm_epochs(epochs, spatial_radius=0.05, picks=["EEG 020", "foo"]))

        # Pick duplicate channels
        with pytest.raises(ValueError, match="`picks` are not unique"):
            next(rdm_epochs(epochs, spatial_radius=0.05, picks=["EEG 020", "EEG 020"]))

    def test_rdm_spatio_temporal(self):
        """Test making RDMs with a searchlight across both sensors and time."""
        epochs = load_epochs()
        rdms = list(rdm_epochs(epochs, spatial_radius=0.05, temporal_radius=0.02))
        assert len(rdms) == (epochs.info["nchan"] - len(epochs.info["bads"])) * (
            len(epochs.times) - 2 * 2
        )
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(
            rdm_epochs(epochs, spatial_radius=0.05, temporal_radius=0.02, noise_cov=cov)
        )
        assert not np.allclose(rdms, rdms_whitened)


class TestEvokedRDMs:
    """Test making RDMs from sensor-level Evoked data."""

    def test_rdm_single_searchlight_patch(self):
        """Test making an RDM with a single searchlight patch."""
        evokeds = load_evokeds()
        rdms_evokeds = list(rdm_evokeds(evokeds))
        assert len(rdms_evokeds) == 1
        assert squareform(rdms_evokeds[0]).shape == (4, 4)

        # Check equivalence of rdm_epochs and rdm_evokeds
        evokeds_as_epochs = mne.EpochsArray(
            np.array([e.data for e in evokeds]),
            info=evokeds[0].info,
            events=np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]]),
        )
        rdms_evokeds_as_epochs = list(rdm_epochs(evokeds_as_epochs))
        assert np.allclose(rdms_evokeds_as_epochs, rdms_evokeds)

        # Invalid input.
        with pytest.raises(ValueError, match="same time points"):
            evokeds_bad = [e.copy() for e in evokeds]
            evokeds_bad[0].crop(0.11, 0.15)  # one evoked is shorter than the others
            next(rdm_evokeds(evokeds_bad))

    def test_rdm_temporal(self):
        """Test making RDMs with a sliding temporal window."""
        evokeds = load_evokeds()

        rdms = list(rdm_evokeds(evokeds, temporal_radius=0.02))  # 2 samples
        assert len(rdms) == len(evokeds[0].times) - 2 * 2
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_evokeds(evokeds, temporal_radius=0.02, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict in time
        rdms = list(rdm_evokeds(evokeds, temporal_radius=0.02, tmin=0.14, tmax=0.161))
        assert len(rdms) == 3

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            next(rdm_evokeds(evokeds, temporal_radius=0.02, tmin=-5))
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            next(rdm_evokeds(evokeds, temporal_radius=0.02, tmax=5))
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            next(rdm_evokeds(evokeds, temporal_radius=0.02, tmax=0.1, tmin=0.2))

    def test_rdm_spatial(self):
        """Test making RDMs with a searchlight across sensors."""
        evokeds = load_evokeds()
        rdms = list(rdm_evokeds(evokeds, spatial_radius=0.05))  # 10 cm
        assert len(rdms) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(rdm_evokeds(evokeds, spatial_radius=0.05, noise_cov=cov))
        assert not np.allclose(rdms, rdms_whitened)

        # Restrict channels
        rdms = list(
            rdm_evokeds(evokeds, spatial_radius=0.05, picks=["EEG 020", "EEG 051"])
        )
        assert len(rdms) == 2

        # Pick non-existing channels
        with pytest.raises(ValueError, match="could not be picked"):
            rdms = list(
                rdm_evokeds(evokeds, spatial_radius=0.05, picks=["EEG 020", "foo"])
            )

        # Pick duplicate channels
        with pytest.raises(ValueError, match="`picks` are not unique"):
            rdms = list(
                rdm_evokeds(evokeds, spatial_radius=0.05, picks=["EEG 020", "EEG 020"])
            )

    def test_rdm_spatio_temporal(self):
        """Test making RDMs with a searchlight across both sensors and time."""
        evokeds = load_evokeds()
        rdms = list(rdm_evokeds(evokeds, spatial_radius=0.05, temporal_radius=0.02))
        assert len(rdms) == (
            evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        ) * (len(evokeds[0].times) - 2 * 2)
        assert squareform(rdms[0]).shape == (4, 4)

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rdms_whitened = list(
            rdm_evokeds(
                evokeds, spatial_radius=0.05, temporal_radius=0.02, noise_cov=cov
            )
        )
        assert not np.allclose(rdms, rdms_whitened)


class TestEpochRSA:
    """Test performing RSA on sensor-level Epoch data."""

    def test_rsa_single_searchlight_patch(self):
        """Test performing RSA with a single searchlight patch."""
        epochs = load_epochs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_epochs(epochs, model_rdm)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()
        assert_equal(rsa_result.round(2), 0.41)

        # Try using different metrics
        rsa_one = rsa_epochs(epochs, model_rdm, epochs_rdm_metric=random_dist)
        assert not np.allclose(rsa_result, rsa_one)
        rsa_tau = rsa_epochs(epochs, model_rdm, rsa_metric="kendall-tau-a")
        assert not np.allclose(rsa_result, rsa_tau)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_epochs(epochs, model_rdm, noise_cov=cov)
        assert not np.allclose(rsa_result, rsa_whitened)

        # Invalid input
        with pytest.raises(ValueError, match="number of items in `rdm_model`"):
            model_bad = np.array([0.5, 1, 0.5])
            rsa_result = rsa_epochs(epochs, model_bad)
        with pytest.raises(ValueError, match="number of items encoded in the `y`"):
            rsa_result = rsa_epochs(epochs, model_rdm, y=[1, 2, 3])

    def test_rsa_temporal(self):
        """Test performing RSA with a sliding temporal window."""
        epochs = load_epochs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_epochs(epochs, model_rdm, temporal_radius=0.02)  # 2 samples
        assert rsa_result.data.shape == (1, len(epochs.times) - 2 * 2)
        assert rsa_result.data.max().round(2) == 0.83
        assert rsa_result.times[0] == 0.12
        assert rsa_result.times[-1] == 0.18

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_epochs(
            epochs, model_rdm, temporal_radius=0.02, noise_cov=cov
        )
        assert not np.allclose(rsa_result.data, rsa_whitened.data)

        # Restrict in time
        rsa_result = rsa_epochs(
            epochs, model_rdm, temporal_radius=0.02, tmin=0.139, tmax=0.161
        )
        print(epochs.times)
        print(rsa_result.times)
        assert rsa_result.data.shape == (1, 3)
        assert rsa_result.times[0] == 0.14
        assert rsa_result.times[-1] == 0.16

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_epochs(epochs, [model_rdm, model_rdm2], temporal_radius=0.02)
        assert len(rsa_result) == 2
        assert rsa_result[0].data.shape == (1, len(epochs.times) - 2 * 2)

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            rsa_epochs(epochs, model_rdm, temporal_radius=0.02, tmin=-5)
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            rsa_epochs(epochs, model_rdm, temporal_radius=0.02, tmax=5)
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            rsa_epochs(epochs, model_rdm, temporal_radius=0.02, tmax=0.1, tmin=0.2)

    def test_rsa_spatial(self):
        """Test making rsas with a searchlight across sensors."""
        epochs = load_epochs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_epochs(epochs, model_rdm, spatial_radius=0.05)  # 10 cm
        assert rsa_result.data.shape == (
            epochs.info["nchan"] - len(epochs.info["bads"]),
            1,
        )
        assert rsa_result.data.max().round(2) == 0.83
        assert len(rsa_result.times) == 1
        assert_equal(rsa_result.times, 0.15)

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_epochs(epochs, model_rdm, spatial_radius=0.05, noise_cov=cov)
        assert not np.allclose(rsa_result.data, rsa_whitened.data)

        # Restrict channels
        rsa_result = rsa_epochs(
            epochs, model_rdm, spatial_radius=0.05, picks=["EEG 020", "EEG 051"]
        )
        assert rsa_result.data.shape == (2, 1)

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_epochs(epochs, [model_rdm, model_rdm2], spatial_radius=0.05)
        assert len(rsa_result) == 2
        assert rsa_result[0].data.shape == (
            epochs.info["nchan"] - len(epochs.info["bads"]),
            1,
        )

        # Pick non-existing channels
        with pytest.raises(ValueError, match="could not be picked"):
            rsa_epochs(epochs, model_rdm, spatial_radius=0.05, picks=["EEG 020", "foo"])

        # Pick duplicate channels
        with pytest.raises(ValueError, match="`picks` are not unique"):
            rsa_epochs(
                epochs, model_rdm, spatial_radius=0.05, picks=["EEG 020", "EEG 020"]
            )

    def test_rsa_spatio_temporal(self):
        """Test making rsas with a searchlight across both sensors and time."""
        epochs = load_epochs()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_epochs(
            epochs, model_rdm, temporal_radius=0.02, spatial_radius=0.05
        )
        assert rsa_result.data.shape == (
            epochs.info["nchan"] - len(epochs.info["bads"]),
            len(epochs.times) - 2 * 2,
        )

        # With noise normalization
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_epochs(
            epochs,
            model_rdm,
            spatial_radius=0.05,
            temporal_radius=0.02,
            noise_cov=cov,
        )
        assert not np.allclose(rsa_result.data, rsa_whitened.data)


class TestEvokedRSA:
    """Test performing RSA on sensor-level Evoked data."""

    def test_rsa_single_searchlight_patch(self):
        """Test performing RSA with a single searchlight patch."""
        evokeds = load_evokeds()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_evokeds(evokeds, model_rdm)
        assert isinstance(rsa_result, np.ndarray)
        assert rsa_result.shape == tuple()
        assert_equal(rsa_result.round(2), 0.62)

        # Try using different metrics
        rsa_one = rsa_evokeds(evokeds, model_rdm, evoked_rdm_metric=random_dist)
        assert not np.allclose(rsa_result, rsa_one)
        rsa_tau = rsa_evokeds(evokeds, model_rdm, rsa_metric="kendall-tau-a")
        assert not np.allclose(rsa_result, rsa_tau)

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_evokeds(evokeds, model_rdm, noise_cov=cov)
        assert not np.allclose(rsa_result, rsa_whitened)

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_evokeds(evokeds, [model_rdm, model_rdm2])
        assert rsa_result.shape == (2,)

        # Invalid input
        with pytest.raises(ValueError, match="number of items in `rdm_model`"):
            model_bad = np.array([0.5, 1, 0.5])
            rsa_result = rsa_evokeds(evokeds, model_bad)
        with pytest.raises(ValueError, match="number of items encoded in the `y`"):
            rsa_result = rsa_evokeds(evokeds, model_rdm, y=[1, 2, 3])
        with pytest.raises(ValueError, match="same time points"):
            evokeds_bad = [e.copy() for e in evokeds]
            evokeds_bad[0].crop(0.11, 0.15)  # one evoked is shorter than the others
            rsa_result = rsa_evokeds(evokeds_bad, model_rdm)

    def test_rsa_temporal(self):
        """Test performing RSA with a sliding temporal window."""
        evokeds = load_evokeds()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_evokeds(evokeds, model_rdm, temporal_radius=0.02)  # 2 samples
        assert rsa_result.data.shape == (1, len(evokeds[0].times) - 2 * 2)
        assert rsa_result.data.max().round(2) == 0.83
        assert rsa_result.times[0] == 0.12
        assert rsa_result.times[-1] == 0.18

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_evokeds(
            evokeds, model_rdm, temporal_radius=0.02, noise_cov=cov
        )
        assert not np.allclose(rsa_result.data, rsa_whitened.data)

        # Restrict in time
        rsa_result = rsa_evokeds(
            evokeds, model_rdm, temporal_radius=0.02, tmin=0.14, tmax=0.161
        )
        print(evokeds[0].times)
        print(rsa_result.times)
        assert rsa_result.data.shape == (1, 3)
        assert rsa_result.times[0] == 0.14
        assert rsa_result.times[-1] == 0.16

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_evokeds(evokeds, [model_rdm, model_rdm2], temporal_radius=0.02)
        assert len(rsa_result) == 2
        assert rsa_result[0].data.shape == (1, len(evokeds[0].times) - 2 * 2)

        # Out of bounds and wrong order of tmin/tmax
        with pytest.raises(ValueError, match="`tmin=-5` is before the first sample"):
            rsa_evokeds(evokeds, model_rdm, temporal_radius=0.1, tmin=-5)
        with pytest.raises(ValueError, match="`tmax=5` is after the last sample"):
            rsa_evokeds(evokeds, model_rdm, temporal_radius=0.1, tmax=5)
        with pytest.raises(ValueError, match="`tmax=0.1` is smaller than `tmin=0.2`"):
            rsa_evokeds(evokeds, model_rdm, temporal_radius=0.1, tmax=0.1, tmin=0.2)

    def test_rsa_spatial(self):
        """Test making rsas with a searchlight across sensors."""
        evokeds = load_evokeds()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_evokeds(evokeds, model_rdm, spatial_radius=0.05)  # 5 cm
        assert rsa_result.data.shape == (
            evokeds[0].info["nchan"] - len(evokeds[0].info["bads"]),
            1,
        )
        assert rsa_result.data.max().round(2) == 0.83
        assert len(rsa_result.times) == 1
        assert_equal(rsa_result.times, 0.15)

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_evokeds(
            evokeds, model_rdm, spatial_radius=0.05, noise_cov=cov
        )
        assert not np.allclose(rsa_result.data, rsa_whitened.data)

        # Restrict channels
        rsa_result = rsa_evokeds(
            evokeds, model_rdm, spatial_radius=0.05, picks=["EEG 020", "EEG 051"]
        )
        assert rsa_result.data.shape == (2, 1)

        # Two model RDMs
        model_rdm2 = np.array([0.2, 0.5, 1, 1, 0.5, 0.2])
        rsa_result = rsa_evokeds(evokeds, [model_rdm, model_rdm2], spatial_radius=0.05)
        assert len(rsa_result) == 2
        assert rsa_result[0].data.shape == (
            evokeds[0].info["nchan"] - len(evokeds[0].info["bads"]),
            1,
        )

        # Pick non-existing channels
        with pytest.raises(ValueError, match="could not be picked"):
            rsa_evokeds(
                evokeds, model_rdm, spatial_radius=0.05, picks=["EEG 020", "foo"]
            )

        # Pick duplicate channels
        with pytest.raises(ValueError, match="`picks` are not unique"):
            rsa_evokeds(
                evokeds, model_rdm, spatial_radius=0.05, picks=["EEG 020", "EEG 020"]
            )

    def test_rsa_spatio_temporal(self):
        """Test making rsas with a searchlight across both sensors and time."""
        evokeds = load_evokeds()
        model_rdm = np.array([0.5, 1, 1, 1, 1, 0.5])
        rsa_result = rsa_evokeds(
            evokeds, model_rdm, temporal_radius=0.02, spatial_radius=0.05
        )
        assert rsa_result.data.shape == (
            evokeds[0].info["nchan"] - len(evokeds[0].info["bads"]),
            len(evokeds[0].times) - 2 * 2,
        )

        # With noise normalization
        epochs = load_epochs()
        cov = mne.compute_covariance(epochs)
        rsa_whitened = rsa_evokeds(
            evokeds,
            model_rdm,
            spatial_radius=0.05,
            temporal_radius=0.02,
            noise_cov=cov,
        )
        assert not np.allclose(rsa_result.data, rsa_whitened.data)
