"""Unit tests for the creation of RDMs."""

import numpy as np
import pytest
from mne_rsa import compute_rdm, compute_rdm_cv, pick_rdm, rdm_array, searchlight
from mne_rsa.rdm import _ensure_condensed, _n_items_from_rdm
from numpy.testing import assert_allclose, assert_equal


class TestDsm:
    """Test computing a RDM."""

    def test_basic(self):
        """Test basic invocation of compute_rdm."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        rdm = compute_rdm(data)
        assert rdm.shape == (1,)
        assert_allclose(rdm, 0, atol=1e-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_rdm."""
        data = np.array([[1], [1]])
        with pytest.raises(ValueError, match="single feature"):
            compute_rdm(data, metric="correlation")

    def test_set_metric(self):
        """Test setting distance metric for computing RDMs."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
        rdm = compute_rdm(data, metric="euclidean")
        assert rdm.shape == (1,)
        assert_allclose(rdm, 5.477226)


class TestDsmCV:
    """Test computing a RDM with cross-validation."""

    def test_basic(self):
        """Test basic invocation of compute_rdm_cv."""
        data = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        rdm = compute_rdm_cv(data)
        assert rdm.shape == (1,)
        assert_allclose(rdm, 0, atol=1e-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_rdm."""
        data = np.array([[[1], [1]]])
        with pytest.raises(ValueError, match="single feature"):
            compute_rdm_cv(data, metric="correlation")

    def test_set_metric(self):
        """Test setting distance metric for computing RDMs."""
        data = np.array([[[1, 2, 3, 4], [2, 4, 6, 8]], [[1, 2, 3, 4], [2, 4, 6, 8]]])
        rdm = compute_rdm_cv(data, metric="euclidean")
        assert rdm.shape == (1,)
        assert_allclose(rdm, 5.477226)


class TestEnsureCondensed:
    """Test the _ensure_condensed function."""

    def test_basic(self):
        """Test basic invocation of _ensure_condensed."""
        rdm = _ensure_condensed(
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), var_name="test"
        )
        assert rdm.shape == (3,)
        assert_equal(rdm, [1, 2, 3])

    def test_list(self):
        """Test invocation of _ensure_condensed on a list."""
        full = [
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
        ]
        rdm = _ensure_condensed(full, var_name="full")
        assert len(rdm) == 2
        assert rdm[0].shape == (3,)
        assert rdm[1].shape == (3,)
        assert_equal(rdm, [[1, 2, 3], [1, 2, 3]])

    def test_condensed(self):
        """Test invocation of _ensure_condensed on already condensed RDM."""
        rdm = _ensure_condensed(np.array([1, 2, 3]), var_name="test")
        assert rdm.shape == (3,)
        assert_equal(rdm, [1, 2, 3])

    def test_invalid(self):
        """Test _ensure_condensed with invalid inputs."""
        # Not a square matrix
        with pytest.raises(ValueError, match="square matrix"):
            _ensure_condensed(np.array([[0, 1], [1, 0], [2, 3]]), var_name="test")

        # Too many dimensions
        with pytest.raises(ValueError, match="Invalid dimensions"):
            _ensure_condensed(np.array([[[[[0, 1, 2, 3]]]]]), var_name="test")

        # Invalid type
        with pytest.raises(TypeError, match="NumPy array"):
            _ensure_condensed([1, 2, 3], var_name="test")


class TestNItemsFromRDM:
    """Test the _n_items_from_rdm function."""

    def test_basic(self):
        """Test basic invocation of _n_items_from_rdm."""
        assert _n_items_from_rdm(np.array([1, 2, 3])) == 3
        assert _n_items_from_rdm(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])) == 3
        with pytest.raises(ValueError):
            _n_items_from_rdm(np.array([[[1, 2, 3]]]))


class TestSelectItemsFromRDM:
    """Test the _select_items_from_rdm function."""

    def test_basic(self):
        """Test basic invocation of _select_items_from_rdm."""
        assert_equal(pick_rdm(np.array([1, 2, 3]), 1), np.array([]))
        assert_equal(
            pick_rdm(np.array([1, 2, 3]), slice(None)),
            np.array([1, 2, 3]),
        )
        assert_equal(pick_rdm(np.array([1, 2, 3]), [0, 1]), np.array([1]))
        assert_equal(pick_rdm(np.array([1, 2, 3]), [0, 2]), np.array([2]))
        assert_equal(pick_rdm(np.array([1, 2, 3]), [1, 2]), np.array([3]))
        assert_equal(
            pick_rdm(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), [1, 2]),
            np.array([[0, 3], [3, 0]]),
        )
        with pytest.raises(ValueError):
            pick_rdm(np.array([[[1, 2, 3]]]), 1)


class TestDsmsSearchlight:
    """Test computing RDMs with searchlight patches."""

    def test_temporal(self):
        """Test computing RDMs using a temporal searchlight."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, temporal_radius=1)
        rdms = rdm_array(data, patches, dist_metric="euclidean")
        assert len(rdms) == len(patches)
        assert rdms.shape == (2, 1)
        assert_equal(list(rdms), [0, 0])

    def test_spatial(self):
        """Test computing RDMs using a spatial searchlight."""
        dist = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rdms = rdm_array(data, patches, dist_metric="euclidean")
        assert len(rdms) == len(patches)
        assert rdms.shape == (4, 1)
        assert_equal(list(rdms), [0, 0, 0, 0])

    def test_spatio_temporal(self):
        """Test computing RDMs using a spatio-temporal searchlight."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rdms = rdm_array(data, patches, dist_metric="correlation")
        assert len(rdms) == len(patches)
        assert rdms.shape == (2, 1, 3)
        assert_allclose(list(rdms), 0, atol=1e-15)

    def test_single_patch(self):
        """Test computing RDMs using a single searchlight patch."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        rdms = rdm_array(data, dist_metric="correlation")
        assert len(rdms) == 1
        assert rdms.shape == (3,)
        assert_allclose(list(rdms), 0, atol=1e-15)

    def test_crossvalidation(self):
        """Test computing RDMs using a searchlight and cross-validation."""
        data = np.array(
            [
                [[1, 2, 3], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 5]],
                [[3, 4, 5], [4, 5, 6]],
                [[1, 2, 3], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 5]],
                [[3, 4, 5], [4, 5, 6]],
            ]
        )
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rdms = rdm_array(data, patches, y=[1, 2, 3, 1, 2, 3], n_folds=2)
        assert len(rdms) == len(patches)
        assert rdms.shape == (2, 1, 3)
        assert_allclose(list(rdms), 0, atol=1e-15)

    def test_generator(self):
        """Test generator behavior when computing RDMs."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, temporal_radius=1)
        rdms = rdm_array(data, patches, dist_metric="euclidean")
        assert len(rdms) == len(patches)
        assert next(rdms).shape == (1,)
        # test restart behavior
        rdm_list1 = [rdm for rdm in iter(rdms)]
        rdm_list2 = [rdm for rdm in iter(rdms)]
        assert len(rdm_list1) == len(rdm_list2) == 2

    def test_match_order(self):
        """Test order matching of rdm_array."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        rdms = rdm_array(data, labels=[0, 0, 1], dist_metric="correlation")
        assert len(rdms) == 1
        assert rdms.shape == (1,)
        assert_allclose(list(rdms), 0, atol=1e-15)

        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        rdms = rdm_array(data, labels=["b", "c", "b"], dist_metric="correlation")
        assert len(rdms) == 1
        assert rdms.shape == (1,)
        assert_allclose(list(rdms), 0, atol=1e-15)
