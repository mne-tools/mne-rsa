"""Unit tests for comparing RDMs (which is called RSA)."""

from types import GeneratorType

import numpy as np
import pytest
from mne_rsa import rsa, rsa_array, rsa_gen, searchlight
from mne_rsa.rsa import _kendall_tau_a, _partial_correlation
from numpy.testing import assert_allclose, assert_equal


def rdm():
    """Create an example RDM."""
    return np.array([1, 2, 3, 4, 5, 6])


def rdm_gen(rdms):
    """Generate RDMs."""
    for rdm in rdms:
        yield np.asarray(rdm)


class TestRSAGen:
    """Test the rsa_gen function."""

    def test_return_type(self):
        """Test return type of rsa_gen."""
        assert isinstance(rsa_gen(rdm_gen([rdm()]), rdm()), GeneratorType)
        assert next(rsa_gen(rdm_gen([rdm()]), rdm())).shape == tuple()
        assert next(rsa_gen(rdm_gen([rdm()]), [rdm()])).shape == (1,)

    def test_spearman(self):
        """Test computing RSA with Spearman correlation."""
        data_rdm = rdm_gen([[1, 2, 3]])
        model_rdm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_rdm, model_rdm, metric="spearman")) == 1.0

        data_rdm = rdm_gen([[np.nan, 2, 3, 4, 5]])
        model_rdm = np.array([2, np.nan, 3.5, 4, 5])
        assert (
            next(rsa_gen(data_rdm, model_rdm, metric="spearman", ignore_nan=True))
            == 1.0
        )

    def test_pearson(self):
        """Test computing RSA with Pearson correlation."""
        data_rdm = rdm_gen([[1, 2, 3]])
        model_rdm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_rdm, model_rdm, metric="pearson")) < 1.0

        data_rdm = rdm_gen([[np.nan, 2, 3, 4, 5]])
        model_rdm = np.array([2, np.nan, 3.5, 4, 5])
        assert (
            next(rsa_gen(data_rdm, model_rdm, metric="pearson", ignore_nan=True)) < 1.0
        )

    def test_kendall_tau_a(self):
        """Test computing RSA with Kendall's Tau Alpha."""
        data_rdm = rdm_gen([[1, 2, 3]])
        model_rdm = np.array([1, 3, 3])  # This metric deals well with ties
        rsa_val = next(rsa_gen(data_rdm, model_rdm, metric="kendall-tau-a"))
        assert rsa_val == 2 / 3

        data_rdm = rdm_gen([[1, np.nan, 2, 3, 4]])
        model_rdm = np.array([1, 2, 3, 3, np.nan])
        rsa_val = next(
            rsa_gen(data_rdm, model_rdm, metric="kendall-tau-a", ignore_nan=True)
        )
        assert rsa_val == 2 / 3

    def test_regression(self):
        """Test computing RSA with regression."""
        model_rdm1 = np.array([-1, 0, 1])
        model_rdm2 = np.array([1, -2, 1])
        data_rdm = rdm_gen([3 * model_rdm1 + 5 * model_rdm2])
        rsa_val = next(rsa_gen(data_rdm, [model_rdm1, model_rdm2], metric="regression"))
        assert_allclose(rsa_val, [3, 5])

        model_rdm1 = np.array([-1, np.nan, 0, 1, 1])
        model_rdm2 = np.array([1, 3, -2, 1, np.nan])
        data_rdm = rdm_gen([3 * model_rdm1 + 5 * model_rdm2])
        rsa_val = next(
            rsa_gen(
                data_rdm, [model_rdm1, model_rdm2], metric="regression", ignore_nan=True
            )
        )
        assert_allclose(rsa_val, [3, 5])

    def test_partial(self):
        """Test computing RSA with partial correlation."""
        # Example taken from https://en.wikipedia.org/wiki/Partial_correlation
        model_rdm1 = np.array([1, 2, 3, 4])
        model_rdm2 = np.array([0, 0, 1, 1])
        data_rdm = rdm_gen([[2, 4, 15, 20]])
        rsa_val = next(rsa_gen(data_rdm, [model_rdm1, model_rdm2], metric="partial"))
        assert_allclose(rsa_val, [0.919145, 0.912871])

        model_rdm1 = np.array([1, np.nan, 2, 3, 4, 4])
        model_rdm2 = np.array([0, 0, 0, 1, 1, np.nan])
        data_rdm = rdm_gen([[2, np.nan, 4, 15, 20, np.nan]])
        rsa_val = next(
            rsa_gen(
                data_rdm, [model_rdm1, model_rdm2], metric="partial", ignore_nan=True
            )
        )
        assert_allclose(rsa_val, [0.919145, 0.912871])

    def test_partial_spearman(self):
        """Test computing RSA with partial spearman correlation."""
        # Example verified with MATLAB's partialcorr function
        model_rdm1 = np.array([1, 2, 3, 4])
        model_rdm2 = np.array([0, 0, 1, 1])
        data_rdm = rdm_gen([[2, 4, 20, 15]])
        rsa_val = next(
            rsa_gen(data_rdm, [model_rdm1, model_rdm2], metric="partial-spearman")
        )
        assert_allclose(rsa_val, [0, 2 / 3], atol=1e-15)

        model_rdm1 = np.array([1, np.nan, 2, 3, 4, 4])
        model_rdm2 = np.array([0, 0, 0, 1, 1, np.nan])
        data_rdm = rdm_gen([[2, np.nan, 4, 20, 15, np.nan]])
        rsa_val = next(
            rsa_gen(
                data_rdm,
                [model_rdm1, model_rdm2],
                metric="partial-spearman",
                ignore_nan=True,
            )
        )
        assert_allclose(rsa_val, [0, 2 / 3], atol=1e-15)

    def test_invalid_metric(self):
        """Test whether computing RSA with an invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid RSA metric"):
            next(rsa_gen(rdm_gen([rdm()]), rdm(), metric="foo"))

        # These metrics only work with multiple model RDMs
        with pytest.raises(ValueError, match="Need more than one model RDM"):
            next(rsa_gen(rdm_gen([rdm()]), rdm(), metric="partial"))
        with pytest.raises(ValueError, match="Need more than one model RDM"):
            next(rsa_gen(rdm_gen([rdm()]), rdm(), metric="partial-spearman"))

    def test_nan(self):
        """Test whether nans are dealt with correctly."""
        assert np.isnan(next(rsa_gen(rdm_gen([[1, 2, np.nan, 4, 5, 6]]), rdm())))
        assert_allclose(
            next(rsa_gen(rdm_gen([[1, 2, np.nan, 4, 5, 6]]), rdm(), ignore_nan=True)),
            1,
            atol=1e-15,
        )


class TestRSA:
    """Test the main RSA function."""

    # Most of the functionality is already tested in TestRSAGen
    def test_return_type(self):
        """Test return type of rsa_gen."""
        assert isinstance(rsa([rdm()], rdm()), np.ndarray)
        assert rsa(rdm(), rdm()).shape == tuple()
        assert rsa(rdm(), [rdm()]).shape == (1,)

    def test_progress_bar(self):
        """Test showing a progress bar for rsa."""
        assert rsa([rdm()], rdm(), verbose=True) == [1.0]
        assert rsa([rdm()], rdm(), verbose=True, n_data_rdms=1) == [1.0]
        assert rsa(rdm_gen([rdm()]), rdm(), verbose=True) == [1.0]

    def test_parallel_processing(self):
        """Test running rsa with multiple cores."""
        assert_equal(rsa([rdm(), rdm()], rdm(), n_jobs=2), [1.0, 1.0])


class TestRSASearchlight:
    """Test computing RSA using searchlight patches."""

    def test_invalid_input(self):
        """Test invalid inputs."""
        data = np.array([[1], [2], [3], [4]])
        model_rdm = rdm()
        with pytest.raises(ValueError, match="There is only a single feature"):
            rsa_array(data, model_rdm)

    def test_rsa_single_searchlight_patch(self):
        """Test RSA with a single searchlight patch."""
        data = np.array([[1], [2], [3], [4]])
        model_rdm = np.array([1, 2, 3, 1, 2, 1])
        rsa_result = rsa_array(data, model_rdm, data_rdm_metric="euclidean")
        assert rsa_result.shape == tuple()  # Single scalar value
        assert rsa_result == 1

    def test_rsa_temp(self):
        """Test RSA with a temporal searchlight."""
        data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        model_rdm = np.array([1, 2, 1])

        # One model RDM, no parallelization
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(data, model_rdm, patches, data_rdm_metric="euclidean")
        assert rsa_result.shape == (2,)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, no parallelization
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_rdm, model_rdm], patches, data_rdm_metric="euclidean"
        )
        assert rsa_result.shape == (2, 2)
        assert_equal(rsa_result, 1)

        # One model RDM, parallelization across 2 CPUs
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, model_rdm, patches, data_rdm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2,)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, parallelization across 2 CPUs
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_rdm, model_rdm], patches, data_rdm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 2)
        assert_equal(rsa_result, 1)

    def test_rsa_spat(self):
        """Test RSA with a spatial searchlight."""
        data = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
        model_rdm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        # One model RDM, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(data, model_rdm, patches, data_rdm_metric="euclidean")
        assert rsa_result.shape == (3,)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, [model_rdm, model_rdm], patches, data_rdm_metric="euclidean"
        )
        assert rsa_result.shape == (2, 3)
        assert_equal(rsa_result, 1)

        # One model RDM, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, model_rdm, patches, data_rdm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (3,)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, [model_rdm, model_rdm], patches, data_rdm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 3)
        assert_equal(rsa_result, 1)

    def test_rsa_spat_temp(self):
        """Test RSA with a spatial-temporal searchlight."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        model_rdm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        # One model RDM, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(data, model_rdm, patches, data_rdm_metric="euclidean")
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_rdm, model_rdm], patches, data_rdm_metric="euclidean"
        )
        assert rsa_result.shape == (2, 2, 1)
        assert_equal(rsa_result, 1)

        # One model RDM, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data, model_rdm, patches, data_rdm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model RDMs, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data,
            [model_rdm, model_rdm],
            patches,
            data_rdm_metric="euclidean",
            n_jobs=2,
            verbose=True,
        )
        assert rsa_result.shape == (2, 2, 1)
        assert_equal(rsa_result, 1)

    def test_nan(self):
        """Test whether nans are dealt with correctly."""
        data = np.array(
            [[1, 2, 3, 4, 5], [2, 3, 4, np.nan, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
        )
        model_rdm = np.array([1, 2, 3, 1, 2, 1])
        patches = searchlight(data.shape, temporal_radius=1)

        # Without ignoring NaNs, the result should contain NaNs where data is missing.
        rsa_result = rsa_array(data, model_rdm, patches, data_rdm_metric="euclidean")
        assert rsa_result.shape == (3,)
        assert ~np.isnan(rsa_result[0])
        assert np.all(np.isnan(rsa_result[1:]))

        # When ignoring NaNs, the result should never contain NaNs.
        rsa_result = rsa_array(
            data, model_rdm, patches, data_rdm_metric="euclidean", ignore_nan=True
        )
        assert rsa_result.shape == (3,)
        assert_equal(rsa_result, 1)

    def test_rsa_match_order(self):
        """Test order matching of rsa_array."""
        data = np.array([[1], [2], [3], [4]])
        model_rdm = np.array([1, 2, 3, 1, 2, 1])

        rsa_result = rsa_array(
            data,
            model_rdm,
            data_rdm_metric="euclidean",
            labels_X=["a", "b", "c", "d"],
            labels_rdm_model=["a", "b", "c", "d"],
        )
        assert rsa_result == 1

        rsa_result = rsa_array(
            X=np.array([[4], [2], [3], [1]]),
            rdm_model=np.array([1, 2, 3, 1, 2, 1]),
            data_rdm_metric="euclidean",
            labels_X=["d", "b", "c", "a"],
            labels_rdm_model=["a", "b", "c", "d"],
        )
        assert rsa_result == 1

        rsa_result = rsa_array(
            X=np.array([[4], [2], [3], [1]]),
            rdm_model=np.array([1, 2, 3, 1, 2, 1]),
            data_rdm_metric="euclidean",
            labels_X=["a", "b", "c", "d"],
            labels_rdm_model=["d", "b", "c", "a"],
        )
        assert rsa_result == 1

        rsa_result = rsa_array(
            X=np.array([[1], [3], [2], [4], [3], [5], [4], [6]]),
            rdm_model=np.array([1, 2, 3, 1, 2, 1]),
            data_rdm_metric="euclidean",
            labels_X=["a", "a", "b", "b", "c", "c", "d", "d"],
            labels_rdm_model=["a", "b", "c", "d"],
        )
        assert rsa_result == 1


class TestKendallTau:
    """Test computing Kendall's Tau Alpha metric."""

    def test_basic(self):
        """Test computing Kendall's Tau Alpha."""
        # This metric deals well with ties
        assert _kendall_tau_a([1, 2, 3], [1, 3, 3]) == 2 / 3

        # Test taken from scipy
        assert _kendall_tau_a([9, 2, 5, 6], [4, 7, 9, 11]) == 0

    def test_sizes(self):
        """Test feeding arrays to Kendall's Tau Alpha of unequal size."""
        with pytest.raises(ValueError, match="must be of the same size"):
            _kendall_tau_a([1, 2, 3], [1, 2, 3, 4, 5])

    def test_empty(self):
        """Test feeding empty arrays to Kendall's Tau Alpha."""
        assert np.isnan(_kendall_tau_a([], []))

    def test_only_ties(self):
        """Test Kendall's Tau Alpha where every value is a tie."""
        assert np.isnan(_kendall_tau_a([1, 1, 1], [2, 2, 2]))


class TestPartialCorrelation:
    """Test partial correlations."""

    def test_basic(self):
        """Test computing partial correlations."""
        data_rdm = [2, 4, 15, 20]
        model_rdms = [[1, 2, 3, 4], [0, 0, 1, 1]]
        assert_allclose(
            _partial_correlation(data_rdm, model_rdms, type="pearson"),
            [0.919145, 0.912871],
        )
        print(_partial_correlation(data_rdm, model_rdms, type="spearman"))
        assert_allclose(
            _partial_correlation(data_rdm, model_rdms, type="spearman"),
            [-1, 0.89442719],
        )

    def test_sizes(self):
        """Test feeding arrays of invalid size to _partial_correlation."""
        with pytest.raises(ValueError, match="Need more than one model RDM"):
            _partial_correlation([1, 2], [[1, 2]])
        with pytest.raises(ValueError, match="Correlation type must be"):
            _partial_correlation([1, 2], [[1, 2], [1, 2]], type="banana")
