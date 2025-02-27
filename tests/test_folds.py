"""Unit tests for the creation of cross-validation folds."""

import numpy as np
import pytest
from mne_rsa.folds import _compute_item_means, _convert_to_one_hot, create_folds
from numpy.testing import assert_equal
from sklearn.model_selection import KFold


class TestCreateFolds:
    """Test the create_folds function."""

    def test_basic(self):
        """Test basic invocation of create_folds."""
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]

        folds = create_folds(data, y, n_folds=1)
        assert_equal(folds, [[2, 2, 2]])

        folds = create_folds(data, y, n_folds=2)
        assert_equal(folds, [[1.5, 1, 1.5], [3, 2.5, 3]])

        folds = create_folds(data, y, n_folds=3)
        assert_equal(folds, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        folds = create_folds(data[:, np.newaxis], y, n_folds=3)
        assert_equal(folds, [[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]])

        # Default value for n_folds is maximum number of folds
        assert_equal(create_folds(data, y), create_folds(data, y, n_folds=3))

        # No folding when y=None
        folds = create_folds(data, y=None)
        assert_equal(folds, [data])

    def test_sklearn_split(self):
        """Test passing a scikit-learn style Split object."""
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        folds = create_folds(data, y, n_folds=KFold(2))  # non-stratified k-fold
        assert_equal(folds, [[1.5, 1.5, 1.0], [3.0, 3.0, 2.5]])
        folds = create_folds(data[:, np.newaxis], y, n_folds=KFold(3))
        assert_equal(folds, [[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]])

    def test_invalid_input(self):
        """Test passing invalid input to create_folds."""
        # Invalid y
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = [1, 2, 3]
        with pytest.raises(ValueError, match="length of y"):
            create_folds(data, y)


class TestConvertToOneHot:
    """Test the _convert_to_one_hot function."""

    def test_basic(self):
        """Test basic invocation of _convert_to_one_hot."""
        y = np.array([1, 2, 3, 1, 2, 3])
        y_one_hot = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        assert_equal(_convert_to_one_hot(y), y_one_hot)
        assert_equal(_convert_to_one_hot(y.tolist()), y_one_hot)
        assert_equal(_convert_to_one_hot(y[:, np.newaxis]), y_one_hot)
        assert_equal(_convert_to_one_hot(y_one_hot), y_one_hot)

    def test_invalid_input(self):
        """Test passing invalid input to _convert_to_one_hot."""
        with pytest.raises(ValueError, match="Wrong number of dimensions"):
            _convert_to_one_hot([[[1], [2], [3]]])


class TestComputeItemMeans:
    """Test the _compute_item_means function."""

    def test_basic(self):
        """Test basic invocation of _compute_item_means."""
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y_one_hot = _convert_to_one_hot([1, 2, 3, 1, 2, 3, 1, 2, 3])
        assert_equal(_compute_item_means(data, y_one_hot), [2, 2, 2])
