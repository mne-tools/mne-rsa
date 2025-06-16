"""Unit tests for the creation of cross-validation folds."""

from functools import partial

import numpy as np
import pytest
from mne_rsa.folds import (
    _compute_item_means,
    _convert_to_one_hot,
    _match_order,
    create_folds,
)
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

        # Scikit-Learn objects should also work.
        folds = create_folds(data, y, n_folds=KFold(3))
        assert_equal(folds, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        # Default value for n_folds is maximum number of folds.
        assert_equal(create_folds(data, y), create_folds(data, y, n_folds=3))

        # No folding when y=None.
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

        assert_equal(_convert_to_one_hot(y)[0], y_one_hot)
        assert_equal(_convert_to_one_hot(y.tolist())[0], y_one_hot)
        assert_equal(_convert_to_one_hot(y[:, np.newaxis])[0], y_one_hot)
        assert_equal(_convert_to_one_hot(y_one_hot)[0], y_one_hot)

    def test_invalid_input(self):
        """Test passing invalid input to _convert_to_one_hot."""
        with pytest.raises(ValueError, match="Wrong number of dimensions"):
            _convert_to_one_hot([[[1], [2], [3]]])


class TestComputeItemMeans:
    """Test the _compute_item_means function."""

    def test_basic(self):
        """Test basic invocation of _compute_item_means."""
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y_one_hot, y = _convert_to_one_hot([1, 2, 3, 1, 2, 3, 1, 2, 3])
        assert_equal(_compute_item_means(data, y_one_hot), [2, 2, 2])
        assert_equal(y, np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))


class TestMatchOrder:
    """Test the _match_order function."""

    def test_basic(self):
        """Test basic invocation of _match_order."""
        invoke = partial(_match_order, len_X=3, len_rdm_model=3)
        assert invoke() is None
        assert_equal(
            invoke(labels_X=[1, 0, 2], labels_rdm_model=None),
            _convert_to_one_hot([1, 0, 2])[0],
        )
        assert_equal(invoke(labels_X=[0, 1, 2], labels_rdm_model=[0, 1, 2]), np.eye(3))
        assert_equal(
            invoke(labels_X=["a", "b", "c"], labels_rdm_model=["a", "b", "c"]),
            np.eye(3),
        )
        assert_equal(
            invoke(labels_X=["a", "c", "b"], labels_rdm_model=["a", "b", "c"]),
            _convert_to_one_hot([0, 2, 1])[0],
        )

        # Test not having to align with labels_rdm_model
        assert _match_order(len_X=3, labels_X=None) is None
        assert_equal(_match_order(len_X=3, labels_X=["a", "b", "c"]), np.eye(3))
        assert_equal(
            _match_order(len_X=4, labels_X=["a", "b", "b", "c"]),
            _convert_to_one_hot([0, 1, 1, 2])[0],
        )

    def test_invalid_input(self):
        """Test passing invalid input to _match_order."""
        with pytest.raises(ValueError, match="you must also specify `labels_X`"):
            _match_order(len_X=3, labels_X=None, labels_rdm_model=[0, 1, 2])
        with pytest.raises(ValueError, match="The number of labels in `labels_X`"):
            _match_order(len_X=3, labels_X=[0, 1])

        invoke = partial(_match_order, len_X=3, len_rdm_model=3)
        with pytest.raises(ValueError, match="The number of labels in `labels_X`"):
            invoke(labels_X=[0, 1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="The number of unique labels in"):
            invoke(labels_X=[0, 1, 1])
        with pytest.raises(ValueError, match="The number of labels in `labels_epochs`"):
            invoke(labels_X=[0, 1], var="labels_epochs")
        with pytest.raises(
            ValueError, match="The number of labels in `labels_rdm_model`"
        ):
            invoke(labels_X=[0, 1, 2], labels_rdm_model=[0, 1, 2, 4, 5])
        with pytest.raises(
            ValueError, match="The number of labels in `labels_rdm_model`"
        ):
            invoke(labels_X=[0, 1, 2], labels_rdm_model=[0, 1, 2, 4, 5])
        with pytest.raises(ValueError, match="Not all labels in `labels_rdm_model`"):
            invoke(labels_X=["a", "b", "c"], labels_rdm_model=["a", "a", "c"])
        with pytest.raises(ValueError, match="Some labels in `labels_X`"):
            invoke(labels_X=["a", "a", "z"], labels_rdm_model=["a", "b", "c"])
        with pytest.raises(ValueError, match="Some labels in `labels_rdm_model`"):
            invoke(labels_X=["a", "a", "c"], labels_rdm_model=["a", "b", "c"])

    def test_many_to_one(self):
        """Test many-to-one mapping of data to rdm_model."""
        invoke = partial(_match_order, len_X=6, len_rdm_model=3)
        assert_equal(
            invoke(labels_X=[0, 0, 1, 1, 2, 2]),
            _convert_to_one_hot([0, 0, 1, 1, 2, 2])[0],
        )
        assert_equal(
            invoke(
                labels_X=["a", "b", "b", "c", "a", "c"],
                labels_rdm_model=["a", "b", "c"],
            ),
            _convert_to_one_hot([0, 1, 1, 2, 0, 2])[0],
        )

    def test_no_model_rdm(self):
        """Test passing labels_X without labels_rdm_model."""
        assert_equal(_match_order(3, labels_X=[1, 2, 3]), np.eye(3))
        assert_equal(
            _match_order(3, labels_X=[2, 1, 3]), _convert_to_one_hot([1, 0, 2])[0]
        )
        assert_equal(_match_order(3, labels_X=["a", "b", "c"]), np.eye(3))
        assert_equal(
            _match_order(3, labels_X=["b", "a", "c"]), _convert_to_one_hot([1, 0, 2])[0]
        )
        with pytest.raises(ValueError, match="The number of labels in `labels_X`"):
            assert_equal(_match_order(5, labels_X=["a", "b", "c"]), np.eye(3))
