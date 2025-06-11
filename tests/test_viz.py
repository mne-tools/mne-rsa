"""Test visualization functions."""

import numpy as np
import pytest
from mne_rsa import plot_rdms
from numpy.testing import assert_equal


class TestPlotRDMs:
    """Test plotting RDMs."""

    def test_plot_single_rdm(self):
        """Test plotting a single rdm."""
        rdm = np.array([0.5, 1, 0.5])
        fig = plot_rdms(rdm, names="My RDM", title="My Title")
        assert fig.get_suptitle() == "My Title"
        assert fig.axes[0].get_title() == "My RDM"
        assert len(fig.axes[0].images) == 1
        data = fig.axes[0].images[0].get_array().data
        assert_equal(data, [[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]])

        # Test giving wrong number of names.
        with pytest.raises(ValueError, match="Number of given names"):
            plot_rdms(rdm, names=["A", "B"])
