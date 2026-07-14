"""Test visualization functions."""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from mne.viz.utils import _fake_click
from mne_rsa import plot_rdms, plot_rdms_topo, plot_roi_map, rdm_evokeds
from numpy.testing import assert_equal


def load_evokeds():
    """Load the MNE-Sample data evokeds."""
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    evokeds = mne.read_evokeds(path / "sample_audvis-ave.fif")
    evokeds = [evoked.pick("eeg").resample(100).crop(0.1, 0.2) for evoked in evokeds]
    return evokeds


class TestPlotRDMs:
    """Test plotting RDMs."""

    def test_plot_single_rdm(self):
        """Test plotting a single RDM."""
        rdm = np.array([0.5, 1, 0.5])
        fig = plot_rdms(rdm, items=["A", "B", "C"], names="My RDM", title="My Title")
        assert fig.get_suptitle() == "My Title"
        assert fig.axes[0].get_title() == "My RDM"
        assert len(fig.axes[0].images) == 1
        data = fig.axes[0].images[0].get_array().data
        assert_equal(data, [[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]])
        for text, item in zip(fig.axes[0].get_xticklabels(), ["A", "B", "C"]):
            assert text.get_text() == item
        for text, item in zip(fig.axes[0].get_yticklabels(), ["A", "B", "C"]):
            assert text.get_text() == item

        # Test giving wrong number of names.
        with pytest.raises(ValueError, match="Number of given names"):
            plot_rdms(rdm, names=["A", "B"])

        # Invalid RDM.
        with pytest.raises(ValueError, match="Invalid shape"):
            plot_rdms(np.array([[[1]]]))

    def test_plot_multiple_rdm(self):
        """Test plotting multiple RDMs."""
        rdm1 = np.array([0.5, 1, 0.5])
        rdm2 = np.array([0.1, 0.5, 0.3])

        fig = plot_rdms(
            [rdm1, rdm2],
            items=["A", "B", "C"],
            names=["RDM1", "RDM2"],
            title="My Title",
        )
        assert fig.get_suptitle() == "My Title"
        assert fig.axes[0].get_title() == "RDM1"
        assert fig.axes[1].get_title() == "RDM2"
        assert len(fig.axes[0].images) == 1
        data = fig.axes[0].images[0].get_array().data
        assert_equal(data, [[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]])
        assert len(fig.axes[1].images) == 1
        data = fig.axes[1].images[0].get_array().data
        assert_equal(data, [[0, 0.1, 0.5], [0.1, 0, 0.3], [0.5, 0.3, 0]])
        for text, item in zip(fig.axes[0].get_xticklabels(), ["A", "B", "C"]):
            assert text.get_text() == item
        for text, item in zip(fig.axes[0].get_yticklabels(), ["A", "B", "C"]):
            assert text.get_text() == item
        for text, item in zip(fig.axes[1].get_xticklabels(), ["A", "B", "C"]):
            assert text.get_text() == item
        assert fig.axes[1].get_yticklabels() == []  # shared axes, no labels here

        # Rows instead of columns
        fig = plot_rdms([rdm1, rdm2], n_rows=2)
        assert len(fig.axes) == 3  # including colorbar

        # When RDMs cannot be evenly divided by the number of rows, some axes will be
        # hidden.
        fig = plot_rdms([rdm1, rdm2, rdm1], n_rows=2)
        assert len(fig.axes) == 5  # including colorbar
        assert not fig.axes[3].get_visible()

        # Test giving wrong number of names.
        with pytest.raises(ValueError, match="Number of given names"):
            plot_rdms([rdm1, rdm2], names="A")
        with pytest.raises(ValueError, match="Number of given names"):
            plot_rdms([rdm1, rdm2], names=["A", "B", "C"])


class TestPlotRDMsTopo:
    """Test plotting RDMs in a topographic map."""

    def test_topo(self):
        """Test basic plotting of RDMs in a topographic map."""
        evokeds = load_evokeds()

        # RDMs as generator
        rdms = rdm_evokeds(evokeds, spatial_radius=0.05)
        fig = plot_rdms_topo(rdms, evokeds[0].info)
        assert len(fig.axes) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        assert fig.axes[0].images[0].get_array().data.shape == (4, 4)
        assert fig.get_suptitle() == "Time point: 0"

        # RDMs as array
        rdms = np.array(list(rdm_evokeds(evokeds, spatial_radius=0.05))).reshape(
            evokeds[0].info["nchan"] - len(evokeds[0].info["bads"]), -1
        )
        fig = plot_rdms_topo(rdms, evokeds[0].info)
        assert len(fig.axes) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])

        # Inside an existing figure
        fig1 = plt.figure()
        fig2 = plot_rdms_topo(rdms, evokeds[0].info, fig=fig1)
        assert fig1 is fig2
        with pytest.raises(TypeError, match="fig has to be matplotlib.pyplot.Figure"):
            plot_rdms_topo(rdms, evokeds[0].info, fig="bla")

        # Invalid shape for RDMs
        with pytest.raises(ValueError, match="RDMs have to be a 2D or 3D ndarray"):
            fig = plot_rdms_topo(np.array([1, 2, 3]), evokeds[0].info)
        with pytest.raises(ValueError, match="RDMs have to be a 2D or 3D ndarray"):
            fig = plot_rdms_topo(
                np.array([1, 2, 3])[:, None, None, None], evokeds[0].info
            )

    def test_topo_timecourse(self):
        """Test basic plotting of RDMs in a topographic map with multiple timepoints."""
        evokeds = load_evokeds()
        rdms = np.array(
            list(rdm_evokeds(evokeds, temporal_radius=0.02, spatial_radius=0.05))
        ).reshape(evokeds[0].info["nchan"] - len(evokeds[0].info["bads"]), -1, 6)
        fig = plot_rdms_topo(rdms, evokeds[0].info)
        assert len(fig.axes) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        assert fig.axes[0].images[0].get_array().data.shape == (4, 4)
        assert fig.get_suptitle() == "From 0 (inclusive) to 7 (exclusive)"

        fig = plot_rdms_topo(rdms, evokeds[0].info, time=3)
        assert len(fig.axes) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        assert fig.get_suptitle() == "Time point: 3"

        fig = plot_rdms_topo(rdms, evokeds[0].info, time=[3, 5])
        assert len(fig.axes) == evokeds[0].info["nchan"] - len(evokeds[0].info["bads"])
        assert fig.get_suptitle() == "From 3 (inclusive) to 5 (exclusive)"

        # Invalid valus for time
        with pytest.raises(TypeError, match="`time` has to be int"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time=[1, 2, 3, 4])
        with pytest.raises(TypeError, match="`time` has to be int"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time=[1, "now", 3.5])
        with pytest.raises(ValueError, match="time window is out of range"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time=30)
        with pytest.raises(ValueError, match="time window is out of range"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time=[0, 30])
        with pytest.raises(ValueError, match="The start of the time window"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time=[3, 2])
        with pytest.raises(TypeError, match="`time` has to be int"):
            fig = plot_rdms_topo(rdms, evokeds[0].info, time="now")

    def test_onclick(self):
        """Test clicking on a topographic map."""
        evokeds = load_evokeds()
        rdms = rdm_evokeds(evokeds, spatial_radius=0.05)
        fig = plot_rdms_topo(rdms, evokeds[0].info)
        _fake_click(fig, fig.axes[0], (0.5, 0.5))
        assert plt.gcf().axes[0].get_title() == "EEG 001"


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skip GL plotting on GitHub Actions"
)
def test_plot_roi_map():
    """Test plotting ROI values on a brain."""
    rois = mne.read_labels_from_annot(
        "sample", subjects_dir=mne.datasets.sample.data_path() / "subjects"
    )
    values = np.arange(len(rois))
    brain = plot_roi_map(
        values,
        rois,
        subject="sample",
        subjects_dir=mne.datasets.sample.data_path() / "subjects",
    )
    assert brain._annots == {"lh": ["annotation"], "rh": ["annotation"]}
