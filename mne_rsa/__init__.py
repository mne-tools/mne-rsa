__version__ = "0.10"

from .source_level import rsa_stcs, rdm_stcs, rsa_stcs_rois, rsa_nifti, rdm_nifti
from .sensor_level import rsa_evokeds, rsa_epochs, rdm_evokeds, rdm_epochs
from .searchlight import searchlight
from .rsa import rsa, rsa_gen, rsa_array
from .rdm import compute_rdm, compute_rdm_cv, rdm_array, pick_rdm
from .viz import plot_rdms, plot_rdms_topo, plot_roi_map
from .folds import create_folds

# This function is useful to have nearby
from scipy.spatial.distance import squareform
