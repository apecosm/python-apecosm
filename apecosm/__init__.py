# -*- coding: utf-8 -*-
"""

Apecosm Python License CeCill

"""

from __future__ import print_function

import os

import pkg_resources  # part of setuptools
try:
    VERSION_FILE = os.path.join(f'{os.path.dirname(__file__)}', '..', 'VERSION')
    with open(VERSION_FILE, 'r') as infile:
        __version__ = infile.read().strip()
except:
    __version__ = pkg_resources.require("apecosm")[0].version

__description__ = "Python package for the manipulation of the Apecosm model"
__author_email__ = "nicolas.barrier@ird.fr"

from .conf import read_config
from .domains import inpolygon, plot_domains, generate_mask
from .extract import extract_ltl_data, extract_time_means, extract_oope_data, extract_weighted_data, open_apecosm_data, open_constants, spatial_mean_to_integral, open_mesh_mask, open_ltl_data, open_fishing_data, read_report_params, extract_oope_data, extract_oope_size_integration
from .extract import extract_mean_size
from .diags import compute_size_cumprop
from .misc import find_percentile, compute_daylength, extract_community_names, size_to_weight, weight_to_size, compute_mean_min_max_ts, extract_fleet_names
from .netcdf import rebuild_restart
from .size_spectra import compute_spectra_ltl, plot_oope_spectra, set_plot_lim
from .mplot import plot_pcolor_map, plot_contour_map, plot_diet_values
from .report import report
from .grid import extract_weight_grid, read_ape_grid, plot_grid_nemo_ape, partial_step_ape
from .habitat import get_tcor, compute_o2, compute_tpref, compute_tlim, compute_lightpref
