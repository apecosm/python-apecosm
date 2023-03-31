# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import os

# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/Datarmor/FileZilla/run-spinup-fisheries/'
apecosm_config_path = data_path+'apecosm-config/'
fishing_config_path = data_path+'cnrm-fishing-config/'
output_path = data_path+'output/'
fishing_path = data_path+'fishing-outputs/'

# ========================== #
# LOAD DATA
# ========================== #
oope_conf_file = apecosm_config_path+'oope.conf'
config = apecosm.read_config(oope_conf_file)

# open mesh file
mesh_file = data_path+'corrected-eORCA1L75_grid.nc'
mesh = apecosm.open_mesh_mask(mesh_file)

# open constant file
const = apecosm.open_constants(output_path)
community_names = apecosm.extract_community_names(const)

# open oope data
data = apecosm.open_apecosm_data(output_path)
oope_data = apecosm.extract_oope_data(data, mesh, const)

market, fleet_maps, fleet_summary, fleet_parameters = apecosm.open_fishing_data(fishing_path)
fleet_names = apecosm.extract_fleet_names(fishing_config_path)

# ========================== #
# REPORT
# ========================== #
apecosm.report(output_path, mesh_file, fishing_path, fishing_config_path)


# ========================== #
# DISPLAY
# ========================== #