# ========================== #
# LIBRARIES
# ========================== #
import apecosm


# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/Datarmor/FileZilla/data_test/'
config_path = data_path+'run-spinup-cycle-1/'
output_path = config_path+'output/'


# ========================== #
# LOAD DATA
# ========================== #
oope_conf_file = config_path+'oope.conf'
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


# ========================== #
# REPORT
# ========================== #
apecosm.report(output_path, mesh_file)


# ========================== #
# DISPLAY
# ========================== #