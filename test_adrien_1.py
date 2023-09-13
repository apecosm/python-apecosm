# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/Datarmor/FileZilla/data_test/'
config_path = data_path+'run-spinup-cycle-1/'
output_path = config_path+'output/'


# ========================== #
# 1.1 Reading config
# ========================== #
oope_conf_file = config_path+'oope.conf'
config = apecosm.read_config(oope_conf_file)


# ========================== #
# 1.2 Reading grid
# ========================== #
try:
    wstep, lstep = apecosm.extract_weight_grid(config)
except:
    print("function apecosm.extract_weight_grid return an error --> deprecated function")
    wstep, lstep = (-1,-1)

vert_grid_fname = data_path+"APECOSM_depth_grid_37.txt"
vert_grid = apecosm.read_ape_grid(vert_grid_fname)
#mesh_nemo
#apecosm.partial_step_ape(vert_grid,mesh_nemo)
#apecosm.plot_grid_nemo_ape()


# ========================== #
# 1.3 Data extraction
# ========================== #

# open mesh file
mesh_file = data_path+'corrected-eORCA1L75_grid.nc'
mesh = apecosm.open_mesh_mask(mesh_file)

# open constant file
const = apecosm.open_constants(output_path)
community_names = apecosm.extract_community_names(const)

# open oope data
data = apecosm.open_apecosm_data(output_path)
oope_data = apecosm.extract_oope_data(data, mesh, const)

# time means
oope_tmean = apecosm.extract_time_means(oope_data, None)

# open lower trophic level data
data = apecosm.open_ltl_data(output_path)
try:
    ltl_data = apecosm.extract_ltl_data(data, 'OOPE', mesh)
except:
    print("apecosm.extract_ltl_data --> ERROR")

BENGUELA = {'lon': [10, 20, 20, 10, 10],
            'lat': [-36, -36, -15, -15, -36]}
benguela_mask = apecosm.generate_mask(mesh, domain=BENGUELA)

# ========================== #
# 1.4 Plots
# ========================== #
apecosm.plot_oope_spectra(oope_data,const)
plt.show()
apecosm.plot_oope_spectra(np.log10(oope_data),const)
plt.show()

ax = plt.axes(projection=ccrs.PlateCarree())
apecosm.plot_pcolor_map(data['OOPE'].isel(time=0, c=0, w=0),mesh,axis=ax)
plt.show()


# ========================== #
# DISPLAY
# ========================== #
print('============ CONFIG ===================')
print(config.keys())
print(config['grid.mask.var.e2v'])
print(config['simulation.restart.file'])
print('===============================')

print('============= WEIGHT/LENGTH ==================')
print(wstep)
print(lstep)
# ERROR
# File "/home/adrien/MEGA/Code/python-apecosm/apecosm/grid.py", line 27, in extract_weight_grid
# lmin = float(config['biology.length.min'])
# TypeError: only size-1 arrays can be converted to Python scalars
print(config['biology.length.min'])
# array([1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05])
print('===============================')

print('============= MESH ==================')
print(mesh.keys())
print('===============================')

print('============= CONSTANT ==================')
print(const.keys())
print(community_names)
print('===============================')

print('============= DATA ==================')
print(data.keys())
print('===============================')

print('============= OOPE DATA==================')
print(oope_data)
print(oope_tmean)
print('===============================')