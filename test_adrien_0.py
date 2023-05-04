# documentation : file:///home/adrien/MEGA/Code/python-apecosm/doc/_build/html/index.html

# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import matplotlib.pyplot as plt


# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/python-apecosm/doc/_static/example/data/'
config_path = data_path+'config/'
output_path = data_path+'apecosm/'


# ========================== #
# 1.1 Reading config
# ========================== #
oope_conf_file = config_path+'oope.conf'
config = apecosm.read_config(oope_conf_file)

print('============ CONFIG ===================')
print(config.keys())
print(config['grid.mask.var.e2v'])
print(config['simulation.restart.file'])
print('===============================')


# ========================== #
# 1.2 Reading grid
# ========================== #
wstep, lstep = apecosm.extract_weight_grid(config)

print('============= WEIGHT/LENGTH ==================')
print(config['biology.length.min']) # 1.e-05
print(wstep)
print(lstep)
print('===============================')


# ========================== #
# 1.3 Data extraction
# ========================== #
mesh_file = data_path+'mesh_mask.nc'
mesh = apecosm.open_mesh_mask(mesh_file)
print('============= MESH ==================')
print(mesh.keys())
print('===============================')

const = apecosm.open_constants(output_path)
print('============= CONSTANTS ==================')
print(const.keys())
print(apecosm.extract_community_names(const))
print('===============================')

data = apecosm.open_apecosm_data(output_path)
print('============= FULL DATA ==================')
print(data.keys())
print('===============================')

oope_data = apecosm.extract_oope_data(data, mesh, const)
print('============= OOPE DATA ==================')
print(oope_data)
print('===============================')

oope_tmean = apecosm.extract_time_means(oope_data, None)
print('===============================')
print(oope_tmean)
print('===============================')


apecosm.plot_oope_spectra(oope_data,const)
plt.show()



data_2 = apecosm.open_ltl_data(dirin)
ltl_data = apecosm.extract_ltl_data(data_2, 'OOPE', mesh)
#  File "/home/adrien/MEGA/Code/python-apecosm/apecosm/extract.py", line 171, in extract_ltl_data
#    zdim, ydim, xdim = data.dims[1:]
#    ^^^^^^^^^^^^^^^^
#ValueError: too many values to unpack (expected 3)

apecosm.report(dirin, mesh_file)

apecosm.plot_oope_map(oope_data,mesh,const)
# The input data array must be 2D
# ERROR conda.cli.main_run:execute(33): Subprocess for 'conda run ['python', '/snap/pycharm-professional/327/plugins/python/helpers/pydev/pydevconsole.py', '--mode=client', '--host=127.0.0.1', '--port=40215']' command failed.  (See above for error)

plt.show()



#apecosm.extract_ltl_data(data,'OOPE', mesh)
#    zdim, ydim, xdim = data.dims[1:]
#    ^^^^^^^^^^^^^^^^
#ValueError: too many values to unpack (expected 3)

apecosm.plot_oope_map(data['OOPE'],mesh,const)
apecosm.plot_oope_map(data,mesh,const)
#/home/adrien/anaconda3/bin/conda run -n apecosm --no-capture-output python /snap/pycharm-professional/327/plugins/python/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=32971
#apecosm.plot_oope_spectra(data, const)