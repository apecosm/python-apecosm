import os
import sys
sys.path.insert(0, os.path.abspath('../python/'))
import apecosm

mesh_file = '_static/example/data/mesh_mask.nc'

const = apecosm.open_constants('_static/example/data/apecosm/')
data = apecosm.open_apecosm_data('_static/example/data/apecosm')
mesh = apecosm.open_mesh_mask(mesh_file)

ape_dat = apecosm.extract_oope_data(data, mesh, const)

print(ape_dat)
