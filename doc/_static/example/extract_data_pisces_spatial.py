import os
import sys
sys.path.insert(0, os.path.abspath('../python/'))
import apecosm

dirin = '_static/example/data/pisces'
pattern = 'data_pisces.nc'
mesh_file = '_static/example/data/mesh_mask.nc'

data = apecosm.open_ltl_data(dirin)
mesh = apecosm.open_mesh_mask(mesh_file)

beng_phy2_dat = apecosm.extract_ltl_data(data, 'PHY2', mesh, None, depth_max=None)

print(beng_phy2_dat)
