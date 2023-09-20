import os
import sys
sys.path.insert(0, os.path.abspath('../python/'))
import apecosm
import pylab as plt

ape_grid_file = "_static/example/data/config/APECOSM_depth_grid_37.txt"
nemo_mesh_file = "_static/example/data/mesh_mask.nc"

ilat, ilon = 65, 21

# Reads the full step Apecosm grid
depth, deltaz = apecosm.read_ape_grid(ape_grid_file)

# Corrects the Apecosm vertical grid to include NEMO partial step
output = apecosm.partial_step_ape(ape_grid_file, nemo_mesh_file)
print(output.keys())
print(output['bottom'].shape)
print(output['depth'].shape)
print(output['deltaz'].shape)
print(output['mask'].shape)

# Plot the Apecosm and NEMO vertical grids for a given point
plt.figure()
apecosm.plot_grid_nemo_ape(ape_grid_file, nemo_mesh_file, ilat, ilon)
plt.ylim(1200, 0)
plt.savefig('_static/example/ape_vertical_grid.svg', bbox_inches='tight')
plt.savefig('_static/example/ape_vertical_grid.pdf', bbox_inches='tight')
