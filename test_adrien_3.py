# ========================== #
# LIBRARIES
# ========================== #
import matplotlib.pyplot as plt

import apecosm
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs


# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/Datarmor/FileZilla/data_test/'
config_path = data_path+'run-spinup-cycle-1/'
output_path = config_path+'output/'
fishing_path = config_path+'fishing_output/'

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

market, fleet_maps, fleet_summary, fleet_parameters = apecosm.open_fishing_data(fishing_path)
fleet_names = apecosm.extract_fleet_names(config_path)

print('========')
print(market.keys())
print('========')
print(fleet_maps[0].keys())
print('========')
print(fleet_summary[0].keys())
print('========')
print(fleet_parameters[0].keys())
print('========')

#plt.figure()
#plt.plot(market.time, market['average_price'].isel(fleet=5, community=1))
#plt.show()

#plt.figure()
#plt.plot(market.time, market['average_landing_rate'].isel(fleet=5, community=1))
#plt.show()

#plt.figure()
#plt.plot(market.time, fleet_summary[0]['average_fuel_use_intensity'])
#plt.show()

tmp = fleet_summary[0]['average_fuel_use_intensity']
T = 365
av, maxi, mini, time = apecosm.compute_mean_min_max_ts(tmp,T)

plt.figure()
plt.plot(market.time, fleet_summary[0]['average_fuel_use_intensity'])
plt.plot(np.arange(0,len(tmp.time),T), av)
plt.plot(np.arange(0,len(tmp.time),T), maxi)
plt.plot(np.arange(0,len(tmp.time),T), mini)
#plt.show()
plt.close()


fig, ax = plt.subplots(3, 3, figsize = (30, 5), dpi = 300)
for i in np.arange(6):
    plt.subplot(3, 2, i+1)
    plt.plot(fleet_summary[i]['time'], fleet_summary[i]['effective_effort'], color='red', linewidth=0.5, label='Fishing')
    plt.plot(fleet_summary[i]['time'], fleet_summary[i]['active_vessels'], color='blue', linewidth=0.5, label='Sailing')
    plt.plot(fleet_summary[i]['time'], fleet_summary[i]['total_vessels'], color='green', linewidth=0.5, label='At port')
    plt.fill_between(fleet_summary[i]['time'], fleet_summary[i]['active_vessels'], fleet_summary[i]['total_vessels'], color='green', alpha=0.5)
    plt.fill_between(fleet_summary[i]['time'], fleet_summary[i]['effective_effort'], fleet_summary[i]['active_vessels'], color='blue', alpha=0.5)
    plt.fill_between(fleet_summary[i]['time'], fleet_summary[i]['effective_effort'], color='red', alpha=0.5)
    plt.grid()
    plt.ylabel("Number of vessels")
fig.tight_layout()
plt.legend(loc="lower left",bbox_to_anchor=(1.0,0.5))
#plt.show()
plt.close()

#crs = ccrs.PlateCarree()
#fig = plt.figure()
#ax = plt.axes(projection=crs)
#plt.plot(fleet_maps[1]['effective_effort_density'].isel(time=0))
##ax.add_feature(cfeature.LAND)
##ax.add_feature(cfeature.COASTLINE)
#plt.show()
#plt.close()

#nb_fleet = len(fleet_maps)
#n_col = 3
#n_row = int(nb_fleet / n_col)

nb_fleet = len(fleet_maps)
n_col = 3
n_row = int(nb_fleet / n_col)

crs = ccrs.PlateCarree()
fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
#for i in np.arange(nb_fleet):
##    axes[int(i/n_col), i % n_col] = plt.axes(projection=crs)
##    axes[int(i/n_col), i % n_col].add_feature(cfeature.LAND)
##    axes[int(i/n_col), i % n_col].add_feature(cfeature.COASTLINE)
##print(axes)
##ax = plt.axes(projection=crs)
##ax.add_feature(cfeature.LAND)
##ax.add_feature(cfeature.COASTLINE)
for i in np.arange(nb_fleet):
    plt.subplot(n_row, n_col, i + 1, projection=crs)
    #print('('+str(int(i/n_col))+') , ('+ str(i % n_col)+')')
    #axes[int(i/n_col), i % n_col] = plt.axes(projection=crs)
    cs = plt.pcolormesh(fleet_maps[i]['effective_effort_density'].sum(dim='time', skipna=True))
    cb = plt.colorbar(cs)
    cb.set_label('T/day-1')
    plt.grid(False)
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.COASTLINE)
    ##axes[int(i/n_col), i % n_col].add_feature(cfeature.LAND)
    ##axes[int(i/n_col), i % n_col].add_feature(cfeature.COASTLINE)

fig.tight_layout()
#plt.show()
plt.close()


#fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
#ax = plt.axes(projection=crs)
#for i in np.arange(nb_fleet):
#    plt.subplot(n_row, n_col, i + 1)
#    plt.plot(fleet_maps[i]['effective_effort_density'].isel(time=10))
#fig.tight_layout()
#plt.show()
#plt.close()

# ========================== #
# REPORT
# ========================== #
apecosm.report(output_path, mesh_file, fishing_path, config_path)


# ========================== #
# DISPLAY
# ========================== #