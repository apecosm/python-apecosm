import os
import sys
sys.path.insert(0, os.path.abspath('../python/'))
import apecosm
import xarray as xr
import pylab as plt

filename_pisces = 'example/data/data_pisces.nc'
filename_apecosm = 'example/data/data_apecosm.nc'

mesh_file = 'example/data/mesh_mask.nc'
domain = 'BENGUELA'

output_var = 'weight'

config = apecosm.read_config('example/data/config/oope.conf')

###################################################### processing of LTL files
for var in ['PHY2', 'ZOO', 'ZOO2', 'GOC']:

    temp = apecosm.extract.extract_ltl_data(filename_pisces, var, mesh_file, domain)
    temp = apecosm.extract.extract_time_means(temp)

    if var == 'PHY2':
        output = temp

    else:
        output[var] = temp[var]

L = [10e-6, 100e-6]
lVec2dPHY2_pisces, rhoLPHY2_pisces = apecosm.compute_spectra_ltl(output['PHY2'], L, output_var=output_var)

L = [20.e-6, 200.e-6]
lVec2dZOO_pisces, rhoLZOO_pisces = apecosm.compute_spectra_ltl(output['ZOO'], L, output_var=output_var)

L = [200.e-6, 2000.e-6]
lVec2dZOO2_pisces, rhoLZOO2_pisces = apecosm.compute_spectra_ltl(output['ZOO2'], L, output_var=output_var)

L = [100e-6, 5000.e-6]
lVec2dGOC_pisces, rhoLGOC_pisces = apecosm.compute_spectra_ltl(output['GOC'], L, N=5000, output_var=output_var)

###################################################### processing of Apecosm files

ape = apecosm.extract.extract_oope_data(filename_apecosm, 'OOPE', mesh_file, domain)
ape = apecosm.extract.extract_time_means(ape)

plt.figure()
ax = plt.gca()
apecosm.plot_oope_spectra(ape, output_var=output_var, config=config)
plt.scatter(lVec2dPHY2_pisces, rhoLPHY2_pisces, label='PHY2')
plt.scatter(lVec2dZOO_pisces, rhoLZOO_pisces, label='ZOO')
plt.scatter(lVec2dZOO2_pisces, rhoLZOO2_pisces, label='ZOO2')
plt.scatter(lVec2dGOC_pisces, rhoLGOC_pisces, label='GOC')
apecosm.set_plot_lim()
plt.legend()
ax.set_yscale("log")
ax.set_xscale("log")
plt.xlabel('Weight ($kg$)')
plt.ylabel('Energy density ($J.kg^{-1}$)')
plt.savefig('example/size_spectra.svg', bbox_inches='tight')
plt.savefig('example/size_spectra.pdf', bbox_inches='tight')
