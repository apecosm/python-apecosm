''' Module that contains some functions for data extractions '''

from __future__ import print_function
import sys
from glob import glob
import xarray as xr
import numpy as np
from .domains import DOMAINS, inpolygon


def extract_ltl_data(file_pattern, varname, meshfile,
                     domain_name, compute_mean=False, depth_max=None):

    '''
    Extraction of LTL values on a given domain.
    LTL is vertically integrated and spatially integrated over
    the domain.

    :param str file_pattern: LTL file pattern
    :param str varname: LTL variable name
    :param str meshfile: Name of the NetCDF meshfile
    :param str domain_name: Name of the domain to extract
    :param bool vvl: True if VVL is on, else False
    :param bool compute_mean: If True, mean is computed. If False, integral is provided.

    :return: A xarray dataset

    '''

    # open the dataset
    data = xr.open_mfdataset(file_pattern)

    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(meshfile)
    surf = mesh['e2t'] * mesh['e1t'] # 1, lat, lon
    surf = surf.values[np.newaxis, :, :, :] # 1, 1, lat, lon

    if 'e3t' in data.variables:
        # if VVL, e3t should be read from data
        e3t = data['e3t']  # time, z, lat, lon
    else:
        e3t = mesh['e3t_0'].values  # 1, z, lat, lon

    tmask = mesh['tmask'].values * mesh['tmaskutil'].values[np.newaxis, :, :, :]  # 1, z, lat, lon
    lon = np.squeeze(mesh['glamt'].values)
    lat = np.squeeze(mesh['gphit'].values)

    # extract the domain coordinates
    if domain_name == 'global':
        maskdom = 1
    else:

        if isinstance(domain_name, str):
            domain = DOMAINS[domain_name]
        else:
            domain = domain_name

        # generate the domain mask
        maskdom = inpolygon(lon, lat, domain['lon'], domain['lat'])

        # add virtual dimensions to domain mask and
        # correct landsea mask
        maskdom = maskdom[np.newaxis, np.newaxis, :, :]  # time, depth, lat, lon

    tmask = tmask * maskdom  #  0 if land or out of domain, else 1
    weight = surf * e3t * tmask  # (1, z, lat, lon) or (time, z, lat, lon)

    # clear unused variables
    del(surf, e3t, tmask, maskdom)

    if depth_max is not None:
        idepth = np.nonzero(np.squeeze(mesh['gdept_1d'].values) < depth_max)[0]
        weight = weight[:, idepth, :, :]
        data = data.isel(z=idepth)

    # integrate spatially and vertically the LTL concentrations
    data = (data[varname] * weight).sum(dim=('y', 'x', 'z'))  # time
    if compute_mean:
        data /= np.sum(weight, axis=(1, 2, 3))

    return data


def extract_time_means(data, time=None):

    r'''
    Extract data time mean.

    :param str file_pattern: File pattern (for instance, "data/\*nc")
    :param str time: Time mean. Can be time average ('tot'),
     yearly means ('year'), seasonal means ('season') or monthly means
     ('monthly')

    :return: A data array with the given time mean
    '''

    if (time is not None) & (time not in ('season', 'month', 'year')):
        message = "Time argument must be set to 'season', 'month', 'year' or None"
        print(message)
        sys.exit(0)

    if 'time_counter' in data.dims:
        dimname = 'time_counter'
    else:
        dimname = 'time'

    if time is None:
        climatology = data.mean(dimname)
    else:
        climatology = data.groupby('time.%s' % time).mean(dimname)

    return climatology

def extract_apecosm_constants(input_dir):

    ''' Extracts APECOSM constant fields

    :param str input_dir: Directory containing Apecosm outputs
    :return: A dataset containing the outputs

     '''

    fileconst = glob('%s/*ConstantFields.nc' %input_dir)

    if len(fileconst) == 0:
        message = 'No ConstantFields file found in directory.'
        print(message)
        sys.exit(1)

    if len(fileconst) > 1:
        message = 'More than  one ConstantFields file found in directory.'
        print(message)
        sys.exit(1)

    constants = xr.open_dataset(fileconst[0])
    return constants


def extract_oope_data(input_dir, meshfile, domain_name, use_wstep=True, compute_mean=True):

    '''
    Extraction of OOPE values on a given domain.
    OOPE is spatially integrated over
    the domain.

    :param str file_pattern: OOPE file pattern
    :param str varname: OOPE variable name
    :param str meshfile: Name of the NetCDF meshfile
    :param str domain_name: Name of the domain to extract
    :param str compute_mean: True if mean, else integral.

    :return: A tuple with the time-value and the LTL time series

    '''

    # Extract constant fields and extract weight_step
    if use_wstep:
        const = extract_apecosm_constants(input_dir)
        wstep = const['weight_step'].values   # weight or (comm, weight)

        # if wstep is 1d, we tile it along the community dimension
        if wstep.ndim == 1:
            wstep = wstep[np.newaxis, :]

        wstep = wstep[np.newaxis, np.newaxis, np.newaxis, :, :]  # time, lat, lon, comm, weight
    else:
        wstep = 1

    # extract the list of OOPE files
    filelist = np.sort(glob("%s/*OOPE*nc" %(input_dir)))

    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(meshfile)
    surf = mesh['e2t'].values * mesh['e1t'].values
    surf = surf[:, :, :, np.newaxis, np.newaxis]  # time, lat, lon, comm, weight

    tmask = mesh['tmaskutil'].values[:, :, :, np.newaxis, np.newaxis]  # time, lat, lon, comm, weight 
    lon = np.squeeze(mesh['glamt'].values)
    lat = np.squeeze(mesh['gphit'].values)

    # extract the domain coordinates
    if isinstance(domain_name, str):
        if domain_name != 'global':
            domain = DOMAINS[domain_name]
    else:
        domain = domain_name

    if domain_name != 'global':
        # generate the domain mask
        maskdom = inpolygon(lon, lat, domain['lon'], domain['lat'])
    else:
        maskdom = 1.

    # add virtual dimensions to domain mask and
    # correct landsea mask
    maskdom = maskdom[np.newaxis, :, :, np.newaxis, np.newaxis]  # time, lat, lon, comm, w
    tmask = tmask * maskdom
    weight = tmask * surf  # time, lat, lon, comm, w


    data = xr.open_mfdataset(fileelist)
    data = data['OOPE']

    if use_wstep:
        data = data * wstep

    data = data.sum(dim=('x', 'y'))  # time, com, w

    if compute_mean:
        data /= np.sum(weight, axis=(1, 2)) 
    
    return data

# if __name__ == '__main__':
#
#     meshfile = 'data/mesh_mask.nc'
#     domain = 'benguela'
#
#     #output = extract_oope_data('data/CMIP2_SPIN_OOPE_EMEAN.nc', 'OOPE', meshfile, domain)
#     #output.to_netcdf('test_ts.nc', format='NETCDF4_CLASSIC')
#
#     test = xr.open_dataset('test_ts.nc')
#     test = test.mean(dim='time')
#     oope = test['OOPE'].values
#     length = test['length'].values
#     comm = test['community'].values.astype(np.int)
#
#     commnames = misc.extract_community_names(test)
#
#     length2d, comm2d = np.meshgrid(length, comm)
#     iok = np.nonzero(oope > 0)
#     lmin, lmax = length2d[iok].min(), length2d[iok].max()
#     rmin, rmax = oope[iok].min(), oope[iok].max()
#
#     file_pattern = 'data/PISCES*nc'
#     time, dataPHY2 = extract_ltl_data(file_pattern, 'PHY2', meshfile, domain)
#     time, dataZOO = extract_ltl_data(file_pattern, 'ZOO', meshfile, domain)
#     time, dataZOO2 = extract_ltl_data(file_pattern, 'ZOO2', meshfile, domain)
#     time, dataGOC = extract_ltl_data(file_pattern, 'GOC', meshfile, domain)
#
#     [dataPHY2, dataZOO, dataZOO2, dataGOC] = map(np.mean, [dataPHY2, dataZOO, dataZOO2, dataGOC])
#
#     import constants
#     mul = 1e-3 * constants.C_E_convert
#
#     L = [10e-6, 100e-6]
#     lVec2dPHY2, rhoLPHY2 = compute_spectra_ltl(mul*dataPHY2, L, out='length')
#
#     L = [20.e-6, 200.e-6]
#     lVec2dZOO, rhoLZOO = compute_spectra_ltl(mul*dataZOO, L, out='length')
#
#     L = [200.e-6, 2000.e-6]
#     lVec2dZOO2, rhoLZOO2 = compute_spectra_ltl(mul*dataZOO2, L, out='length')
#
#     L = [100e-6, 50000.e-6]
#     lVec2dGOC, rhoLGOC = compute_spectra_ltl(mul*dataGOC, L, out='length')
#
#
#     import pylab as plt
#     cmap = plt.cm.Spectral
#     plt.figure()
#     ax = plt.gca()
#     for icom in comm:
#         color = cmap(float(icom) / len(comm))
#         plt.scatter(length2d[icom], oope[icom], c=color, label=commnames[icom])
#
#
#     plt.scatter(lVec2dPHY2, rhoLPHY2, label='PHY2')
#     plt.scatter(lVec2dZOO, rhoLZOO, label='ZOO')
#     plt.scatter(lVec2dZOO2, rhoLZOO2, label='ZOO2')
#     plt.scatter(lVec2dGOC, rhoLGOC, label='GOC')
#
#     xmin = np.min(map(np.min, [lVec2dPHY2, lVec2dZOO, lVec2dZOO2, lVec2dGOC, length2d]))
#     xmax = np.max(map(np.max, [lVec2dPHY2, lVec2dZOO, lVec2dZOO2, lVec2dGOC, length2d]))
#
#     ymin = np.min(map(np.min, [rhoLPHY2, rhoLZOO, rhoLZOO2, rhoLGOC, oope]))
#     ymax = np.max(map(np.max, [rhoLPHY2, rhoLZOO, rhoLZOO2, rhoLGOC, oope]))
#
#     ax.set_yscale("log")
#     ax.set_xscale("log")
#     plt.xlabel('Size (m)')
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#     plt.ylabel(r'$\displaystyle \rho$')
#     plt.legend()
#     plt.title(domain)
#
#     plt.savefig('size_spectra_%s.png' %domain, bbox_inches='tight')
