''' Module that contains some functions for data extractions '''

from __future__ import print_function
import sys
import xarray as xr
import numpy as np
from glob import glob
from .domains import DOMAINS, inpolygon


def extract_ltl_data(file_pattern, varname, meshfile, domain_name, depth_max=None):

    '''
    Extraction of LTL values on a given domain.
    LTL is vertically integrated and spatially integrated over
    the domain.

    :param str file_pattern: LTL file pattern
    :param str varname: LTL variable name
    :param str meshfile: Name of the NetCDF meshfile
    :param str domain_name: Name of the domain to extract

    :return: A xarray dataset

    '''

    # open the dataset
    dataset = xr.open_mfdataset(file_pattern)
    data = dataset[varname].values
    data = np.ma.masked_where(np.isnan(data), data)

    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(meshfile)
    e3t = mesh['e3t'].values
    e2t = mesh['e2t'].values
    e1t = mesh['e1t'].values

    surf = e1t * e2t
    surf = surf[np.newaxis, :, :, :]

    tmask = mesh['tmask'].values
    lon = np.squeeze(mesh['glamt'].values)
    lat = np.squeeze(mesh['gphit'].values)

    # extract the domain coordinates
    if isinstance(domain_name, str):
        domain = DOMAINS[domain_name]
    else:
        domain = domain_name

    # generate the domain mask
    maskdom = inpolygon(lon, lat, domain['lon'], domain['lat'])

    # add virtual dimensions to domain mask and
    # correct landsea mask
    maskdom = maskdom[np.newaxis, np.newaxis, :, :]
    tmask = tmask * maskdom

    temp = surf * e3t * tmask * data

    if depth_max is not None:
        iok = np.nonzero(np.squeeze(mesh['gdept_0'].values) < depth_max)[0]
        temp = temp[:, iok, :, :]

    # integrate spatially and vertically the LTL concentrations
    data = np.sum(temp, axis=(1, 2, 3))
    # output the time series as a Dataset in order to keep track of
    # the coordinates (community and weight)
    output = xr.Dataset({varname: (['time_counter'], data)})
    output['time_counter'] = dataset['time_counter']

    return output


def extract_time_means(data, time=None):

    ''' 
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

    if 'time_counter' in data.dims.keys():
        dimname = 'time_counter'
    else:
        dimname = 'time'

    if time is None:
        climatology = data.mean(dimname)
    else:
        climatology = data.groupby('time.%s' % time).mean(dimname)

    return climatology

def extract_apecosm_constants(input_dir): 
    
    fileconst = glob('%s/*GYRE_ConstantFields.nc' %input_dir)
    
    if(len(fileconst) == 0): 
        message = 'No ConstantFields file found in directory.'
        sys.exit(1)
       
    if(len(fileconst) > 1): 
        message = 'More than  one ConstantFields file found in directory.'
        sys.exit(1)    

    constants = xr.open_dataset(fileconst[0])
    return constants

def extract_oope_data(input_dir, meshfile, domain_name, use_wstep=True):

    '''
    Extraction of OOPE values on a given domain.
    OOPE is spatially integrated over
    the domain.

    :param str file_pattern: OOPE file pattern
    :param str varname: OOPE variable name
    :param str meshfile: Name of the NetCDF meshfile
    :param str domain_name: Name of the domain to extract

    :return: A tuple with the time-value and the LTL time series

    '''
    
    # Extract constant fields and extract weight_step
    if(use_wstep):
        const = extract_apecosm_constants(input_dir)
        wstep = const['weight_step'].values   # weight
        wstep = wstep[np.newaxis, np.newaxis, np.newaxis, :]  #time, lat, lon, comm, weight
    else:
        wstep = 1
        
    # extract the list of OOPE files
    filelist = np.sort(glob("%s/*OOPE*nc" %(input_dir)))
    
    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(meshfile)
    e2t = mesh['e2t'].values
    e1t = mesh['e1t'].values

    surf = e1t * e2t
    surf = surf[:, :, :, np.newaxis, np.newaxis]  # time, lat, lon, comm, weight

    tmask = mesh['tmask'].values  # time, depth, lat, lon
    tmaskutil = mesh['tmaskutil'].values    # time, lat, lon
    tmask = tmask * tmaskutil[:, np.newaxis, :, :]
    lon = np.squeeze(mesh['glamt'].values)
    lat = np.squeeze(mesh['gphit'].values)
   
    # extract the domain coordinates
    if(isinstance(domain_name, str)):
        if(domain_name != 'global'):
            domain = DOMAINS[domain_name]
    else:
        domain = domain_name

    if(domain_name != 'global'): 
        # generate the domain mask
        maskdom = inpolygon(lon, lat, domain['lon'], domain['lat'])
    else:
        maskdom = np.ones(lon.shape)

    # add virtual dimensions to domain mask and
    # correct landsea mask
    maskdom = maskdom[np.newaxis, np.newaxis, :, :]  # time, depth, lat, lon
    tmask = tmask * maskdom
    tmask = tmask[:, 0, :, :, np.newaxis, np.newaxis]  # time, lat, lon, com, length

    output = []
    timeout = []
    for f in filelist:
        dataset = xr.open_dataset(f)
        data = dataset['OOPE'].to_masked_array()
        timeout.extend(dataset['time'].values)
        output.extend(np.sum(surf * tmask * data * wstep, axis=(1, 2)))

    # integrate spatially the OOPE concentrations
    output = np.array(output)
    timeout = np.array(timeout)
 
    # output the time series as a Dataset in order to keep track of
    # the coordinates (community and weight)
    output = xr.Dataset({'OOPE': (['time', 'community', 'weight'], output)})
    output['time'] = timeout

    return output
    

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
