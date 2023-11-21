''' Module that contains some functions for data extractions '''

import os
import sys
from glob import glob
import xarray as xr
import numpy as np


def open_mesh_mask(meshfile, replace_dims=None):

    '''

    Opens a NEMO mesh mask. It removes the `t` dimension from
    the dataset.

    :param mesh_file: Full path of the mesh file.
    :type mesh_file: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    :return: A dataset containing the variables of the
        mesh mask.
    :rtype: :class:`xarray.Dataset`
    '''

    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(meshfile)
    if replace_dims is not None:
        mesh = mesh.rename(replace_dims)
    if 't' in mesh.dims:
        mesh = mesh.isel(t=0)

    return mesh


def open_constants(dirin, replace_dims=None):

    '''
    Opens an Apecosm constant file. It will look for
    a file that contains `Constant`.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    '''

    path = os.path.join(dirin, '*Constant*.nc')
    constant = xr.open_mfdataset(path)
    if replace_dims is not None:
        constant = constant.rename(replace_dims)
    return constant

def _check_file(f, varlist):
    
    for v in varlist:
        if v in f:
            return True

    return False


def open_apecosm_data(dirin, replace_dims=None, varlist=None, **kwargs):

    '''
    Opens Apecosm outputs.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    '''

    pattern = os.path.join(dirin, '*.nc.*')
    filelist = glob(pattern)
    if len(filelist) == 0:
        pattern = os.path.join(dirin, '*.nc')
        filelist = glob(pattern)
        filelist = [f for f in filelist if 'Constant' not in f]

    # open the dataset
    filelist.sort()

    if varlist is not None:
        if isinstance(varlist, str):
            varlist = [varlist]
        filelist = [f for f in filelist if _check_file(f, varlist)] 
    
    data = xr.open_mfdataset(filelist, **kwargs)
    if replace_dims is not None:
        data = data.rename(replace_dims)
    return data


def open_ltl_data(dirin, replace_dims=None, **kwargs):

    '''
    Opens NEMO/PISCES outputs.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    '''

    pattern = os.path.join(dirin, '*.nc')
    filelist = glob(pattern)
    filelist.sort()

    data = xr.open_mfdataset(filelist, **kwargs)
    if replace_dims is not None:
        data = data.rename(replace_dims)
    return data


def extract_ltl_data(data, varname, mesh,
                     maskdom=None, compute_mean=False,
                     depth_max=None):

    '''
    Extraction of LTL values on a given domain.
    LTL is vertically integrated and spatially integrated over
    the domain.

    :param str file_pattern: LTL file pattern
    :param str varname: LTL variable name
    :param str meshfile: Name of the NetCDF meshfile
    :param str domain_name: Name of the domain to extract
    :param bool vvl: True if VVL is on, else False
    :param bool compute_mean: If True, mean is computed.
        If False, integral is provided.

    :return: A xarray dataset

    '''

    surf = _squeeze_variable(mesh['e2t']) * _squeeze_variable(mesh['e1t'])

    if 'e3t' in data.variables:
        # if VVL, e3t should be read from data
        e3t = data['e3t']  # time, z, lat, lon
    elif 'e3t_0' in mesh.variables:
        e3t = mesh['e3t_0']  # 1, z, lat, lon
    else:
        e3t = mesh['e3t_1d']

    data = data[varname]
    data = _rename_z_dim(data)

    if 'gdept_0' in mesh.variables:
        depth = mesh['gdept_0']  # 1, z, lat, lon
    else:
        depth = mesh['gdept_1d']

    tmask = mesh['tmask']

    if 'tmaskutil' in mesh.variables:
        tmask *= mesh['tmaskutil']

    lat = _squeeze_variable(mesh['gphit'])

    # extract the domain coordinates
    if maskdom is None:
        maskdom = np.ones(lat.shape)

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    tmask = tmask * maskdom  # 0 if land or out of domain, else 1
    weight = surf * e3t * tmask  # (1, z, lat, lon) or (time, z, lat, lon)

    # clear unused variables
    del(surf, e3t, tmask, maskdom)

    if depth_max is not None:
        weight = weight.where(depth <= depth_max)

    zdim, ydim, xdim = data.dims[1:]

    # integrate spatially and vertically the LTL concentrations
    data = (data * weight).sum(dim=(zdim, ydim, xdim))  # time
    if compute_mean:
        data /= weight.sum(dim=(zdim, ydim, zdim))

    return data


def _rename_z_dim(var):

    for dims in ['olevel', 'depth', 'deptht', 'depthu', 'depthv']:
        if dims in var.dims:
            var = var.rename({dims: 'z'})
    return var


def extract_oope_size_integration(data, const, lmin=None, lmax=None):

    oope = data['OOPE']
    weight_step = const['weight_step']
    length = const['length'] * 100
    if lmin is not None:
        # if lmin is not None, check_lmin is True if length greater than Lmin
        check_lmin = (length >= lmin)
    else:
        # if lmin is None, check_lmin is true everywhere
        check_lmin = (length >= 0)

    if lmax is not None:
        # if lmax is not None, check_lmax is True if length greater than lmax
        check_lmax = (length <= lmax)
    else:
        # if lmax is None, check_lmax is true everywhere
        check_lmax = (length >= 0)

    check_size = (check_lmin & check_lmax)

    output = (oope * weight_step).where(check_size).sum(dim='w')
    return output

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
        message = "Time argument must be set to \
            'season', 'month', 'year' or None"
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


def extract_mean_size(data, const, mesh, varname,
                      maskdom=None, aggregate=False):

    '''
    Extracts the mean length or weight.

    :param data: Apecosm dataset
    :type data: :class:`xarray.Dataset`
    :param const: Apecosm constants dataset
    :type const: :class:`xarray.Dataset`
    :param mesh: Mesh grid dataset
    :type mesh: :class:`xarray.Dataset`
    :param varname: Name of the variable
        to extract (`length` or `weight`)
    :type varname: str
    :param maskdom: Array of domain mask
    :type maskdom: :class:`numpy.array`, optional
    :param aggregate: True if community is included
        in the mean. If False, output depends on community.
    :type aggregate: bool, optional

    '''

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask']

    tmask = _squeeze_variable(tmask)
    surf = _squeeze_variable(mesh['e1t'] * mesh['e2t'])

    # extract the domain coordinates
    if maskdom is None:
        maskdom = np.ones(tmask.shape)

    oope = data['OOPE']

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])
    tmask = tmask * maskdom

    # time, lat, lon, comm, w
    weight = tmask * surf * oope * const['weight_step']

    dims = ['x', 'y', 'w']
    if aggregate:
        dims += ['c']

    variable = (const[varname] * weight).sum(dims)
    variable /= weight.sum(dims)

    return variable


def extract_weighted_data(data, const, mesh, varname,
                          maskdom=None):

    '''
    Extracts data outputs and weight them using
    biomass outputs.

    :param data: Apecosm dataset
    :type data: :class:`xarray.Dataset`
    :param const: Apecosm constants dataset
    :type const: :class:`xarray.Dataset`
    :param mesh: Mesh grid dataset
    :type mesh: :class:`xarray.Dataset`
    :param varname: Name of the variable
        to extract (`repfonct_day` for instance)
    :type varname: str
    :param maskdom: Array of domain mask
    :type maskdom: :class:`numpy.array`, optional

    '''

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask']

    tmask = _squeeze_variable(tmask)
    surf = _squeeze_variable(mesh['e1t'] * mesh['e2t'])

    # extract the domain coordinates
    if maskdom is None:
        maskdom = np.ones(tmask.shape)

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])
    tmask = tmask * maskdom

    oope = data['OOPE']

    # time, lat, lon, comm, w
    weight = tmask * surf * oope * const['weight_step']

    dims = ['y', 'x']

    output = (data[varname] * weight).sum(dims) / weight.sum(dims)
    return output


def extract_oope_data(data, mesh, const, maskdom=None,
                      use_wstep=True, compute_mean=False):

    '''
    Extraction of OOPE values on a given domain.
    OOPE is spatially integrated over
    the domain.

    :param data: Apecosm dataset
    :param mesh: Mesh grid dataset
    :param const: Apecosm constants dataset
    :param maskdom: Array containing the area mask
    :param use_wstep: True if data must be multiplied
        by weight step (conversion from :math:`J.m^{-2}.kg^{-1}`
        to :math:`J.m^{-2}`)
    :param compute_mean: True if mean, else integral.

    :type data: :class:`xarray.Dataset`
    :type mesh: :class:`xarray.Dataset`
    :type const: :class:`xarray.Dataset`
    :type maskdom: :class:`numpy.array`, optional
    :type use_wstep: bool, optional
    :type compute_mean: bool, optional


    :return: A tuple with the time-value and the LTL time series

    '''

    # Extract constant fields and extract weight_step
    if use_wstep:
        wstep = const['weight_step']
    else:
        wstep = 1

    surf = _squeeze_variable(mesh['e2t']) * _squeeze_variable(mesh['e1t'])

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask']

    tmask = _squeeze_variable(tmask)

    # extract the domain coordinates
    if maskdom is None:
        maskdom = np.ones(tmask.shape)

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    # add virtual dimensions to domain mask and
    # correct landsea mask
    tmask = tmask * maskdom
    weight = tmask * surf  # time, lat, lon, comm, w

    data = data['OOPE']

    if use_wstep:
        data = data * wstep

    data = (data * weight).sum(dim=('x', 'y'))  # time, com, w

    if compute_mean:
        data /= weight.sum(dim=['x', 'y'])

    return data


def _squeeze_variable(variable):

    r'''
    If a variable which is supposed to be 2D (dims=['x', 'y']) but
    is in fact 3D, we remove the spurious dimensions.

    :return: A data array with the given time mean
    '''

    dictout = {}
    for dim in variable.dims:
        if dim not in ['x', 'y']:
            dictout[dim] = 0
        else:
            dictout[dim] = slice(None, None)
    return variable.isel(**dictout)
