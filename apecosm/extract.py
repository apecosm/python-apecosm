""" Module that contains some functions for data extractions """

import os
import sys
from glob import glob
import xarray as xr
import numpy as np


def open_mesh_mask(mesh_file, replace_dims=None):

    """
    Opens a NEMO mesh mask. It removes the `t` dimension from
    the dataset.

    :param str mesh_file: Full path of the mesh file.
    :param replace_dims: Dictionnary that is used to
        rename dimension names by other names
    :type replace_dims: dict, optional
    :return: A dataset containing the variables of the
        mesh mask.
    :rtype: :class:`xarray.Dataset`
    """

    # open the mesh file, extract tmask, lonT and latT
    mesh = xr.open_dataset(mesh_file)
    if replace_dims is not None:
        mesh = mesh.rename(replace_dims)
    if 't' in mesh.dims:
        mesh = mesh.isel(t=0)

    return mesh


def open_constants(dirin, replace_dims=None):

    """
    Opens an Apecosm constant file. It will look for
    a file that contains `Constant`.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    """

    path = os.path.join(dirin, '*Constant*.nc')
    constant = xr.open_mfdataset(path)
    if replace_dims is not None:
        constant = constant.rename(replace_dims)
    return constant


def open_apecosm_data(dirin, replace_dims=None, **kwargs):

    """
    Opens Apecosm outputs.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    """

    pattern = os.path.join(dirin, '*.nc.*')
    filelist = glob(pattern)
    if len(filelist) == 0:
        pattern = os.path.join(dirin, '*.nc')
        filelist = glob(pattern)
        filelist = [f for f in filelist if 'Constant' not in f]

    # open the dataset
    filelist.sort()
    data = xr.open_mfdataset(filelist, **kwargs)
    if replace_dims is not None:
        data = data.rename(replace_dims)
    return data


def open_ltl_data(dirin, replace_dims=None, **kwargs):

    """
    Opens NEMO/PISCES outputs.

    :param dirin: Input directory.
    :type dirin: str

    :param replace_dims: Dictionnary that is used to
        rename dimensions
    :type replace_dims: dict, optional
    """

    pattern = os.path.join(dirin, '*.nc')
    filelist = glob(pattern)
    filelist.sort()

    data = xr.open_mfdataset(filelist, **kwargs)
    if replace_dims is not None:
        data = data.rename(replace_dims)
    return data


def extract_ltl_data(data, mesh, varname,
                     mask_dom=None, depth_min=None,
                     depth_max=None):

    """
    Extraction of LTL values on a given domain.
    LTL is vertically integrated and spatially averaged over
    the domain.

    :param data: Apecosm dataset
    :type data: :class:`xarray.Dataset`
    :param mesh: Mesh grid dataset
    :type mesh: :class:`xarray.Dataset`
    :param str varname: LTL variable name
    :param mask_dom: Mask array. If None, full domain is considered
    :type mask_dom: :class:`numpy.array`
    :param int depth_min: Minimum depth
    :param int depth_max: Maximum depth

    :return: A xarray dataset
    """

    surf = mesh['e2t'] * mesh['e1t']

    data_array = data[varname]

    if 'gdept_0' in mesh.variables:
        depth = mesh['gdept_0']  # 1, z, lat, lon
    else:
        depth = mesh['gdept_1d']

    tmask = mesh['tmask']

    if 'tmaskutil' in mesh.variables:
        tmask *= mesh['tmaskutil']

    if data_array.ndim == 4:

        # If data is 4d, assume a depth dimension. We integrate over this dimension

        # First, we extract the cell thickness for the integration
        if 'e3t' in data.variables:
            # if VVL, e3t should be read from data
            e3t = data['e3t']  # time, z, lat, lon
        elif 'e3t_0' in mesh.variables:
            e3t = mesh['e3t_0']  # 1, z, lat, lon
        else:
            e3t = mesh['e3t_1d']

        # Create a vertical_weight mask
        vertical_weight = e3t * tmask  # (1, z, lat, lon) or (time, z, lat, lon)

        # If a maximum depth is provide, we mask data below
        if depth_max is None:
            depth_max =  sys.float_info.max

        if depth_min is None:
            depth_min = 0

        vertical_weight = vertical_weight.where((depth >= depth_min) & (depth <= depth_max))

        # Replace NaN with 0, especially if VVL is used
        vertical_weight = vertical_weight.fillna(0)

        # Data is vertically integrated
        data_array = data_array.weighted(vertical_weight).sum(dim='z') # time

    # extract the domain coordinates
    if mask_dom is None:
        mask_dom = np.ones(tmask[0].shape)

    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])
    tmask = tmask * mask_dom  # 0 if land or out of domain, else 1

    horizontal_weight = (surf * tmask).isel(z=0).fillna(0)

    # Horizonal average of the biomass
    output = data_array.weighted(horizontal_weight).mean(dim=('x', 'y')) # time

    output.attrs['horizontal_norm_weight'] = float(horizontal_weight.sum(dim=('x', 'y')).compute().values)

    return output


def spatial_mean_to_integral(data):

    '''
    Converts a spatial mean into an integral by multiplying by the total surface
    '''

    norm_data = data * data.attrs['horizontal_norm_weight']

    return norm_data


def extract_oope_size_integration(data, const, lmin=None, lmax=None):

    '''
    Integrates the biomass density between two lengths. The provided data can be either
    a map of a time-series.

    :param data: Apecosm dataset. **Units must be in :math:`J.kg^{-1}`**
    :type data: :class:`xarray.DataArray`
    :param const: Apecosm constant dataset. It must contain `weight_step` and `length`
    :type const: :class:`xarray.Dataset`
    :param float lmin: Minimum size to consider (cm). If None, integrates from the beginning of the size spectra
    :param float lmax: Maximum size to consider (cm). If None, integrates to the end of the size spectra

    :return: A xarray dataset
    '''

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

    output = (data * weight_step).where(check_size).sum(dim='w')
    return output

def extract_time_means(data, time=None):

    r"""
    Extract data time mean.

    :param data: Apecosm dataset
    :type data: :class:`xarray.Dataset`
    :param str time: Time mean. Can be time average ('tot'), yearly means ('year'),
     seasonal means ('season') or monthly means ('monthly')

    :return: A data array with the given time mean
    """

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


def compute_cumulated_biomass(spatial_integrated_biomass, const):

    '''
    Computes the cumulated biomass, i.e. the biomass proportion
    for each size class.

    :param spatial_integrated_biomass: OOPE variable
    :param xarray.Dataset const: Apecosm constant dataset

    :return: A xarray.DataArray

    '''

    spatial_integrated_biomass = spatial_integrated_biomass * const['weight_step']
    size_prop = spatial_integrated_biomass.cumsum(dim='w') / spatial_integrated_biomass.sum(dim='w') * 100
    return size_prop

def extract_mean_size(spatially_integrated_biomass, const, varname, ):

    """
    Extracts the mean length or weight.

    :param spatially_integrated_biomass: Biomass integrated over a given region (dim: time, c, w). Must be in :math:`J.kg^{-1}`
    :type data: :class:`xarray.Dataset`
    :param const: Apecosm constants dataset
    :type const: :class:`xarray.Dataset`
    :param mesh: Mesh grid dataset
    :type mesh: :class:`xarray.Dataset`
    :param varname: Name of the variable
        to extract (`length` or `weight`)
    :type varname: str
    :param mask_dom: Array of domain mask
    :type mask_dom: :class:`numpy.array`, optional
    :param aggregate: True if community is included
        in the mean. If False, output depends on community.
    :type aggregate: bool, optional

    """

    # time, lat, lon, comm, w
    weight = (spatially_integrated_biomass * const['weight_step']).fillna(0)
    output = const[varname].weighted(weight).mean(dim=['w'])
    output['community_weight'] = weight.sum(dim='w')

    return output

def compute_community_mean(data):

    '''
    Computes the community mean.

    :param data: Data array that contains a ``community_weight`` attribute.
    :type data: :class:`xarray.DataArray`
    return: The mean computed over the communities.

    '''

    weight = data['community_weight']
    output = data.weighted(weight).mean(dim='c')
    return output

def extract_weighted_data(data, const, mesh, varname,
                          mask_dom=None):

    """
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
    :param mask_dom: Array of domain mask
    :type mask_dom: :class:`numpy.array`, optional

    """

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask'].isel(z=0)

    surf = mesh['e1t'] * mesh['e2t']

    # extract the domain coordinates
    if mask_dom is None:
        mask_dom = np.ones(tmask.shape)

    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])
    tmask = tmask * mask_dom

    oope = data['OOPE']

    # time, lat, lon, comm, w
    weight = (tmask * surf * oope * const['weight_step']).fillna(0)

    dims = ['y', 'x']

    output = (data[varname].weighted(weight)).mean(dims)
    output.attrs['average_weight'] = weight.sum(dim=dims).compute()
    return output


def extract_oope_data(data, mesh, mask_dom=None):

    """
    Extraction of OOPE values on a given domain.
    OOPE is spatially integrated over
    the domain.

    :param data: Apecosm dataset
    :param mesh: Mesh grid dataset
    :param const: Apecosm constants dataset
    :param mask_dom: Array containing the area mask
    :param use_wstep: True if data must be multiplied
        by weight step (conversion from :math:`J.m^{-2}.kg^{-1}`
        to :math:`J.m^{-2}`)

    :type data: :class:`xarray.Dataset`
    :type mesh: :class:`xarray.Dataset`
    :type const: :class:`xarray.Dataset`
    :type mask_dom: :class:`numpy.array`, optional
    :type use_wstep: bool, optional

    :return: A tuple with the time-value and the LTL time series

    """

    surf = mesh['e2t'] * mesh['e1t']

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask']

    tmask = tmask

    # extract the domain coordinates
    if mask_dom is None:
        mask_dom = np.ones(tmask.shape)

    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])

    # add virtual dimensions to domain mask and
    # correct landsea mask
    tmask = tmask * mask_dom
    weight = (tmask * surf).fillna(0)  # time, lat, lon, comm, w

    output = data.weighted(weight).mean(dim=('x', 'y'))  # time, com, w
    output.attrs['horizontal_norm_weight'] = weight.sum(dim=['x', 'y'])
    output.name = data.name

    return output

def open_fishing_data(dirin):

    """
        Opens Apecosm fishing output files : market_result.nc; fleet_maps_2d_X.nc;
        fleet_summary_X.nc; fleet_parameters_X.nc

        :param dirin: Directory of Apecosm fishing outputs.

        :type dirin: str
    """

    market = xr.open_dataset(os.path.join(dirin, "market_results.nc"))
    nb_fleet = len(market['fleet'])

    fleet_maps = {}
    fleet_summary = {}
    fleet_parameters = {}
    for i in np.arange(nb_fleet):
        fleet_maps[i] = xr.open_dataset(os.path.join(dirin, 'fleet_maps_2d_' + str(i) + '.nc'))
        fleet_summary[i] = xr.open_dataset(os.path.join(dirin, 'fleet_summary_' + str(i) + '.nc'))
        fleet_parameters[i] = xr.open_dataset(os.path.join(dirin, 'fleet_parameters_' + str(i) + '.nc'))

    return market, fleet_maps, fleet_summary, fleet_parameters


def read_report_params(csv_file_name):
    file = open(csv_file_name)
    report_parameters = {'output_dir':'', 'mesh_file':'', 'FONT_SIZE':'', 'LABEL_SIZE':'', 'THIN_LWD':'', 'REGULAR_LWD':'',
                         'THICK_LWD':'','COL_GRID':'','REGULAR_TRANSP':'','HIGH_TRANSP':'', 'FIG_WIDTH':'','FIG_HEIGHT':'',
                         'FIG_DPI':'','CB_SHRINK':'','COL_MAP':'','FISHING_PERIOD':'','APECOSM_PERIOD':'','fishing_output_dir':'', 'fishing_config_dir':''}
    for line in file:
        fields = line.strip().split(',')
        if fields[0] == 'output_dir':
            report_parameters['output_dir'] = fields[1].replace(" ", "")
        elif fields[0] == 'mesh_file':
            report_parameters['mesh_file'] = fields[1].replace(" ", "")
        elif fields[0] == 'FONT_SIZE':
            report_parameters['FONT_SIZE'] = int(fields[1])
        elif fields[0] == 'LABEL_SIZE':
            report_parameters['LABEL_SIZE'] = int(fields[1])
        elif fields[0] == 'THIN_LWD':
            report_parameters['THIN_LWD'] = int(fields[1])
        elif fields[0] == 'REGULAR_LWD':
            report_parameters['REGULAR_LWD'] = int(fields[1])
        elif fields[0] == 'THICK_LWD':
            report_parameters['THICK_LWD'] = int(fields[1])
        elif fields[0] == 'COL_GRID':
            report_parameters['COL_GRID'] = (int(fields[1].replace(' (', '').split('/')[0])/256, int(fields[2].replace(' ', '').split('/')[0])/256, int(fields[3].replace(') ','').split('/')[0])/256)
        elif fields[0] == 'REGULAR_TRANSP':
            report_parameters['REGULAR_TRANSP'] = float(fields[1])
        elif fields[0] == 'HIGH_TRANSP':
            report_parameters['HIGH_TRANSP'] = float(fields[1])
        elif fields[0] == 'FIG_WIDTH':
            report_parameters['FIG_WIDTH'] = float(fields[1])
        elif fields[0] == 'FIG_HEIGHT':
            report_parameters['FIG_HEIGHT'] = float(fields[1])
        elif fields[0] == 'FIG_DPI':
            report_parameters['FIG_DPI'] = int(fields[1])
        elif fields[0] == 'CB_SHRINK':
            report_parameters['CB_SHRINK'] = float(fields[1])
        elif fields[0] == 'CB_THRESH':
            report_parameters['CB_THRESH'] = int(fields[1])
        elif fields[0] == 'COL_MAP':
            report_parameters['COL_MAP'] = fields[1].replace(" ", "")
        elif fields[0] == 'FISHING_PERIOD':
            report_parameters['FISHING_PERIOD'] = int(fields[1])
        elif fields[0] == 'APECOSM_PERIOD':
            report_parameters['APECOSM_PERIOD'] = int(fields[1])
        elif fields[0] == 'fishing_output_dir':
            report_parameters['fishing_output_dir'] = fields[1].replace(" ", "")
        elif fields[0] == 'fishing_config_dir':
            report_parameters['fishing_config_dir'] = fields[1].replace(" ", "")

    return report_parameters
