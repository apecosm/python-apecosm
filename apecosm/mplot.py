'''
Module that contains functions for plotting
OOPE maps. The Ngl module is needed. It
can be used by using a virtual environment
(see https://www.pyngl.ucar.edu/)
'''

from __future__ import print_function
import sys
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl import geoaxes
import matplotlib.pyplot as plt
import numpy as np
from .misc import extract_community_names
from .constants import LTL_NAMES

plt.rcParams['text.usetex'] = False

PROJIN = ccrs.PlateCarree()

def plot_diet_values(diet_data, const, community_index, draw_legend=False, legend_args=None, **kwargs):

    r'''
    Draws the diet matrix.

    :param diet_data: DataArray containing the diet matrix. The time
    and space dimensions must have been removed.
    :param const: Dataset containing the Apecosm constant variables
    :param community_index: Index of the community to draw
    :param draw_legend: True if the legend must be added
    :param legend_args: Dictionnary containing additional legend arguments
    :param \**kwargs: Additional arguments of the stackplot function

    '''

    if legend_args is None:
        legend_args = {}

    community_names = extract_community_names(const)
    n_community = len(community_names)

    legend = LTL_NAMES.copy()
    for c in range(n_community):
        legend.append(community_names[c])

    ax = plt.gca()
    length = const['length'].isel(c=community_index)
    diet = diet_data.isel(c=community_index)#.compute()
    repf = diet.sum(dim='prey_group')
    l = ax.stackplot(length, diet.T, edgecolor='k', **kwargs)
    ax.set_xscale('log')
    ax.set_xlim(length.min(), length.max())
    ax.set_ylim(0, repf.max())
    ax.set_title(community_names[community_index])
    if draw_legend:
        ax.legend(legend, **legend_args)
    return l

def plot_pcolor_map(data, mesh, axis=None, draw_land=True, **kwargs):

    r'''
    Draws 2D OOPE maps.

    :param data: Data to plot
    :type data: :class:`xarray.DataArray`
    :param mesh: Mesh dataset
    :type mesh: :class:`xarray.Dataset`
    :param ax: Axis on which to draw
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot` or
        :class:`cartopy.mpl.geoaxes.GeoAxesSubplot`, optional
    :param \**kwargs: Additional arguments to the `pcolormesh` function

    :return: The output quad mesh.
    :rtype: :class:`matplotlib.collections.QuadMesh`

    '''

    if not isinstance(data, xr.DataArray):
        message = 'The input must be a "xarray.DataArray" '
        message += f'Currently, it is a {type(data)} object'
        print(message)
        sys.exit(1)

    if data.ndim != 2:
        message = 'The input data array must be 2D'
        print(message)
        sys.exit(1)

    if 'x' not in data.dims:
        message = 'The input data array must have a "x" dim. '
        message += f'Dimensions are {data.dims}'
        print(message)
        sys.exit(1)

    if 'y' not in data.dims:
        message = 'The input data array must have a "y" dim.'
        message += f'Dimensions are {data.dims}'
        print(message)
        sys.exit(1)

    if axis is None:
        axis = plt.gca()

    lonf = mesh['glamf'].values
    latf = mesh['gphif'].values
    tmask = mesh['tmask'].isel(z=0).values

    lonf, latf, tmask, var_to_plot = _reconstuct_variable(lonf, latf, tmask, data)

    if isinstance(axis, geoaxes.GeoAxesSubplot):
        projected = True
        quadmesh = axis.pcolormesh(lonf, latf, var_to_plot[1:, 1:],
                                  transform=PROJIN, **kwargs)
    else:
        projected = False
        quadmesh = axis.pcolormesh(var_to_plot, **kwargs)
    if projected and draw_land:
        axis.add_feature(cfeature.LAND, zorder=1000)
        axis.add_feature(cfeature.COASTLINE, zorder=1001)

    return quadmesh

def plot_contour_map(data, mesh, filled=False, axis=None, draw_land=False, **kwargs):

    r'''
    Draws 2D OOPE maps.

    :param data: Data to plot
    :type data: :class:`xarray.DataArray`
    :param mesh: Mesh dataset
    :type mesh: :class:`xarray.Dataset`
    :param ax: Axis on which to draw
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot` or
        :class:`cartopy.mpl.geoaxes.GeoAxesSubplot` , optional
    :param \**kwargs: Additional arguments to the `pcolormesh` function

    :return: The output quad mesh.
    :rtype: :class:`matplotlib.collections.QuadMesh`

    '''



    if not isinstance(data, xr.DataArray):
        message = 'The input must be a "xarray.DataArray" '
        message += f'Currently, it is a {type(data)} object'
        print(message)
        sys.exit(1)

    if data.ndim != 2:
        message = 'The input data array must be 2D'
        print(message)
        sys.exit(1)

    if 'x' not in data.dims:
        message = 'The input data array must have a "x" dim. '
        message += f'Dimensions are {data.dims}'
        print(message)
        sys.exit(1)

    if 'y' not in data.dims:
        message = 'The input data array must have a "y" dim.'
        message += f'Dimensions are {data.dims}'
        print(message)
        sys.exit(1)

    if axis is None:
        axis = plt.gca()

    if filled:
        contour_function = axis.contourf
        proj_contour_function = axis.tricontourf
    else:
        contour_function = axis.contour
        proj_contour_function = axis.tricontour

    lont = mesh['glamt'].values
    latt = mesh['gphit'].values
    tmask = mesh['tmask'].isel(z=0).values

    lont, latt, tmask, var_to_plot = _reconstuct_variable(lont, latt, tmask, data)

    if isinstance(axis, geoaxes.GeoAxesSubplot):
        projected = True
        # Extract non masked data and associated coordinates, store them
        # in a 1D array
        iok = np.nonzero(np.ma.getmaskarray(var_to_plot) == 0)
        lon1d = lont[iok]
        lat1d = latt[iok]
        tp1d = var_to_plot[iok]

        # Convert the geographical coordinates into map coords
        projout = axis.projection
        _temp = projout.transform_points(PROJIN, lon1d, lat1d)
        lonout = _temp[..., 0]
        latout = _temp[..., 1]
        cl = proj_contour_function(lonout, latout, tp1d, **kwargs)
    else:
        projected = False
        cl = contour_function(var_to_plot, **kwargs)
    if projected and draw_land:
        axis.add_feature(cfeature.LAND, zorder=1000)
        axis.add_feature(cfeature.COASTLINE, zorder=1001)

    return cl



def _reconstuct_variable(lonf, latf, tmask, data):

    var_to_plot_temp = data.values
    nlat_grid, nlon_grid = lonf.shape
    nlat_data, nlon_data = var_to_plot_temp.shape

    # If the data array does not have the save dimension as the grid,
    # i.e. new Nemo format, we reconstruct the zonal cyclicity
    if (nlon_data == nlon_grid - 2) & (nlat_data == nlat_grid - 1):

        # init an array of the same size as the grid except for
        # the northfold band (top row)
        var_to_plot = np.zeros((nlat_data, nlon_grid), dtype=float)

        # fill inner bound
        var_to_plot[:, 1:-1] = var_to_plot_temp

        # add cyclicity
        var_to_plot[:, 0] = var_to_plot[:, -2]
        var_to_plot[:, -1] = var_to_plot[:, 1]

        # Remove upper row
        lonf = lonf[:-1, :]
        latf = latf[:-1, :]
        tmask = tmask[:-1, :]

    else:
        var_to_plot = var_to_plot_temp

    var_to_plot = np.ma.masked_where((tmask == 0) | (np.isnan(var_to_plot)), var_to_plot)

    return lonf, latf, tmask, var_to_plot



# if __name__ == '__main__':

#     DIRIN = '../doc/_static/example/data/'
#     MESH = xr.open_dataset('%s/mesh_mask.nc' % DIRIN).isel(t=0)

#     DATA = xr.open_dataset('%s/apecosm/apecosm_OOPE.nc' % DIRIN)
#     DATA1 = DATA['OOPE'].mean(dim='time').isel(community=0, weight=0)
#     DATA2 = DATA['OOPE'].mean(dim='time').isel(x=0, y=0)

#     fig = plt.figure()
#     axes = plt.axes()
#     plot_pcolor_map(DATA1, MESH)
#     plt.savefig('maps1.png', bbox_inches='tight')
#     plt.close(fig)

#     fig = plt.figure()
#     axes = plt.axes(projection=ccrs.PlateCarree())
#     plot_pcolor_map(DATA1, MESH)
#     plt.savefig('maps2.png', bbox_inches='tight')
#     plt.close(fig)
