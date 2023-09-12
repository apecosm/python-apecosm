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
from .misc import extract_community_names
from .constants import LTL_NAMES
from math import ceil
import numpy as np

plt.rcParams['text.usetex'] = False

PROJIN = ccrs.PlateCarree()

def plot_diet_values(diet_data, const, community_index, draw_legend=False, legend_args={}, **kwargs):

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

def plot_oope_map(data, mesh, axis=None, draw_land=True, **kwargs):

    '''
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
        message += 'Currently, it is a %s object' % type(data)
        print(message)
        sys.exit(1)

    if data.ndim != 2:
        message = 'The input data array must be 2D'
        print(message)
        sys.exit(1)

    if 'x' not in data.dims:
        message = 'The input data array must have a "x" dim. '
        message += 'Dimensions are %s' % str(data.dims)
        print(message)
        sys.exit(1)

    if 'y' not in data.dims:
        message = 'The input data array must have a "y" dim.'
        message += 'Dimensions are %s' % str(data.dims)
        print(message)
        sys.exit(1)

    if axis is None:
        axis = plt.gca()
    lonf = mesh['glamf'].values
    latf = mesh['gphif'].values
    tmask = mesh['tmask'].isel(z=0).values
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

    var_to_plot = np.ma.masked_where(tmask == 0, var_to_plot)

    if isinstance(axis, geoaxes.GeoAxesSubplot):
        projected = True
        quadmesh = plt.pcolormesh(lonf, latf, var_to_plot[1:, 1:],
                                  transform=PROJIN, **kwargs)
    else:
        projected = False
        quadmesh = plt.pcolormesh(var_to_plot, **kwargs)
    if projected and draw_land:
            axis.add_feature(cfeature.LAND, zorder=1000)
            axis.add_feature(cfeature.COASTLINE, zorder=1001)

    return quadmesh


if __name__ == '__main__':

    DIRIN = '../doc/_static/example/data/'
    MESH = xr.open_dataset('%s/mesh_mask.nc' % DIRIN).isel(t=0)

    DATA = xr.open_dataset('%s/apecosm/apecosm_OOPE.nc' % DIRIN)
    DATA1 = DATA['OOPE'].mean(dim='time').isel(community=0, weight=0)
    DATA2 = DATA['OOPE'].mean(dim='time').isel(x=0, y=0)

    fig = plt.figure()
    axes = plt.axes()
    plot_oope_map(DATA1, MESH)
    plt.savefig('maps1.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    axes = plt.axes(projection=ccrs.PlateCarree())
    plot_oope_map(DATA1, MESH)
    plt.savefig('maps2.png', bbox_inches='tight')
    plt.close(fig)
