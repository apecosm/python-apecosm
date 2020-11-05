'''
Module that contains pre-defined domains and functions
related to domains
'''

import numpy as np
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
    

# Defintions of some domains (list to be completed)
BENGUELA = {'lon': [10, 20, 20, 10, 10], 'lat': [-36, -36, -15, -15, -36]}
OMZ_PERU = {'lon': [-90, -70, -70, -90, -90], 'lat': [-20, -20, 5, 5, -20]}
ARCTIC = {'lon': [-180, 180, 180, -180, -180], 'lat': [66, 66, 90, 90, 66]}
ANTARCTIC = {'lon': [-180, 180, 180, -180, -180], 'lat': [-55, -55, -90, -90, -55]}
ETP = {'lon': [-160, -70., -70, -160, -160], 'lat': [-20, -20, 25, 25, -20]}


DOMAINS = {'BENGUELA': BENGUELA,
           'OMZ-PERU': OMZ_PERU,
           'ARCTIC': ARCTIC,
           'ANTARCTIC': ANTARCTIC,
           'ETP': ETP}


def inpolygon(xin_2d, yin_2d, x_pol, y_pol):

    """
    Determines whether points of a 2D-grid are within a polygon.

    Equivalent to the inpolygon function of Matlab.

    .. note:: If the input polygon is not closed, it is automatically closed

    :param numpy.array xin_2d: 2-D array with the x-coords of the domain
    :param numpy.array yin_2d: 2-D array with the y-coords of the domain
    :param numpy.array x_pol: 1-D array with the x-coords of the polygon
    :param numpy.array y_pol: 1-D array with the y-coords of the polygon

    :return: A 2D array (same shape as xin_2d and yin_2d)
     with 1 when the point is within the polygon, else 0.
    :rtype: numpy.array

    """

    from matplotlib import path

    x_pol = np.array(x_pol)
    y_pol = np.array(y_pol)

    if xin_2d.ndim != 2:
        raise ValueError("The xin_2d argument must be 2D. %d dimensions" %xin_2d.ndim)

    if yin_2d.ndim != 2:
        raise ValueError("The yin_2d argument must be 2D. %d dimensions" %yin_2d.ndim)

    if x_pol.ndim != 1:
        raise ValueError("The x_pol argument must be 1D. %d dimensions" %x_pol.ndim)

    if y_pol.ndim != 1:
        raise ValueError("The y_pol argument must be 1D. %d dimensions" %y_pol.ndim)

    x_pol = np.array(x_pol)
    y_pol = np.array(y_pol)

    # If the polynom is not closed, we close it
    if (x_pol[0] != x_pol[-1]) | (y_pol[0] != y_pol[-1]):
        x_pol = np.append(x_pol, x_pol[0])
        y_pol = np.append(y_pol, y_pol[0])

    nx, ny = xin_2d.shape

    # creation the input of the path.Path command:
    # [(x1, y1), (x2, y2), (x3, y3)]
    path_input = [(xtemp, ytemp) for xtemp, ytemp in zip(x_pol, y_pol)]

    # initialisation of the path object
    temppath = path.Path(path_input)

    # creation of the list of all the points within the domain
    # it must have a N x 2 shape
    list_of_points = np.array([np.ravel(xin_2d), np.ravel(yin_2d)]).T

    # Calculation of the mask (True if within the polygon)
    mask = temppath.contains_points(list_of_points)

    # reconverting the mask into a nx by ny array
    mask = np.reshape(mask, (nx, ny))

    return mask


def plot_domains(filename='apecosm_domains.pdf', ncol=2, leg_font_size=8):

    '''
    Plots the pre-defined domains on a global map.

    :param str filename: Name of the output image file
    :param int ncol: Number of columns in the legend
    :param float leg_font_size: Legend font size

    '''

    cmap = getattr(plt.cm, plt.rcParams['image.cmap'])
    plt.figure()
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    cpt = 0.0
    for k, v in DOMAINS.items():
        col = cmap(cpt / len(DOMAINS.items()))
        plt.plot(v['lon'], v['lat'], label=k, linewidth=1, color=col, transform=projection)
        cpt += 1.0

    ax.coastlines(linewidth=0.5)
    plt.legend(fontsize=leg_font_size, ncol=ncol)
    plt.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':

    plot_domains()



# if __name__ == '__main__':
#
#     from envtoolkit import map as nbmap
#     import pylab as plt
#
#     data = xr.open_dataset("data/dyna_grid_T.nc")
#     sst = data['sosstsst'].values
#     lon = data['nav_lon'].values
#     lat = data['nav_lat'].values
#     sst = np.ma.masked_where(sst==0, sst)
#
#     sst = sst[0]
#
#     temp = arctic
#     temp = antarctic
#     temp = benguela
#     temp = omz_peru
#     temp = etp
#
#     mask = inpolygon(lon, lat, temp['lon'], temp['lat'])
#     print mask.shape
#
#     plt.figure()
#     plt.subplot(211)
#     cs = plt.imshow(sst, origin='lower', interpolation='none')
#     plt.colorbar(cs)
#     plt.subplot(212)
#     sst = np.ma.masked_where(mask==0, sst)
#     cs = plt.imshow(sst, origin='lower', interpolation='none')
#     plt.colorbar(cs)
#     plt.savefig('toto')
#
#     plt.figure()
#     lon = [-180, 180]
#     lat = [-90, 90]
#     m = nbmap.make_bmap(lon, lat)
#     m.plot(benguela['lon'], benguela['lat'], label='Benguela')
#     m.plot(etp['lon'], etp['lat'], label='etp')
#     m.plot(omz_peru['lon'], omz_peru['lat'], label='OMZ')
#     m.plot(arctic['lon'], arctic['lat'], label='Arctic')
#     m.plot(antarctic['lon'], antarctic['lat'], label='Antarctic')
#     m.drawcoastlines()
#     plt.legend()
#     plt.savefig('domains.png')
#
#     plot_domains()
