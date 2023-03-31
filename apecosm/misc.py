''' Module that contains some miscellaneous functions '''

import numpy as np
from .constants import ALLOM_W_L
import os
import xarray as xr


def find_percentile(data, percentage=1):

    '''
    Extract percentile to saturate the colormaps.
    They are computed from unmasked arrays

    :param data: Data array
    :type data: :class:`numpy.array`
    :param percentage: Percentage used to
        saturate the colormap.
    :type percentege: float

    :return: A tuple containing the lower
        and upper bounds (cmin, cmax)

    '''

    data = np.ma.masked_where(np.isnan(data), data)
    iok = np.nonzero(np.logical_not(np.ma.getmaskarray(data)))
    temp = data[iok]

    cmin = np.percentile(np.ravel(temp), percentage)
    cmax = np.percentile(np.ravel(temp), 100 - percentage)

    return cmin, cmax


def compute_daylength(lat, nlon=None):

    '''
    Computes the day-length fraction providing a latitude array by
    using the same formulation as in APECOSM.

    :param lat: Latitude array (either 1D or 2D)
    :type lat: :class:`numpy.array`
    :param nlon: Number of longitudes
    :type nlon: int, optional

    :return: A 2D array with the daylength fraction
    '''

    lat = np.squeeze(lat)

    # If the number of dimensions for lat is 1,
    # tile it with dimensions (nlat, nlon)
    if lat.ndim == 1:
        lat = np.tile(lat, (nlon, 1)).T

    time = np.arange(0, 365)

    nlon = lat.shape[1]

    lat = lat[np.newaxis, :, :]
    time = time[:, np.newaxis, np.newaxis]

    p_val = 0.833

    theta = 0.2163108 + 2 * \
        np.arctan(0.9671396 * np.tan(0.00860 * (time + 1 - 186)))  # eq. 1
    phi = np.arcsin(0.39795 * np.cos(theta))  # eq. 2
    a_values = (np.sin(p_val * np.pi / 180.) + np.sin(lat * np.pi / 180.) *
         np.sin(phi)) / (np.cos(lat * np.pi / 180.) * np.cos(phi))
    a_values[a_values >= 1] = 1
    a_values[a_values <= -1] = -1
    daylength = 1.0 - (1.0 / np.pi) * np.arccos(a_values)
    daylength[daylength < 0] = 0
    daylength[daylength > 1] = 1

    return daylength


def extract_community_names(const):

    '''
    Extracts community names from the units attribute in
    the OOPE NetCDF file. It uses regular expressions to extract
    the names. `community` must be a variable in the NetCDF
    variable.

    :param xarray.Dataset data: xarray dataset that
     is returned when using xr.open_dataset on the
     output file.

    :return: The list of community names
    '''

    comnames = {}
    attrlist = [attr for attr in const.attrs if attr.startswith('Community_')]
    if len(attrlist) != 0:
        for attr in attrlist:
            comnames[attr.replace('_', ' ')] = const.attrs[attr]
    else:
        for community_index in range(const.dims['c']):
            name = 'Community %d' % community_index
            comnames[name] = name
    return comnames


def size_to_weight(size):

    r'''
    Converts size (in m) into weight (in kg) #

    .. math::

        W = A L^3

    '''

    return ALLOM_W_L * np.power(size, 3)


def weight_to_size(weight):

    r'''
    Converts weight (in kg) into size (in m)

    ..math::

        L = \left(\frac{W}{A}\right)^{-3}

    '''

    return np.power(weight / ALLOM_W_L, 1/3.)


def compute_mean_min_max_ts(ts, period):

    '''
        Compute the mean, min and max value of timeserie ts over time of length period

        :param ts: timeserie (containing a time field).
        :param period: number of time step upon which are computed mean, min and max value of ts.

        :type ts: xarray with a field named time
        :type period: int
    '''

    n = int(len(ts.time)/period)

    average = np.zeros(n)
    maxi = np.zeros(n)
    mini = np.zeros(n)
    time = np.zeros(n)
    cpt = 0
    while cpt < n:
        average[cpt] = ts[(cpt * period):((cpt + 1) * period - 1)].mean()
        maxi[cpt] = ts[(cpt * period):((cpt + 1) * period - 1)].max()
        mini[cpt] = ts[(cpt * period):((cpt + 1) * period - 1)].min()
        time[cpt] = ts.time[cpt*period]
        cpt = cpt + 1

    return average, maxi, mini, time


def extract_fleet_names(dirin):

    '''
    Extracts fleet names from the fishing configuration file.

    :param str: configuration path

    :return: The list of fleet names
    '''

    fishing_model = os.path.join(dirin, 'fishing_model.conf')
    with open(fishing_model) as f:
        for line in f:
            if "nb_fishing_fleets" in line.strip():
                _, nb_fleet_str = line.split('=')
                nb_fleet = int(nb_fleet_str)

    fleet = {}
    fleet_names = {}
    for i in np.arange(nb_fleet):
        fleet_fname = os.path.join(dirin, 'fleet_' + str(i) + '.conf')
        with open(fleet_fname) as f:
            for line in f:
                if "fleet_name" in line.strip():
                    _, fleet_name_str = line.split('=')
                    fleet_names[i] = fleet_name_str[:-1]

    return fleet_names



if __name__ == '__main__':

    X = 1e-6
    W = size_to_weight(X)
    XTEST = weight_to_size(W)
