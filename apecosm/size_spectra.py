'''
Module dedicated to the computation and
plotting of size spectra
'''


from __future__ import print_function
import sys
import numpy as np
try:
    import pylab as plt
except ImportError:
    pass
import xarray as xr
import apecosm.constants as co
import apecosm.misc
import apecosm.conf
from .misc import extract_community_names


def compute_spectra_ltl(data, L, N=100, conv=1e-3, output_var='weight', **kwargs):

    r'''
    Computes the size/weight spectra for lower trophic levels
    (i.e. PISCES variables).

    **Weight spectra**:

    .. math::

        P(w) = a w^{-1}

    .. math::

        PHY2 = a\int_{w_1}^{w_2} w^{-1} dw = a \ln\left(\frac{w_2}{w_1}\right)

    .. math::

        a = \frac{PHY2}{\ln\left(\frac{w_2}{w_1}\right)}

    **Length spectra**:

    .. math::

        P(l) = b l^{-1}

    .. math::

        PHY2 = b\int_{l_1}^{l_2} l^{-1} dl = a \ln\left(\frac{l_2}{l_1}\right)

    .. math::

        b = \frac{PHY2}{\ln\left(\frac{l_2}{l_1}\right)}


    :param numpy.array|float data: LTL content (number of moles), either a float or a 1D array
    :param list L: Size limits of the LTL class
    :param int N: Number of weight class to draw.
    :param float conv: Conversion factor to move from LTL units to mol/m3 (Apecosm units)
    :param str out: 'weight' if weight spectra should be drawn, else 'length' for a
     size spectra

    :return: A tuple containing the weight, length array and the corresponding biomass

    '''

    # multiply conversion factor from C to E
    conv *= co.C_E_CONVERT
    data = data * conv

    # if isinstance(data, xr.DataArray):
    #     print('Data is converted into a numpy array')
    #     data = data.values
    #     data = np.ma.masked_where(np.isnan(data), data)
    #     data = np.atleast_1d(data)

    # L is a 2d array
    L = np.sort(L)
    if (L.ndim != 1) | (len(L) != 2):
        message = 'The "L" argument must be a list or array with two values'
        print(message)
        sys.exit(0)

    # Convert L into W using allometric formulae
    W = apecosm.misc.size_to_weight(L)

    # generate a vector of weights and length
    wvec = xr.DataArray(data=np.linspace(W[0], W[1], N), dims='l')
    lvec = xr.DataArray(data=np.linspace(L[0], L[1], N), dims='l')

    # broadcast arrays
    data, lvec2 = xr.broadcast(data, lvec)
    data, wvec2 = xr.broadcast(data, wvec)

    alpha = data / np.log(W[1] / W[0])
    beta = data / np.log(L[1] / L[0])
    rho = alpha * np.power(wvec2, -1)
    rhoL = beta * np.power(lvec2, -1)

    if output_var == 'weight':
        x = wvec
        y = rho
    else:
        x = lvec
        y = rhoL

    l = plt.plot(x, y.T, **kwargs)
    return l[0]


def plot_oope_spectra(data, const, output_var='weight', **kwargs):

    r'''
    Plots the OOPE size spectra. Since OOPE data are
    stored in :math:`J.m^{-3}.kg^{-1}`, they must me converted
    into :math:`J.m^{-3}.m^{-1}`:

    .. math::

        a \int_{w_i}^{w_{i+1}} w^{-1} dw = b \int_{l_i}^{l_{i+1}} l^{-3} dl

    .. math::

        \xi(w_i) \delta w_i = \xi(l_i) \delta l_i

    .. math::

        \xi(l_i) = \xi(w_i) \frac{\delta w_i}{\delta l_i}

    :param xarray.Dataset data: A OOPE dataset that must be
     2-dimensional (community, weight)
    :param str output_var: 'length' if size spectra should
     be plotted, else weight spectra is plotted.

    return None

    '''

    message = 'The input dataset must be have (community, weight) dimensions '

    if 'c' not in data.dims:
        print(message)
        sys.exit(0)

    if 'w' not in data.dims:
        print(message)
        sys.exit(0)

    # recovers the default cmap
    cmap = getattr(plt.cm, plt.rcParams['image.cmap'])

    weight = const['weight']
    length = const['length']

    if output_var == 'length':
        wstep = const['weight_step']  # kg
        lstep = const['length_step']  # l
        data = data * wstep / lstep
        xvar = length
    else:
        xvar = weight

    comnames = extract_community_names(const)

    ax = plt.gca()
    l = []
    for icom in data['c'].values:
        color = cmap(float(icom) / len(data['c']))
        lll = ax.plot(xvar.isel(c=icom), data.isel(c=icom).T, color=color, label=comnames[icom], **kwargs)
        l.append(lll[0])
    return l

def set_plot_lim():

    '''
    Automatically sets the plot limits.
    It loops over all the scatter plots and
    line plots and extracts the bounding plot

    '''

    ymin = np.Inf
    xmin = np.Inf

    ymax = np.NINF
    xmax = np.NINF

    # Loop over all the collections, i.e. over all the scatter
    # plots
    for coll in plt.gca().collections:

        path = coll.get_offsets()

        xmin = np.min([path[:, 0].min(), xmin])
        ymin = np.min([path[:, 1].min(), ymin])
        xmax = np.max([path[:, 0].max(), xmax])
        ymax = np.max([path[:, 1].max(), ymax])

    # Loop over all the lines
    for l in plt.gca().get_lines():

        path = np.array([l.get_xdata(), l.get_ydata()]).T

        xmin = np.min([path[:, 0].min(), xmin])
        ymin = np.min([path[:, 1].min(), ymin])
        xmax = np.max([path[:, 0].max(), xmax])
        ymax = np.max([path[:, 1].max(), ymax])

    plt.gca().set_ylim(ymin, ymax)
    plt.gca().set_xlim(xmin, xmax)

# if __name__ == '__main__':
#
#     from extract import extract_time_means, extract_ltl_data
#     import xarray as xr
#
#     config = apecosm.conf.read_config('/home/nbarrier/Modeles/apecosm/svn-apecosm/trunk/tools/config/gyre/oope.conf')
#     dirin = '../../../doc_user/data/'
#
#     meshfile = '%s/mesh_mask.nc' % dirin
#     domain = 'benguela'
#
#     test = xr.open_dataset('test_ts.nc')
#     test = test.mean(dim='time')
#
#     file_pattern = '%s/PISCES*nc' % dirin
#     dataPHY2 = extract_ltl_data(file_pattern, 'PHY2', meshfile, domain)
#     dataZOO = extract_ltl_data(file_pattern, 'ZOO', meshfile, domain)
#     dataZOO2 = extract_ltl_data(file_pattern, 'ZOO2', meshfile, domain)
#     dataGOC = extract_ltl_data(file_pattern, 'GOC', meshfile, domain)
#
#     [dataPHY2, dataZOO, dataZOO2, dataGOC] = map(extract_time_means, [dataPHY2, dataZOO, dataZOO2, dataGOC])
#
#     output_var = 'weight'
#
#     L = [10e-6, 100e-6]
#     lVec2dPHY2, rhoLPHY2 = compute_spectra_ltl(dataPHY2['PHY2'], L, output_var=output_var)
#
#     L = [20.e-6, 200.e-6]
#     lVec2dZOO, rhoLZOO = compute_spectra_ltl(dataZOO['ZOO'], L, output_var=output_var)
#
#     L = [200.e-6, 2000.e-6]
#     lVec2dZOO2, rhoLZOO2 = compute_spectra_ltl(dataZOO2['ZOO2'], L, output_var=output_var)
#
#     L = [100e-6, 50000.e-6]
#     lVec2dGOC, rhoLGOC = compute_spectra_ltl(dataGOC['GOC'], L, output_var=output_var)
#
#     plt.figure()
#     ax = plt.gca()
#     plot_oope_spectra(test, output_var=output_var, config=config)
#
#     ax.scatter(lVec2dPHY2, rhoLPHY2, label='PHY2')
#     ax.scatter(lVec2dZOO, rhoLZOO, label='ZOO')
#     ax.scatter(lVec2dZOO2, rhoLZOO2, label='ZOO2')
#     ax.scatter(lVec2dGOC, rhoLGOC, label='GOC')
#     set_plot_lim()
#     ax.set_yscale("log")
#     ax.set_xscale("log")
#     plt.legend()
#
#     plt.savefig('size_spectra_%s_%s.png' % (domain, output_var), bbox_inches='tight')
