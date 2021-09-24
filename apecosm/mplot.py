'''
Module that contains functions for plotting
OOPE maps. The Ngl module is needed. It
can be used by using a virtual environment
(see https://www.pyngl.ucar.edu/)
'''

from __future__ import print_function
import sys
import os.path
import numpy as np
import xarray as xr
import apecosm.extract
import apecosm.misc as misc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.feature as cfeature
plt.rcParams['text.usetex'] = False

def plot_oope_map(data, weight_step, length, lonf, latf, figname, size_class=None, projection=None, percentage=1, features=None, figargs={}):

    ''' Draws 2D OOPE maps.

    :param xarray.Dataset data: 2D OOPE array. Dims must be (y, x, comm, size)
    :param numpy.array weight_step: Weight step array.
    :param numpy.array length: Length array (cm).
    :param numpy.array lonf: Longitude of the ``F`` points (variable `glamf` of grid files)
    :param numpy.array latf: Latitude of the ``F`` points (variable `gphif` of grid files)
    :param str figname: Name of the figure file (must end by .png or .pdf)
    :param list size_class: Size classes to output (in cm)
    :param float percentage: percentage used to saturate colorbar from percentile.
     Colorbar is saturated from values of the (X) and (100 - X) percentile.

    :return: None
    
    '''

    if weight_step.ndim == 1:
        weight_step = weight_step[np.newaxis, :] # comm, w

    if size_class is None:
        size_class = [1e-3, 1e-2, 1e-1, 1]

    # sort size class in ascending order, and add 0 and infinity as size bounds
    size_class = np.sort(size_class)

    if size_class[0] != 0:
        size_class = np.concatenate(([0], size_class), axis=0)
    if size_class[-1] != np.Inf:
        size_class = np.concatenate((size_class, [np.Inf]), axis=0)

    # Check that the OOPE dataset has 4 dimensions (i.e. no time dimension)
    ndims = len(data['OOPE'].dims)

    if ndims != 4:
        message = 'Data must have dimensions of size (lat, lon, comm, wei)'
        print(message)
        sys.exit(0)

    oope = data['OOPE'].to_masked_array()  # lat, lon, comm, wei
    oope = np.ma.transpose(oope (2, 3, 0, 1)) # comm, weight, lat, lon
    oope = oope * weight_step[:, :, np.newaxis, np.newaxis]  # multiply by weight step

    comm = data['community'][:].values.astype(np.int)

    comm_string = misc.extract_community_names(data)

    if not(figname.endswith('pdf')):
        message = 'Figure name should end with pdf'
        print(message)
        sys.exit(0)

    if(projection is None):
        projection = ccrs.PlateCarree()

    with PdfPages(figname) as pdf:

        # Loop over communities
        for icom in comm:
            # Loop over size classes

            if length.ndim ==1:
                ltemp = length
            else:
                ltemp = length[icom]

            for isize in range(0, len(size_class) - 1):

                # Extract sizes comprised between the size class bound
                iw = np.nonzero((ltemp >= size_class[isize]) & (ltemp < size_class[isize + 1]))[0]
                if iw.size == 0:
                    continue

                # Integrate OOPE for the given community and given size class
                temp = data[icom, iw, :, :] # weight, :, :
                temp = np.ma.sum(temp, axis=0)
                
                # Finds the colorbar limits
                cmin, cmax = misc.find_percentile(temp, percentage=1)
    
                plt.figure(**figargs)

                ax = plt.axes(projection=projection)
                
                cs = plt.pcolormesh(lonf, latf, temp[1:, 1:], transform=ccrs.PlateCarree())
                cs.set_clim(cmin, cmax)
                cb = plt.colorbar(cs, orientation='horizontal')
                cb.set_label(r"OOPE ($J.m^{-2}$)")

                # add title
                title = r'Community=%s, L = [%.2E cm, %.2E cm[' % (comm_string[icom], size_class[isize], size_class[isize + 1])
                plt.title(title)

                csorder = cs.get_zorder() + 1
                if features is None: 
                    ax.add_feature(cfeature.LAND, zorder=csorder)
                    ax.add_feature(cfeature.COASTLINE, zorder=csorder + 1, linewidth=0.5)
                else:
                    for feat in features:
                        ax.add_feature(feat, zorder=csorder)
                        csorder += 1

                pdf.savefig()
                plt.close()



def plot_season_oope(file_pattern, figname, percentage=1):

    ''' Plot seasonal means

    :param str file_pattern: File pattern (for instance, "data/\*nc")
    :param str figname: Figure name
    :param str percentage: Percentile for colormap saturation

    :return: None

    '''

    fig_dir = os.path.dirname(figname)
    fig_name = os.path.basename(figname)

    data = xr.open_mfdataset(file_pattern)

    clim = extract.extract_time_means(data, time='season')
    for s in clim['season'].values:

        print('++++++++++++++++ Drawing season %s ' % s)

        temp = clim.sel(season=s)
        outfile = '%s/%s_%s' % (fig_dir, s, fig_name)
        plot_oope_map(temp, outfile, percentage=percentage)

    Ngl.end()


if __name__ == '__main__':

    data = xr.open_dataset('/home/barrier/Work/apecosm/analyse_passive_active/data/OOPE.nc')
    data = data.isel(time=0)

    #plot_oope_map(data, 'test.pdf', size_class=None, percentage=1, projection=ccrs.Mollweide())
    projection = ccrs.Mollweide()
    #projection = None
    plot_oope_map(data, 'test.pdf', size_class=None, percentage=1, projection=projection)



    #plot_season_oope('data/CMIP2_SPIN_OOPE_EMEAN.nc', './OOPE_mean.pdf')
