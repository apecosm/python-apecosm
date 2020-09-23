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
try:
    import Ngl
except ImportError:
    pass
import apecosm.extract
import apecosm.misc as misc


def plot_oope_map(data, figname, size_class=None, percentage=1):

    ''' Draws 2D OOPE maps.

    :param xarray.Dataset data: 2D OOPE array. Dims must be (y, x, comm, size)
    :param str figname: Name of the figure file (must end by .png or .pdf)
    :param list size_class: Size classes to output (in m)
    :param float percentage: percentage used to saturate colorbar from percentile.
     Colorbar is saturated from values of the (X) and (100 - X) percentile.

    :return: None
    
    '''

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

    # Recover data variables
    length = data['length'].values
    oope = data['OOPE'].values
    lon = data['longitude'][:].values
    lat = data['latitude'][:].values
    comm = data['community'][:].values.astype(np.int)

    # mask oope where land
    oope = np.ma.masked_where(np.isnan(oope), oope)

    comm_string = misc.extract_community_names(data)

    if figname.endswith('png'):
        form = 'png'
    elif figname.endswith('pdf'):
        form = 'pdf'
    else:
        message = 'Figure name should end with png or pdf'
        print(message)
        sys.exit(0)

    # opens the document
    wks = Ngl.open_wks(form, figname)

    # set the document colormap
    # resngl = Ngl.Resources()
    # resngl.wkColorMap = 'precip2_15lev'
    # Ngl.set_values(wks, resngl)

    # Add gray to the workspace
    Ngl.new_color(wks, 0.7, 0.7, 0.7)

    # init the plot resources
    # For a detailed description, see https://www.ncl.ucar.edu/Document/Graphics/Resources/
    res = Ngl.Resources()

    # not necessary, just a good habit
    res.nglDraw = False
    res.nglFrame = False

    # Set map resources.
    res.mpLimitMode = "LatLon"     # limit map via lat/lon
    res.mpMinLatF = lat.min()         # map area
    res.mpMaxLatF = lat.max()         # latitudes
    res.mpMinLonF = lon.min()         # and
    res.mpMaxLonF = lon.max()         # longitudes
    res.mpFillOn = True
    res.mpLandFillColor = "LightGray"
    res.mpOceanFillColor = -1
    res.mpInlandWaterFillColor = "LightBlue"
    res.mpGeophysicalLineThicknessF = 1  # thickness of coastlines

    # coordinates for contour plots
    res.sfXArray = lon
    res.sfYArray = lat

    res.cnFillOn = True  # filled contour
    res.cnLinesOn = False  # no lines
    res.cnLineLabelsOn = False  # no labels
    res.cnInfoLabelOn = False  # no info about contours

    res.cnFillMode = 'CellFill'  # contourf=AreaFill, pcolor="CellFill" or "RasterFill"
    res.lbOrientation = "Horizontal"  # colorbar orientation
    res.lbLabelFontHeightF = 0.012  # colorbar label fontsize
    res.lbTitlePosition = "Bottom"  # position of colorbar title
    res.lbTitleFontHeightF = 0.012  # title font height

    # res.cnFillPalette = "wgne15"
    res.cnFillPalette = "WhiteBlueGreenYellowRed"

    txres = Ngl.Resources()
    txres.txJust = "BottomCenter"
    txres.txFontHeightF = 0.02

    res.cnLevelSelectionMode = 'ExplicitLevels'
    res.cnMaxLevelCount = 41
    res.mpGridAndLimbOn = True

    # Equations are complicated with NCARG
    # see https://www.ncl.ucar.edu/Applications/fcodes.shtml
    res.lbTitleString = "OOPE (J.kg~S~-1~N~.m~S~-2~N~)"  # title of colorbar

    # Loop over communities
    for icom in comm:
        # Loop over size classes
        for isize in xrange(0, len(size_class) - 1):

            # Extract sizes comprised between the size class bound
            iw = np.nonzero((length >= size_class[isize]) & (length < size_class[isize+1]))[0]
            if iw.size == 0:
                continue

            # Integrate OOPE for the given community and given size class
            temp = oope[:, :, icom, iw]
            temp = np.sum(temp, axis=-1)

            # Finds the colorbar limits
            cmin, cmax = misc.find_percentile(temp, percentage=1)

            # draw the contour maps
            # defines the contour
            res.cnLevels = np.linspace(cmin, cmax, res.cnMaxLevelCount)
            mapplot = Ngl.contour_map(wks, temp, res)

            # add title
            title = 'Community=%s, %.2E m <= L < %.2E m' % (comm_string[icom], size_class[isize], size_class[isize + 1])
            Ngl.text_ndc(wks, title, 0.5, 0.85, txres)

            # draws the map
            Ngl.draw(mapplot)

            # add a page to the pdf output
            Ngl.frame(wks)

            # WARNING!!! After calls to this function,
            # Ngl.end() function has to be called!


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

    plot_season_oope('data/CMIP2_SPIN_OOPE_EMEAN.nc', './OOPE_mean.pdf')
