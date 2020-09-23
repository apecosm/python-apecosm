''' Module containing some functions related to NetCDF files'''

from __future__ import print_function
from glob import glob
import sys
import os
import xarray as xr
import numpy as np


def rebuild_restart(dirin):

    '''
    Rebuilds a global restart file from individual restart
    files. Usefull if the tile file has been deleted by mistake.

    :param str dirin: Name of the restart directory.

    '''

    filelist = np.sort(glob("%s/*restart*.nc.*" % dirin))

    if filelist.size == 0:
        print("No restart files was found in %s" % dirin)
        print("The program will be stopped")
        sys.exit(os.EX_NOINPUT)

    file_number = 0

    # Loop over all the tiles
    for filename in filelist[:]:

        if file_number == 0:
            # The first file is used to initialize output array
            print("Initializing the output variable from " + filename)
            data = xr.open_dataset(filename)
            output_name = filename.replace('.nc.000', '.nc')
        else:
            # The next files (1, 2, ..., ntile - 1) are used to concatenate
            # the output array
            print("Concatenating the output variable from " + filename)
            temp = xr.open_dataset(filename)   # open the current tile restart
            data = xr.concat([data, temp], 'cell')   # append it to the data variable

        file_number += 1

    # Save the output into a NetCDF file
    print('Reconstructed restart is in %s' % output_name)
    data.to_netcdf(output_name)
