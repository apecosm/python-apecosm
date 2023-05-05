
.. _data_extraction:

=================================
Data extraction
=================================

**********************************************************
Opening the files
**********************************************************

The Apecosm package provides some tools to open the files from an Apecosm simulation. First, the modules are loaded:

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt

.. ipython:: python

    import os
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import matplotlib.pyplot as plt
    import apecosm

Then, the mesh file containing the grid informations is loaded:

.. ipython:: python

    mesh_file = 'data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

The :py:func:`apecosm.open_mesh_mask` opens the file and remove an eventual unsused time dimension.

Then, the Apecosm constant file, which contains  biological constants (length, weight, etc.) are extracted as follows:

.. ipython:: python

    const = apecosm.open_constants('data/apecosm/')
    const

Note that in this case, **a folder** is provided as argument.

Finally, the Apecosm data are loaded as follows:

.. ipython:: python

    data = apecosm.open_apecosm_data('data/apecosm')
    data

In this example, the dataset only contains one variable, `OOPE`, i.e. biomass density.

.. ipython:: python

    data['OOPE']

In case of very heavy simulations (global simulations for instance), you may want to use Dask parallel capacities. To do so, an additionnal argument
can be provided in the :py:func:`apecosm.open_apecosm_data` function:

.. ipython:: python

    data_chunked = apecosm.open_apecosm_data('data/apecosm',  chunks={'time': 1, 'x': 50, 'y': 50})
    data_chunked

In this case, the chunk size is now `(1, 50, 50, 5, 100)`, while it was `(12, 108, 163, 5, 100)` in the above.

.. warning::

    The ``const``, ``mesh`` and ``data`` objects must have the same dimension names. If it is not the case, use the ``replace_dims`` arguments
    to rename the dimensions. Commonly accepted dimensions are ``time``, ``y``, ``x``, ``c``, ``w``.

.. _spatial_inte:

**********************************************************
Spatial integration
**********************************************************

OOPE output data can be extracted over a given geographical by using the :py:func:`apecosm.extract_oope_data` function as follows:

.. ipython:: python

    spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh)

This function returns:

.. math::

    X(t, c, w) = \int\limits_{(i, j)\in S} M(i, j) \times OOPE(t, i, j, c, w) \times dS(i, j)

with :math:`S` the domain where data are extracted, :math:`M` the value of the land-sea mask and :math:`dS` the surface
of the :math:`(i, j)` cell, :math:`c` is the community and :math:`w` is the size-class:

.. ipython:: python

    spatial_integral

Note that in this case, the spatial integral is computed. In order to obtain the mean:

.. ipython:: python

    spatial_mean = apecosm.normalize_data(spatial_integral)
    spatial_mean

In addition, there is the possibility to provide a regional mask in order to extract the area over a given region. For instance, if we have a file containing
different domains:

.. ipython:: python

    domain_ds = xr.open_dataset('data/domains.nc')
    domain = domain_ds['domain_1']

.. ipython:: python
    :suppress:

    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
    domain_ds = xr.open_dataset('data/domains.nc')
    domain = domain_ds['domain_1'] * mesh['tmaskutil']
    ax.pcolormesh(mesh['glamf'], mesh['gphif'], domain.isel(x=slice(1, None), y=slice(1, None)), transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    plt.savefig('_static/domains.jpg', bbox_inches='tight')
    plt.savefig('_static/domains.pdf', bbox_inches='tight')

.. figure::  _static/domains.*
    :align: center

    Domains example

We can extract the integrated biomass over this domain as follows:

.. ipython:: python

    regional_spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh, domain)
    regional_spatial_integral


**********************************************************
Extraction of biogeochemical data
**********************************************************

The 3D extraction of biogeochemical forcing data is achieved by using the :py:func:`apecosm.extract_ltl_data` function as follows:

.. ipython:: python

    ltl_data = apecosm.open_ltl_data('data/pisces',
                                    replace_dims={'olevel': 'z'},
                                    drop_variables=['bounds_nav_lat',
                                                    'bounds_nav_lon',
                                                    'nav_lat',
                                                    'nav_lon'])
    ltl_data

.. ipython:: python

    spatial_integrated_phy2 = apecosm.extract_ltl_data(ltl_data, mesh, 'PHY2')
    spatial_integrated_phy2

.. note::

    In this case, the output data is also an xarray Dataarray, however it contains only one dimension since there is no other dimensions than depth, latitude, longitude.

As in the above, this computes the 3D integral. If the mean is needed:

.. ipython:: python

    spatial_mean_phy2 = apecosm.normalize_data(spatial_integrated_phy2)
    spatial_mean_phy2

**********************************************************
Time average
**********************************************************

There is also the possibility to compute time averages. This is done by using
the :py:func:`apecosm.extract_time_means` function. It allows the possibily to compute either
full time average, yearly, seasonal or monthly averages.

For the time-average over the entire simulation:

.. ipython:: python

    data_temporal_mean = apecosm.extract_time_means(data)
    data_temporal_mean

To compute the yearly mean, the function must be called with a ``year`` argument:

.. ipython:: python
    :okwarning:

    data_yearly_mean = apecosm.extract_time_means(data, 'year')
    data_yearly_mean
