
.. _data_extraction:

=================================
Data extraction
=================================

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt
    import os
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import matplotlib.pyplot as plt
    import apecosm

    mesh_file = 'data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)

    const = apecosm.open_constants('data/apecosm/')

    data = apecosm.open_apecosm_data('data/apecosm')

    ltl_data = apecosm.open_ltl_data('data/pisces',
                                    replace_dims={'olevel': 'z'})

**********************************************************
Spatial integration over the entire domain
**********************************************************

Apecosm outputs can be extracted over a given geographical by using the :py:func:`apecosm.extract_oope_data` function.

This function returns:

.. math::

    X_{int}(t, c, w) = \int\limits_{(y, x)\in S} M(y, x) \times X(t, y, x, c, w) \times dS(y, x)

with :math:`S` the domain where data are extracted, :math:`M` the value of the land-sea mask and :math:`dS` the surface
of the :math:`(i, j)` cell, :math:`c` is the community and :math:`w` is the size-class. It is called as follows:

.. ipython:: python

    spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_integral

with the first argument being the DataArray from which we extract the integral.

In order to extract the mean instead of the integral, the :py:func:`apecosm.normalize_data` function need to be called:

.. ipython:: python

    spatial_mean = apecosm.normalize_data(spatial_integral)
    spatial_mean

.. _spatial_inte:

**********************************************************
Spatial integration over the a given subregion
**********************************************************

In addition, there is the possibility to provide a regional mask in order to extract the area over a given region. For instance, if we have a file containing
different domains:

.. ipython:: python

    domain_ds = xr.open_dataset('data/domains.nc')
    domain = domain_ds['domain_1']

.. ipython:: python
    :suppress:

    fig = plt.figure()
    lonf = mesh['glamf']
    latf = mesh['gphif']
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
    domain_ds = xr.open_dataset('data/domains.nc')
    domain = domain_ds['domain_1'] * mesh['tmaskutil']
    ax.pcolormesh(lonf, latf, domain.isel(x=slice(1, None), y=slice(1, None)),
                  transform=ccrs.PlateCarree(), cmap=plt.cm.get_cmap('binary'))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.set_extent([lonf.min(), lonf.max(), latf.min(), latf.max()], crs=ccrs.PlateCarree())
    plt.savefig('_static/domains.jpg', bbox_inches='tight')
    plt.savefig('_static/domains.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/domains.*
    :align: center

    Example of a spatial domain

We can extract the integrated biomass over this domain as follows:

.. ipython:: python

    regional_spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh, domain)
    regional_spatial_integral


.. _extract_ltl:

**********************************************************
Extraction of biogeochemical data
**********************************************************

The 3D extraction of biogeochemical forcing data is achieved by using the :py:func:`apecosm.extract_ltl_data` function as follows:

.. ipython:: python

    spatial_mean_phy2 = apecosm.extract_ltl_data(ltl_data, mesh, 'PHY2')
    spatial_mean_phy2

This function will first vertically **integrate** the LTL biomass (converting from :math:`mmol/m3` into :math:`mmol/m2`). And then
compute the horizontal **average**. This choice has been made to be consistent with Apecosm outputs. Indeed, OOPE is provided as a vertically
integrated biomass.

However, it remains possible to convert the horizontal average into an horizontal integral as follows:

.. ipython:: python

    spatial_integral_phy2 = apecosm.spatial_mean_to_integral(spatial_mean_phy2)
    spatial_integral_phy2
