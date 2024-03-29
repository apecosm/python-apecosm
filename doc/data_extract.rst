
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
    import matplotlib as mpl
    import apecosm

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm'))

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))

    ltl_data = apecosm.open_ltl_data(os.path.join('doc', 'data', 'pisces'),
                                    replace_dims={'olevel': 'z'})

**********************************************************
Spatial average of Apecosm outputs
**********************************************************

Apecosm outputs can be extracted over a given
geographical by using the :py:func:`apecosm.extract_oope_data` function.

This function returns:

.. math::
    :label: oope_mean

    X_{mean}(t, c, w) = \dfrac
    {\int\limits_{(y, x)\in S} X(t, y, x, c, w) \times dS(y, x)}
    {\int\limits_{(y, x)\in S} dS(y, x)}

with :math:`S` the domain where data are extracted, and :math:`dS` the surface
of the :math:`(i, j)` cell, :math:`c` is the community and
:math:`w` is the size-class. It is called as follows:

.. ipython:: python

    spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_mean

with the first argument being the ``DataArray`` from which we extract the mean
and the second argument being the mesh object
(obtained with the :py:func:`apecosm.open_mesh_mask` function).

In order to extract the integral instead of the mean,
the :py:func:`apecosm.spatial_mean_to_integral` function need
to be called. This function multiply the above calculation by
the denominator of :eq:`oope_mean`, which is stored as
the ``horizontal_norm_weight`` attribute

.. ipython:: python

    spatial_integral = apecosm.spatial_mean_to_integral(spatial_mean)
    spatial_integral

In addition, there is the possibility to provide a regional
mask in order to extract the data over a given region. For instance, if we
have a file containing different domains masks:

.. ipython:: python

    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain_ds

We can extract the mean biomass over this domain as follows:

.. ipython:: python
    .. ipython:: python

    regional_spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh, domain_ds['domain_1'])
    regional_spatial_mean

.. ipython:: python
    :suppress:

    fig = plt.figure()
    lonf = mesh['glamf']
    latf = mesh['gphif']
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=180))
    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain = domain_ds['domain_1'] * mesh['tmaskutil']
    ax.pcolormesh(lonf, latf, domain.isel(x=slice(1, None), y=slice(1, None)),
                  transform=ccrs.PlateCarree(), cmap=mpl.colormaps['binary'])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.set_extent([lonf.min(), lonf.max(), latf.min(), latf.max()], crs=ccrs.PlateCarree())
    plt.savefig(os.path.join('doc', '_static', 'domains.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', '_static', 'domains.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/domains.*
    :align: center

    Example of a spatial domain


.. _extract_ltl:

**********************************************************
Spatial average of NEMO/Pisces outputs
**********************************************************

The extraction of 3D biogeochemical forcing data is achieved
by applying the following formula:

.. math::
    :label: pisces_mean

    X_{mean}(t) = \dfrac
    {\int\limits_{(y, x)\in S} \left[\sum_{z=z_{min}}^{z_{max}}X(t, z, y, x) dZ(z)\right]\times dS(y, x)}
    {\int\limits_{(y, x)\in S} dS(y, x)}


It is achieved by using the :py:func:`apecosm.extract_ltl_data`
function:

.. ipython:: python

    spatial_mean_phy2 = apecosm.extract_ltl_data(ltl_data['PHY2'], mesh)
    spatial_mean_phy2

This function will first vertically **integrate** the LTL biomass
(converting from :math:`mmol/m3` into :math:`mmol/m2`, term in
brackets in :eq:`pisces_mean`). Then
the horizontal **average** is computed. This choice has been made to
be consistent with Apecosm outputs. Indeed, OOPE is provided as a vertically
integrated biomass. Therefore, vertical integration need to be performed
on LTL outputs in order to draw the size-spectra.

.. ipython:: python
    :suppress:

    fig = plt.figure()
    spatial_mean_phy2.plot()
    plt.savefig(os.path.join('doc', '_static', 'mean_phy2.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', '_static', 'mean_phy2.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_phy2.*
    :align: center

    Mean concentration of PHY2

However, it remains possible to convert the horizontal average into an
horizontal integral as follows:

.. ipython:: python

    spatial_integral_phy2 = apecosm.spatial_mean_to_integral(spatial_mean_phy2)
    spatial_integral_phy2

There is also the possibility to control the depth at which the
average is performed and the domain used for the averaging.
For instance, to compute the average between 0 and 200m over the
domain defined above:

.. ipython:: python

    spatial_0_200_reg_mean_phy2 = apecosm.extract_ltl_data(ltl_data['PHY2'],
                                                           mesh,
                                                           mask_dom=domain_ds['domain_1'],
                                                           depth_limits=[0, 200])
    spatial_0_200_reg_mean_phy2


.. ipython:: python
    :suppress:

    fig = plt.figure()
    spatial_0_200_reg_mean_phy2.plot()
    plt.savefig(os.path.join('doc', '_static', 'mean_phy2_reg_0_200.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', '_static', 'mean_phy2_reg_0_200.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_phy2_reg_0_200.*
    :align: center

    Mean concentration of PHY2 over the subregion and
    between 0 and 200m
