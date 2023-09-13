**********************************************************
Quadmesh plots
**********************************************************

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    import apecosm
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np

    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain = domain_ds['domain_1']

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

The :py:func:`apecosm.plot_pcolor_plot` allows to easily draw 2D maps.
For instance, in order to plot the total biomass, we first integrate
over the full size-spectra, in order to get the biomass:

.. ipython:: python

    biomass = apecosm.extract_oope_size_integration(data['OOPE'], const)
    biomass

.. ipython:: python
    :suppress:

    biomass = biomass.compute()


Now, we sum the biomass over all  the communities:

.. ipython:: python

    tot_biomass = biomass.sum(dim='c')
    tot_biomass

.. ipython:: python
    :suppress:

    tot_biomass = tot_biomass.compute()

Then, we compute the time-average as as follows:

.. ipython:: python

    tavg_tot_biomass = apecosm.extract_time_means(tot_biomass)
    tavg_tot_biomass

Now that we have a DataArray with dimensions ``x`` and ``y``, we
can call the :py:func:`apecosm.plot_pcolor_plot` function:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    test = tavg_tot_biomass.where(tavg_tot_biomass != 0)
    cs = apecosm.plot_pcolor_map(test, mesh)
    cb = plt.colorbar(cs)
    cb.set_label('Biomass density (J/m2)')


.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_raw.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_raw.pdf'), bbox_inches='tight')
    plt.close(fig)


.. figure::  _static/pcolor_raw.*
    :align: center

    Example of a pcolor map drawn with the Apecosm
    package.

Note that in this example, the map is not projected. Therefore,
X and Y axis show the cell indexes, not the real coordinates.
In order to draw the outputs on a projected map,
a projected axes must first be initialized, as follows:

.. ipython:: python

    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    cs = apecosm.plot_pcolor_map(tavg_tot_biomass, mesh, draw_land=True)
    cb = plt.colorbar(cs, shrink=0.5, location='bottom')
    cb.set_label('Biomass density (J/m2)')


.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_projected.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_projected.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/pcolor_projected.*
    :align: center

    Example of a **projected** pcolor map drawn with the Apecosm
    package.

The function will automatically detect that the current axes is
a ``Geoaxes`` object. Consequently, it will manage the mapping.
Note that the :py:func:`plot_pcolor_map` function can take
all the arguments of the `pcolormesh` function. For instance,
the colormap normalization can be changed.

To plot the biomass in log scale, we can first extract the minimum
and maximum value to plot:

.. ipython:: python

    import matplotlib.colors as color
    temp = tavg_tot_biomass.values
    temp = np.ma.masked_where((temp == 0), temp)
    vmin = temp.min()
    vmax = temp.max()
    vmin, vmax

Then, we can define the log normalization of the colormap as follows:

.. ipython:: python

    norm = color.LogNorm(vmin=vmin, vmax=vmax)
    norm

Finally, we can plot the map as follows:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    cs = apecosm.plot_pcolor_map(tavg_tot_biomass, mesh, draw_land=True, norm=norm, cmap='jet')
    cb = plt.colorbar(cs, shrink=0.5, location='bottom')
    cb.set_label('Biomass density (J/m2)')

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_projected_log.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'pcolor_projected_log.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/pcolor_projected_log.*
    :align: center

    Example of a **projected** pcolor map drawn with the Apecosm
    package, with modified colormap and normalization
