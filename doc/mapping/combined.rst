**********************************************************
Combined
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
    import cartopy.crs as ccrs

    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain = domain_ds['domain_1']

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    ltl_data = apecosm.open_ltl_data(os.path.join('doc', 'data', 'pisces'),
                                     replace_dims={'olevel': 'z'})

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

    biomass = apecosm.extract_oope_size_integration(data['OOPE'], const)
    biomass = biomass.compute()

    tot_biomass = biomass.sum(dim='c')
    tot_biomass

    tot_biomass = tot_biomass.compute()

    tavg_tot_biomass = apecosm.extract_time_means(tot_biomass)
    tavg_tot_biomass = tavg_tot_biomass.compute()

There is the possibility to combine both quadmesh and contour plots.
For instance, if we want to superimpose the sea-surface temperature
on top of the biomass, plotted in a log-scale, we first extract
time-average sea-surface temperature:

.. ipython:: python

    sst = ltl_data['thetao'].isel(z=0)
    sst

.. ipython:: python
    :suppress:

    sst = sst.compute()

.. ipython:: python

     sst_mean = apecosm.extract_time_means(sst)
     sst_mean

.. ipython:: python
    :suppress:

    sst_mean = sst_mean.compute()

Now we prepare the quadmesh plot of biomass, as done in
:numref:`quadmesh`:

.. ipython:: python

    import matplotlib.colors as color
    temp = tavg_tot_biomass.values
    temp = np.ma.masked_where((temp == 0), temp)
    vmin = temp.min()
    vmax = temp.max()

    norm = color.LogNorm(vmin=vmin, vmax=vmax)
    norm

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    cs = apecosm.plot_pcolor_map(tavg_tot_biomass, mesh, draw_land=True, norm=norm, cmap='jet')
    cb = plt.colorbar(cs, shrink=0.5, location='bottom')
    cb.set_label('Biomass density (J/m2)')

    cl = apecosm.plot_contour_map(sst_mean, mesh, filled=False, draw_land=False,
                                  levels=21, colors='k', linewidths=1)
    plt.clabel(cl)

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'combined_contour_pcolor.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'combined_contour_pcolor.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/combined_contour_pcolor.*
    :align: center

    Example of a projected map of SST contours and
    biomass quadmesh plot