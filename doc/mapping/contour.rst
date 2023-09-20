**********************************************************
Contour plots
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


The :py:func:`apecosm.plot_contour_plot` allows to easily draw 2D contour
maps. Using the same array as plotted in :numref:`quadmesh`, the contour
plot can be achieved as follows:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    cl = apecosm.plot_contour_map(tavg_tot_biomass, mesh)
    cb = plt.colorbar(cl)
    cb.set_label('Biomass density (J/m2)')


.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_lines_raw.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_lines_raw.pdf'), bbox_inches='tight')
    plt.close(fig)


.. figure::  _static/contour_lines_raw.*
    :align: center

    Example of a contour plot map drawn with the Apecosm
    package.

In order to use filled contours, set the ``filled`` to ``True``:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    cl = apecosm.plot_contour_map(tavg_tot_biomass, mesh, filled=True)
    cb = plt.colorbar(cl)
    cb.set_label('Biomass density (J/m2)')


.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_filled_raw.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_filled_raw.pdf'), bbox_inches='tight')
    plt.close(fig)


.. figure::  _static/contour_filled_raw.*
    :align: center

    Example of a filled contour plot map drawn with the Apecosm
    package.

Note that the :py:func:`apecosm.plot_contour_map` can take all the parameters
of the Matplotlib contour and contourf functions. For instance, to control
the widths, colors and levels of the contour plots:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    cl = apecosm.plot_contour_map(tavg_tot_biomass, mesh, filled=False, draw_land=False,
                                  levels=21, colors='k', linewidths=2)
    cb = plt.colorbar(cl, orientation='horizontal')
    cb.set_label('Biomass density (J/m2)')

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_lines_raw_params.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_lines_raw_params.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/contour_lines_raw_params.*
    :align: center

    Example of a contour plot map drawn with the Apecosm
    package, with the control of the contour parameters.


Note that in these examples, the map is not projected. Therefore,
X and Y axis show the cell indexes, not the real coordinates.

In order to draw the outputs on a projected map,
a projected axes must first be initialized, as follows:


.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    cl = apecosm.plot_contour_map(tavg_tot_biomass, mesh, filled=True, draw_land=True)
    cb = plt.colorbar(cl, orientation='horizontal')
    cb.set_label('Biomass density (J/m2)')

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_filled_projected.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'mapping', '_static', 'contour_filled_projected.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/contour_filled_projected.*
    :align: center

    Example of a projected filled contour plot map drawn with the Apecosm
    package.