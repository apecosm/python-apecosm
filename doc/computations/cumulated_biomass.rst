**********************************************************
Cumulated biomass
**********************************************************

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    import apecosm
    import xarray as xr
    import matplotlib.pyplot as plt

    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain = domain_ds['domain_1']

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

    spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_integral = spatial_integral.compute()

The cumulated proportion of fish biomass can be computed by using the
:py:func:`compute_size_cumprop` function. It returns:

.. math::

    B_{cum}(t, c, w_0) = \dfrac{\sum_{w=0}^{w_0} B(t, c, w) \times \Delta W (c, w)}{\sum_{w=0}^{w_{max}} B(t, c, w) \times \Delta W (c, w)}

This function is called as follows:

.. ipython:: python

    com_cum_biom = apecosm.compute_size_cumprop(spatial_integral, const)
    com_cum_biom

This function returns the cumulated biomass for each community.

.. danger::

    It is not possible to obtain the cumulated biomass
    including all the community, since it supposes that
    all the communities have the same size-classes, which
    is not always the case. However, it may be possible to
    compute it manually.


We can now draw the cumulated biomass proportion averaged
over the entire simulation period. First,
we extract the temporal mean of the cumulated biomass.

.. ipython:: python

    mean_com_cum_biom = apecosm.extract_time_means(com_cum_biom)

Now we can draw the cumulated biomass.

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        plt.fill_between(const['length'].isel(c=c), 0, mean_com_cum_biom.isel(c=c), color='lightgray')
        plt.plot(const['length'].isel(c=c), mean_com_cum_biom.isel(c=c), color='black')
        ax.set_title('Cumulated proportion, c = %d' %c)
        ax.set_ylabel('%')
        ax.set_ylim(0, 100)
        ax.set_xscale('log')
        ax.grid(True)

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'computations', '_static', 'cumulated_biomass.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'cumulated_biomass.pdf'), bbox_inches='tight')

.. figure::  _static/cumulated_biomass.*
    :align: center

    Mean cumulated biomass