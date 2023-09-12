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

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

    spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_mean = spatial_mean.compute()

    regional_spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh, domain)
    regional_spatial_mean = regional_spatial_mean.compute()


**********************************************************
Size-integration
**********************************************************

The :py:func:`apecosm.extract_oope_size_integration` integrates the
biomass along the size-dimension.

.. math::

    B(t, y, x, c) = \sum_{w}  OOPE(t, y, x, c, w) \times \Delta W(c, w)


.. ipython:: python

    biomass_maps = apecosm.extract_oope_size_integration(data['OOPE'], const)
    biomass_maps

The function can also be called on spatially averaged biomass density,
computed using the :py:func:`apecosm.extract_oope_data`:

.. math::

    B(t, c) = \sum_{w}  OOPE(t, c, w) \times \Delta W(c, w)

.. ipython:: python

    biomass_timeseries = apecosm.extract_oope_size_integration(spatial_mean, const)
    biomass_timeseries

.. ipython:: python
    :suppress:

    biomass_timeseries = biomass_timeseries.compute()

The biomass time-series can then be plotted as folllows:

.. ipython:: python

    comnames = apecosm.extract_community_names(const)

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(len(comnames)):
        ax = plt.subplot(3, 2, c + 1)
        biomass_timeseries.isel(c=c).plot()
        ax.set_title('Averaged biomass, c = %s' %comnames[c])
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

.. ipython:: python
    :suppress:

    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass.pdf'), bbox_inches='tight')
    plt.close(fig)


.. figure::  _static/mean_biomass.*
    :align: center

    Averaged biomass


In this case, the integration is performed along all the size-classes. It is also possible to provide
length boundaries (**in cm**), using the ``lmin`` and ``lmax`` dimensions. For biomass between 0 and 3cm :

.. ipython:: python

    biomass_timeseries_0_3 = apecosm.extract_oope_size_integration(spatial_mean,
                                                                   const, lmax=3)
    biomass_timeseries_0_3

.. ipython:: python
    :suppress:

    biomass_timeseries_0_3 = biomass_timeseries_0_3.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        biomass_timeseries_0_3.isel(c=c).plot()
        ax.set_title('0-3cm biomass, c = %s' %comnames[c])
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_0_3.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_0_3.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_biomass_0_3.*
    :align: center

    Averaged biomass between 0 and 3cm

For biomass between 3 and 20 cm

.. ipython:: python

    mean_biomass_3_20 = apecosm.extract_oope_size_integration(spatial_mean,
                                                                    const, lmin=3, lmax=20)
    mean_biomass_3_20

.. ipython:: python
    :suppress:

    mean_biomass_3_20 = mean_biomass_3_20.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        mean_biomass_3_20.isel(c=c).plot()
        ax.set_title('3-20cm biomass, c = %s' %comnames[c])
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_3_20.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_3_20.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_biomass_3_20.*
    :align: center

    Integrated biomass between 3 and 20cm

For biomass greater than 20 cm:

.. ipython:: python

    mean_biomass_20_inf = apecosm.extract_oope_size_integration(spatial_mean,
                                                                      const, lmin=20)
    mean_biomass_20_inf

.. ipython:: python
    :suppress:

    mean_biomass_20_inf = mean_biomass_20_inf.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        mean_biomass_20_inf.isel(c=c).plot()
        ax.set_title('>20cm biomass, c = %s' %comnames[c])
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_20_inf.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_biomass_20_inf.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_biomass_20_inf.*
    :align: center

    Integrated biomass > 20cm

.. danger::

    Size-integration must only be applied to ``OOPE``
