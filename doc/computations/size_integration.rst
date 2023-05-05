.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import xarray as xr
    import matplotlib.pyplot as plt

    domain_ds = xr.open_dataset('data/domains.nc')
    domain = domain_ds['domain_1']

    mesh_file = 'data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    const = apecosm.open_constants('data/apecosm/')
    const

    data = apecosm.open_apecosm_data('data/apecosm')
    data

    spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_integral = spatial_integral.compute()

    regional_spatial_integral = apecosm.extract_oope_data(data['OOPE'], mesh, domain)
    regional_spatial_integral = regional_spatial_integral.compute()


**********************************************************
Size-integration
**********************************************************

The :py:func:`apecosm.extract_oope_size_integration` integrates the biomass along the size-dimension.

.. math::

    B(t, y, x, c) = \sum_{w}  OOPE(t, y, x, c, w) \times \Delta W(c, w)


.. ipython:: python

    biomass = apecosm.extract_oope_size_integration(data['OOPE'], const)
    biomass

The function can also be called on spatially integrated biomass density:

.. math::

    B(t, c) = \sum_{w}  OOPE(t, c, w) \times \Delta W(c, w)

.. ipython:: python

    integrated_biomass = apecosm.extract_oope_size_integration(spatial_integral, const)
    integrated_biomass

.. ipython:: python
    :suppress:

    integrated_biomass = integrated_biomass.compute()

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        integrated_biomass.isel(c=c).plot()
        ax.set_title('Integrated biomass, c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/integrated_biomass.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/integrated_biomass.pdf', bbox_inches='tight')
    plt.close(fig)


.. figure::  _static/integrated_biomass.*
    :align: center

    Integrated biomass


In this case, the integration is performed along all the size-classes. It is also possible to provide
length boundaries (**in cm**), using the ``lmin`` and ``lmax`` dimensions. For biomass between 0 and 3cm :

.. ipython:: python

    integrated_biomass_0_3 = apecosm.extract_oope_size_integration(spatial_integral,
                                                                   const, lmax=3)
    integrated_biomass_0_3

.. ipython:: python
    :suppress:

    integrated_biomass_0_3 = integrated_biomass_0_3.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        integrated_biomass_0_3.isel(c=c).plot()
        ax.set_title('0-3cm biomass, c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig('computations/_static/integrated_biomass_0_3.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/integrated_biomass_0_3.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/integrated_biomass_0_3.*
    :align: center

    Integrated biomass between 0 and 3cm

For biomass between 3 and 20 cm

.. ipython:: python

    integrated_biomass_3_20 = apecosm.extract_oope_size_integration(spatial_integral,
                                                                    const, lmin=3, lmax=20)
    integrated_biomass_3_20

.. ipython:: python
    :suppress:

    integrated_biomass_3_20 = integrated_biomass_3_20.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        integrated_biomass_3_20.isel(c=c).plot()
        ax.set_title('3-20cm biomass, c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig('computations/_static/integrated_biomass_3_20.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/integrated_biomass_3_20.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/integrated_biomass_3_20.*
    :align: center

    Integrated biomass between 3 and 20cm

For biomass greater than 20 cm:

.. ipython:: python

    integrated_biomass_20_inf = apecosm.extract_oope_size_integration(spatial_integral,
                                                                      const, lmin=20)
    integrated_biomass_20_inf

.. ipython:: python
    :suppress:

    integrated_biomass_20_inf = integrated_biomass_20_inf.compute()

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        integrated_biomass_20_inf.isel(c=c).plot()
        ax.set_title('>20cm biomass, c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')

    plt.savefig('computations/_static/integrated_biomass_20_inf.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/integrated_biomass_20_inf.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/integrated_biomass_20_inf.*
    :align: center

    Integrated biomass > 20cm

.. danger::

    Size-integration must be applied to variables whose units are in :math:`kg^{-1}`, like ``OOPE``