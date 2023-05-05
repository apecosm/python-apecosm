
=================================
Calculations
=================================

.. ipython:: python
    :suppress:

    import os
    import apecosm
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

Using the different variables defined in :numref:`data_extraction`, there is different diagnostics available in the Apecosm python package.

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

In this case, the integration is performed along all the size-classes. It is also possible to provide
length boundaries (**in cm**), using the ``lmin`` and ``lmax`` dimensions. For biomass between 0 and 3cm :

.. ipython:: python

    integrated_biomass_0_3 = apecosm.extract_oope_size_integration(spatial_integral, const, lmax=3)
    integrated_biomass_0_3

For biomass between 3 and 20 cm

.. ipython:: python

    integrated_biomass_3_20 = apecosm.extract_oope_size_integration(spatial_integral, const, lmin=3, lmax=20)
    integrated_biomass_3_20

For biomass greater than 20 cm:

.. ipython:: python

    integrated_biomass_20_inf = apecosm.extract_oope_size_integration(spatial_integral, const, lmin=20)
    integrated_biomass_20_inf

.. danger::

    Size-integration must be applied to variables whose units are in :math:`kg^{-1}`, like ``OOPE``

**********************************************************
Computation of mean length
**********************************************************

The :py:func:`apecosm.extract_mean_size` computes the mean length or weight over a given area. It takes as argument the output
of the :py:func:`apecosm.extract_oope_data` function applied on OOPE.

.. math::

    L_{mean}(t, c) = \dfrac{\sum_{w}  OOPE(t, c, w) \times \Delta W(c, w) \times L(c, w)}{\sum_{w}  OOPE(t, c, w) \times \Delta W(c, w)}

.. math::

    W_{mean}(t, c) = \dfrac{\sum_{w}  OOPE(t, c, w) \times \Delta W(c, w) \times W(c, w)}{\sum_{w}  OOPE(t, c, w) \times \Delta W(c, w)}

To compute the mean length over the entire basin:

.. ipython:: python

    com_mean_length = apecosm.extract_mean_size(spatial_integral, const, 'length')
    com_mean_length

To compute the mean weight:

.. ipython:: python

    com_mean_weight = apecosm.extract_mean_size(spatial_integral, const, 'weight')
    com_mean_weight

To compute the mean length over a given basin, such as the one defined in :numref:`spatial_inte`, the argument
must be the integral over this given region:

.. ipython:: python

    com_reg_mean_length = apecosm.extract_mean_size(regional_spatial_integral, const, 'length')
    com_reg_mean_length

Note that the :py:func:`apecosm.extract_mean_size` returns the mean for each community. The :py:func:`compute_community_mean` allows to average
over the communities

.. ipython:: python

    mean_length = apecosm.compute_community_mean(com_mean_length)
    mean_length