

**********************************************************
Computation of mean length
**********************************************************

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

Note that the :py:func:`apecosm.extract_mean_size` returns the mean for each community. The :py:func:`compute_community_mean` allows to average
over the communities

.. ipython:: python

    mean_length = apecosm.compute_community_mean(com_mean_length)
    mean_length

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        com_mean_length.isel(c=c).plot()
        ax.set_title('Mean length (m), c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    mean_length.plot()
    ax.set_title('Mean length (m), all com.')
    ax.grid(True)

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/mean_length.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/mean_length.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_length.*
    :align: center

    Mean length

To compute the mean weight:

.. ipython:: python

    com_mean_weight = apecosm.extract_mean_size(spatial_integral, const, 'weight')
    com_mean_weight

.. ipython:: python

    mean_weight = apecosm.compute_community_mean(com_mean_weight)
    mean_weight

.. ipython:: python
    :suppress:

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        com_mean_weight.isel(c=c).plot()
        ax.set_title('Mean weight (kg), c = %d' %c)
        ax.grid(True)
    ax = plt.subplot(3, 2, 6)
    mean_weight.plot()
    ax.set_title('Mean weight (kg), all com.')
    ax.grid(True)
    plt.savefig('computations/_static/mean_weight.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/mean_weight.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_weight.*
    :align: center

    Mean weight

To compute the mean length over a given basin, such as the one defined in :numref:`spatial_inte`, the argument
must be the integral over this given region:

.. ipython:: python

    com_reg_mean_length = apecosm.extract_mean_size(regional_spatial_integral, const, 'length')
    com_reg_mean_length