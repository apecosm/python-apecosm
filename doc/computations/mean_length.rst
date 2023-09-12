

**********************************************************
Computation of mean length
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

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

    spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_mean = spatial_mean.compute()

    regional_spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh, domain)
    regional_spatial_mean = regional_spatial_mean.compute()

    spatial_integral = apecosm.spatial_mean_to_integral(spatial_mean)
    regional_spatial_integral = apecosm.spatial_mean_to_integral(regional_spatial_mean)

The mean length of each community can be computed by using a weighted mean,
in which the weight is provided by the biomass in each grid cell:

.. math::

    L_{mean}(t, c) = \dfrac
    {\sum_{y, x, w}  OOPE(t, y, x, c, w) \times \Delta W(c, w) \times dS(x, y) \times L(c, w)}
    {\sum_{y, x, w}  OOPE(t, y, x, c, w) \times \Delta W(c, w) \times dS(x, y)}

Since :math:`L(c,w)` does not depend on :math:`x` and :math:`y`, it
can be rewritten as follows:

.. math::
    :label: size_mean

    L_{mean}(t, c) =
    \dfrac
    {\sum_w L(c, w) \times \Delta W(c, w) \left[{\sum_{y, x} OOPE(t, y, x, c, w)  \times dS(x, y)}\right]}
    {\sum_w \Delta W(c, w) \left[{\sum_{y, x} OOPE(t, y, x, c, w)  \times dS(x, y)}\right]}

We notice that the values in brackets are the outputs of the successive calls
of :py:func:`apecosm.extract_oope_data` and
:py:func:`apecosm.spatial_mean_to_integral`. Therefore, the mean length
over a given domain can be computed by first computing the spatial integral:

.. ipython:: python

    spatial_mean = apecosm.extract_oope_data(data['OOPE'], mesh)
    spatial_integral = apecosm.spatial_mean_to_integral(spatial_mean)
    spatial_integral

When this is done, the integration of :eq:`size_mean` is achieved
by using the :py:func:`apecosm.extract_mean_size` function as follows:

.. ipython:: python

    com_mean_length = apecosm.extract_mean_size(spatial_integral, const, 'length')
    com_mean_length

Note that the :py:func:`apecosm.extract_mean_size` returns the mean
for each community. The :py:func:`apecosm.compute_community_mean` allows
to average over the communities, hence returning an array that only depends
on time:

.. ipython:: python

    mean_length = apecosm.compute_community_mean(com_mean_length)
    mean_length

Finally, the mean length time-series can be plotted as follows:

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

    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_length.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_length.pdf'), bbox_inches='tight')
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
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_weight.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_weight.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_weight.*
    :align: center

    Mean weight

To compute the mean length over a given basin,
such as the one defined in :numref:`spatial_inte`, the argument
must be the integral over this given region:

.. ipython:: python

    com_reg_mean_length = apecosm.extract_mean_size(regional_spatial_integral, const, 'length')
    com_reg_mean_length
