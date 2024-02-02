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

************************
Weighted variable means
************************

The computation of the horizontal average of a given variable :math:`V`
(functional response, growth rate or diet matrix), must be done by
computed a weighted average, where
the weight is provided by the biomass:

.. math::

    \overline{V(t, c, w)} = \dfrac
    {\sum_{x,y} V(t, x, y, c, w) \times OOPE(t, x, y, c, w) \times \Delta W(c, w) \times dS(x,y)}
    {\sum_{x,y} OOPE(t, x, y, c, w) \times \Delta W(c, w)  \times dS(x,y)}

where :math:`V` is the variable, :math:`OOPE` is the spatial OOPE,
:math:`\times dS(x,y)` is the weight step and :math:`dS(x,y)` is the surface
of the cell.

In the Apecosm package, this is achieved by
using the :py:func:`apecosm.extract_weighted_data` function. For instance,
to extract the mean functional response over the entire domain:

.. ipython:: python

    mean_repfunct = apecosm.extract_weighted_data(data, const, mesh, 'repfonct_day')
    mean_repfunct

Note that there is also the possibility to compute the mean over
a given subregion as follows:

.. ipython:: python

    regional_mean_repfunct = apecosm.extract_weighted_data(data,
                                                           const,
                                                           mesh,
                                                           'repfonct_day',
                                                           mask_dom=domain)
    regional_mean_repfunct

It is now possible to plot the mean functional response as follows.
First, we compute the time average:

.. ipython:: python

    time_avg_mean_repfunct = apecosm.extract_time_means(mean_repfunct)
    time_avg_mean_repfunct

.. ipython:: python
    :suppress:

    time_avg_mean_repfunct.compute()

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    for c in range(5):
        l = const['length'].isel(c=c)
        ax = plt.subplot(3, 2, c + 1)
        ax.plot(l, time_avg_mean_repfunct.isel(c=c))
        ax.set_xlim(const['length'].min(), const['length'].max())
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_title('Repfonct, c = %d' %c)
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_repfonct.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'mean_repfonct.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/mean_repfonct.*
    :align: center

    Mean functional response
