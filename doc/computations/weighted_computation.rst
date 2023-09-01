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
(functional response, growth rate or diet matrix), must be done by computed a weighted average, where
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

    regional_mean_repfunct = apecosm.extract_weighted_data(data, const, mesh, 'repfonct_day', mask_dom=domain)
    regional_mean_repfunct