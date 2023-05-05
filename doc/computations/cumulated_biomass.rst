**********************************************************
Cumulated biomass
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

The cumulated proportion of fish biomass can be computed by using the
:py:func:`compute_size_cumprop` function. It returns:

.. math::

    B_{cum}(t, c, w_0) = \dfrac{\sum_{w=0}^{w_0} B(t, c, w) \times \Delta W (c, w)}{\sum_{w=0}^{w_{max}} B(t, c, w) \times \Delta W (c, w)}

To extract the cumulated biomass for each size-class