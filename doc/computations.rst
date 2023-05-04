
=================================
Calculations
=================================

**********************************************************
Size-integration
**********************************************************

Using the different variables defined in :numref:`data_extraction`, there is different diagnostics available in the Apecosm python package

.. ipython:: python
    :suppress:

    import os
    import apecosm
    import xarray as xr
    import matplotlib.pyplot as plt

    mesh_file = '_static/example/data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    const = apecosm.open_constants('_static/example/data/apecosm/')
    const

    data = apecosm.open_apecosm_data('_static/example/data/apecosm')
    data

    spatial_integral = apecosm.extract_oope_data(data, mesh, const)
    spatial_integral = spatial_integral.compute()
