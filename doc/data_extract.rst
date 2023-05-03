
=================================
Data extraction
=================================

**********************************************************
Opening the files
**********************************************************

The Apecosm package provides some tools to open the files from an Apecosm simulation. First, the modules are loaded:

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt

.. ipython:: python

    import os
    import apecosm
    import xarray as xr
    import matplotlib.pyplot as plt

Then, the mesh file containing the grid informations is loaded:

.. ipython:: python

    mesh_file = '_static/example/data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

The :py:func:`apecosm.open_mesh_mask` opens the file and remove an eventual unsused time dimension.

Then, the Apecosm constant file, which contains  biological constants (length, weight, etc.) are extracted as follows:

.. ipython:: python

    const = apecosm.open_constants('_static/example/data/apecosm/')
    const

Note that in this case, **a folder** is provided as argument.

Finally, the Apecosm data are loaded as follows:

.. ipython:: python

    data = apecosm.open_apecosm_data('_static/example/data/apecosm')
    data

In this example, the dataset only contains one variable, `OOPE`, i.e. biomass density.

.. ipython:: python

    data['OOPE']

In case of very heavy simulations (global simulations for instance), you may want to use Dask parallel capacities. To do so, an additionnal argument
can be provided in the :py:func:`apecosm.open_apecosm_data` function:

.. ipython:: python

    xarray_args = {'chunks' : {'time': 1, 'x': 50, 'y': 50}}
    data_chunked = apecosm.open_apecosm_data('_static/example/data/apecosm', **xarray_args)
    data_chunked

In this case, the chunk size is now `(1, 50, 50, 5, 100)`, while it was `(12, 108, 163, 5, 100)` in the above.

.. warning::

    The `const`, `mesh` and `data` objects must have the same dimension names. If it is not the case, use the `replace_dims` arguments
    to rename the dimensions. Commonly accepted dimensions are `time`, `y`, `x`, `c`, `w`.

**********************************************************
Spatial extraction of OOPE
**********************************************************

OOPE output data can be extracted over a given geographical by using the :py:func:`apecosm.extract_oope_data` function as follows:

.. ipython:: python

    spatial_integral = apecosm.extract_oope_data(data, mesh, const)
    spatial_integral = spatial_integral.compute()

This function returns:

.. math::

    X(t, c, w) = \int\limits_{(i, j)\in S} M(i, j) \times OOPE(t, i, j, c, w) \times dS(i, j)

with :math:`S` the domain where data are extracted, :math:`M` the value of the land-sea mask and :math:`dS` the surface
of the :math:`(i, j)` cell, :math:`c` is the community and :math:`w` is the size-class:

.. ipython:: python

    spatial_integral

Note that in this case, the spatial integral is computed. In order to obtain the mean:

.. ipython:: python

    spatial_mean = apecosm.normalize_data(spatial_integral)
    spatial_mean

In addition, there is the possibility to provide a regional mask in order to extract the area over a given region. For instance, if we have a file containing
different domains:

.. code-block:: python

    import xarray as xr
    domain_ds = xr.open_dataset('_static/example/data/domains.nc')
    domain = domain_ds['domain_1'] * mesh['tmaskutil']
    domain.plot()

.. ipython:: python
    :suppress:

    import xarray as xr
    domain_ds = xr.open_dataset('_static/example/data/domains.nc')
    domain = domain_ds['domain_1'] * mesh['tmaskutil']
    domain.plot()
    plt.savefig('_static/domains.jpg')
    plt.savefig('_static/domains.pdf')

.. figure::  _static/domains.*
    :align: center

    Domains example

We can extract the integrated biomass over this domain as follows:

.. ipython:: python

    regional_spatial_integral = apecosm.extract_oope_data(data, mesh, const, domain)
    regional_spatial_integral


**********************************************************
Spatial extraction of biogeochemical data
**********************************************************

The output is provided as a `xarray <http://xarray.pydata.org/en/stable/>`_ Dataset object.

To extraction of biogeochemical forcing data is achieved by using the :py:func:`apecosm.extract_ltl_data` function as follows:

.. literalinclude::  _static/example/extract_data_pisces_spatial.py

.. program-output:: python _static/example/extract_data_pisces_spatial.py

.. note

    In this case, the output data is also an xarray Dataset, however it contains only one dimension since there is no other dimensions than depth, latitude, longitude.

**********************************************************
Time extraction
**********************************************************

There is also the possibility to extract time averages. This is done by using the :py:func:`apecosm.extract_time_means` function. It allows the possibily to compute either full time average, yearly, seasonal or monthly averages.

.. literalinclude::  _static/example/extract_data_ape_temporal.py

.. program-output:: python _static/example/extract_data_ape_temporal.py

.. warning::

    This function takes as argument xarray.Dataset objects. They are either obtained by using the :py:func:`apecosm.extract_oope_data`, :py:func:`apecosm.extract_ltl_data` or
    by using the :py:func:`xarray.open_dataset` functions.
