
.. _opening_extraction:

=================================
Opening data
=================================

**********************************************************
Opening the mesh file
**********************************************************

The Apecosm package provides some tools to open the files
from an Apecosm simulation. First, the modules are loaded:

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt

.. ipython:: python

    import os
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import matplotlib.pyplot as plt
    import apecosm

Then, the mesh file containing the grid informations is loaded:

.. ipython:: python

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

The :py:func:`apecosm.open_mesh_mask` opens the file and
remove an eventual unsused time dimension. Dimension names can be
replaced using the `replace_dims` argument, which is a dictionary
with keys being the current dimension names and values being the new names:

.. code-block:: python

    mesh = apecosm.open_mesh_mask(mesh_file, replace_dims={'x': 'X', 'y': 'Y'})
    mesh


**********************************************************
Opening the Apecosm constant file
**********************************************************

Then, the Apecosm constant file, which contains
biological constants (length, weight, etc.) are extracted
as follows:

.. ipython:: python

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

Note that in this case, **a folder** is provided as argument.
This folder must contain the full Apecosm outputs. This
function also includes a `replace_dims` argument to eventually
change the name of the dimensions.

**********************************************************
Opening the Apecosm data files
**********************************************************

Finally, the Apecosm data are loaded as follows:

.. ipython:: python

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

This function will read all the available outputs
in the folder. In this example, the available
outputs are:

-  ``OOPE``, i.e. biomass density.
-  ``community_diet_values``, i.e. diet matrix
-  ``gamma1``, i.e. the growth rate
-  ``mort_day``, i.e. the daily mortality rate
-  ``repfonct_day``, i.e. the functional response

To access one variable:

.. ipython:: python

    data['OOPE']

In order to facilitate the reading of
multiple files, the arguments of the
:py:func:`xarray.open_mfdataset` function
can be used in the :py:func:`open_apecosm_data` one.

For instance in case of very heavy simulations
(global simulations for instance), chunking options used to paralellize
the processing can be provided as follows:

.. ipython:: python

    data_chunked = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'),  chunks={'time': 1, 'x': 50, 'y': 50})
    data_chunked

In this case, the chunk size is now `(1, 50, 50, 5, 100)`, while it
was `(12, 108, 163, 5, 100)` in the above.

.. danger::

    The ``const``, ``mesh`` and ``data`` objects must have the same dimension names. If it is not the case, use the ``replace_dims`` arguments
    to rename the dimensions. Generally expected dimension names
    are ``time``, ``y``, ``x``, ``c``, ``w``.


**********************************************************
Opening the Pisces data files
**********************************************************

The :py:func:`apecosm.open_ltl_data` function
extracts NEMO/Pisces data files:

.. ipython:: python

    ltl_data = apecosm.open_ltl_data(os.path.join('doc', 'data', 'pisces'),
                                    replace_dims={'olevel': 'z'})
    ltl_data

The ``replace_dims`` arguments allows to replace
dimension names, in order to make the name consistent
with the dimensions in the mesh file. In this case,
the `olevel` variable is replaced by `z`.

As for :py:func:`open_apecosm_data`, arguments of
the :py:func:`xarray.open_mfdataset` function can be included in the
:py:func:`apecosm.open_ltl_data` one.