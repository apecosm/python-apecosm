
.. _opening_extraction:

=================================
Opening data
=================================

In this section, functions that allow to open
the NEMO/Pisces grid and data files and the Apecosm
output files are described

**********************************************************
Opening the mesh file
**********************************************************

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt

The NEMO/Pisces grid file is required to analyse Apecosm outputs, since
it contains usefull variables such as longitudes and latitudes of the grid
points, surface of the grid cells, land-sea mask, etc.

The mesh mask is opened by using the :py:func:`apecosm.open_mesh_mask`
function:

.. ipython:: python

    import os
    import apecosm

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

The :py:func:`apecosm.open_mesh_mask` opens the file and
removes an eventual time dimension.

Dimension names can be
replaced using the `replace_dims` argument, which is a dictionary
with keys being the current dimension names and values being the new names:

.. code-block:: python

    mesh = apecosm.open_mesh_mask(mesh_file, replace_dims={'x': 'X', 'y': 'Y'})
    mesh

.. warning::

    Dimensions should be ``z``, ``y`` and ``x``


**********************************************************
Opening the Apecosm constant file
**********************************************************

Another import file is the Apecosm constant file, which
contains biological constants (length, weight, etc.)
that will be used to perform size-integration for instance.

It is openened by using the :py:func:`apecosm.open_constants`
function:

.. ipython:: python

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

The argument of this fonction is **a folder** that points to the
location of the Apecosm outputs. This
function also includes a `replace_dims` argument to eventually
change the name of the dimensions.

.. warning::

    Dimensions should be ``c`` (community), ``w`` (size-class),
    ``y`` and ``x``

**********************************************************
Opening the Apecosm data files
**********************************************************

Apecosm data are opened using the :py:func:`apecosm.open_apecosm_data`.

.. ipython:: python

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

This function will read all the available outputs
in the folder, except the constant file. In this example, the available
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

    data_chunked = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'),
                                             chunks={'time': 1, 'x': 50, 'y': 50})
    data_chunked

In this case, the chunk size is now `(1, 50, 50, 5, 100)`, while it
was `(12, 108, 163, 5, 100)` in the above.

.. danger::

    The ``const``, ``mesh`` and ``data`` objects must have the same dimension names.
    If it is not the case, use the ``replace_dims`` arguments
    to rename the dimensions. Dimension names are expected
    to be ``time``, ``y``, ``x``, ``c``, ``w``.


**********************************************************
Opening the NEMO/Pisces data files
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

.. danger::

    Dimension names are expected to be ``z``, ``y``, ``x``.
