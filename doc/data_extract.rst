
=================================
Data extraction
=================================

**********************************************************
Spatial extraction of OOPE
**********************************************************

OOPE output data can be extracted over a given geographical polygon by using the :py:func:`apecosm.extract_oope_data` function as follows:

.. literalinclude::  _static/example/extract_data_ape_spatial.py

.. program-output:: python _static/example/extract_data_ape_spatial.py

This function returns:

.. math::

    X = \int\limits_{(i, j)\in S} M(i, j) \times OOPE(i, j) \times dS(i, j)

with :math:`S` the domain where data are extracted, :math:`M` the value of the land-sea mask and :math:`dS` the surface of the :math:`(i, j)` cell.

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

    
