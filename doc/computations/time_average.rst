=================================
Time averaging
=================================

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    import apecosm

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'))
    data

There is the possibility to compute time averages. This is done by using
the :py:func:`apecosm.extract_time_means` function, which allows to compute
either full time average, yearly, seasonal or monthly averages.

**********************************************************
Full time average
**********************************************************

For the time-average over the entire simulation:

.. ipython:: python

    data_temporal_mean = apecosm.extract_time_means(data)
    data_temporal_mean

with ``data`` the output of the :py:func:`apecosm.open_apecosm_data`
function.

In this example, there is no time dimension anymore

**********************************************************
Yearly average
**********************************************************

To compute the yearly mean, the function must be
called with a ``year`` argument:

.. ipython:: python

    data_yearly_mean = apecosm.extract_time_means(data, 'year')
    data_yearly_mean

In this case, the ``time`` dimension has been replaced by a `year` dimension, which
contains only one element.

**********************************************************
Seasonal average
**********************************************************

To compute the seasonal mean, the function must be
called with a ``season`` argument:

.. ipython:: python

    data_yearly_mean = apecosm.extract_time_means(data, 'season')
    data_yearly_mean

In this case, the ``time`` dimension has been replaced by a `season` dimension, which
contains 4 elements. Note that the user does not have the choice of the seasons,
which are ``DJF``, ``JJA`` ``MAM`` and ``SON``

**********************************************************
Monthly average
**********************************************************

To compute the monthly mean, the function must be
called with a ``month`` argument:

.. ipython:: python

    data_yearly_mean = apecosm.extract_time_means(data, 'month')
    data_yearly_mean

In this case, the ``time`` dimension has been replaced by a `month`
dimension, which contains 12 elements.


.. warning::

    The computation of seasonal and monthly averages are available
    only when Apecosm outputs are provided with at least a monthly
    time step.
