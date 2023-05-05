=================================
Time averaging
=================================

.. ipython:: python
    :suppress:

    import apecosm

    data = apecosm.open_apecosm_data('data/apecosm')
    data

There is also the possibility to compute time averages. This is done by using
the :py:func:`apecosm.extract_time_means` function. It allows the possibily to compute either
full time average, yearly, seasonal or monthly averages.

**********************************************************
Full average
**********************************************************

For the time-average over the entire simulation:

.. ipython:: python

    data_temporal_mean = apecosm.extract_time_means(data)
    data_temporal_mean

**********************************************************
Yearly average
**********************************************************

To compute the yearly mean, the function must be called with a ``year`` argument:

.. ipython:: python

    data_yearly_mean = apecosm.extract_time_means(data, 'year')
    data_yearly_mean