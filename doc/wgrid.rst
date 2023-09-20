
=================================
Apecosm grid
=================================

****************************************
Weight grid
****************************************

In order to convert OOPE (:math:`J.kg^{-1}.m^{-2}`) into a
energy (:math:`J.m^{-2}`), the user needs to extract the Apecosm
weight grid (which contains the :math:`\Delta W` values). This is done
by using the :py:func:`apecosm.extract_weight_grid` function as follows:

.. ipython:: python
    :suppress:

    import os
    import sys
    sys.path.insert(0, os.path.abspath('../python/'))
    import apecosm

.. ipython:: python

    config = apecosm.read_config('_static/example/data/config/oope.conf')
    wstep, lstep = apecosm.extract_weight_grid(config)

.. ipython:: python

    wstep

.. ipython:: python

    lstep