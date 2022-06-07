
=================================
Apecosm grid
=================================

****************************************
Weight grid
****************************************

In order to convert OOPE (:math:`J.kg^{-1}.m^{-2}`) into a energy (:math:`J.m^{-2}`), the user needs to extract the Apecosm weight grid (which contains the :math:`\Delta W` values). This is done by using the
:py:func:`apecosm.extract_weight_grid` function as follows:

.. literalinclude::  _static/example/extract_wstep.py

.. program-output:: python _static/example/extract_wstep.py

    

