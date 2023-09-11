
.. _configuration:

=================================
Reading configuration files
=================================

Apecosm configuration files can be read by using the :py:func:`apecosm.read_config` function. It reads the configuration file recursively. The program tries to convert the parameters into int, then into float. If both conversion fails, the parameter is kept as a string.

.. ipython:: python
    :suppress:

    import os
    import sys
    import apecosm

.. ipython:: python

    config = apecosm.read_config(os.path.join('doc', 'data', 'config', 'oope.conf'))
    config.keys()

.. ipython:: python

    config['grid.mask.var.e2v']
