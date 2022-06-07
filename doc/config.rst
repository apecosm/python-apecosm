
.. _configuration:

=================================
Reading configuration files
=================================

Apecosm configuration files can be read by using the :py:func:`apecosm.read_config` function. It reads the configuration file recursively. The program tries to convert the parameters into int, then into float. If both conversion fails, the parameter is kept as a string.

.. literalinclude::  _static/example/read_config.py

.. program-output:: python _static/example/read_config.py

    

