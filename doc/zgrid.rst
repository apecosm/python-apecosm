
*********************************
Vertical grid
*********************************

The extraction of the Apecosm vertical grid is achieved by using the :py:func:`apecosm.read_ape_grid` function. This function reads the :samp:`.txt` file, which provides the depth of the Apecosm vertical points, and extracts the :samp:`depth` and :samp:`deltaz` functions, as done in the :samp:`Apecosm` model.

.. literalinclude::  _static/example/extract_zgrid.py
    :lines: 1, 3-4, 8-10

In the new version of Apecosm, partial steps have been implemented; the depth and width of the last ocean cells may vary depending on the local bathymetry, as done in most OGCMS. As a consequence, the :samp:`deltaz` and :samp:`depth` variables are no more 1D but 3D. The application of partial step is done by using the :py:func:`apecosm.partial_step_ape` function as follows:

.. literalinclude::  _static/example/extract_zgrid.py
    :lines: 1, 3-5, 12-18

.. ipython:: python
    :suppress:

    import os
    import subprocess
    cwd = os.getcwd()
    fpath = "_static/example/extract_zgrid.py"
    with open(fpath) as f:
        with open(os.devnull, "w") as DEVNULL:
            subprocess.call(["python", fpath])

.. program-output:: python _static/example/extract_zgrid.py

In order to have an insight on the vertical structure at a given point, the :py:func:`apecosm.plot_grid_nemo_ape` function can be drawn. It shows, side by side, the vertical structure of the Apecosm and of the input (NEMO) grid. The result is shown below.

.. literalinclude::  _static/example/extract_zgrid.py
    :lines: 1-7, 19-

.. figure:: _static/example/ape_vertical_grid.*
   :align: center

   Output figure of the :py:func:`plot_grid_nemo_ape` function.

