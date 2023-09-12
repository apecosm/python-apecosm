.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    import apecosm
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib

    domain_ds = xr.open_dataset(os.path.join('doc', 'data', 'domains.nc'))
    domain = domain_ds['domain_1']

    mesh_file = os.path.join('doc', 'data', 'pacific_mesh_mask.nc')
    mesh = apecosm.open_mesh_mask(mesh_file)
    mesh

    const = apecosm.open_constants(os.path.join('doc', 'data', 'apecosm/'))
    const

    data = apecosm.open_apecosm_data(os.path.join('doc', 'data', 'apecosm'),
                                     replace_dims={'community': 'c'})
    data

**********************
Diet calculation
**********************

If the diet outputs are available in the Apecosm
simulations, it is possible to plot the diet matrix. This is
achieved as follows.

First, the spatial mean diet is computed using the
:py:func:`apecosm.extract_weighted_data` function.

.. ipython:: python

    mean_diet = apecosm.extract_weighted_data(data, const, mesh, 'community_diet_values')
    mean_diet

.. ipython:: python
    :suppress:

    mean_diet = mean_diet.compute()

Then, the time average is computed using the
:py:func:`apecosm.extract_time_means` function:

.. ipython:: python

    time_average_mean_diet = apecosm.extract_time_means(mean_diet)
    time_average_mean_diet

.. ipython:: python
    :suppress:

    time_average_mean_diet = time_average_mean_diet.compute()

Now, the drawing of the diet matrix is done by using the
:py:func:`plot_diet_values` function:

.. ipython:: python

    fig = plt.figure()
    ax = plt.gca()
    l = apecosm.plot_diet_values(time_average_mean_diet, const, 0)
    plt.savefig(os.path.join('doc', 'computations', '_static', 'diet_com_0.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'diet_com_0.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/diet_com_0.*
    :align: center

    Mean diet for community 0


The first argument is the spatially and time averaged  diet matrix,
the second argument is the dataset containing all the Apecosm constants
and the third argument is the community index.

Note that there is the possibility to control the legend layout by
providing a `legend_args` argument, which is a dictionary containing
the legend arguments.

Furthermore, the arguments of the Matplotlib ``stackplot`` function
can also be included in the   :py:func:`apecosm.plot_diet_values` function.

For instance:

.. ipython:: python

    cmap = matplotlib.colormaps['jet']
    colors = [cmap(i / 10) for i in range(11)]

    fig = plt.figure()
    ax = plt.gca()
    l = apecosm.plot_diet_values(time_average_mean_diet, const, 0,
                             colors=colors, alpha=0.5,
                             legend_args={'ncol': 2, 'fontsize': 6})

    plt.savefig(os.path.join('doc', 'computations', '_static', 'upt_diet_com_0.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'upt_diet_com_0.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/upt_diet_com_0.*
    :align: center

    Mean diet for community 0 with additional arguments for
    controling the legend and stackplot display

To draw the diet matrix for all communities, a loop
must be done over the `c` dimension:

.. ipython:: python

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)

    for c in range(5):
        ax = plt.subplot(3, 2, c + 1)
        draw_legend = (c == 0)
        l = apecosm.plot_diet_values(time_average_mean_diet, const, c,
                                     colors=colors, alpha=0.5, draw_legend=draw_legend,
                                     legend_args={'ncol': 2, 'fontsize': 6, 'framealpha': 1})
        ax.set_xlabel('Length (m)')
        ax.set_xlim(const['length'].min(), const['length'].max())

    plt.savefig(os.path.join('doc', 'computations', '_static', 'full_diet.jpg'), bbox_inches='tight')
    plt.savefig(os.path.join('doc', 'computations', '_static', 'full_diet.pdf'), bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/full_diet.*
    :align: center

    Mean diet for all communities with additional arguments for
    controling the legend and stackplot display
