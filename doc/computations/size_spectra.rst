**********************************************************
Cumulated biomass
**********************************************************

.. ipython:: python
    :suppress:

    import sys
    import os
    sys.path.insert(0, os.path.abspath('../'))
    import matplotlib.pyplot as plt
    import os
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import matplotlib.pyplot as plt
    import apecosm

    mesh_file = 'data/pacific_mesh_mask.nc'
    mesh = apecosm.open_mesh_mask(mesh_file)

    const = apecosm.open_constants('data/apecosm/')

    data = apecosm.open_apecosm_data('data/apecosm')

    ltl_data = apecosm.open_ltl_data('data/pisces',
                                    replace_dims={'olevel': 'z'})


Plotting of LTL size spectra
###############################################

The size-spectra associated with a given LTL can be plotted using the
:py:func:`compute_spectra_ltl` function.

First, the spatial integral is computed, as in :numref:`extract_ltl`:

.. ipython:: python

    ltl_phy2 = apecosm.extract_ltl_data(ltl_data, mesh, 'PHY2')
    ltl_phy2

.. ipython:: python
    :suppress:

    ltl_phy2 = ltl_phy2.compute()

And we compute the time mean:

.. ipython:: python

    ltl_phy2_mean = apecosm.extract_time_means(ltl_phy2)
    ltl_phy2_mean

Then, the size-spectra is plotted as follows:

.. ipython:: python

    fig = plt.figure()
    ax = plt.gca()
    L = [10e-6, 100e-6]
    apecosm.compute_spectra_ltl(ltl_phy2_mean, L, output_var='weight', label='PHY2')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Biomass")
    ax.set_xlabel('Weight (kg)')
    ax.set_title('PHY2 size-spectra')

The ``L`` variable is the lower and upper bound of the LTL length, as used in the Apecosm configuration
(``forcing.XXX.length.min`` and ``forcing.XXX.length.max`` parameters).

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/spectra_ltl_phy2_weight.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/spectra_ltl_phy2_weight.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/spectra_ltl_phy2_weight.*
    :align: center

    Diatoms size-spectra as a function of weight

Note that the size-spectra can also be plotted as a function of length:

.. ipython:: python

    fig = plt.figure()
    ax = plt.gca()
    L = [10e-6, 100e-6]
    apecosm.compute_spectra_ltl(ltl_phy2_mean, L, output_var='length', label='PHY2')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Biomass")
    ax.set_xlabel('Length (m)')
    ax.set_title('PHY2 size-spectra')

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/spectra_ltl_phy2_length.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/spectra_ltl_phy2_length.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/spectra_ltl_phy2_length.*
    :align: center

    Diatoms size-spectra as a function of length


Plotting of Apecosm size spectra
###############################################

Apecosm size-spectra is plotting using the :py:func:`apecosm.plot_oope_spectra` function.

First, we extract the Apecosm biomass on a given region region:

.. ipython:: python

    ts = apecosm.extract_oope_data(data['OOPE'], mesh)

.. ipython:: python
    :suppress:

    ts = ts.compute()

Then, we compute the time mean:

.. ipython:: python

    tsmean = apecosm.extract_time_means(ts)

.. ipython::

    fig = plt.figure()
    ax = plt.gca()
    cs = apecosm.plot_oope_spectra(tsmean, const, output_var='weight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e7, 1e23)
    ax.set_ylabel("Biomass")
    ax.set_xlabel('Weight (kg)')
    ax.set_title('Apecosm size-spectra')
    plt.legend()

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/spectra_apecosm_weight.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/spectra_apecosm_weight.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/spectra_apecosm_weight.*
    :align: center

    Apecosm size-spectra as a function of weight


Size-spectra with all variables
###############################################

What has been done for ``PHY2`` can be done also for the othe LTL variables. First, the other
variables are extracted and time averaged:

.. ipython:: python

    ltl_zoo2 = apecosm.extract_ltl_data(ltl_data, mesh, 'ZOO2', depth_max=depth_max).compute()
    ltl_zoo = apecosm.extract_ltl_data(ltl_data, mesh, 'ZOO', depth_max=depth_max).compute()
    ltl_goc = apecosm.extract_ltl_data(ltl_data, mesh, 'GOC', depth_max=depth_max).compute()

    ltl_goc_mean = ltl_goc.mean(dim='time_counter')
    ltl_zoo_mean = ltl_zoo.mean(dim='time_counter')
    ltl_zoo2_mean = ltl_zoo2.mean(dim='time_counter')

.. ipython:: python
    :suppress:

    ltl_goc_mean = ltl_goc_mean.compute()
    ltl_zoo_mean = ltl_zoo_mean.compute()
    ltl_zoo2_mean = ltl_zoo2_mean.compute()


.. ipython:: python

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    cs = apecosm.plot_oope_spectra(tsmean, const, output_var=output_var)

    L = [10e-6, 100e-6]
    apecosm.compute_spectra_ltl(ltl_phy2_mean, L, output_var=output_var, label='PHY2')

    L = [20.e-6, 200.e-6]
    apecosm.compute_spectra_ltl(ltl_zoo_mean, L, output_var=output_var, label='ZOO')

    L = [200.e-6, 2000.e-6]
    apecosm.compute_spectra_ltl(ltl_zoo2_mean, L, output_var=output_var, label='ZOO2')

    L = [100e-6, 50000.e-6]
    apecosm.compute_spectra_ltl(ltl_goc_mean, L, output_var=output_var, label='GOC')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e7, 1e32)
    ax.set_ylabel("Biomass")
    ax.set_xlabel('Weight (kg)')
    ax.set_title('All variables size-spectra')
    plt.legend()

.. ipython:: python
    :suppress:

    plt.savefig('computations/_static/spectra_allvars_weight.jpg', bbox_inches='tight')
    plt.savefig('computations/_static/spectra_allvars_weight.pdf', bbox_inches='tight')
    plt.close(fig)

.. figure::  _static/spectra_allvars_weight.*
    :align: center

   Size-spectra of LTL and Apecosm variables as a function of weight