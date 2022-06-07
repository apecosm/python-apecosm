.. _api:

#################
API
#################

.. automodule:: apecosm

Reading configuration
-----------------------

.. autosummary::
   :toctree: generated

   read_config

Reading grid
-------------------

.. autosummary::
   :toctree: generated

   extract_weight_grid
   read_ape_grid 
   partial_step_ape
   plot_grid_nemo_ape

Data extraction
-----------------------

.. autosummary::
   :toctree: generated

   extract_ltl_data
   extract_time_means
   extract_oope_data
   generate_mask

Habitat
-----------------------

.. autosummary::
   :toctree: generated

   get_tcor
   compute_o2
   compute_tpref
   compute_tlim
   compute_lightpref

Miscellaneous
-----------------------

.. autosummary::
   :toctree: generated

   size_to_weight
   weight_to_size
   find_percentile
   compute_daylength
   extract_community_names

Maps
-----------------------

.. warning:: 

    The following functions need the PyNgl library. This could be done by
    setting up a virtual environment.

    For more details, please visit `https://www.pyngl.ucar.edu/Download/ <https://www.pyngl.ucar.edu/Download/>`_

.. autosummary::
   :toctree: generated

   plot_season_oope
   plot_oope_map


Size spectra
-----------------------

.. autosummary::
   :toctree: generated

   compute_spectra_ltl
   plot_oope_spectra


