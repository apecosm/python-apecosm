# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Apecosm report

# + tags=["remove_input"]
import apecosm
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

input_dir = '/home/barrier/Codes/apecosm/git-apecosm-config/gyre_multicom/output_apecosm'
input_mesh = '/home/barrier/Codes/apecosm/git-apecosm-config/gyre/mesh_mask.nc'
# -

# ## Time-series

apecosm.plot_report_ts(input_dir, input_mesh)

# ## Maps

apecosm.plot_report_map(input_dir, input_mesh)

# ## Size-spectras

apecosm.plot_report_size_spectra(input_dir, input_mesh)
