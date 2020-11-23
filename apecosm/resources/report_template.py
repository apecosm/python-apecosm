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

# + tags=["remove_input", "remove_output", "remove_cell"]
import warnings
warnings.filterwarnings('ignore')
import apecosm
import matplotlib.pyplot as plt
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# + tags=["parameters", "remove_input", "remove_output", "remove_cell"]
input_dir = '/home/barrier/Codes/apecosm/git-apecosm-config/gyre_multicom/output_apecosm'
input_mesh = '/home/barrier/Codes/apecosm/git-apecosm-config/gyre/mesh_mask.nc'
# -

# ## Time-series

# + tags=["remove_input"]
apecosm.plot_report_ts(input_dir, input_mesh)
# -

# ## Maps

# + tags=["remove_input"]
apecosm.plot_report_map(input_dir, input_mesh)
# -

# ## Size-spectras

# + tags=["remove_input"]
apecosm.plot_report_size_spectra(input_dir, input_mesh)
