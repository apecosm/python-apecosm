# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import memory_profiler
import line_profiler
import psutil
from dask.diagnostics import ProgressBar

# ========================== #
# FUNCTIONS
# ========================== #

def _savefig(report_dir, fig_name, pic_format):
    img_file = os.path.join(report_dir, 'html', 'images', fig_name)
    plt.savefig(img_file, format=pic_format, bbox_inches='tight')
    return os.path.join('images', fig_name)

def _plot_mean_maps_2(report_dir, mesh, data, const, crs_out, mask_dom, dom_name):

    crs_in = ccrs.PlateCarree()

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)

    if mask_dom is None:
        mask_dom = np.ones(lon_f.shape)
    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])

    community_names = apecosm.extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    # Computation of the time average for OOPE -> (y, x, c, w)

    output = (data['OOPE'] * const['weight_step']).sum(dim=['w'])
    with ProgressBar():
        output = output.compute()
    print('++++++++++ Size integration: check')

    output = output.mean(dim='time')
    with ProgressBar():
        output = output.compute()
    print('++++++++++ Time mean: check')

    output = output.fillna(0)
    output = output.where(output > 0, drop=False)
    output = output.where(mask_dom > 0, drop=False)
    total = output.sum(dim='c')
    total = total.where(total > 0, drop=False)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            print("i =", i)
            print("j =", j)
            print("cpu% = ", psutil.cpu_percent())
            print("mem% = ", psutil.virtual_memory().percent)
            cpt = cpt+1
            if cpt == 1:
                #ax = plt.subplot(n_row, n_col, cpt, projection=crs_out)
                cs = axes[i, j].pcolormesh(lon_f, lat_f, total[1:, 1:], cmap=COL_MAP, transform=crs_in)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i, j].set_title('Total', fontsize=FONT_SIZE)
                axes[i, j].add_feature(cfeature.LAND, zorder=100)
                axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
            elif 2 <= cpt <= n_plot:
                #ax = plt.subplot(n_row, n_col, cpt, projection=crs_out)
                cs = axes[i, j].pcolormesh(lon_f, lat_f, output.isel(c=c)[1:, 1:], cmap=COL_MAP, transform=crs_in)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].add_feature(cfeature.LAND, zorder=100)
                axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
                c = c+1
            else:
                #ax = plt.subplot(n_row, n_col, cpt)
                axes[i, j].axis('off')
    del output
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'mean_maps_com_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_mean_maps(report_dir, mesh, data, const, crs_out, mask_dom, dom_name):

    crs_in = ccrs.PlateCarree()

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)

    if mask_dom is None:
        mask_dom = np.ones(lon_f.shape)
    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])

    community_names = apecosm.extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    output = (data['OOPE'] * const['weight_step']).mean(dim='time').sum(dim=['w'])
    #output = output.where(output>0)
    #output = output.where(mask_dom>0, drop=False)
    #output_raw = output_raw.where(output_raw>0, drop=False)
    #output_raw = output_raw.where(mask_dom>0, drop=False)
    #output = output_raw.compute()
    #total_raw = output_raw.sum(dim='c')
    #total_raw = total_raw.where(total_raw > 0)
    #total = total_raw.compute()
    #del output_raw, total_raw

    print("def output/total")
    output = output.where(output > 0, drop=False)
    output = output.where(mask_dom > 0, drop=False)
    total = output.sum(dim='c')
    total = total.where(total > 0)

    print(output.chunks)
    print("OUTPUT",output)
    print("TOTAL",total)

    print("start compute output/total")
    output = output.compute(chunks={'time': 1, 'x': 10, 'y': 10, 'c': 1})
    total = total.compute(chunks={'time': 1, 'x': 10, 'y': 10})
    print("end compute output/total")

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    c = 0
    for i in range(n_row):
        for j in range(n_col):
            print("i =",i)
            print("j =",j)
            print("cpu% = ",psutil.cpu_percent())
            print("mem% = ",psutil.virtual_memory().percent)
            if i+j == 0:
                print("loop 1")
                cs = axes[i, j].pcolormesh(lon_f, lat_f, total[1:, 1:], cmap=COL_MAP, transform=crs_in, rasterized=True)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i,j].set_title('Total', fontsize=FONT_SIZE)
                axes[i,j].add_feature(cfeature.LAND, zorder=100)
                axes[i,j].add_feature(cfeature.COASTLINE, zorder=101)
                del total, cs, cb
            elif 2 <= i+j+1 <= n_plot:
                print("loop 2")
                print("c =",c)
                cs = axes[i,j].pcolormesh(lon_f, lat_f, output.isel(c=c)[1:,1:], cmap=COL_MAP, transform=crs_in, rasterized=True)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i,j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i,j].add_feature(cfeature.LAND, zorder=100)
                axes[i,j].add_feature(cfeature.COASTLINE, zorder=101)
                c = c+1
                del cs, cb
            else:
                print("loop 3")
                axes[i,j].axis('off')
    del output
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'mean_maps_com_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


# ========================== #
# LOAD
# ========================== #
report_parameters = apecosm.read_report_params('report_params_conf1.csv')
mesh_file = report_parameters['mesh_file']
output_dir = report_parameters['output_dir']
fishing_output_dir = report_parameters['fishing_output_dir']
fishing_config_dir = report_parameters['fishing_config_dir']
report_dir='report'

global FONT_SIZE, LABEL_SIZE, THIN_LWD, REGULAR_LWD, THICK_LWD, COL_GRID, REGULAR_TRANSP, HIGH_TRANSP, FIG_WIDTH, FIG_HEIGHT, FIG_DPI, CB_SHRINK, CB_THRESH, COL_MAP
FONT_SIZE = report_parameters['FONT_SIZE']
LABEL_SIZE = report_parameters['LABEL_SIZE']
THIN_LWD = report_parameters['THIN_LWD']
REGULAR_LWD = report_parameters['REGULAR_LWD']
THICK_LWD = report_parameters['THICK_LWD']
COL_GRID = report_parameters['COL_GRID']
REGULAR_TRANSP = report_parameters['REGULAR_TRANSP']
HIGH_TRANSP = report_parameters['HIGH_TRANSP']
FIG_WIDTH = report_parameters['FIG_WIDTH']
FIG_HEIGHT = report_parameters['FIG_HEIGHT']
FIG_DPI = report_parameters['FIG_DPI']
CB_SHRINK = report_parameters['CB_SHRINK']
CB_THRESH = report_parameters['CB_THRESH']
COL_MAP = report_parameters['COL_MAP']


# ========================== #
# CPU/RAM TRACKING
# ========================== #
mesh = apecosm.open_mesh_mask(mesh_file)
const = apecosm.open_constants(output_dir)
data = apecosm.open_apecosm_data(output_dir)
crs = ccrs.Mollweide()
domains = None
mask_dom = np.ones(mesh['nav_lon'].shape)
dom_name = 'global'
mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])

outputs = {}
outputs['maps_figs'] = _plot_mean_maps_2(report_dir, mesh, data, const, crs, mask_dom, dom_name)


