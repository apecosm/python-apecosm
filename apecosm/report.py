import os
import sys
import urllib
import numpy as np
import pkg_resources
import jinja2
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import ceil
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from apecosm.constants import LTL_NAMES
from .diags import compute_size_cumprop
from .extract import extract_oope_data, extract_time_means, open_apecosm_data, open_constants, open_mesh_mask, extract_weighted_data, extract_mean_size, open_fishing_data
from .misc import extract_community_names, compute_mean_min_max_ts, extract_fleet_names
from .size_spectra import plot_oope_spectra
plt.rcParams['text.usetex'] = False
import psutil

global FONT_SIZE, LABEL_SIZE, THIN_LWD, REGULAR_LWD, THICK_LWD, COL_GRID, REGULAR_TRANSP, HIGH_TRANSP, FIG_WIDTH, FIG_HEIGHT, FIG_DPI, CB_SHRINK, COL_MAP
FONT_SIZE = 25
LABEL_SIZE = 25
THIN_LWD = 1
REGULAR_LWD = 2
THICK_LWD = 4
COL_GRID = (190/256, 190/256, 190/256)
REGULAR_TRANSP = 0.80
HIGH_TRANSP = 0.25
FIG_WIDTH = 12
FIG_HEIGHT = 6
FIG_DPI = 300
CB_SHRINK = 0.7
COL_MAP = 'RdYlBu_r'


def report(input_dir, mesh_file, fishing_path, config_path, domain_file=None, crs=ccrs.PlateCarree(), output_dir='report', filecss='default', xarray_args={}):

    mesh = open_mesh_mask(mesh_file)
    const = open_constants(input_dir)
    data = open_apecosm_data(input_dir, **xarray_args)

    # If a domain file is provided, extracts it and store it into a dictionary
    if domain_file is None:
        domains = {}
    else:
        domains = {}
        nc_domains = xr.open_dataset(domain_file)
        for v in nc_domains.variables:
            domains[v] = nc_domains[v]

    # create the output architecture
    # first create html folder
    html_dir = os.path.join(output_dir, 'html')
    os.makedirs(html_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, 'html/images')
    os.makedirs(images_dir, exist_ok=True)

    # create css folder
    css_dir = os.path.join(output_dir, 'css')
    os.makedirs(css_dir, exist_ok=True)

    # process the banner file, by adding as many tabs as domains
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("banner.html")

    outputs = {'domains': domains}
    render = template.render(**outputs)
    output_file = os.path.join(output_dir, 'html', 'banner.html')
    with open(output_file, "w") as f:
        f.write(render)

    if filecss is None:
        css = ''

    # load default value (one in package)
    elif filecss == 'default':
        filecss = pkg_resources.resource_filename('apecosm', os.path.join('templates', 'styles.css'))
        with open(filecss) as fin:
            css = fin.read()

    # load web resource
    elif filecss.startswith('http'):
        with urllib.request.urlopen(filecss) as fin:
            css = fin.read().decode('utf-8')

    else:
        with open(filecss) as fin:
            css = fin.read()

    with open(os.path.join(css_dir, 'styles.css'), 'w') as fout:
        fout.write(css)

    _make_meta_template(output_dir, css, data, const)
    print('meta - cpu % used:', psutil.cpu_percent())
    print('meta - memory % used:', psutil.virtual_memory()[2])
    _make_config_template(output_dir, css, data, const)
    print('conf - cpu % used:', psutil.cpu_percent())
    print('conf - memory % used:', psutil.virtual_memory()[2])
    _make_result_template(output_dir, css, data, const, mesh, crs)
    print('res - cpu % used:', psutil.cpu_percent())
    print('res - memory % used:', psutil.virtual_memory()[2])
    _make_fisheries_template(output_dir, css, fishing_path, config_path, mesh, crs)
    print('fisheries - cpu % used:', psutil.cpu_percent())
    print('fisheries - memory % used:', psutil.virtual_memory()[2])
    for domname in domains:
        _make_result_template(output_dir, css, data, const, mesh, crs, domains, domname)


    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("template.html")

    outputs = {}
    outputs['css'] = css

    render = template.render(**outputs)

    with open(os.path.join(output_dir, 'index.html'), "w") as f:
        f.write(render)


def _savefig(output_dir, fig_name, format):
    img_file = os.path.join(output_dir, 'html', 'images', fig_name)
    plt.savefig(img_file, format=format, bbox_inches='tight')
    return os.path.join('images', fig_name)


def _make_result_template(output_dir, css, data, const, mesh, crs, domains=None, domname=None):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("template_results.html")

    if domains is not None:
        maskdom = domains[domname]
    else:
        maskdom = np.ones(mesh['nav_lon'].shape)
        domname = 'global'
    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    outputs = {}
    outputs['css'] = css
    outputs['domain_figs'] = _plot_domain_maps(output_dir, mesh, crs, maskdom, domname) #ok
    #outputs['ts_figs'] = _plot_time_series(output_dir, mesh, data, const, maskdom, domname) #ok
    outputs['mean_length_figs'] = _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, 'length')  # ok
    outputs['mean_weight_figs'] = _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, 'weight')  # ok
    outputs['cumbiom_figs'] = _plot_integrated_time_series(output_dir, mesh, data, const, maskdom, domname) #ok
    #outputs['maps_figs'] = _plot_mean_maps(output_dir, mesh, data, const, crs, maskdom, domname) # nok--> os kill when plotting
    outputs['spectra_figs'] = _plot_size_spectra(output_dir, mesh, data, const, maskdom, domname) #ok
    if 'repfonct_day' in data.variables:
        outputs['repfonct_figs'] = _plot_weighted_values(output_dir, mesh, data, const, 'repfonct_day', maskdom, domname)
    if 'mort_day' in data.variables:
        outputs['mort_figs'] = _plot_weighted_values(output_dir, mesh, data, const, 'mort_day', maskdom, domname)
    if 'community_diet_values' in data.variables:
        outputs['diet_figs'] = _plot_diet_values(output_dir, mesh, data, const, maskdom, domname)

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'results_report_%s.html' %domname)
    with open(output_file, "w") as f:
        f.write(render)


def _plot_domain_maps(output_dir, mesh, crs_out, maskdom, domname):

    crs_in = ccrs.PlateCarree()

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)
    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask'].isel(z=0)

    tmask = tmask.values.copy()
    tmask = np.ma.masked_where(tmask == 0, tmask)
    test = (tmask == 1) & (maskdom.values == 1)
    tmask[~test] -= 1

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax = plt.axes(projection=crs_out)
    cs = plt.pcolormesh(lonf, latf, tmask[1:, 1:].astype(int), cmap=COL_MAP, transform=crs_in)
    cs.set_clim(0, 1)
    cb = plt.colorbar(cs, shrink=CB_SHRINK)
    cb.ax.tick_params(labelsize=LABEL_SIZE)
    cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
    plt.title('%s mask' %domname, fontsize=FONT_SIZE)
    ax.add_feature(cfeature.LAND, zorder=100)
    ax.add_feature(cfeature.COASTLINE, zorder=101)
    fig_name = _savefig(output_dir, 'domain_map_%s.svg' %domname, 'svg')
    plt.close(fig)
    return fig_name


def _plot_time_series(output_dir, mesh, data, const, maskdom, domname):

    output = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=True, compute_mean=False)
    output = output.sum(dim='w')
    total = output.sum(dim='c')

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1==1:
            total.plot.line(linewidth=THICK_LWD)
            plt.title('Total', fontsize=FONT_SIZE)
            plt.ylabel('Joules', fontsize=FONT_SIZE)
            plt.xticks(rotation=30, ha='right')
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.xlabel('')
        elif 2 <= i+1 <= n_plot:
            c = i-1
            output.isel(c=c).plot.line(linewidth=THICK_LWD)
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.ylabel('Joules', fontsize=FONT_SIZE)
            plt.xticks(rotation=30, ha='right')
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.xlabel('')
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'time_series_com_%s.svg' %domname, 'svg')
    plt.close(fig)

    return fig_name


def _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, varname):

    mean_size_tot = extract_mean_size(data, const, mesh, varname, maskdom=maskdom, aggregate=True)
    mean_size = extract_mean_size(data, const, mesh, varname, maskdom=maskdom)

    if varname == 'weight':
        mean_size *= 1000
        mean_size_tot *= 1000
        ylabel = 'Weight (g)'
    else:
        ylabel = 'Length (cm)'
        mean_size *= 100  # conversion in cm
        mean_size_tot *= 100  # conversion in cm

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 == 1:
            mean_size_tot.plot.line(linewidth=THICK_LWD)
            plt.title('Total', fontsize=FONT_SIZE)
            plt.ylabel(ylabel, fontsize=FONT_SIZE)
            plt.xlabel('')
            plt.xticks(rotation=30, ha='right')
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        elif 2 <= i+1 <= n_plot:
            c = i-1
            toplot = mean_size.isel(c=c)
            toplot.plot.line(linewidth=THICK_LWD)
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.ylabel(ylabel, fontsize=FONT_SIZE)
            plt.xlabel('')
            plt.xticks(rotation=30, ha='right')
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'mean_%s_comunities_and_total_%s.svg' %(varname, domname), 'svg')
    plt.close(fig)

    return fig_name


def _plot_integrated_time_series(output_dir, mesh, data, const, maskdom, domname):

    size_prop = compute_size_cumprop(mesh, data, const, maskdom=maskdom)
    size_prop = extract_time_means(size_prop)

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            c = i
            l = const['length'].isel(c=c)
            toplot = size_prop.isel(c=c)
            plt.fill_between(l, 0, toplot, edgecolor='k', facecolor='lightgray', linewidth=THIN_LWD)
            ax.set_xscale('log')
            plt.xlim(l.min(), l.max())
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.xlabel('Length (log-scale)', fontsize=FONT_SIZE)
            plt.ylabel('Proportion (%)', fontsize=FONT_SIZE)
            plt.ylim(0, 100)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'biomass_cumsum_bycom_%s.svg' %domname, 'svg')
    plt.close(fig)
    return fig_name


def _plot_mean_maps(output_dir, mesh, data, const, crs_out, maskdom, domname):

    crs_in = ccrs.PlateCarree()

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    if maskdom is None:
        maskdom = np.ones(lonf.shape)
    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    output = (data['OOPE'] * const['weight_step']).mean(dim='time').sum(dim=['w'])
    output = output.where(output > 0)
    output = output.where(maskdom > 0, drop=False)
    total = output.sum(dim='c')
    total = total.where(total > 0)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, c+1, projection=crs_out)
        if i+1 == 1:
            cs = plt.pcolormesh(lonf, latf, total.isel(y=slice(1, None), x=slice(1, None)), cmap=COL_MAP, transform=crs_in)
            cb = plt.colorbar(cs, shrink=CB_SHRINK)
            cb.ax.tick_params(labelsize=LABEL_SIZE)
            cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            cb.set_label('J/m2', fontsize=FONT_SIZE)
            plt.title('Total', fontsize=FONT_SIZE)
            ax.add_feature(cfeature.LAND, zorder=100)
            ax.add_feature(cfeature.COASTLINE, zorder=101)
        elif 2 <= i+1 <= n_plot:
            c = i-1
            cs = plt.pcolormesh(lonf, latf, output.isel(c=c, y=slice(1, None), x=slice(1, None)), cmap=COL_MAP, transform=crs_in)
            ax.add_feature(cfeature.LAND, zorder=100)
            ax.add_feature(cfeature.COASTLINE, zorder=101)
            cb = plt.colorbar(cs, shrink=CB_SHRINK)
            cb.ax.tick_params(labelsize=LABEL_SIZE)
            cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            cb.set_label('Joules/m2', fontsize=FONT_SIZE)
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'mean_maps_com_%s.svg' %domname, 'svg')
    plt.close(fig)
    return fig_name


def _plot_size_spectra(output_dir, mesh, data, const, maskdom, domname):

    # extract data in the entire domain, integrates over space
    data = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=False, compute_mean=False)
    data = extract_time_means(data)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    plot_oope_spectra(data, const, output_var='length', linewidth=THICK_LWD)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('Length (m)', fontsize=FONT_SIZE)
    plt.ylabel('OOPE (J/m)', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=LABEL_SIZE)
    plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
    plt.legend(fontsize=FONT_SIZE)
    fig_name = _savefig(output_dir, 'size_spectra_%s.svg' %domname, 'svg')
    plt.close(fig)
    return fig_name


def _plot_weighted_values(output_dir, mesh, data, const, varname, maskdom, domname):

    output = extract_weighted_data(data, const, mesh, varname, maskdom)
    output = extract_time_means(output)

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            c = i
            l = const['length'].isel(c=c)
            toplot = output.isel(c=c)
            plt.plot(l, toplot, color='k', linewidth=THICK_LWD)
            ax.set_xscale('log')
            plt.xlim(l.min(), l.max())
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.xlabel('Length (log-scale)', fontsize=FONT_SIZE)
            plt.ylabel(varname, fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.ylim(toplot.min(), toplot.max())
        else:
            ax.axis('off')
    fig_name = _savefig(output_dir, 'weighted_%s_by_com_%s.svg' %(varname, domname), 'svg')
    plt.close(fig)
    return fig_name


def _plot_diet_values(output_dir, mesh, data, const, maskdom, domname):

    if 'community' in data.dims:
        data = data.rename({'community' : 'c'})

    diet = extract_weighted_data(data, const, mesh, 'community_diet_values', maskdom=maskdom)
    diet = extract_time_means(diet)

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    legend = LTL_NAMES.copy()
    for c in range(n_community):
        legend.append(community_names['Community ' + str(c)])

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            c = i
            l = const['length'].isel(c=c)
            toplot = diet.isel(c=c)
            repf = toplot.sum(dim='prey_group')
            plt.stackplot(l, toplot.T, edgecolor='k', linewidth=THIN_LWD)
            plt.ylim(0, repf.max())
            plt.xlim(l.min(), l.max())
            ax.set_xscale('log')
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.legend(legend, fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig_name = _savefig(output_dir, 'diets_com_%s.svg' %domname, 'svg')
    plt.close(fig)
    return fig_name


def _make_meta_template(output_dir, css, data, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("template_meta.html")

    community_names = extract_community_names(const)
    #fleet_names = extract_fleet_names(fishing_config_path)
    #outputs['fleet_names'] = fleet_names
    dims = data.dims
    list_dims = [d for d in data.dims if 'prey' not in d]

    outputs = {}
    outputs['css'] = css
    outputs['comnames'] = community_names
    outputs['dims'] = dims
    outputs['list_dims'] = list_dims
    outputs['start_date'] = data['time'][0].values
    outputs['end_date'] = data['time'][-1].values

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'config_meta.html')
    with open(output_file, "w") as f:
        f.write(render)


def _make_config_template(output_dir, css, data, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("template_config.html")

    outputs = {}
    outputs['css'] = css
    outputs['length_figs'] = _plot_wl_community(output_dir, const, 'length', 'meters')
    outputs['weight_figs'] = _plot_wl_community(output_dir, const, 'weight', 'kilograms')
    outputs['trophic_figs'] = _plot_trophic_interactions(output_dir, const)
    outputs['select_figs'] = _plot_ltl_selectivity(output_dir, const)

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'config_report.html')
    with open(output_file, "w") as f:
        f.write(render)


def _plot_wl_community(output_dir, data, varname, units):

    community_names = extract_community_names(data)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            c = i
            length = data[varname].isel(c=c)
            plt.plot(length.values, linewidth=THICK_LWD)
            plt.xlim(0, length.shape[0]-1)
            plt.ylabel('%s' %units, fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        else:
            ax.axis('off')
    fig_name = _savefig(output_dir, '%s_by_com.svg' %varname, 'svg')
    plt.close(fig)

    return fig_name


def _plot_trophic_interactions(output_dir, data):

    trophic_interact = data['troph_interaction'].values

    community_names = extract_community_names(data)
    xlabel = []
    for c in range(0, trophic_interact[0][0].shape[0]):
        xlabel.append(community_names['Community ' + str(c)])

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.8)
    nd, npred, nprey = trophic_interact.shape
    title = ['Day', 'Night']
    for d in range(2):
        ax = plt.subplot(1, 2, d+1)
        cs = plt.imshow(trophic_interact[d], origin='lower', interpolation='none', cmap=plt.cm.jet)
        for i in range(nprey + 1):
            plt.axvline(i - 0.5, linestyle='--', linewidth=THIN_LWD, color='w')
            plt.axhline(i - 0.5, linestyle='--', linewidth=THIN_LWD, color='w')
        plt.title(title[d])
        cs.set_clim(0, 1)
        plt.xlabel('Prey')
        plt.ylabel('Predator')
        ax.set_xticks(np.arange(nprey))
        ax.set_yticks(np.arange(npred))
        ax.set_xticklabels(xlabel, rotation=45)
        ax.set_yticklabels(xlabel, rotation=45)
        ax.set_aspect('equal', 'box')
    fig_name = _savefig(output_dir, 'trophic_interactions.svg', 'svg')
    plt.close(fig)
    return fig_name


def _plot_ltl_selectivity(output_dir, data):

    community_names = extract_community_names(data)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            c = i
            length = data['length'].isel(c=c)
            varlist = [v for v in data.variables if v.startswith('select_')]
            for v in varlist:
                temp = data[v].isel(c=0)
                plt.plot(length, temp, label=v, linewidth=THICK_LWD)
            plt.legend(fontsize=FONT_SIZE)
            plt.xlim(length.min(), length.max())
            ax.set_xscale('log')
            plt.title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
        else:
            ax.axis('off')
    fig_name = _savefig(output_dir, 'selectivity_com_by_com.svg', 'svg')
    plt.close(fig)

    return fig_name


def _make_fisheries_template(output_dir, css, fishing_path, fishing_config_path, mesh, crs):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"), autoescape=jinja2.select_autoescape())
    template = env.get_template("template_fisheries.html")

    market, fleet_maps, fleet_summary, fleet_parameters = open_fishing_data(fishing_path)
    fleet_names = extract_fleet_names(fishing_config_path)
    #fleet_names = ['east_pacific_ps', 'atlantic_ps', 'indian_ps', 'west_pacific_ps', 'longline', 'indian_ss']

    outputs = {}
    outputs['css'] = css
    outputs['fleet_size'] = _plot_fleet_size(output_dir, fleet_summary, fleet_names) #ok
    outputs['fishing_effective_effort'] = _plot_fishing_effective_effort(output_dir, fleet_maps, fleet_names, mesh, crs) #ok
    outputs['landing_rate_eez_hs'] = _plot_landing_rate_eez_hs(output_dir, fleet_summary, fleet_names) #ok
    outputs['landing_rate_total'] = _plot_landing_rate_total(output_dir, fleet_summary, fleet_names) #ok
    outputs['landing_rate_by_vessels'] = _plot_landing_rate_by_vessels(output_dir, fleet_maps, fleet_names, mesh, crs) #ok
    outputs['landing_rate_density'] = _plot_landing_rate_density(output_dir, fleet_maps, fleet_names, mesh, crs) #ok
    outputs['average_fishing_distance'] = _plot_average_fishing_distance(output_dir, fleet_summary, fleet_names) #ok
    outputs['fuel_use_intensity'] = _plot_fuel_use_intensity(output_dir, fleet_summary, fleet_names) #ok
    outputs['yearly_profit'] = _plot_yearly_profit(output_dir, fleet_summary, fleet_names) #ok
    outputs['savings'] = _plot_savings(output_dir, fleet_summary, fleet_names) #ok
    outputs['fish_price'] = _plot_fish_price(output_dir, market, fleet_names) #ok
    outputs['capture_landing_rate'] = _plot_capture_landing_rate(output_dir, fleet_summary, fleet_names) #ok
    outputs['cost_revenue_by_vessels'] = _plot_cost_revenue_by_vessels(output_dir, fleet_summary, fleet_names) #ok
    outputs['fishing_time_fraction'] = _plot_fishing_time_fraction(output_dir, fleet_summary, fleet_names) #ok

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'fisheries_report.html')
    with open(output_file, "w") as f:
        f.write(render)


def _plot_fleet_size(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (241/256, 140/256, 141/256)
    col_2 = (154/256, 190/256, 219/256)
    col_3 = (165/256, 215/256, 164/256)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, sharex=True)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            av_1, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['effective_effort'], 365)
            av_2, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['active_vessels'], 365)
            av_3, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['total_vessels'], 365)
            plt.plot(time, av_1, color='black', linewidth=THIN_LWD)
            plt.plot(time, av_2, color='black', linewidth=THIN_LWD)
            plt.plot(time, av_3, color='black', linewidth=THIN_LWD)
            plt.fill_between(time, av_1, color=col_1, alpha=REGULAR_TRANSP, label='Fishing')
            plt.fill_between(time, av_2, av_1, color=col_2, alpha=REGULAR_TRANSP, label='Sailing')
            plt.fill_between(time, av_3, av_2, color=col_3, alpha=REGULAR_TRANSP, label='At port')
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Number of vessels', fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    plt.legend(loc='best', fontsize=FONT_SIZE)
    fig_name = _savefig(output_dir, 'fleet_size.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_fishing_effective_effort(output_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    cb_qu = 15

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1, projection=crs_out)
        if i+1 <= n_plot:
            raw = fleet_maps[i]['effective_effort_density'].isel(time=-1)

            raw_val = raw.values.copy()
            raw_val[raw_val == 0] = sys.float_info.min
            extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
            data = np.log10(raw_val)
            cb_lb = 0
            cb_ub = 1
            if len(data[extract]) > 0:
                cb_lb = np.percentile(data[extract], cb_qu)
                cb_ub = np.percentile(data[extract], 100-cb_qu)

            cs = plt.pcolormesh(lonf, latf, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
            # cs = plt.pcolormesh(lonf[:-1, :-1], latf[:-1, :-1], test[1:-1, 1:-1],  cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            cb = plt.colorbar(cs, shrink=CB_SHRINK)
            cb.ax.tick_params(labelsize=LABEL_SIZE)
            cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            ax.coastlines(zorder=101)
            ax.add_feature(cfeature.LAND, zorder=100)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'fishing_effective_effort.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_landing_rate_eez_hs(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (241/256, 140/256, 141/256)
    col_2 = (154/256, 190/256, 219/256)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, sharex=True)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            av_1, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * fleet_summary[i]['current_total_landings_rate_from_EEZ'], 365)
            av_2, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * (fleet_summary[i]['step_landings']-fleet_summary[i]['current_total_landings_rate_from_EEZ']), 365)
            plt.plot(time, av_1, color='black', linewidth=THIN_LWD)
            plt.plot(time, av_1+av_2, color='black', linewidth=THIN_LWD)
            plt.fill_between(time, av_1+av_2, av_1, color=col_2, alpha=REGULAR_TRANSP, label='HS')
            plt.fill_between(time, av_1, color=col_1, alpha=REGULAR_TRANSP, label='EEZ')
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Landing rate (MT.years-1)', fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    plt.legend(loc='best', fontsize=FONT_SIZE)
    fig_name = _savefig(output_dir, 'landing_rate_eez_hs.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_landing_rate_total(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * fleet_summary[i]['step_landings'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Landing rate (MT.years-1)', fontsize=FONT_SIZE)
            plt.title('%s - last year landing rate : %.2f MT.years-1' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'landing_rate_total.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_landing_rate_by_vessels(output_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    cb_qu = 15

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1, projection=crs_out)
        if i+1 <= n_plot:
            raw = fleet_maps[i]['landing_rate_by_vessel'].isel(time=-1)

            raw_val = raw.values.copy()
            raw_val[raw_val == 0] = sys.float_info.min
            extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
            data = np.log10(raw_val)
            cb_lb = 0
            cb_ub = 1
            if len(data[extract]) > 0:
                cb_lb = np.percentile(data[extract], cb_qu)
                cb_ub = np.percentile(data[extract], 100-cb_qu)

            cs = plt.pcolormesh(lonf, latf, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
            # cs = plt.pcolormesh(lonf[:-1, :-1], latf[:-1, :-1], test[1:-1, 1:-1],  cmap=COL_MAP, transform=crs_in, vmin=-13.5, vmax=-10)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            cb = plt.colorbar(cs, shrink=CB_SHRINK)
            cb.set_label('T/day-1', fontsize=FONT_SIZE)
            cb.ax.tick_params(labelsize=LABEL_SIZE)
            cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            ax.coastlines(zorder=101)
            ax.add_feature(cfeature.LAND, zorder=100)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'landing_rate_by_vessels.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_landing_rate_density(output_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    cb_qu = 15

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1, projection=crs_out)
        if i+1 <= n_plot:
            raw = fleet_maps[i]['landing_rate'].isel(time=-1)

            raw_val = raw.values.copy()
            raw_val[raw_val == 0] = sys.float_info.min
            extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
            data = np.log10(raw_val)
            cb_lb = 0
            cb_ub = 1
            if len(data[extract]) > 0:
                cb_lb = np.percentile(data[extract], cb_qu)
                cb_ub = np.percentile(data[extract], 100-cb_qu)

            cs = plt.pcolormesh(lonf, latf, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
            # cs = plt.pcolormesh(lonf[:-1, :-1], latf[:-1, :-1], test[1:-1, 1:-1],  cmap=COL_MAP, transform=crs_in, vmin=-13.5, vmax=-10)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            cb = plt.colorbar(cs, shrink=CB_SHRINK)
            cb.ax.tick_params(labelsize=LABEL_SIZE)
            cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
            ax.coastlines(zorder=101)
            ax.add_feature(cfeature.LAND, zorder=100)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'landing_rate_density.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_average_fishing_distance(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (255/255, 0/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['average_fishing_distance_to_ports_of_active_vessels'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Average fishing distance (km)', fontsize=FONT_SIZE)
            plt.title('%s - last year distance : %.1f km' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'average_fishing_distance.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_fuel_use_intensity(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (160/255, 32/255, 240/255)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['average_fuel_use_intensity'], 365)
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Fuel use intensity (kL.T-1)', fontsize=FONT_SIZE)
            plt.title('%s - last year FUI : %.1f kL.T-1' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'fuel_use_intensity.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_yearly_profit(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (255/255, 165/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(0.001 * 365 * fleet_summary[i]['step_profits'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Yearly profit (M$.years-1)', fontsize=FONT_SIZE)
            plt.title('%s - last year profit : %.1f M$.years-1' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'yearly_profit.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_savings(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 100/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(0.001 * fleet_summary[i]['savings'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Savings (M$)', fontsize=FONT_SIZE)
            plt.title('%s - last year savings : %.1f M$' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'savings.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_fish_price(output_dir, market, fleet_names):

    n_fleet = len(market['fleet'])
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (167/255, 61/255, 11/255)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            if i == 4:
                average, maxi, mini, time = compute_mean_min_max_ts(market['average_price'].isel(fleet=i, community=4), 365)
            else:
                average, maxi, mini, time = compute_mean_min_max_ts(market['average_price'].isel(fleet=i, community=1), 365)
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Fish price ($.kg-1)', fontsize=FONT_SIZE)
            plt.title('%s - last year fish price : %.1f $.kg-1' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'fish_price.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_capture_landing_rate(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)
    col_2 = (255/255, 0/255, 0/255)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        if i+1 <= n_plot:
            average_1, maxi_1, mini_1, time_1 = compute_mean_min_max_ts(fleet_summary[i]['average_capture_rate_by_active_vessel'], 365)
            average_2, maxi_2, mini_2, time_2 = compute_mean_min_max_ts(fleet_summary[i]['average_landing_rate_by_active_vessel'], 365)
            plt.subplot(n_row, n_col, i+1)
            plt.plot(time_1, average_1, linewidth=5, color=col_1, label='Capture')
            plt.fill_between(time_1, mini_1, average_1, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time_1, average_1, maxi_1, color=col_1, alpha=HIGH_TRANSP)
            plt.plot(time_2, average_2, linewidth=5, color=col_2, label='Landing')
            plt.fill_between(time_2, mini_2, average_2, color=col_2, alpha=HIGH_TRANSP)
            plt.fill_between(time_2, average_2, maxi_2, color=col_2, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Capture and landing rate (T.day-1)', fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    plt.legend(loc='best', fontsize=FONT_SIZE)
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'capture_landing_rate.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_cost_revenue_by_vessels(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)
    col_2 = (255/255, 0/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average_1, maxi_1, mini_1, time_1 = compute_mean_min_max_ts(fleet_summary[i]['average_cost_by_active_vessels'], 365)
            average_2, maxi_2, mini_2, time_2 = compute_mean_min_max_ts(fleet_summary[i]['average_profit_by_active_vessels']+fleet_summary[i]['average_cost_by_active_vessels'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.plot(time_1, average_1, linewidth=5, color=col_1, label='Cost')
            plt.fill_between(time_1, mini_1, average_1, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time_1, average_1, maxi_1, color=col_1, alpha=HIGH_TRANSP)
            plt.plot(time_2, average_2, linewidth=5, color=col_2, label='Revenue')
            plt.fill_between(time_2, mini_2, average_2, color=col_2, alpha=HIGH_TRANSP)
            plt.fill_between(time_2, average_2, maxi_2, color=col_2, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.title(fleet_names[i], fontsize=FONT_SIZE)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Cost and revenue by active vessels (k$.day-1)', fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    plt.legend(loc='best', fontsize=FONT_SIZE)
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'cost_revenue_by_vessels.png', 'png')
    plt.close(fig_name)
    return fig_name


def _plot_fishing_time_fraction(output_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 104/255, 139/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    for i in range(n_row*n_col):
        ax = plt.subplot(n_row, n_col, i+1)
        if i+1 <= n_plot:
            average, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['average_fishing_time_fraction_of_active_vessels'], 365)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.plot(time, average, linewidth=5, color=col_1)
            plt.fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
            plt.fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
            plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            plt.tick_params(axis='both', labelsize=LABEL_SIZE)
            plt.xlabel('Time (years)', fontsize=FONT_SIZE)
            plt.ylabel('Fishing time fraction', fontsize=FONT_SIZE)
            plt.ylim([-0.05, 1.05])
            plt.title('%s - last year fishing time fraction : %.2f' % (fleet_names[i], average[-1]), fontsize=FONT_SIZE)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig_name = _savefig(output_dir, 'fishing_time_fraction.png', 'png')
    plt.close(fig_name)
    return fig_name


if __name__ == '__main__':

    #pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    print("toto")
