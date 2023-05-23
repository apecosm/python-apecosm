import os
import sys
import urllib
from math import ceil
import numpy as np
import pkg_resources
import jinja2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import xarray as xr
from pyhtml2pdf import converter
from pypdf import PdfMerger
from dask.diagnostics import ProgressBar
from .constants import LTL_NAMES
from .extract import extract_oope_data, extract_time_means, open_apecosm_data, open_constants, open_mesh_mask, extract_weighted_data, open_fishing_data
from .misc import extract_community_names, compute_mean_min_max_xr, extract_fleet_names
from .size_spectra import plot_oope_spectra
plt.rcParams['text.usetex'] = False
matplotlib.use('Agg')


def report(report_parameters, domain_file=None, crs=ccrs.PlateCarree(), report_dir='report', file_css='default', xarray_args={}):

    # read report parameters
    mesh_file = report_parameters['mesh_file']
    output_dir = report_parameters['output_dir']
    fishing_output_dir = report_parameters['fishing_output_dir']
    fishing_config_dir = report_parameters['fishing_config_dir']
    use_fishing = fishing_output_dir != '' and fishing_config_dir != ''
    global FONT_SIZE, LABEL_SIZE, THIN_LWD, REGULAR_LWD, THICK_LWD, COL_GRID, REGULAR_TRANSP, HIGH_TRANSP, FIG_WIDTH, FIG_HEIGHT, FIG_DPI, CB_SHRINK, CB_THRESH, COL_MAP, FISHING_PERIOD, APECOSM_PERIOD, LAND_BACKGROUND
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
    FISHING_PERIOD = report_parameters['FISHING_PERIOD']
    APECOSM_PERIOD = report_parameters['APECOSM_PERIOD']
    LAND_BACKGROUND = report_parameters['LAND_BACKGROUND']

    mesh = open_mesh_mask(mesh_file)
    const = open_constants(output_dir)
    data = open_apecosm_data(output_dir, **xarray_args)

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
    html_dir = os.path.join(report_dir, 'html')
    os.makedirs(html_dir, exist_ok=True)

    images_dir = os.path.join(report_dir, 'html/images')
    os.makedirs(images_dir, exist_ok=True)

    # create css folder
    css_dir = os.path.join(report_dir, 'css')
    os.makedirs(css_dir, exist_ok=True)

    # process the banner file, by adding as many tabs as domains
    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('banner.html')

    outputs = {'domains': domains}
    outputs = {'use_fishing': use_fishing}
    render = template.render(**outputs)
    output_file = os.path.join(report_dir, 'html', 'banner.html')
    with open(output_file, 'w') as f:
        f.write(render)

    if file_css is None:
        css = ''

    # load default value (one in package)
    elif file_css == 'default':
        file_css = pkg_resources.resource_filename('apecosm', os.path.join('templates', 'styles.css'))
        with open(file_css) as fin:
            css = fin.read()

    # load web resource
    elif file_css.startswith('http'):
        with urllib.request.urlopen(file_css) as fin:
            css = fin.read().decode('utf-8')

    else:
        with open(file_css) as fin:
            css = fin.read()

    with open(os.path.join(css_dir, 'styles.css'), 'w') as fout:
        fout.write(css)

    _make_meta_template(report_dir, fishing_config_dir, css, data, const)
    _make_config_template(report_dir, css, const)
    _make_result_template(report_dir, css, data, const, mesh, crs)
    for dom_name in domains:
        _make_result_template(report_dir, css, data, const, mesh, crs, domains, dom_name)
    if use_fishing:
        _make_fisheries_template(report_dir, css, fishing_output_dir, fishing_config_dir, mesh, crs)

    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('template.html')

    outputs = {}
    outputs['css'] = css

    render = template.render(**outputs)

    with open(os.path.join(report_dir, 'index.html'), 'w') as f:
        f.write(render)

    # Convert html report in pdf report
    _html2pdf(report_dir)


def _savefig(report_dir, fig_name, pic_format):
    img_file = os.path.join(report_dir, 'html', 'images', fig_name)
    plt.savefig(img_file, format=pic_format, bbox_inches='tight')
    return os.path.join('images', fig_name)


def _html2pdf(report_dir):
    html_files = ['/html/config_meta.html', '/html/config_report.html', '/html/fisheries_report.html', '/html/results_report_global.html']
    for i in range(len(html_files)):
        file_data = open(report_dir + html_files[i], 'r').read()
        file_data = file_data.replace('width="150%"', 'width="100%"')
        open(report_dir+html_files[i], 'w').write(file_data)
        converter.convert(f'file:///{os.path.abspath(report_dir + html_files[i])}', report_dir+'/report_p%d.pdf' % i,print_options={'scale': 0.85})
        file_data = file_data.replace('width="100%"', 'width="150%"')
        open(report_dir + html_files[i], 'w').write(file_data)
    merger = PdfMerger()
    for i in range(len(html_files)):
        merger.append(report_dir + '/report_p%d.pdf' % i)
        os.remove(report_dir + '/report_p%d.pdf' % i)
    merger.write(report_dir + '/report.pdf')
    merger.close()


def _make_result_template(report_dir, css, data, const, mesh, crs, domains=None, dom_name=None):

    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('template_results.html')

    if domains is not None:
        mask_dom = domains[dom_name]
    else:
        mask_dom = np.ones(mesh['nav_lon'].shape)
        dom_name = 'global'
    mask_dom = xr.DataArray(data=mask_dom, dims=['y', 'x'])

    if 'tmaskutil' in mesh.variables:
        tmask = mesh['tmaskutil']
    else:
        tmask = mesh['tmask']

    # Computation of the full ocean surface within the domain considered
    surf_ocean = (mesh['e1t'] * mesh['e2t'] * mask_dom * tmask).sum(dim=['x', 'y']).compute()

    # Do some pre-calculations
    # We integrate the biomass over the entire domain: integration, no weight step. J/kg/m2 -> J
    # Output dimensions: time, c, w
    spatial_integrated_biomass = extract_oope_data(data, mesh, const, mask_dom=mask_dom)
    with ProgressBar():
        spatial_integrated_biomass = spatial_integrated_biomass.compute()
    print('+++++++++++ Pre-processing of spatial integral: check')

    outputs = {}
    outputs['css'] = css
    outputs['domain_figs'] = _plot_domain_maps(report_dir, mesh, crs, mask_dom, dom_name)
    print('+++++++++++ Plotting domain_figs: check')
    outputs['ts_figs'] = _plot_time_series(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name)
    print('+++++++++++ Plotting ts_figs: check')
    outputs['mean_length_figs'] = _plot_mean_size(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name, 'length')
    print('+++++++++++ Plotting mean_length_figs: check')
    outputs['mean_weight_figs'] = _plot_mean_size(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name, 'weight')
    print('+++++++++++ Plotting mean_weight_figs: check')
    outputs['cumbiom_figs'] = _plot_integrated_time_series(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name)
    print('+++++++++++ Plotting cumbiom_figs: check')
    if domains is None:
        outputs['maps_figs'] = _plot_mean_maps(report_dir, mesh, data, const, crs, mask_dom, dom_name)
        print('+++++++++++ Plotting maps_figs: check')
    outputs['spectra_figs'] = _plot_size_spectra(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name)
    print('+++++++++++ Plotting spectra_figs: check')
    if 'repfonct_day' in data.variables:
        outputs['repfonct_figs'] = _plot_weighted_values(report_dir, mesh, data, const, 'repfonct_day', mask_dom, dom_name)
        print('+++++++++++ Plotting repfonct_figs: check')
    if 'mort_day' in data.variables:
        outputs['mort_figs'] = _plot_weighted_values(report_dir, mesh, data, const, 'mort_day', mask_dom, dom_name)
        print('+++++++++++ Plotting mort_day: check')
    if 'community_diet_values' in data.variables:
        outputs['diet_figs'] = _plot_diet_values(report_dir, mesh, data, const, mask_dom, dom_name)
        print('+++++++++++ Plotting diet_figs: check')

    render = template.render(**outputs)

    output_file = os.path.join(report_dir, 'html', 'results_report_%s.html' %dom_name)
    with open(output_file, 'w') as f:
        f.write(render)


def _plot_domain_maps(report_dir, mesh, crs_out, mask_dom, dom_name):

    crs_in = ccrs.PlateCarree()

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)
    if 'tmaskutil' in mesh.variables:
        t_mask = mesh['tmaskutil']
    else:
        t_mask = mesh['tmask'].isel(z=0)

    t_mask = t_mask.values.copy()
    t_mask = np.ma.masked_where(t_mask == 0, t_mask)
    test = (t_mask == 1) & (mask_dom.values == 1)
    t_mask[~test] -= 1

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax = plt.axes(projection=crs_out)
    cs = plt.pcolormesh(lon_f, lat_f, t_mask[1:, 1:].astype(int), cmap=COL_MAP, transform=crs_in)
    cs.set_clim(0, 1)
    cb = plt.colorbar(cs, shrink=CB_SHRINK)
    cb.ax.tick_params(labelsize=LABEL_SIZE)
    cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
    plt.title('%s mask' %dom_name, fontsize=FONT_SIZE)
    if LAND_BACKGROUND:
        ax.add_feature(cfeature.LAND, zorder=100)
        ax.add_feature(cfeature.COASTLINE, zorder=101)
    fig_name = _savefig(report_dir, 'domain_map_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_time_series(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name):

    output = (spatial_integrated_biomass * const['weight_step'])
    output = output.sum(dim='w').compute()
    total = output.sum(dim='c').compute()

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community+1
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt == 1:
                total.plot.line(ax=axes[i, j], linewidth=THICK_LWD)
                axes[i, j].set_title('Total', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Joules', fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('')
                plt.setp(axes[i, j].get_xticklabels(), rotation=30, ha='right')
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].yaxis.get_offset_text().set(size=FONT_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            elif 2 <= cpt <= n_plot:
                output.isel(c=c).plot.line(ax=axes[i, j], linewidth=THICK_LWD)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Joules', fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('')
                plt.setp(axes[i, j].get_xticklabels(), rotation=30, ha='right')
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].yaxis.get_offset_text().set(size=FONT_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'time_series_com_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_mean_size(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name, varname):

    spatial_integrated_biomass = (spatial_integrated_biomass * const['weight_step']).compute()

    mean_size_tot = (spatial_integrated_biomass * const[varname]).sum(dim=['c', 'w']) / spatial_integrated_biomass.sum(dim=['c', 'w'])
    mean_size = (spatial_integrated_biomass * const[varname]).sum(dim=['w']) / spatial_integrated_biomass.sum(dim=['w'])
    with ProgressBar():
        mean_size = mean_size.compute()

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
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt == 1:
                mean_size_tot.plot.line(ax=axes[i, j], linewidth=THICK_LWD)
                axes[i, j].set_title('Total', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel(ylabel, fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('')
                plt.setp(axes[i, j].get_xticklabels(), rotation=30, ha='right')
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].yaxis.get_offset_text().set(size=FONT_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
            elif 2 <= cpt <= n_plot:
                mean_size.isel(c=c).plot.line(ax=axes[i, j], linewidth=THICK_LWD)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_ylabel(ylabel, fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('')
                plt.setp(axes[i, j].get_xticklabels(), rotation=30, ha='right')
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].yaxis.get_offset_text().set(size=FONT_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'mean_%s_comunities_and_total_%s.jpg' %(varname, dom_name), 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_integrated_time_series(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name):

    spatial_integrated_biomass = spatial_integrated_biomass * const['weight_step']
    size_prop = spatial_integrated_biomass.cumsum(dim='w') / spatial_integrated_biomass.sum(dim='w') * 100
    size_prop = extract_time_means(size_prop)
    with ProgressBar():
        size_prop = size_prop.compute()

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                l = const['length'].isel(c=c)
                axes[i, j].fill_between(l, 0, size_prop.isel(c=c), edgecolor='k', facecolor='lightgray', linewidth=THIN_LWD)
                axes[i, j].set_xscale('log')
                axes[i, j].set_xlim(l.min(), l.max())
                axes[i, j].set_ylim(0, 100)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Length (log-scale)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Proportion (%)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'biomass_cumsum_bycom_%s.jpg' %dom_name, 'jpg')
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

    community_names = extract_community_names(const)

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

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * FIG_WIDTH, n_row * FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt == 1:
                cs = axes[i, j].pcolormesh(lon_f, lat_f, total[1:, 1:], cmap=COL_MAP, transform=crs_in)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i, j].set_title('Total', fontsize=FONT_SIZE)
                if LAND_BACKGROUND:
                    axes[i, j].add_feature(cfeature.LAND, zorder=100)
                    axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
            elif 2 <= cpt <= n_plot:
                cs = axes[i, j].pcolormesh(lon_f, lat_f, output.isel(c=c)[1:, 1:], cmap=COL_MAP, transform=crs_in)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                cb.set_label('J/m2', fontsize=FONT_SIZE)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                if LAND_BACKGROUND:
                    axes[i, j].add_feature(cfeature.LAND, zorder=100)
                    axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
                c = c+1
            else:
                axes[i, j].axis('off')
    del output
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'mean_maps_com_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_size_spectra(spatial_integrated_biomass, report_dir, mesh, const, mask_dom, dom_name):

    data = extract_time_means(spatial_integrated_biomass)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    plot_oope_spectra(data, const, output_var='length', linewidth=THICK_LWD)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('Length (m)', fontsize=FONT_SIZE)
    plt.ylabel('OOPE (J/m)', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=LABEL_SIZE)
    plt.grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
    plt.legend(fontsize=FONT_SIZE)
    fig_name = _savefig(report_dir, 'size_spectra_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_weighted_values(report_dir, mesh, data, const, varname, mask_dom, dom_name):

    output = extract_weighted_data(data, const, mesh, varname, mask_dom)
    with ProgressBar():
        output = output.compute()
    output = extract_time_means(output)

    community_names = extract_community_names(const)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                l = const['length'].isel(c=c)
                output.isel(c=c).plot.line(ax=axes[i, j], color='k', linewidth=THICK_LWD)
                axes[i, j].set_xscale('log')
                axes[i, j].set_xlim(l.min(), l.max())
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Length (log-scale)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel(varname, fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig_name = _savefig(report_dir, 'weighted_%s_by_com_%s.jpg' %(varname, dom_name), 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_diet_values(report_dir, mesh, data, const, mask_dom, dom_name):

    if 'community' in data.dims:
        data = data.rename({'community' : 'c'})

    diet = extract_weighted_data(data, const, mesh, 'community_diet_values', mask_dom=mask_dom)
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
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                l = const['length'].isel(c=c)
                repf = diet.isel(c=c).sum(dim='prey_group')
                axes[i, j].stackplot(l, diet.isel(c=c).T, edgecolor='k', linewidth=THIN_LWD)
                axes[i, j].set_xscale('log')
                axes[i, j].set_xlim(l.min(), l.max())
                axes[i, j].set_ylim(0, repf.max())
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Length (log-scale)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                axes[i, j].legend(legend, fontsize=FONT_SIZE)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig_name = _savefig(report_dir, 'diets_com_%s.jpg' %dom_name, 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _make_meta_template(report_dir, fishing_config_dir, css, data, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('template_meta.html')

    community_names = extract_community_names(const)
    outputs = {}
    outputs['css'] = css
    outputs['community_names'] = community_names
    if fishing_config_dir != '':
        outputs['fleet_names'] = extract_fleet_names(fishing_config_dir)
    outputs['dims'] = data.dims
    outputs['list_dims'] = [d for d in data.dims if 'prey' not in d]
    outputs['start_date'] = data['time'][0].values
    outputs['end_date'] = data['time'][-1].values

    render = template.render(**outputs)

    output_file = os.path.join(report_dir, 'html', 'config_meta.html')
    with open(output_file, 'w') as f:
        f.write(render)


def _make_config_template(report_dir, css, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('template_config.html')

    outputs = {}
    outputs['css'] = css
    outputs['length_figs'] = _plot_wl_community(report_dir, const, 'length', 'meters')
    outputs['weight_figs'] = _plot_wl_community(report_dir, const, 'weight', 'kilograms')
    outputs['trophic_figs'] = _plot_trophic_interactions(report_dir, const)
    outputs['select_figs'] = _plot_ltl_selectivity(report_dir, const)

    render = template.render(**outputs)

    output_file = os.path.join(report_dir, 'html', 'config_report.html')
    with open(output_file, 'w') as f:
        f.write(render)


def _plot_wl_community(report_dir, data, varname, units):

    community_names = extract_community_names(data)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                length = data[varname].isel(c=c)
                axes[i, j].plot(length.values, linewidth=THICK_LWD)
                axes[i, j].set_xlim(0, length.shape[0]-1)
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('%s' %units, fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig_name = _savefig(report_dir, '%s_by_com.jpg' %varname, 'jpg')
    fig.clear()
    plt.close(fig)

    return fig_name


def _plot_trophic_interactions(report_dir, data):

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
    fig_name = _savefig(report_dir, 'trophic_interactions.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_ltl_selectivity(report_dir, data):

    community_names = extract_community_names(data)

    n_community = len(community_names)
    n_plot = n_community
    n_col = 3
    n_row = ceil(n_plot/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    c = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                length = data['length'].isel(c=c)
                var_list = [v for v in data.variables if v.startswith('select_')]
                for v in var_list:
                    temp = data[v].isel(c=0)
                    axes[i, j].plot(length, temp, label=v, linewidth=THICK_LWD)
                axes[i, j].legend(fontsize=FONT_SIZE)
                axes[i, j].set_xlim(length.min(), length.max())
                axes[i, j].set_xscale('log')
                axes[i, j].set_title(community_names['Community ' + str(c)], fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                c = c+1
            else:
                axes[i, j].axis('off')
    fig_name = _savefig(report_dir, 'selectivity_com_by_com.jpg', 'jpg')
    fig.clear()
    plt.close(fig)

    return fig_name


def _make_fisheries_template(report_dir, css, fishing_output_dir, fishing_config_dir, mesh, crs):

    env = jinja2.Environment(loader=jinja2.PackageLoader('apecosm'), autoescape=jinja2.select_autoescape())
    template = env.get_template('template_fisheries.html')

    market, fleet_maps, fleet_summary, fleet_parameters = open_fishing_data(fishing_output_dir)
    fleet_names = extract_fleet_names(fishing_config_dir)
    #fleet_names = ['east_pacific_ps', 'atlantic_ps', 'indian_ps', 'west_pacific_ps', 'longline', 'indian_ss']

    outputs = {}
    outputs['css'] = css
    outputs['fleet_size'] = _plot_fleet_size(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting fleet_size: check')
    outputs['fishing_effective_effort'] = _plot_fishing_effective_effort(report_dir, fleet_maps, fleet_names, mesh, crs)
    print('+++++++++++ Plotting fishing_effective_effort: check')
    outputs['landing_rate_eez_hs'] = _plot_landing_rate_eez_hs(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting landing_rate_eez_hs: check')
    outputs['landing_rate_total'] = _plot_landing_rate_total(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting landing_rate_total: check')
    outputs['landing_rate_by_vessels'] = _plot_landing_rate_by_vessels(report_dir, fleet_maps, fleet_names, mesh, crs)
    print('+++++++++++ Plotting landing_rate_by_vessels: check')
    outputs['landing_rate_density'] = _plot_landing_rate_density(report_dir, fleet_maps, fleet_names, mesh, crs)
    print('+++++++++++ Plotting landing_rate_density: check')
    outputs['average_fishing_distance'] = _plot_average_fishing_distance(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting average_fishing_distance: check')
    outputs['fuel_use_intensity'] = _plot_fuel_use_intensity(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting fuel_use_intensity: check')
    outputs['yearly_profit'] = _plot_yearly_profit(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting yearly_profit: check')
    outputs['savings'] = _plot_savings(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting savings: check')
    outputs['fish_price'] = _plot_fish_price(report_dir, market, fleet_names)
    print('+++++++++++ Plotting fish_price: check')
    outputs['capture_landing_rate'] = _plot_capture_landing_rate(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting capture_landing_rate: check')
    outputs['cost_revenue_by_vessels'] = _plot_cost_revenue_by_vessels(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting cost_revenue_by_vessels: check')
    outputs['fishing_time_fraction'] = _plot_fishing_time_fraction(report_dir, fleet_summary, fleet_names)
    print('+++++++++++ Plotting fishing_time_fraction: check')

    render = template.render(**outputs)

    output_file = os.path.join(report_dir, 'html', 'fisheries_report.html')
    with open(output_file, 'w') as f:
        f.write(render)


def _plot_fleet_size(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (241/256, 140/256, 141/256)
    col_2 = (154/256, 190/256, 219/256)
    col_3 = (165/256, 215/256, 164/256)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, sharex='all')
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                av_1, _, _, time = compute_mean_min_max_xr(fleet_summary[f]['effective_effort'], FISHING_PERIOD)
                av_2, _, _, _ = compute_mean_min_max_xr(fleet_summary[f]['active_vessels'], FISHING_PERIOD)
                av_3, _, _, _ = compute_mean_min_max_xr(fleet_summary[f]['total_vessels'], FISHING_PERIOD)
                axes[i, j].plot(time, av_1, linewidth=THIN_LWD, color='black')
                axes[i, j].plot(time, av_2, linewidth=THIN_LWD, color='black')
                axes[i, j].plot(time, av_3, linewidth=THIN_LWD, color='black')
                axes[i, j].fill_between(time, 0, av_1, color=col_1, alpha=REGULAR_TRANSP, label='Fishing')
                axes[i, j].fill_between(time, av_2, av_1, color=col_2, alpha=REGULAR_TRANSP, label='Sailing')
                axes[i, j].fill_between(time, av_3, av_2, color=col_3, alpha=REGULAR_TRANSP, label='At port')
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Number of vessels', fontsize=FONT_SIZE)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                if cpt == n_plot:
                    axes[i, j].legend(loc='best', fontsize=FONT_SIZE)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'fleet_size.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_fishing_effective_effort(report_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                raw = fleet_maps[f]['effective_effort_density'].isel(time=-1)
                raw_val = raw.values.copy()
                raw_val[raw_val == 0] = sys.float_info.min
                extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
                data = np.log10(raw_val)
                cb_lb = 0
                cb_ub = 1
                if len(data[extract]) > 0:
                    cb_lb = np.percentile(data[extract], CB_THRESH)
                    cb_ub = np.percentile(data[extract], 100-CB_THRESH)
                cs = axes[i, j].pcolormesh(lon_f, lat_f, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
                # cs = axes[i, j].pcolormesh(lon_f[:-1, :-1], lat_f[:-1, :-1], test[1:-1, 1:-1],  cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                if LAND_BACKGROUND:
                    axes[i, j].add_feature(cfeature.LAND, zorder=100)
                    axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'fishing_effective_effort.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_landing_rate_eez_hs(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (241/256, 140/256, 141/256)
    col_2 = (154/256, 190/256, 219/256)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, sharex='all')
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                av_1, _, _, time = compute_mean_min_max_xr(0.000001*365*fleet_summary[f]['current_total_landings_rate_from_EEZ'], FISHING_PERIOD)
                av_2, _, _, _ = compute_mean_min_max_xr(0.000001*365*(fleet_summary[f]['step_landings'] - fleet_summary[f]['current_total_landings_rate_from_EEZ']), FISHING_PERIOD)
                axes[i, j].plot(time, av_1, linewidth=THIN_LWD, color='black')
                axes[i, j].plot(time, av_1+av_2, linewidth=THIN_LWD, color='black')
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes[i, j].fill_between(time, av_1 + av_2, av_1, color=col_2, alpha=REGULAR_TRANSP, label='HS')
                axes[i, j].fill_between(time, av_1, color=col_1, alpha=REGULAR_TRANSP, label='EEZ')
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Landing rate (MT.years-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                if cpt == n_plot:
                    axes[i, j].legend(loc='best', fontsize=FONT_SIZE)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'landing_rate_eez_hs.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_landing_rate_total(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(0.000001*365*fleet_summary[f]['step_landings'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year landing rate : %.2f MT.years-1' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Landing rate (MT.years-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'landing_rate_total.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_landing_rate_by_vessels(report_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt + 1
            if cpt <= n_plot:
                raw = fleet_maps[f]['landing_rate_by_vessel'].isel(time=-1)
                raw_val = raw.values.copy()
                raw_val[raw_val == 0] = sys.float_info.min
                extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
                data = np.log10(raw_val)
                cb_lb = 0
                cb_ub = 1
                if len(data[extract]) > 0:
                    cb_lb = np.percentile(data[extract], CB_THRESH)
                    cb_ub = np.percentile(data[extract], 100 - CB_THRESH)
                cs = axes[i, j].pcolormesh(lon_f, lat_f, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.set_label('T/day-1', fontsize=FONT_SIZE)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                if LAND_BACKGROUND:
                    axes[i, j].add_feature(cfeature.LAND, zorder=100)
                    axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'landing_rate_by_vessels.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_landing_rate_density(report_dir, fleet_maps, fleet_names, mesh, crs_out):

    crs_in = ccrs.PlateCarree()

    n_fleet = len(fleet_maps)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    lon_f = np.squeeze(mesh['glamf'].values)
    lat_f = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI, subplot_kw={'projection': crs_out})
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt + 1
            if cpt <= n_plot:
                raw = fleet_maps[f]['landing_rate'].isel(time=-1)
                raw_val = raw.values.copy()
                raw_val[raw_val == 0] = sys.float_info.min
                extract = (raw_val != sys.float_info.min) & (np.isnan(raw_val) == False)
                data = np.log10(raw_val)
                cb_lb = 0
                cb_ub = 1
                if len(data[extract]) > 0:
                    cb_lb = np.percentile(data[extract], CB_THRESH)
                    cb_ub = np.percentile(data[extract], 100 - CB_THRESH)
                cs = axes[i, j].pcolormesh(lon_f, lat_f, data[1:, 1:], cmap=COL_MAP, transform=crs_in, vmin=cb_lb, vmax=cb_ub)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                cb = plt.colorbar(cs, shrink=CB_SHRINK)
                cb.ax.tick_params(labelsize=LABEL_SIZE)
                cb.ax.yaxis.get_offset_text().set(size=FONT_SIZE)
                if LAND_BACKGROUND:
                    axes[i, j].add_feature(cfeature.LAND, zorder=100)
                    axes[i, j].add_feature(cfeature.COASTLINE, zorder=101)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'landing_rate_density.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_average_fishing_distance(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (255/255, 0/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(fleet_summary[f]['average_fishing_distance_to_ports_of_active_vessels'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year distance : %.1f km' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Average fishing distance (km)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'average_fishing_distance.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_fuel_use_intensity(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (160/255, 32/255, 240/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(fleet_summary[f]['average_fuel_use_intensity'], FISHING_PERIOD)
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year FUI : %.1f kL.T-1' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Fuel use intensity (kL.T-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'fuel_use_intensity.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_yearly_profit(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (255/255, 165/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(0.001*365*fleet_summary[f]['step_profits'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year profit : %.1f M$.years-1' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Yearly profit (M$.years-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'yearly_profit.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_savings(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 100/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(0.001*fleet_summary[f]['savings'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year savings : %.1f M$' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Savings (M$)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'savings.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_fish_price(report_dir, market, fleet_names):

    n_fleet = len(market['fleet'])
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (167/255, 61/255, 11/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                if f == 4:
                    average, maxi, mini, time = compute_mean_min_max_xr(market['average_price'].isel(fleet=f, community=4), FISHING_PERIOD)
                else:
                    average, maxi, mini, time = compute_mean_min_max_xr(market['average_price'].isel(fleet=f, community=1), FISHING_PERIOD)
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year fish price : %.1f $.kg-1' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Fish price ($.kg-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'fish_price.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_capture_landing_rate(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)
    col_2 = (255/255, 0/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average_1, maxi_1, mini_1, time = compute_mean_min_max_xr(fleet_summary[f]['average_capture_rate_by_active_vessel'], FISHING_PERIOD)
                average_2, maxi_2, mini_2, _ = compute_mean_min_max_xr(fleet_summary[f]['average_landing_rate_by_active_vessel'], FISHING_PERIOD)
                axes[i, j].plot(time, average_1, linewidth=THICK_LWD, color=col_1, label='Capture')
                axes[i, j].fill_between(time, mini_1, average_1, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average_1, maxi_1, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].plot(time, average_2, linewidth=THICK_LWD, color=col_2, label='Landing')
                axes[i, j].fill_between(time, mini_2, average_2, color=col_2, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average_2, maxi_2, color=col_2, alpha=HIGH_TRANSP)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Capture and landing rate (T.day-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                if cpt == n_plot:
                    axes[i, j].legend(loc='best', fontsize=FONT_SIZE)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'capture_landing_rate.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_cost_revenue_by_vessels(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 0/255, 255/255)
    col_2 = (255/255, 0/255, 0/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average_1, maxi_1, mini_1, time = compute_mean_min_max_xr(fleet_summary[f]['average_cost_by_active_vessels'], FISHING_PERIOD)
                average_2, maxi_2, mini_2, _ = compute_mean_min_max_xr(fleet_summary[f]['average_cost_by_active_vessels']+fleet_summary[f]['average_cost_by_active_vessels'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%d'))
                axes[i, j].plot(time, average_1, linewidth=THICK_LWD, color=col_1, label='Cost')
                axes[i, j].fill_between(time, mini_1, average_1, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average_1, maxi_1, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].plot(time, average_2, linewidth=THICK_LWD, color=col_2, label='Revenue')
                axes[i, j].fill_between(time, mini_2, average_2, color=col_2, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average_2, maxi_2, color=col_2, alpha=HIGH_TRANSP)
                axes[i, j].set_title(fleet_names[f], fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Cost and revenue by active vessels (k$.day-1)', fontsize=FONT_SIZE)
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                if cpt == n_plot:
                    axes[i, j].legend(loc='best', fontsize=FONT_SIZE)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'cost_revenue_by_vessels.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


def _plot_fishing_time_fraction(report_dir, fleet_summary, fleet_names):

    n_fleet = len(fleet_summary)
    n_plot = n_fleet
    n_col = 3
    n_row = ceil(n_plot/n_col)

    col_1 = (0/255, 104/255, 139/255)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*FIG_WIDTH, n_row*FIG_HEIGHT), dpi=FIG_DPI)
    f = 0
    cpt = 0
    for i in range(n_row):
        for j in range(n_col):
            cpt = cpt+1
            if cpt <= n_plot:
                average, maxi, mini, time = compute_mean_min_max_xr(fleet_summary[f]['average_fishing_time_fraction_of_active_vessels'], FISHING_PERIOD)
                axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axes[i, j].plot(time, average, linewidth=THICK_LWD, color=col_1)
                axes[i, j].fill_between(time, mini, average, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].fill_between(time, average, maxi, color=col_1, alpha=HIGH_TRANSP)
                axes[i, j].set_title('%s - last year fishing time fraction : %.2f' % (fleet_names[f], average[-1]), fontsize=FONT_SIZE)
                axes[i, j].set_xlabel('Time (years)', fontsize=FONT_SIZE)
                axes[i, j].set_ylabel('Fishing time fraction', fontsize=FONT_SIZE)
                axes[i, j].set_ylim([-0.05, 1.05])
                axes[i, j].tick_params(axis='both', labelsize=LABEL_SIZE)
                axes[i, j].grid(color=COL_GRID, linestyle='dashdot', linewidth=REGULAR_LWD)
                f = f+1
            else:
                axes[i, j].axis('off')
    fig.tight_layout()
    fig_name = _savefig(report_dir, 'fishing_time_fraction.jpg', 'jpg')
    fig.clear()
    plt.close(fig)
    return fig_name


if __name__ == '__main__':

    #pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    print('toto')
