# from traitlets.config import Config
# from jupyter_core.command import main as jupymain
# from nbconvert.exporters import HTMLExporter, PDFExporter
# from nbconvert.preprocessors import TagRemovePreprocessor
from functools import total_ordering
import subprocess
from apecosm.constants import LTL_NAMES
from .diags import compute_size_cumprop
from .extract import extract_oope_data, extract_time_means, open_apecosm_data, open_constants, open_mesh_mask, extract_weighted_data, extract_mean_size
from .misc import extract_community_names, find_percentile
from .size_spectra import plot_oope_spectra
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
# import papermill as pm
import pkg_resources
import os
import jinja2
import os
import io
import tempfile
import urllib
plt.rcParams['text.usetex'] = False
import cartopy.feature as cfeature
import shutil
from glob import glob


def report(input_dir, mesh_file, domain_file=None, crs=ccrs.PlateCarree(), output_dir='report', filecss='default', xarray_args={}):

    mesh = open_mesh_mask(mesh_file)
    const = open_constants(input_dir)
    data = open_apecosm_data(input_dir, **xarray_args)

    # If a domain file is provided, extracts it and
    # store it into a dictionnary
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

    # process the banner file, by adding as many tabs as do,ains
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
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
    _make_config_template(output_dir, css, data, const)
    _make_result_template(output_dir, css, data, const, mesh, crs)
    for domname in domains:
        _make_result_template(output_dir, css, data, const, mesh, crs, domains, domname)


    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template.html")

    outputs = {}
    outputs['css'] = css

    render = template.render(**outputs)

    with open(os.path.join(output_dir, 'index.html'), "w") as f:
        f.write(render)

def _make_result_template(output_dir, css, data, const, mesh, crs, domains=None, domname=None):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template_results.html")

    if domains is not None:
        maskdom = domains[domname]
    else:
        maskdom = np.ones(mesh['nav_lon'].shape)
        domname = 'global'

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    outputs = {}
    outputs['css'] = css

    outputs['domain_figs'] = _plot_domain_maps(output_dir, mesh, crs, maskdom, domname)

    outputs['ts_figs'] = _plot_time_series(output_dir, mesh, data, const, maskdom, domname)
    outputs['cumbiom_figs'] = _plot_integrated_time_series(output_dir, mesh, data, const, maskdom, domname)
    outputs['maps_figs'] = _plot_mean_maps(output_dir, mesh, data, const, crs, maskdom, domname)

    outputs['mean_length_figs'] = _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, 'length')
    outputs['mean_weight_figs'] = _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, 'weight')

    if 'repfonct_day' in data.variables:
        outputs['repfonct_figs'] = _plot_weighted_values(output_dir, mesh, data, const, 'repfonct_day', maskdom, domname)

    if 'mort_day' in data.variables:
        outputs['mort_figs'] = _plot_weighted_values(output_dir, mesh, data, const, 'mort_day', maskdom, domname)

    if 'community_diet_values' in data.variables:
        outputs['diet_figs'] = _plot_diet_values(output_dir, mesh, data, const, maskdom, domname)

    outputs['spectra_figs'] = _plot_size_spectra(output_dir, mesh, data, const, maskdom, domname)

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'results_report_%s.html' %domname)
    with open(output_file, "w") as f:
        f.write(render)

def _plot_mean_size(output_dir, mesh, data, const, maskdom, domname, varname):

    mean_size_tot = extract_mean_size(data, const, mesh, varname, maskdom=maskdom, aggregate=True)
    mean_size = extract_mean_size(data, const, mesh, varname, maskdom=maskdom)

    filenames = {}

    if varname == 'weight':
        mean_size *= 1000
        mean_size_tot *= 1000
        ylabel = 'Weight (g)'
    else:
        ylabel = 'Length (cm)'
        mean_size *= 100  # conversion in cm
        mean_size_tot *= 100  # conversion in cm

    fig = plt.figure()
    ax = plt.gca()
    mean_size_tot.plot()
    plt.title('Total')
    plt.ylabel(ylabel)
    filenames['Total'] = _savefig(output_dir, 'mean_%s_tot_%s.svg' %(varname, domname))
    plt.close(fig)

    community_names = extract_community_names(const)

    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.gca()
        toplot = mean_size.isel(c=c)
        toplot.plot()
        plt.title(community_names['Community ' + str(c)])
        plt.ylabel(ylabel)
        filenames['Community ' + str(c)] = _savefig(output_dir, 'mean_%s_com_%d_%s.svg' %(varname, c, domname))
        plt.close(fig)

    return filenames

def _plot_diet_values(output_dir, mesh, data, const, maskdom, domname):

    if 'community' in data.dims:
        data = data.rename({'community' : 'c'})

    diet = extract_weighted_data(data, const, mesh, 'community_diet_values', maskdom=maskdom, replace_dims={})
    diet = extract_time_means(diet)

    community_names = extract_community_names(const)

    legend = LTL_NAMES.copy()
    for c in range(data.dims['c']):
        legend.append(community_names['Community ' + str(c)])

    filenames = {}
    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.gca()
        l = const['length'].isel(c=c)
        toplot = diet.isel(c=c)
        repf = toplot.sum(dim='prey_group')
        plt.stackplot(l, toplot.T, edgecolor='k', linewidth=0.5)
        plt.ylim(0, repf.max())
        plt.xlim(l.min(), l.max())
        ax.set_xscale('log')
        plt.title(community_names['Community ' + str(c)])
        plt.legend(legend)

        filenames['Community ' + str(c)] = _savefig(output_dir, 'diets_com_%d_%s.svg' %(c, domname))
        plt.close(fig)

    return filenames

def _make_meta_template(output_dir, css, data, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template_meta.html")

    comnames = extract_community_names(const)

    outputs = {}

    outputs['css'] = css

    dims = data.dims
    list_dims = [d for d in data.dims if 'prey' not in d]

    outputs['comnames'] = comnames
    outputs['dims'] = dims
    outputs['list_dims'] = list_dims
    outputs['start_date'] = data['time'][0].values
    outputs['end_date'] = data['time'][-1].values

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'config_meta.html')
    with open(output_file, "w") as f:
        f.write(render)

def _plot_trophic_interactions(output_dir, data):

    trophic_interact = data['troph_interaction'].values

    comnames = extract_community_names(data)
    xlabel = []
    for c in range(0, trophic_interact[0][0].shape[0]):
        xlabel.append(comnames['Community ' + str(c)])

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.8)
    nd, npred, nprey = trophic_interact.shape
    title = ['Day', 'Night']
    for d in range(2):
        ax = plt.subplot(1, 2, d + 1)
        cs = plt.imshow(trophic_interact[d], origin='lower', interpolation='none', cmap=plt.cm.jet)
        for i in range(nprey + 1):
            plt.axvline(i - 0.5, linestyle='--', linewidth=1, color='w')
            plt.axhline(i - 0.5, linestyle='--', linewidth=1, color='w')
        plt.title(title[d])
        cs.set_clim(0, 1)
        plt.xlabel('Prey')
        plt.ylabel('Predator')
        ax.set_xticks(np.arange(nprey))
        ax.set_yticks(np.arange(npred))
        ax.set_xticklabels(xlabel, rotation=45)
        ax.set_yticklabels(xlabel, rotation=45)
        ax.set_aspect('equal', 'box')
    output = _savefig(output_dir, 'trophic_interactions.svg')
    plt.close(fig)
    return output

def _make_config_template(output_dir, css, data, const):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template_config.html")

    outputs = {}

    outputs['css'] = css

    dims = data.dims
    list_dims = [d for d in data.dims if 'prey' not in d]

    outputs['dims'] = dims
    outputs['list_dims'] = list_dims
    outputs['start_date'] = data['time'][0].values
    outputs['end_date'] = data['time'][-1].values

    outputs['length_figs'] = _plot_wl_community(output_dir, const, 'length', 'meters')
    outputs['weight_figs'] = _plot_wl_community(output_dir, const, 'weight', 'kilograms')
    outputs['select_figs'] = _plot_ltl_selectivity(output_dir, const)

    outputs['trophic_figs'] = _plot_trophic_interactions(output_dir, const)

    render = template.render(**outputs)

    output_file = os.path.join(output_dir, 'html', 'config_report.html')
    with open(output_file, "w") as f:
        f.write(render)

def _plot_weighted_values(output_dir, mesh, data, const, varname, maskdom, domname):

    output = extract_weighted_data(data, const, mesh, varname, maskdom)
    output = extract_time_means(output)

    comnames = extract_community_names(const)

    filenames = {}
    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.gca()
        l = const['length'].isel(c=c)
        toplot = output.isel(c=c)
        plt.plot(l, toplot, color='k')
        ax.set_xscale('log')
        plt.xlim(l.min(), l.max())
        plt.title(comnames['Community ' + str(c)])
        plt.xlabel('Length (log-scale)')
        plt.ylabel(varname)
        plt.ylim(toplot.min(), toplot.max())
        filenames['Community ' + str(c)] = _savefig(output_dir, 'weighted_%s_com_%d_%s.svg' %(varname, c, domname))
        plt.close(fig)
    return filenames


def _plot_integrated_time_series(output_dir, mesh, data, const, maskdom, domname):

    filenames = {}
    size_prop = compute_size_cumprop(mesh, data, const, maskdom=maskdom)
    size_prop = extract_time_means(size_prop)

    comnames = extract_community_names(const)

    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.gca()
        l = const['length'].isel(c=c)
        toplot = size_prop.isel(c=c)
        plt.fill_between(l, 0, toplot, edgecolor='k', facecolor='lightgray')
        ax.set_xscale('log')
        plt.xlim(l.min(), l.max())
        plt.title(comnames['Community ' + str(c)])
        plt.xlabel('Length (log-scale)')
        plt.ylabel('Proportion (%)')
        plt.ylim(0, 100)
        filenames['Community ' + str(c)] = _savefig(output_dir, 'biomass_cumsum_com_%d_%s.svg' %(c, domname))
        plt.close(fig)
    return filenames

def _plot_time_series(output_dir, mesh, data, const, maskdom, domname):

    filenames = {}

    output = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=True, compute_mean=False)
    output = output.sum(dim='w')
    total = output.sum(dim='c')
    comnames = extract_community_names(const)

    fig = plt.figure()
    total.plot()
    plt.title('Total')
    plt.ylabel('Joules  ')
    plt.xticks(rotation=30, ha='right')
    plt.grid()
    plt.xlabel('')
    filenames['Total'] = _savefig(output_dir, 'time_series_total.svg')
    plt.close(fig)

    for c in range(data.dims['c']):
        fig = plt.figure()
        output.isel(c=c).plot()
        plt.title(comnames['Community ' + str(c)])
        plt.ylabel('Joules')
        plt.xticks(rotation=30, ha='right')
        plt.xlabel('')
        plt.grid()
        filenames['Community ' + str(c)] = _savefig(output_dir, 'time_series_com_%d_%s.svg' %(c, domname))
        plt.close(fig)

    return filenames

def _plot_domain_maps(output_dir, mesh, crs, maskdom, domname):

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

    fig = plt.figure()
    ax = plt.axes(projection=crs)
    cs = plt.pcolormesh(lonf, latf, tmask[1:, 1:].astype(int), cmap=plt.cm.jet)
    cs.set_clim(0, 1)
    cb = plt.colorbar(cs)
    plt.title('%s mask' %domname)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    fileout = _savefig(output_dir, 'domain_map_%s.svg' %domname)
    plt.close(fig)
    return fileout

def _plot_mean_maps(output_dir, mesh, data, const, crs, maskdom, domname):

    filenames = {}

    if maskdom is None:
        maskdom = np.ones(lonf.shape)

    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    mesh = mesh.where(maskdom > 0, drop=True)
    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    output = (data['OOPE'] * const['weight_step']).mean(dim='time').sum(dim=['w'])
    output = output.where(output > 0)
    output = output.where(maskdom > 0, drop=True)
    total = output.sum(dim='c')
    total = total.where(total > 0)

    comnames = extract_community_names(const)

    fig = plt.figure()
    ax = plt.axes(projection=crs)
    cs = plt.pcolormesh(lonf, latf, total.isel(y=slice(1, None), x=slice(1, None)))
    cb = plt.colorbar(cs)
    cb.set_label('J/m2')
    plt.title("Total")
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    filenames['Total'] = _savefig(output_dir, 'mean_maps_total_%s.svg' %domname)
    plt.close(fig)

    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.axes(projection=crs)
        cs = plt.pcolormesh(lonf, latf, output.isel(c=c, y=slice(1, None), x=slice(1, None)))
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        cb = plt.colorbar(cs)
        cb.set_label('Joules/m2')
        plt.title(comnames['Community ' + str(c)])
        filenames['Community ' + str(c)] = _savefig(output_dir, 'mean_maps_com_%d_%s.svg' %(c, domname))
        plt.close(fig)

    return filenames


def _savefig(output_dir, figname):

    img_file = os.path.join(output_dir, 'html', 'images', figname)
    plt.savefig(img_file, format="svg", bbox_inches='tight')
    return os.path.join('images', figname)

def _plot_ltl_selectivity(output_dir, data):

    comnames = extract_community_names(data)

    output = {}
    for c in range(data.dims['c']):

        length = data['length'].isel(c=c)
        varlist = [v for v in data.variables if v.startswith('select_')]

        fig = plt.figure()
        ax = plt.gca()
        for v in varlist:
            temp = data[v].isel(c=0)
            plt.plot(length, temp, label=v)
        plt.legend()
        plt.xlim(length.min(), length.max())
        ax.set_xscale('log')
        plt.title(comnames['Community ' + str(c)])
        plt.grid()
        output[c] = _savefig(output_dir, 'selectivity_com_%d.svg' %c)
        plt.close(fig)

    return output

def _plot_wl_community(output_dir, data, varname, units):

    output = {}

    comnames = extract_community_names(data)

    for c in range(data.dims['c']):

        length = data[varname].isel(c=c)

        fig = plt.figure()
        plt.plot(length.values)
        plt.xlim(0, length.shape[0] - 1)
        plt.ylabel('%s' %units)
        plt.title(comnames['Community ' + str(c)])
        plt.grid()
        output[c] = _savefig(output_dir, '%s_com_%d.svg' %(varname, c))
        plt.close(fig)

    return output


def _plot_size_spectra(output_dir, mesh, data, const, maskdom, domname):

     # extract data in the entire domain, integrates over space
    data = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=False, compute_mean=False, replace_dims={}, replace_const_dims={})
    data = extract_time_means(data)

    fig = plt.figure()
    plot_oope_spectra(data, const, output_var='length')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('Length (m)')
    plt.ylabel('OOPE (J/m)')
    plt.legend()
    figname = _savefig(output_dir, 'size_spectra_%s.svg' %domname)
    plt.close(figname)

    return figname


if __name__ == '__main__':

    #pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    print("toto")
