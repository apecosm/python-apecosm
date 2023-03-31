from apecosm.constants import LTL_NAMES
from .diags import compute_size_cumprop
from .extract import extract_oope_data, extract_time_means, open_apecosm_data, open_constants, open_mesh_mask, extract_weighted_data, extract_mean_size, open_fishing_data
from .misc import extract_community_names, compute_mean_min_max_ts, extract_fleet_names
from .size_spectra import plot_oope_spectra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import xarray as xr
import cartopy.crs as ccrs
import pkg_resources
import jinja2
import os
import urllib
plt.rcParams['text.usetex'] = False
import cartopy.feature as cfeature


def report(input_dir, mesh_file, fishing_path, config_path, domain_file=None, crs=ccrs.PlateCarree(), output_dir='report', filecss='default', xarray_args={}):

    mesh = open_mesh_mask(mesh_file)
    if('X' in mesh.dims and 'Y' in mesh.dims and 'Z' in mesh.dims):
        mesh = open_mesh_mask(mesh_file, replace_dims={'X': 'x', 'Y': 'y', 'Z': 'z'})
    const = open_constants(input_dir)
    data = open_apecosm_data(input_dir, **xarray_args)

    # If a domain file is provided, extracts it and
    # store it into a dictionary
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
    #_make_result_template(output_dir, css, data, const, mesh, crs)
    _make_fisheries_template(output_dir, css, fishing_path, config_path, mesh, crs)
    #for domname in domains:
    #    _make_result_template(output_dir, css, data, const, mesh, crs, domains, domname)


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

    diet = extract_weighted_data(data, const, mesh, 'community_diet_values', maskdom=maskdom)
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
    try:
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
    except:
        pass
    fileout = _savefig(output_dir, 'domain_map_%s.svg' %domname)
    plt.close(fig)
    return fileout

def _plot_mean_maps(output_dir, mesh, data, const, crs, maskdom, domname):

    filenames = {}
    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    if maskdom is None:
        maskdom = np.ones(lonf.shape)
    maskdom = xr.DataArray(data=maskdom, dims=['y', 'x'])

    print(mesh.dims)
    #mesh = mesh.drop_dims(["X","Y"])
    mesh = mesh.where(maskdom > 0, drop=True)

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
    try:
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
    except:
        pass
    filenames['Total'] = _savefig(output_dir, 'mean_maps_total_%s.svg' %domname)
    plt.close(fig)

    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.axes(projection=crs)
        cs = plt.pcolormesh(lonf, latf, output.isel(c=c, y=slice(1, None), x=slice(1, None)))
        try:
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE)
        except:
            pass
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
    #data = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=False, compute_mean=False, replace_dims={}, replace_const_dims={})
    data = extract_oope_data(data, mesh, const, maskdom=maskdom, use_wstep=False, compute_mean=False)
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


def _make_fisheries_template(output_dir, css, fishing_path, config_path, mesh, crs):

    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template_fisheries.html")

    market, fleet_maps, fleet_summary, fleet_parameters = open_fishing_data(fishing_path)
    fleet_names = extract_fleet_names(config_path)

    outputs = {}
    outputs['css'] = css

    outputs['fleet_size'] = _plot_fleet_size(output_dir, fleet_summary, fleet_names) #ok
    outputs['fishing_effective_effort'] = _plot_fishing_effective_effort(output_dir, fleet_maps, fleet_names, mesh, crs) #nok
    outputs['landing_rate_eez_hs'] = _plot_landing_rate_eez_hs(output_dir, fleet_summary, fleet_names) #ok
    outputs['landing_rate_total'] = _plot_landing_rate_total(output_dir, fleet_summary, fleet_names) #ok
    outputs['landing_rate_by_vessels'] = _plot_landing_rate_by_vessels(output_dir, fleet_maps, fleet_names, mesh, crs) #nok
    outputs['landing_rate_density'] = _plot_landing_rate_density(output_dir, fleet_maps, fleet_names, mesh, crs) #nok
    outputs['average_fishing_distance'] = _plot_average_fishing_distance(output_dir, fleet_summary, fleet_names) #nok
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
    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet/n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize = (n_col*10, n_row*10), dpi = 300, sharex=True)
    for i in np.arange(nb_fleet):
        plt.subplot(n_row, n_col, i + 1)
        av_1, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['effective_effort'], 365)
        av_2, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['active_vessels'], 365)
        av_3, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['total_vessels'], 365)
        plt.plot(time, av_1, color='red', linewidth=2, label='Fishing')
        plt.plot(time, av_1+av_2, color='blue', linewidth=2, label='Sailing')
        plt.plot(time, av_1+av_2+av_3, color='green', linewidth=2, label='At port')
        plt.fill_between(time, av_1+av_2+av_3, av_1+av_2, color='green', alpha=0.25)
        plt.fill_between(time, av_1+av_2, av_1, color='blue', alpha=0.25)
        plt.fill_between(time, av_1, color='red', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Number of vessels', fontsize=25)
    fig.tight_layout()
    plt.legend(loc='best', fontsize=25)
    figname = _savefig(output_dir, 'fleet_size.svg')
    plt.close(figname)
    return figname


def _plot_fishing_effective_effort(output_dir, fleet_maps, fleet_names, mesh, crs):

    nb_fleet = len(fleet_maps)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 12, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        ax = plt.subplot(n_row, n_col, i + 1, projection=crs)
        cs = plt.pcolormesh(lonf, latf, fleet_maps[i]['effective_effort_density'].sum(dim='time', skipna=True)[1:,1:])
        plt.title(fleet_names[i], fontsize=25)
        cb = plt.colorbar(cs, shrink=0.4)
        cb.set_label('T/day-1', fontsize=25)
        cb.ax.tick_params(labelsize=25)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        plt.grid()
    fig.tight_layout()
    figname = _savefig(output_dir, 'fishing_effective_effort.svg')
    plt.close(figname)
    return figname


def _plot_landing_rate_eez_hs(output_dir, fleet_summary, fleet_names):
    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300, sharex=True)
    for i in np.arange(nb_fleet):
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        av_1, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * fleet_summary[i]['current_total_landings_rate_from_EEZ'], 365)
        av_2, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * (fleet_summary[i]['step_landings']-fleet_summary[i]['current_total_landings_rate_from_EEZ']), 365)
        plt.plot(time, av_1, color='red', linewidth=2,label='EEZ')
        plt.plot(time, av_1+av_2, color='blue', linewidth=2,label='HS')
        plt.fill_between(time, av_1+av_2, av_1, color='blue', alpha=0.25)
        plt.fill_between(time, av_1, color='red', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Landing rate (million T.years-1)', fontsize=25)
    fig.tight_layout()
    plt.legend(loc='best', fontsize=25)
    figname = _savefig(output_dir, 'landing_rate_eez_hs.svg')
    plt.close(figname)
    return figname


def _plot_landing_rate_total(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * fleet_summary[i]['step_landings'], 365)
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Landing rate (MT.years-1)', fontsize=25)
        plt.title('last year landing rate : %.2f MT.years-1' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'landing_rate_total.svg')
    plt.close(figname)
    return figname


def _plot_landing_rate_by_vessels(output_dir, fleet_maps, fleet_names, mesh, crs):
    nb_fleet = len(fleet_maps)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 12, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        ax = plt.subplot(n_row, n_col, i + 1, projection=crs)
        cs = plt.pcolormesh(lonf, latf, fleet_maps[i]['landing_rate_by_vessel'].sum(dim='time', skipna=True)[1:,1:])
        plt.title(fleet_names[i], fontsize=25)
        cb = plt.colorbar(cs, shrink=0.4)
        cb.set_label('T/day-1', fontsize=25)
        cb.ax.tick_params(labelsize=25)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        plt.grid()
    fig.tight_layout()
    figname = _savefig(output_dir, 'landing_rate_by_vessels.svg')
    plt.close(figname)
    return figname


def _plot_landing_rate_density(output_dir, fleet_maps, fleet_names, mesh, crs):
    nb_fleet = len(fleet_maps)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 12, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        ax = plt.subplot(n_row, n_col, i + 1, projection=crs)
        cs = plt.pcolormesh(lonf, latf, fleet_maps[i]['landing_rate'].sum(dim='time', skipna=True)[1:,1:])
        plt.title(fleet_names[i], fontsize=25)
        cb = plt.colorbar(cs, shrink=0.4)
        cb.set_label('T/day-1', fontsize=25)
        cb.ax.tick_params(labelsize=25)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        plt.grid()
    fig.tight_layout()
    figname = _savefig(output_dir, 'landing_rate_density.svg')
    plt.close(figname)
    return figname


def _plot_average_fishing_distance(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(0*fleet_summary[i]['step_landings'], 365)
        plt.subplot(n_row, n_col, i + 1)
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Average fishing distance (km)', fontsize=25)
        plt.title('last year distance : %.1f km' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'average_fishing_distance.svg')
    plt.close(figname)
    return figname


def _plot_fuel_use_intensity(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['average_fuel_use_intensity'], 365)
        plt.subplot(n_row, n_col, i + 1)
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Fuel use intensity (kL.T-1)', fontsize=25)
        plt.title('last year FUI : %.1f kL.T-1' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'fuel_use_intensity.svg')
    plt.close(figname)
    return figname


def _plot_yearly_profit(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(0.000001 * 365 * fleet_summary[i]['step_profits'], 365)
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Yearly profit (G$.years-1)', fontsize=25)
        plt.title('last year profit : %.1f G$.years-1' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'yearly_profit.svg')
    plt.close(figname)
    return figname


def _plot_savings(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(0.000001 * fleet_summary[i]['savings'], 365)
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Savings (G$)', fontsize=25)
        plt.title('last year savings : %.1f G$' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'savings.svg')
    plt.close(figname)
    return figname


def _plot_fish_price(output_dir, market, fleet_names):
    nb_fleet = len(market['fleet'])
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(market['average_price'].isel(fleet=i), 365)
        plt.subplot(n_row, n_col, i + 1)
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Fish price ($.kg-1)', fontsize=25)
        plt.title('last year fish price : %.1f $.kg-1' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'fish_price.svg')
    plt.close(figname)
    return figname


def _plot_capture_landing_rate(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average_1, maxi_1, mini_1, time_1 = compute_mean_min_max_ts(fleet_summary[i]['average_capture_rate_by_active_vessel'], 365)
        average_2, maxi_2, mini_2, time_2 = compute_mean_min_max_ts(fleet_summary[i]['average_landing_rate_by_active_vessel'], 365)
        plt.subplot(n_row, n_col, i + 1)
        plt.plot(time_1, average_1, linewidth=2, color='blue', label='Capture')
        plt.fill_between(time_1, mini_1, average_1, color='blue', alpha=0.25)
        plt.fill_between(time_1, average_1, maxi_1, color='blue', alpha=0.25)
        plt.plot(time_2, average_2, linewidth=2, color='red', label='Landing')
        plt.fill_between(time_2, mini_2, average_2, color='red', alpha=0.25)
        plt.fill_between(time_2, average_2, maxi_2, color='red', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Capture and landing rate (T.day-1)', fontsize=25)
    plt.legend(loc='best', fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'capture_landing_rate.svg')
    plt.close(figname)
    return figname


def _plot_cost_revenue_by_vessels(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average_1, maxi_1, mini_1, time_1 = compute_mean_min_max_ts(fleet_summary[i]['average_cost_by_active_vessels'], 365)
        average_2, maxi_2, mini_2, time_2 = compute_mean_min_max_ts(fleet_summary[i]['average_profit_by_active_vessels'], 365)
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.plot(time_1, average_1, linewidth=2, color='blue', label='Cost')
        plt.fill_between(time_1, mini_1, average_1, color='blue', alpha=0.25)
        plt.fill_between(time_1, average_1, maxi_1, color='blue', alpha=0.25)
        plt.plot(time_2, average_2, linewidth=2, color='red', label='Revenue')
        plt.fill_between(time_2, mini_2, average_2, color='red', alpha=0.25)
        plt.fill_between(time_2, average_2, maxi_2, color='red', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Cost and revenue by active vessels (k$.day-1)', fontsize=25)
    plt.legend(loc='best', fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'cost_revenue_by_vessels.svg')
    plt.close(figname)
    return figname


def _plot_fishing_time_fraction(output_dir, fleet_summary, fleet_names):

    nb_fleet = len(fleet_summary)
    n_col = 3
    n_row = int(nb_fleet / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 10), dpi=300)
    for i in np.arange(nb_fleet):
        average, maxi, mini, time = compute_mean_min_max_ts(fleet_summary[i]['average_fishing_time_fraction_of_active_vessels'], 365)
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.plot(time, average, linewidth=2, color='blue')
        plt.fill_between(time, mini, average, color='blue', alpha=0.25)
        plt.fill_between(time, average, maxi, color='blue', alpha=0.25)
        plt.grid()
        plt.title(fleet_names[i], fontsize=25)
        plt.tick_params(axis='both', labelsize=25)
        plt.xlabel('Time (years)', fontsize=25)
        plt.ylabel('Fishing time fraction', fontsize=25)
        plt.ylim([0,1])
        plt.title('last year fishing time fraction : %.2f' % average[-1], fontsize=25)
    fig.tight_layout()
    figname = _savefig(output_dir, 'fishing_time_fraction.svg')
    plt.close(figname)
    return figname


if __name__ == '__main__':

    #pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    print("toto")
