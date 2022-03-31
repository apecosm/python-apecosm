# from traitlets.config import Config
# from jupyter_core.command import main as jupymain
# from nbconvert.exporters import HTMLExporter, PDFExporter
# from nbconvert.preprocessors import TagRemovePreprocessor
import subprocess
import nbformat as nbf
from .diags import compute_size_cumprop
from .extract import extract_oope_data, extract_time_means, open_apecosm_data, open_constants, open_mesh_mask
from .misc import extract_community_names, find_percentile
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


def report(input_dir, mesh_file, crs=ccrs.PlateCarree(), output_dir='report', filecss='default', xarray_args={}):
    
    mesh = open_mesh_mask(mesh_file)
    const = open_constants(input_dir)
    data = open_apecosm_data(input_dir)

    # create the output architecture
    
    # first create html folder
    html_dir = os.path.join(output_dir, 'html')
    os.makedirs(html_dir, exist_ok=True)
    
    images_dir = os.path.join(output_dir, 'html/images')
    os.makedirs(images_dir, exist_ok=True)
    
    # create css folder
    css_dir = os.path.join(output_dir, 'css')
    os.makedirs(css_dir, exist_ok=True)   

    filebanner = pkg_resources.resource_filename('apecosm', os.path.join('templates', 'banner.html'))
    shutil.copyfile(filebanner, os.path.join(output_dir, 'html', 'banner.html'))
    
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
    
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template.html")
    
    outputs = {}
    outputs['css'] = css
    
    render = template.render(**outputs)
    
    with open(os.path.join(output_dir, 'index.html'), "w") as f:
        f.write(render)
        
def _make_result_template(output_dir, css, data, const, mesh, crs):
    
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template_results.html")
    
    outputs = {}
    outputs['css'] = css
    
    outputs['ts_figs'] = _plot_time_series(output_dir, mesh, data, const)
    outputs['cumbiom_figs'] = _plot_integrated_time_series(output_dir, mesh, data, const)
    outputs['maps_figs'] = _plot_mean_maps(output_dir, mesh, data, const, crs)
           
    render = template.render(**outputs)
    
    output_file = os.path.join(output_dir, 'html', 'results_report.html')
    with open(output_file, "w") as f:
        f.write(render)
        
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
    trophic_interact.shape

    fig = plt.figure()
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
    
def _plot_integrated_time_series(output_dir, mesh, data, const):
    
    filenames = {}
    size_prop = compute_size_cumprop(mesh, data, const, maskdom=None)
    size_prop = extract_time_means(size_prop)
    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.gca()
        l = const['length'].isel(c=c)
        toplot = size_prop.isel(c=c)
        plt.fill_between(l, 0, toplot, edgecolor='k', facecolor='lightgray')
        ax.set_xscale('log')
        plt.xlim(l.min(), l.max())
        plt.title('Community ' + str(c))
        plt.xlabel('Length (log-scale)')
        plt.ylabel('Proportion (%)')
        filenames['Community ' + str(c)] = _savefig(output_dir, 'biomass_cumsum_com_%d.svg' %c)
        plt.close(fig)
    return filenames                 
    
def _plot_time_series(output_dir, mesh, data, const):
    
    filenames = {}
    
    output = (data['OOPE'] * const['weight_step'] * mesh['e1t'] * mesh['e2t']).sum(dim=['w', 'x', 'y'])
    output = extract_oope_data(data, mesh, const, maskdom=None, use_wstep=True, compute_mean=False, replace_dims={}, replace_const_dims={})
    output = output.sum(dim='w')
    total = output.sum(dim='c')
    
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
        plt.title('Community ' + str(c))
        plt.ylabel('Joules')
        plt.xticks(rotation=30, ha='right')
        plt.xlabel('')
        plt.grid()
        filenames['Community ' + str(c)] = _savefig(output_dir, 'time_series_com_%d.svg' %c)
        plt.close(fig)
    
    return filenames

def _plot_mean_maps(output_dir, mesh, data, const, crs):
    
    filenames = {}
    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)
    
    output = (data['OOPE'] * const['weight_step']).mean(dim='time').sum(dim=['w'])
    output = output.where(output > 0)
    total = output.sum(dim='c')
    total = total.where(total > 0)
  
    fig = plt.figure()
    ax = plt.axes(projection=crs)
    cs = plt.pcolormesh(lonf, latf, total.isel(y=slice(1, None), x=slice(1, None)))
    cb = plt.colorbar(cs)
    cb.set_label('J/m2')
    plt.title("Total")
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    filenames['Total'] = _savefig(output_dir, 'mean_maps_total.svg')
    plt.close(fig)
    
    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.axes(projection=crs)
        cs = plt.pcolormesh(lonf, latf, output.isel(c=c, y=slice(1, None), x=slice(1, None)))
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        cb = plt.colorbar(cs)
        cb.set_label('Joules/m2')
        plt.title('Community ' + str(c))
        filenames['Community ' + str(c)] = _savefig(output_dir, 'mean_maps_com_%d.svg' %c)
        plt.close(fig)
    
    return filenames
    
    
def _savefig(output_dir, figname):
    
    img_file = os.path.join(output_dir, 'html', 'images', figname)
    plt.savefig(img_file, format="svg", bbox_inches='tight')
    return os.path.join('images', figname)
        
def _plot_ltl_selectivity(output_dir, data):
    
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
        plt.title('Community ' + str(c))
        plt.grid()
        output[c] = _savefig(output_dir, 'selectivity_com_%d.svg' %c)
        plt.close(fig)
        
    return output
  
def _plot_wl_community(output_dir, data, varname, units):
    
    output = {}
    
    for c in range(data.dims['c']):
        
        length = data[varname].isel(c=c)
    
        fig = plt.figure()
        plt.plot(length.values)
        plt.xlim(0, length.shape[0] - 1)
        plt.ylabel('%s' %units)
        plt.title('Community ' + str(c))
        plt.grid()
        output[c] = _savefig(output_dir, '%s_com_%d.svg' %(varname, c))
        plt.close(fig)
        
    return output

        

# def report(input_dir, meshfile, output):

#     suffindex = output.rfind(".")
#     fmt = output[suffindex + 1:]
    
#     tempfile = output.replace(fmt, "ipynb")
    
#     if fmt not in ['pdf', 'html']:
#         message = "The output format must be 'html' or 'pdf'. The program will stop"
#         print(message)
#         sys.exit(1)

#     template_file = pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
#     param = dict(input_dir=input_dir, input_mesh=meshfile)
#     pm.execute_notebook(template_file, tempfile, parameters=param)

#     c = Config()

#     c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
#     c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
#     c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)    
#     c.TagRemovePreprocessor.enabled = True

#     if(fmt == 'html'):
#         c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
#         exporter = HTMLExporter(config=c)
#     else:
#         c.PDFExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
#         exporter = PDFExporter(config=c)
    
#     strout, objout = exporter.from_filename(tempfile)
#     if(type(strout) == bytes):
#         wmode = "wb"
#     else:
#         wmode = "w"
    
#     with open(output, wmode) as fout:
#         fout.write(strout)
        
#     os.remove(tempfile)
    
def plot_report_ts(input_dir, input_mesh):

    # extract data in the entire domain, integrates over space
    data = extract_oope_data(input_dir, input_mesh, domain_name='global')
    time = data['time'].values
    comm = data['community'].values.astype(np.int)
    weight = data['weight'].values
    oope = data['OOPE'].values
    
    try:
        date = [d.strftime("%Y-%m") for d in time]
        date = np.array(date)
        rotation = 45
        ha = 'right'
    except:
        date = time
        rotation = 0
        ha = 'center'

    ntime = len(time)
    stride = ntime // 20
    stride = max(1, stride)
    labelindex = np.arange(0, ntime, stride)

    # move time back into time
    time = np.arange(ntime)

    print('Mean total biomass, all communities: %e J' %(np.mean(np.sum(oope, axis=(1, 2)), axis=0)))
    for i in range(0, len(comm)):
        print('Mean total biomass, community %d: %e J' %(i + 1, np.mean(np.sum(oope[:, i, :], axis=(-1)), axis=0)))
    
    # plot total biomass
    fig = plt.figure()
    plt.plot(time, np.sum(oope, axis=(1, 2)))
    plt.title('Total biomass, all communities')
    plt.xlabel('Time')
    plt.ylabel('OOPE (J)')
    plt.gca().set_xticks(time[labelindex])
    plt.gca().set_xticklabels(date[labelindex], rotation=rotation, ha=ha)
    plt.show()
    plt.close(fig)
        
    for i in range(0, len(comm)):
        fig = plt.figure()
        plt.plot(time, np.sum(oope[:, i, :], axis=(1)))
        plt.title('Total biomass, community %d' %(i + 1))
        plt.xlabel('Time')
        plt.ylabel('OOPE (J)')
        plt.gca().set_xticks(time[labelindex])
        plt.gca().set_xticklabels(date[labelindex], rotation=rotation, ha=ha)
        plt.show()
        plt.close(fig)
    

def plot_report_map(input_dir, input_mesh, draw_features=True):
    
    constants = open_constants(input_dir)
    wstep = constants['weight_step'].values
    wstep = wstep[np.newaxis, np.newaxis, :]
    
    dataset = xr.open_mfdataset("%s/*OOPE*nc" %input_dir, combine='by_coords')
    dataset = extract_time_means(dataset)
    oope = dataset['OOPE'] * wstep   #  conversion in J/m2  (lat, lon, comm, weight)
    oope = oope.sum(dim=('w')).to_masked_array()   # sum over all weight classes: (lat, lon, com)
    
    mesh = xr.open_dataset(input_mesh)
    lon = mesh['glamf'].values[0]
    lat = mesh['gphif'].values[0]
    tmask = mesh['tmask'].values[0, 0]
    
    projection = ccrs.PlateCarree()
    
    fig = plt.figure()
    ax = plt.axes(projection=projection)
    if(draw_features):
        ax.add_feature(cfeature.LAND, zorder=1000)
        ax.add_feature(cfeature.COASTLINE, zorder=1001)
    temp = np.sum(oope, axis=-1)
    temp = np.log10(temp, where=(temp > 0))
    cmin, cmax = find_percentile(temp)
    cs = plt.pcolormesh(lon, lat, temp[1:, 1:], transform=projection)
    cs.set_clim(cmin, cmax)
    cb = plt.colorbar(cs, orientation='horizontal')
    cb.set_label('LOG10 OOPE (J/m2), all communities')
    plt.show()
    plt.close(fig)   
    
    for i in range(oope.shape[-1]):
        fig = plt.figure()
        ax = plt.axes(projection=projection)
        if(draw_features):
            ax.add_feature(cfeature.LAND, zorder=1000)
            ax.add_feature(cfeature.COASTLINE, zorder=1001)
        temp = oope[:, :, i]
        temp = np.log10(temp, where=(temp > 0))
        cmin, cmax = find_percentile(temp)
        cs = plt.pcolormesh(lon, lat, temp[1:, 1:], transform=projection)
        cs.set_clim(cmin, cmax)
        cb = plt.colorbar(cs, orientation='horizontal')
        cb.set_label('LOG10 OOPE (J/m2), community %d' %(i + 1))
        plt.show()
        plt.close(fig)   

def plot_report_size_spectra(input_dir, input_mesh):
    
     # extract data in the entire domain, integrates over space
    data = extract_oope_data(input_dir, input_mesh, domain_name='global', use_wstep=False)
    data = extract_time_means(data)
    data = data['OOPE']  # community, w
    ncom, nw = data.shape
    
    const = extract_apecosm_constants(input_dir)
    weight = const['weight'].values
    length = const['length'].values

    if(weight.ndim == 1):
        weight = np.tile(weight, (ncom, 1))
        length = np.tile(length, (ncom, 1))
    
    ncom, nweight = data.shape
    
    cmap = getattr(plt.cm, plt.rcParams['image.cmap'])
    
    fig = plt.figure()
    ax = plt.gca()
    for icom in range(ncom):
        color = cmap(icom / (ncom - 1))
        ax.plot(weight[icom], data[icom], marker='o', linestyle='none', color=color, label='Community %d' %(icom + 1))
    ax.loglog()
    plt.legend()
    plt.xlabel('Weight (g)')
    plt.ylabel('OOPE (J/kg)')
    plt.show()
    plt.close(fig)
    


if __name__ == '__main__':

    #pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    print("toto")
