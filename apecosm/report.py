# from traitlets.config import Config
# from jupyter_core.command import main as jupymain
# from nbconvert.exporters import HTMLExporter, PDFExporter
# from nbconvert.preprocessors import TagRemovePreprocessor
import subprocess
import nbformat as nbf
from .extract import extract_oope_data, extract_time_means, extract_apecosm_constants
from .misc import find_percentile
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


def report(input_dir, mesh_file, crs=ccrs.PlateCarree(), output_file='report.html', filecss='default', xarray_args={}):
    
    mesh = xr.open_dataset(mesh_file)
    
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
    
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template.html")
    
    outputs = {}
    
    outputs['css'] = css
        
    data = xr.open_mfdataset(os.path.join(input_dir, '*.nc'), **xarray_args)
    dims = data.dims
    list_dims = [d for d in data.dims if 'prey' not in d]
    
    outputs['dims'] = dims
    outputs['list_dims'] = list_dims
    outputs['start_date'] = data['time'][0].values
    outputs['end_date'] = data['time'][-1].values

    outputs['length_figs'] = _plot_wl_community(data, 'length', 'meters')
    outputs['weight_figs'] = _plot_wl_community(data, 'weight', 'kilograms')
    outputs['select_figs'] = _plot_ltl_selectivity(data)
    
    outputs['ts_figs'] = _plot_time_series(data)
    outputs['maps_figs'] = _plot_mean_maps(mesh, data, crs)
    
    render = template.render(**outputs)
    
    with open(output_file, "w") as f:
        f.write(render)
    
def _plot_time_series(data):
    
    filenames = {}
    
    output = (data['OOPE'] * data['weight_step']).sum(dim=['w', 'x', 'y'])
    total = output.sum(dim='c')
    
    fig = plt.figure()
    total.plot()
    plt.title('Total')
    filenames['Total'] = _savefig()
    plt.close(fig)
    
    for c in range(data.dims['c']):
        fig = plt.figure()
        output.isel(c=c).plot()
        plt.title('Community ' + str(c))
        filenames['Community ' + str(c)] = _savefig()
        plt.close(fig)
    
    return filenames

def _plot_mean_maps(mesh, data, crs):
    
    filenames = {}
    lonf = np.squeeze(mesh['glamf'].values)
    latf = np.squeeze(mesh['gphif'].values)
    
    output = (data['OOPE'] * data['weight_step']).mean(dim='time').sum(dim=['w'])
    output = output.where(output > 0)
    total = output.sum(dim='c')
  
    fig = plt.figure()
    ax = plt.axes(projection=crs)
    cs = plt.pcolormesh(lonf, latf, total.isel(y=slice(1, None), x=slice(1, None)))
    plt.colorbar(cs)
    plt.title("Total")
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    filenames['Total'] = _savefig()
    plt.close(fig)
    
    for c in range(data.dims['c']):
        fig = plt.figure()
        ax = plt.axes(projection=crs)
        cs = plt.pcolormesh(lonf, latf, output.isel(c=c, y=slice(1, None), x=slice(1, None)))
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        plt.colorbar(cs)
        plt.title('Community ' + str(c))
        filenames['Community ' + str(c)] = _savefig()
        plt.close(fig)
    
    return filenames
    
    
def _savefig():
    
    buf = io.BytesIO()
    plt.savefig(buf, format="svg")
    fp = tempfile.NamedTemporaryFile() 
    with open(f"{fp.name}.svg", 'wb') as ff:
        ff.write(buf.getvalue()) 

    buf.close()
    
    return f"{fp.name}.svg"
    
        
def _plot_ltl_selectivity(data):
    
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
        output[c] = _savefig()
        plt.close(fig)
        
    return output
  
def _plot_wl_community(data, varname, units):
    
    output = {}
    
    for c in range(data.dims['c']):
        
        length = data[varname].isel(c=c)
    
        fig = plt.figure()
        plt.plot(length.values)
        plt.xlim(0, length.shape[0] - 1)
        plt.ylabel('[%s]' %units)
        plt.title('Community ' + str(c))
        output[c] = _savefig()
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
    
    constants = extract_apecosm_constants(input_dir)
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