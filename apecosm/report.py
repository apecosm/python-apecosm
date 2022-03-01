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


def report(input_dir, mesh_file, output_file='report.html', filecss='default'):
    
    if(filecss == 'default'):
        filecss = pkg_resources.resource_filename('apecosm', os.path.join('templates', 'styles.css'))
        with open(filecss) as fin:
            css = fin.read()
    
    elif filecss.startswith('http'):
        with urllib.request.urlopen(filecss) as fin:
            css = fin.read().decode('utf-8')
    
    env = jinja2.Environment(loader=jinja2.PackageLoader("apecosm"),  autoescape=jinja2.select_autoescape())
    template = env.get_template("template.html")
    
    arguments = {}
    
    arguments['css'] = css
        
    data = xr.open_mfdataset(os.path.join(input_dir, '*.nc'))
    dims = data.dims
    list_dims = [d for d in data.dims if 'prey' not in d]
    
    arguments['dims'] = dims
    arguments['list_dims'] = list_dims
    arguments['start_date'] = data['time'][0].values
    arguments['end_date'] = data['time'][-1].values

    arguments['length_figs'] = _plot_wl_community(data, 'length', 'meters')
    arguments['weight_figs'] = _plot_wl_community(data, 'weight', 'kilograms')
    arguments['select_figs'] = _plot_ltl_selectivity(data)

    render = template.render(**arguments)
    
    with open(output_file, "w") as f:
        f.write(render)
        
        
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
        buf = io.BytesIO()
        plt.savefig(buf, format="svg")
        plt.close(fig)
        
        fp = tempfile.NamedTemporaryFile() 
      
        with open(f"{fp.name}.svg", 'wb') as ff:
            ff.write(buf.getvalue()) 

        buf.close()
        output[c] = f"{fp.name}.svg"
        
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
        buf = io.BytesIO()
        plt.savefig(buf, format="svg")
        plt.close(fig)
        
        fp = tempfile.NamedTemporaryFile() 
      
        with open(f"{fp.name}.svg", 'wb') as ff:
            ff.write(buf.getvalue()) 

        buf.close()
        output[c] = f"{fp.name}.svg"
        
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