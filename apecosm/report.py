from traitlets.config import Config
from jupyter_core.command import main as jupymain
from nbconvert.exporters import HTMLExporter, PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import subprocess
import nbformat as nbf
from .extract import extract_oope_data, extract_time_means, extract_apecosm_constants
from .misc import find_percentile
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import papermill as pm
import pkg_resources
import os

def report(input_dir, meshfile, output):
        
    suffindex = output.rfind(".")
    fmt = output[suffindex + 1:]
    
    tempfile = output.replace(fmt, "ipynb")
    
    if fmt not in ['pdf', 'html']:
        message = "The output format must be 'html' or 'pdf'. The program will stop"
        print(message)
        sys.exit(1)

    template_file = pkg_resources.resource_filename('apecosm', 'resources/report_template.ipynb')
    param = dict(input_dir=input_dir, input_mesh=meshfile)
    pm.execute_notebook(template_file, tempfile, parameters=param)

    c = Config()

    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)    
    c.TagRemovePreprocessor.enabled = True

    if(fmt == 'html'):
        c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
        exporter = HTMLExporter(config=c)
    else:
        c.PDFExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
        exporter = PDFExporter(config=c)
    
    strout, objout = exporter.from_filename(tempfile)
    if(type(strout) == bytes):
        wmode = "wb"
    else:
        wmode = "w"
    
    with open(output, wmode) as fout:
        fout.write(strout)
        
    os.remove(tempfile)
    
def plot_report_ts(input_dir, input_mesh):

    # extract data in the entire domain, integrates over space
    data = extract_oope_data(input_dir, input_mesh, domain_name='global')
    time = data['time'].values
    comm = data['community'].values.astype(np.int)
    weight = data['weight'].values
    oope = data['OOPE'].values
    
    print('Mean total biomass, all communities: %e J' %(np.mean(np.sum(oope, axis=(1, 2)), axis=0)))
    for i in range(0, len(comm)):
        print('Mean total biomass, community %d: %e J' %(i + 1, np.mean(np.sum(oope[:, i, :], axis=(-1)), axis=0)))
    
    # plot total biomass
    fig = plt.figure()
    plt.plot(time, np.sum(oope, axis=(1, 2)))
    plt.title('Total biomass, all communities')
    plt.xlabel('Time')
    plt.ylabel('OOPE (J)')
    plt.show()
    plt.close(fig)
        
    for i in range(0, len(comm)):
        fig = plt.figure()
        plt.plot(time, np.sum(oope[:, i, :], axis=(1)))
        plt.title('Total biomass, community %d' %(i + 1))
        plt.xlabel('Time')
        plt.ylabel('OOPE (J)')
        plt.show()
        plt.close(fig)
    

def plot_report_map(input_dir, input_mesh):
    
    constants = extract_apecosm_constants(input_dir)
    wstep = constants['weight_step'].values
    wstep = wstep[np.newaxis, np.newaxis, :]
    
    dataset = xr.open_mfdataset("%s/*OOPE*nc" %input_dir, combine='by_coords')
    dataset = extract_time_means(dataset)
    oope = dataset['OOPE'] * wstep   #  conversion in J/m2  (lat, lon, comm, weight)
    oope = oope.sum(dim=('w')).to_masked_array()   # sum over all weight classes: (lat, lon, com)
    
    mesh = xr.open_dataset(input_mesh)
    lon = mesh['glamt'].values[0]
    lat = mesh['gphit'].values[0]
    tmask = mesh['tmask'].values[0, 0]
    
    projection = ccrs.PlateCarree()
    
    fig = plt.figure()
    ax = plt.axes(projection=projection)
    temp = np.sum(oope, axis=-1)
    temp = np.log10(temp, where=(temp > 0))
    cmin, cmax = find_percentile(temp)
    cs = plt.pcolormesh(lon, lat, temp, transform=projection)
    cs.set_clim(cmin, cmax)
    cb = plt.colorbar(cs, orientation='horizontal')
    cb.set_label('LOG10 OOPE (J/m2), all communities')
    plt.show()
    plt.close(fig)   
    
    for i in range(oope.shape[-1]):
        fig = plt.figure()
        ax = plt.axes(projection=projection)
        temp = oope[:, :, i]
        temp = np.log10(temp, where=(temp > 0))
        cmin, cmax = find_percentile(temp)
        cs = plt.pcolormesh(lon, lat, temp, transform=projection)
        cs.set_clim(cmin, cmax)
        cb = plt.colorbar(cs, orientation='horizontal')
        cb.set_label('LOG10 OOPE (J/m2), community %d' %(i + 1))
        plt.show()
        plt.close(fig)   

def plot_report_size_spectra(input_dir, input_mesh):
    
     # extract data in the entire domain, integrates over space
    data = extract_oope_data(input_dir, input_mesh, domain_name='global', use_wstep=False)
    data = extract_time_means(data)
    data = data['OOPE']
    
    const = extract_apecosm_constants(input_dir)
    weight = const['weight'].values
    length = const['length'].values
    
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

    print('++++++++++++++++++++++++ processing')
    report('output dir', 'mesh file', 'html')
    report('output dir', 'mesh file', 'pdf')
