import xarray as xr
from .extract import extract_oope_data

def compute_size_cumprop(mesh, data, const, maskdom=None):
    
    output = extract_oope_data(data, mesh, const, maskdom, use_wstep=True)
    output = output.cumsum(dim='w') / output.sum(dim='w') * 100
    return output
                                        