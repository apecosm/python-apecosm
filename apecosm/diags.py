"""
Module for analysing Apecosm outputs
"""

from .extract import extract_oope_data


def compute_size_cumprop(mesh, data, const, mask_dom=None):

    """
    Computes the cumulated biomass proportion as
    a function of length.

    :param data: Apecosm dataset
    :type data: :class:`xarray.Dataset`
    :param mesh: Mesh grid dataset
    :type mesh: :class:`xarray.Dataset`
    :param const: Apecosm constants datasets
    :type const: :class:`xarray.Dataset`
    :param mask_dom: Mask array. If None, full domain is considered
    :type mask_dom: :class:`numpy.array`
    """

    output = extract_oope_data(data, mesh, const, mask_dom, use_wstep=True)
    output = output.cumsum(dim='w') / output.sum(dim='w') * 100
    return output
