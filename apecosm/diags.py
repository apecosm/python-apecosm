"""
Module for analysing Apecosm outputs
"""

from .extract import extract_oope_data


def compute_size_cumprop(data, const):

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

    data = (data * const['weight_step'])
    output = data.cumsum(dim='w') / data.sum(dim='w') * 100
    output['community_weight'] = data.sum(dim='w')

    return output
