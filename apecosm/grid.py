'''
Module that contains some functions related to the
vertical grid
'''

#from __future__ import print_function
import numpy as np
import pandas as pd
import xarray as xr
try:
    import pylab as plt
    from matplotlib import patches
except ImportError:
    pass


def extract_weight_grid(config):

    '''
    Extracts weight grid attributes.
    Returns the weight and length step used for integration
    '''

    omega_var_min = float(config["biology.weight.omega.min"])
    omega_var_step = float(config['biology.weight.omega.incr'])
    n_weight_class = int(config['biology.weight.nclass'])
    lmin = float(config['biology.length.min'])
    lmax = float(config['biology.length.max'])
    alpha = float(config['biology.weight.alpha'])
    allom_w_l = float(config['biology.allometric.coeff'])
    c_fonct_w_dep = float(config['biology.functional_response.half_saturation.dependence'])

    omega_var_max = omega_var_min + (n_weight_class - 1) * omega_var_step
    beta = (lmin - lmax * np.power(alpha, (omega_var_min - omega_var_max - 1))) \
        / (1. - np.power(alpha, (omega_var_min - omega_var_max - 1)))
    gam = np.log((lmin - lmax) / (np.power(alpha, omega_var_min) \
        - np.power(alpha, omega_var_max + 1))) / np.log(alpha)

    tmp_length = np.zeros(n_weight_class + 1)
    tmp_weight = np.zeros(n_weight_class + 1)
    weight = np.zeros(n_weight_class)
    length = np.zeros(n_weight_class)
    weight_step = np.zeros(n_weight_class)
    length_step = np.zeros(n_weight_class)
    weight_step2 = np.zeros(n_weight_class + 1)
    w_pow_c = np.zeros(n_weight_class)
    w_pow_m1_3 = np.zeros(n_weight_class)

    for w in range(0, n_weight_class + 1):
        omega_var = omega_var_min + w * omega_var_step
        tmp_length[w] = np.power(alpha, (omega_var + gam)) + beta
        tmp_weight[w] = allom_w_l * np.power(tmp_length[w], 3.)

    for w in range(0, n_weight_class):
        weight[w] = (tmp_weight[w] + tmp_weight[w + 1]) / 2.
        length[w] = np.power((weight[w] / allom_w_l), (1. / 3.))
        w_pow_c[w] = np.power(weight[w], c_fonct_w_dep)
        w_pow_m1_3[w] = np.power(weight[w], (-1. / 3.))


    # computes weigh intervals
    for w in range(0, n_weight_class):
        length_step[w] = tmp_length[w + 1] - tmp_length[w]
        weight_step[w] = tmp_weight[w + 1] - tmp_weight[w]

    weight_step2[0] = weight[0] - tmp_weight[0]
    for w in range(1, n_weight_class):
        weight_step2[w] = weight[w] - weight[w - 1]
    weight_step2[n_weight_class] = tmp_weight[n_weight_class] - weight[n_weight_class - 1]

    return weight_step, length_step


def read_ape_grid(filename):

    '''
    Reads the Apecosm text file containing depth coordinates of
    Apecosm T-grid points.

    :param str filename: Apecosm vertical grid file

    :return: A tuple contaning the depth and thicknessess
     of the cell points

     .. todo::

        Add correction: in the new version, the :samp:`.txt` file
        provides the lower edge of the T grid

    '''

    data = pd.read_csv(filename, header=None)
    adepth = np.squeeze(data.values)
    nlevel = len(adepth)
    deltaz = np.zeros(nlevel)
    output = np.zeros(nlevel)

    adepth = np.concatenate(([0], adepth))

    for i in range(0, nlevel):
        deltaz[i] = adepth[i + 1] - adepth[i]
        output[i] = 0.5 * (adepth[i + 1] + adepth[i])

    return (output, deltaz)


def partial_step_ape(ape_grid, mesh_file_nemo):

    '''
    Reads the NEMO mesh file to extract the
    Apecosm vertical grid

    :param ape_grid: CSV file containing the Apecosm
    vertical levels
    :param mesh_file_nemo: NetCDF file containing the NEMO mesh
    file

    '''

    # Reads the NEMO mesh file to extract all
    # the data that are needed.
    data = xr.open_dataset(mesh_file_nemo)
    tmask = np.squeeze(data['tmask'].values)
    tmask = tmask[0]

    adepth, deltaz = read_ape_grid(ape_grid)
    nlevel = len(adepth)
    nlat, nlon = tmask.shape

    output_depth = np.zeros((nlevel, nlat, nlon), dtype=float) - 1000
    output_deltaz = np.zeros((nlevel, nlat, nlon), dtype=float) - 1000
    output_bottom = np.zeros((nlat, nlon), dtype=int)
    output_mask = np.zeros((nlevel, nlat, nlon), dtype=int)

    iok = np.nonzero(tmask == 1)

    for ilat, ilon in zip(iok[0], iok[1]):
        temp = _partial_step_ape_grid(adepth, deltaz, data, ilat, ilon)
        output_depth[:, ilat, ilon] = temp['apecosm']['depth']
        output_deltaz[:, ilat, ilon] = temp['apecosm']['deltaz']
        output_bottom[ilat, ilon] = temp['apecosm']['bottom']
        output_mask[:temp['apecosm']['bottom'], ilat, ilon] = 1

    output = {'bottom': output_bottom,
              'depth': output_depth,
              'deltaz': output_deltaz,
              'mask': output_mask}

    return output


def _partial_step_ape_grid(adepth, deltaz, data_mesh_file_nemo, ilat, ilon):

    '''
    Extracts the vertical NEMO and Apecosm grids at
    point `ilat`, `ilon`.

    '''

    # Hard copy of the adepth and deltaz arguments
    # prevents overwritting
    adepth = adepth.copy()
    deltaz = deltaz.copy()

    nlevel = len(adepth)

    if 'e3t_1d' in data_mesh_file_nemo.data_vars.keys():
        # new nemo formatting
        #print('New nemo formatting for vertical scale factors')
        e3t_1d_varname = 'e3t_1d'
        depth_1d_varname = 'gdept_1d'
        e3t_varname = 'e3t_0'
    else:
        # old nemo formatting
        #print('Old nemo formatting for vertical scale factors')
        e3t_1d_varname = 'e3t_0'
        depth_1d_varname = 'gdept_0'
        e3t_varname = 'e3t'

    e3t_1d = data_mesh_file_nemo[e3t_1d_varname].values[0, :]
    znemo = data_mesh_file_nemo[depth_1d_varname].values[0, :]
    tmask = data_mesh_file_nemo['tmask'].values[0, :, ilat, ilon]

    if tmask[0] == 0:
        message = 'The point is in land. Nothing will be done'
        print(message)
        return None

    # check if NEMO grid is in partial step or not
    # nemo_partial_step = 'e3t_0' in data.data_vars.keys()
    nemo_partial_step = e3t_varname in data_mesh_file_nemo.data_vars.keys()
    #print('NEMO is in partial step mode')

    if nemo_partial_step:
        corrz = data_mesh_file_nemo['hdept'].values[0, ilat, ilon]
        e3t_f = data_mesh_file_nemo[e3t_varname].values[0, :, ilat, ilon]
        e3t = e3t_f
    else:
        e3t = e3t_1d

    # 1D array of size (nlevel+1)
    edges_a = np.append(0, np.cumsum(deltaz))

    # 3D array of size (nlevel_opa+1, NLAT, NLON)
    edges_n = np.append(0, np.cumsum(e3t, axis=0))

    # bottom_opa is the first land point in the vertical
    iok = np.nonzero(tmask == 1)[0]
    bottom_opa = iok[-1] + 1

    nlevel_opa = len(znemo)

    for k in range(0, nlevel_opa):
        if tmask[k] == 0:
            bottom_opa = k
            # if partial step in NEMO, correct the depth of the last ocean point
            if nemo_partial_step:
                znemo[k - 1] = corrz
            break

    # recovers the limit of the land point (i.e. the depth
    # of the NEMO sea/rock interface.
    limit = edges_n[bottom_opa]

    bottom_nico = nlevel
    for k in range(0, nlevel):
        if edges_a[k] > limit:
            bottom_nico = k
            break

    if edges_a[bottom_nico] > limit:
        k = bottom_nico
        edges_a[k] = limit    # // move the edge of the last ocean cell upward

    for k in range(1, nlevel + 1):
        deltaz[k - 1] = edges_a[k] - edges_a[k - 1]   # update in the deltaz value of the last ocean point
        adepth[k - 1] = 0.5 * (edges_a[k] + edges_a[k - 1])  # update in the depth value for the last ocean point

    apecosm = {'deltaz': deltaz, 'depth': adepth, 'edges': edges_a, 'bottom': bottom_nico}
    nemo = {'deltaz': e3t, 'depth': znemo, 'edges': edges_n, 'bottom': bottom_opa, 'limit': limit}
    output = {'nemo': nemo, 'apecosm': apecosm}

    return output


def plot_grid_nemo_ape(ape_grid, mesh_file_nemo, ilat, ilon):

    '''
    Draws the NEMO and Apecosm vertical grid
    at points `ilat`, `ilon`.
    '''

    # Reads the NEMO mesh file to extract all
    # the data that are needed.
    data = xr.open_dataset(mesh_file_nemo)
    adepth, deltaz = read_ape_grid(ape_grid)

    output = _partial_step_ape_grid(adepth, deltaz, data, ilat, ilon)
    if output is None:
        return

    nemo = output['nemo']
    apecosm = output['apecosm']
    bottom_nico = apecosm['bottom']
    edges_a = apecosm['edges']
    adepth = apecosm['depth']
    deltaz = apecosm['deltaz']

    bottom_opa = nemo['bottom']
    edges_n = nemo['edges']
    limit = nemo['limit']
    znemo = nemo['depth']
    corrz = znemo[bottom_opa - 1]

    plt.figure(figsize=(4, 10))
    plt.subplots_adjust(right=0.95, left=0.16, bottom=0.1, top=0.95)
    ax1 = plt.gca()

    plt.plot([0]*len(adepth[:bottom_nico]), adepth[:bottom_nico], marker='o', linestyle='none', color='k')
    for i in range(0, len(adepth[:bottom_nico])):
        plt.plot(np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
                 [edges_a[i], edges_a[i], edges_a[i + 1], edges_a[i + 1], edges_a[i]],
                 color='k')

    plt.plot([0]*len(adepth[bottom_nico:]), adepth[bottom_nico:], marker='o', linestyle='none')
    for i in range(bottom_nico, len(adepth)):
        plt.plot(np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
                 [edges_a[i], edges_a[i], edges_a[i + 1], edges_a[i + 1], edges_a[i]],
                 color='k')
        rect = patches.Rectangle((-0.5, edges_a[i]), 1, edges_a[i+1] - edges_a[i], color='lightgray', alpha=0.5)
        ax1.add_patch(rect)

    # plot unmasked NEMO
    plt.plot([2 - 0.5] * len(znemo[:bottom_opa - 1]), znemo[:bottom_opa - 1], marker='o', linestyle='none', color='k')
    for i in range(0, len(znemo[:bottom_opa])):
        plt.plot(np.array([1.5, 2.5, 2.5, 1.5, 1.5]) - 0.5,
                 [edges_n[i], edges_n[i], edges_n[i+1], edges_n[i+1], edges_n[i]], color='k')

    # plot masked NEMO
    plt.plot([2 - 0.5] * len(znemo[bottom_opa:]), znemo[bottom_opa:], marker='o', linestyle='none', color='k')
    for i in range(bottom_opa, len(znemo)):
        plt.plot(np.array([1.5, 2.5, 2.5, 1.5, 1.5]) - 0.5,
                 [edges_n[i], edges_n[i], edges_n[i + 1], edges_n[i + 1], edges_n[i]], color='k')
        rect = patches.Rectangle((1, edges_n[i]), 1, edges_n[i+1] - edges_n[i], color='lightgray', alpha=0.5)
        ax1.add_patch(rect)

    plt.plot([2-0.5], corrz, marker='o', linestyle='none', color='firebrick')
    for i in range(bottom_opa-1, bottom_opa):
        plt.plot(np.array([1.5, 2.5, 2.5, 1.5, 1.5]) - 0.5,
                 [edges_n[i], edges_n[i], edges_n[i + 1], edges_n[i + 1], edges_n[i]], color='r', linestyle='--', linewidth=1)
    for i in range(bottom_opa-1, bottom_opa):
        rect = patches.Rectangle((1, edges_n[i+1]), 1, edges_n[i + 1] - edges_n[i + 1], color='lightgray', alpha=0.5)
        ax1.add_patch(rect)

    plt.axhline(limit, color='gold', linestyle='--', linewidth=0.5)

    ylim = edges_n[bottom_opa + 1]
    plt.ylim(0, ylim)

    plt.title('Partial step')
    plt.ylabel('Depth (m)')
    ax1.set_xticks([0, 1.5])
    ax1.set_xticklabels(['Apecosm', 'NEMO'])
    ax1.set_ylim(ax1.get_ylim()[::-1])


# if __name__ == '__main__':
#
#     filename = 'APECOSM_depth_grid_37.txt'
#     mesh_file = 'data/mesh_mask.nc'
#     ilat, ilon = 115, 73
#     #ilat, ilon = 131, 19
#     ilat, ilon = 131, 44
#
#     #mesh_file = '/home/nbarrier/Modeles/apecosm/svn-apecosm/trunk/tools/config/gyre/mesh_mask.nc'
#     #ilat, ilon = 12,12
#
#     adepth, deltaz = read_ape_grid(filename)
#     output = partial_step_ape_grid(filename, mesh_file, ilat, ilon)
#     plot_grid_nemo_ape(filename, mesh_file, ilat, ilon)
#     plt.show()
