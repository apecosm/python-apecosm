import sys
import numpy as np
import xarray as xr
from apecosm.misc import compute_daylength
import sys

def get_tcor(temp_array, ta=5020., tref=298.15):
    
    r''' 
    Computes the Ahrrenius temperature as follows:

    .. math::

        \exp^{\frac{T_a}{T_{ref}} - \frac{T_a}{T}}

    :param numpy.array temp_array: Temperature (in :math:`K`)
    '''

    return np.exp(ta / tref - ta / temp_array);


def compute_o2(oxy_array, tcor=1.0, oxyresp=1e5, oxylim=1e-4):

    r''' 
    Computes the oxygen habitat function:

    .. math:: 
       
        H_{oxy}=\frac{1}{1+\exp{\left[OXYRESP \times (T_{cor} \times OXYLIM - Oxy)\right]}}

    :param numpy.array oxy_array: Oxygen (in :math:`mol.L^{-1}`)
    :param numpy.array tcor: Ahrrenius temperature (obtained from the :py:func:`get_tcor` function)
    '''
    
    output = 1. / (1. + np.exp(oxyresp * (tcor * oxylim - oxy_array)));

    return output


def compute_tpref(tcor, sigm_tcor=0.1):
    
    r''' 
    Computes the temperature preference function:

    .. math:: 

        H_{tpref}=\exp{\left[-0.5\left(\frac{\frac{T_{cor}(z)}{T_{cor}(z=0)}-1}{SIGM_{TCOR}}\right)^2\right]} 

    :param numpy.array tcor: Ahrrenius temperature (obtained from the :py:func:`get_tcor` function)

    .. danger:: 

        The temperature array should be 4D, with z as the second dimension.
    
    '''

    if(tcor.ndim != 4):
        message = 'The number of dims must be 4, with z as the second dimension'
        print(message)
        sys.exit(1)

    tcor0 = tcor[:, 0:1, :, :]

    output = np.exp(-.5 * np.power(((tcor / tcor0 - 1.) / sigm_tcor), 2.));

    return output


def compute_tlim(temp_array, t_sup=280.15, t_inf=268.5, ta_lim=200000):

    r'''
    Computes the temperature limitation function:

    .. math:: 

        H_{tlim} = \frac{1}{1+\exp\left(\frac{Ta_{LIM}}{temper}-\frac{Ta_{LIM}}{T_{inf}}\right)} \times \frac{1}{1+\exp\left(\frac{Ta_{LIM}}{T_{sup}}-\frac{Ta_{LIM}}{temper}\right)}

    '''

    output = (1. / (1. + np.exp(ta_lim / temp_array - ta_lim / t_inf))) / (1. + np.exp(ta_lim / t_sup - ta_lim / temp_array));

    return output


def compute_lightpref(par, daylen, opt_light=1.0e2, sigm_light=1.7e2, same_daynight=False, nfact=1e-6):

    r'''
    Computes the light preference function.

    .. math:: 

        \begin{eqnarray}
            & & light & = & \frac{PAR}{DAYLENGTH}*nfactor+EPS \\ 
            H_{lpref} & = & H_{lpred} & = & \frac{mode}{light} \times \exp\left[\frac{(\log(mode)-mu)^2-(\log(light)-mu)^2}{2\times ssigm} \right] 
        \end{eqnarray}

    :param numpy.array par: Photosynthetically Active Radiation (in :math:`W.m^{-2}`). **Should be 4D (time, z, y, x).**
    :param numpy.array daylen: Fraction of day within 24h (can be obtained from the :py:func:`compute_daylength` function). Its shape should allow broadcasting
     with the :samp:`par` array (for instance, if par=(time, depth, y, x), daylen should be (time, 1, y, x). Should correspond with the time step of the PAR files.
    :param bool same_daynight: Whether light habitat should distinguish day and night cycles.
    :param float nfactor: Proportionality coefficient between light intensity at night and during the day.

    :return: An array of dimensions (dn, time, z, y, x), with dn=0 for daytime, dn=1 for night time.

    .. note:: 

        This function can also be used to compute the light contribution to the functional response.

    '''

    nfactor = np.ones((2), dtype=np.float)   # init the nfactor with ones 
    
    if(not(same_daynight)):
        nfactor[1] = nfact    # if not same_daynight -> change nfactor for night (dn = 1)

    nfactor = nfactor[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]   # dn, time, z, y, x

    mu = np.log(opt_light) - .5 * np.log1p((sigm_light * sigm_light) / (opt_light * opt_light));
    sigm = np.sqrt(np.log1p(sigm_light * sigm_light / (opt_light * opt_light)));
    ssigm = sigm * sigm;
    mode = np.exp(mu - ssigm);

    par = np.ma.masked_where(np.isnan(par), par)
    par = par[np.newaxis, :, :, :, :]
    daylen = daylen[np.newaxis, :, :, :, :]
    daylen[daylen < 1./24.] = 1 / 24.
    daylen = np.ma.masked_where(par[:, :, 0:1, :, :].mask, daylen)
    tmplight = np.divide(par, daylen, where=(par.mask == False)) * nfactor + 2 * sys.float_info.min 

    output = np.divide(mode, tmplight, where=(tmplight.mask==False)) * np.exp((np.power((np.log(mode) - mu), 2.) - np.power((np.log(tmplight, where=(tmplight.mask==False)) - mu), 2.)) / (2. * ssigm));

    return output



if __name__ == '__main__':

    dirin = '/home/nbarrier/Modeles/apecosm/svn-apecosm/trunk/tools/config/gyre_nico/data'
    filelist = "%s/clim_grid_T.nc" %dirin
    data = xr.open_dataset(filelist)
    temp = data['votemper'].values + 273.15   # converted into K
    lat = data['nav_lat'].values

    # computation
    tcor = get_tcor(temp)

    filelist = "%s/clim_ptrc_T.nc" %dirin
    data = xr.open_dataset(filelist)
    oxy = data['O2'].values * 1e-6   # converted into mmol/L
    par = data['PAR'].values 
    
    ######################## group = 0
    tpref = compute_tpref(tcor)
    tpref = np.ma.masked_where(temp == 273.15, tpref)
    o2hab = compute_o2(oxy, tcor)
    
    ######################## group = 0
    daylen = compute_daylength(lat)   # doy, lat, lon
    time = np.arange(par.shape[0]) * 5 + 2.5
    time = time.round().astype(np.int)
    daylen = daylen[time, np.newaxis, :, :]
    daylen[daylen < 1 / 24.] = 1/24.

    lightpref = compute_lightpref(par, daylen, opt_light=8e-4, sigm_light=8e-3, same_daynight=True)
    lightpref = compute_lightpref(par, daylen, opt_light=8e-4, sigm_light=8e-3, same_daynight=False)

    '''
    import pylab as plt
    temp = np.ma.masked_where(temp == 273.15, temp)
    oxy = np.ma.masked_where(oxy == 0, oxy)
    o2hab = np.ma.masked_where(oxy.mask, o2hab)
    print o2hab.min(), o2hab.max()
    print oxy.min(), oxy.max()

    iz = 0
    it = 50

    plt.figure()
    ax = plt.subplot(211)
    cs = plt.imshow(oxy[it, iz])
    plt.colorbar(cs)
    ax = plt.subplot(212)
    cs = plt.imshow(o2hab[it, iz])
    cs.set_clim(0, 1)
    plt.colorbar(cs)
    plt.savefig('toto.png')
    '''


