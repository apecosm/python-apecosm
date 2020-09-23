import numpy as np

def get_monthly_clim(var):
    
    ntime = var.shape[0]

    nyears = ntime // 12

    index = np.arange(12)

    for i in range(nyears):
        if(i == 0):
            clim = var[index]
        else:
            clim += var[index]
        index += 12
    clim /= nyears

    anom = np.zeros(var.shape)
    index = np.arange(12)
    for i in range(nyears):
        anom[index] = var[index] - clim
        index += 12

    return clim, anom
