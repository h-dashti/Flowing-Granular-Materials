#from scipy.interpolate import interp1d
from scipy.interpolate import interp1d
import scipy.stats as stats
import numpy as np
from scipy.optimize import curve_fit
###############################################################
def interp(x, y, x0):

    inter = interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    return inter(x0)
###############################################################

def loglog_slope(x, y, full=True):

    x, y = np.log(x), np.log(y)

    if full:
        popt, pcov = np.polyfit(x, y, deg=1, cov=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt = np.polyfit(x, y, deg=1, cov=False)
       
    expo = popt[0]
    c = np.exp(popt[1])
    
    if full:
        return expo, c, perr[0], c * perr[1]
    else:
        return expo, c
    
  ############################################################### 
def loglog_slope2(x, y, z, bounds=(-np.inf, np.inf)):  # z=yerr


    x, y, z = np.log(x), np.log(y), np.abs(np.log(z))


    linfunc = lambda x, b, a : b * x + a       


    popt, pcov = curve_fit(linfunc, x, y, 
                           sigma=z, 
                           absolute_sigma=False,
                           bounds=bounds,
                           )
    perr = np.sqrt(np.diag(pcov))
    
    
    def proper_popt(popt, perr=None):
        
        popt[1] = np.exp(popt[1])
        
        if perr is not None:
            perr[1] = popt[1] * perr[1]
            return popt, perr
        else:
            return popt
    
    popt, perr = proper_popt(popt, perr)
        
    return popt, perr
   

def lin_slope2(x, y, z, intercept=0):  # z=yerr

    if intercept is None:
        linfunc = lambda x, m, c : m * x + c 
    else:
        linfunc = lambda x, m : m * x + intercept 

    popt, pcov = curve_fit(linfunc, x, y, 
                           sigma=z, 
                           absolute_sigma=False
                           )
    perr = np.sqrt(np.diag(pcov))    
    return popt, perr
   


        
        
def loglog_cov(x, y):

    x, y = np.log(x), np.log(y)

    popt, pcov = np.polyfit(x, y, deg=1, cov=True)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr
#########################################

def lin_slope(x, y, full=True):

    if full:
        popt, pcov = np.polyfit(x, y, deg=1, cov=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt = np.polyfit(x, y, deg=1, cov=False)
    
    if full:
        return popt[0], popt[1], perr[0], perr[1] 
    else:
        return popt
#########################################

def linregress(x, y):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return stats.linregress(x, y)

#########################################


if  __name__ == '__main__':
    pass