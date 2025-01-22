import numpy as np
import os, sys
from math import log10, sqrt

sys.path.insert(0, '../aux-codes')
from params import fntsize, fntsize_info, markers, \
    get_mycolors, fillstyles
from scipy.optimize import curve_fit

########################################################################
idir = '../../main/plot_fc/DATA'
fname = 'N_gammadot_fav(phi=0.86)--by-rheology.dat'
fpath = os.path.join(idir, fname)
########################################################################

def get_data(fpath):
    data = np.loadtxt(fpath)

    res = {}

    for d in data:
        L, N, gammadot, allf_ave, allf_var, allf_mode, fbar_ave = d[:7]
        N = int(N)
        if N not in res:
            res[N] = {}

        res[N][gammadot] = (sqrt(allf_var),)

    return res
########################################################################

def plot(ax,
         plot_fit=True,
         show_legend=True,
         plt_xlabel=False,
         plt_ylabel=True,
         sub_indx_fit = None,
         what_am_i_doing = 'plotting allf_var by rheology',
         ):


    print (what_am_i_doing)
    
    res = get_data(fpath)


    i = 0
    for N in res.keys():
        
        xy = np.array([[k, *res[N][k]] for k in res[N]])
        xy = xy[xy[:,0].argsort()]

        ax.plot(xy[:,0], xy[:,1], #xy[:, 2],
                    label=r'$N={:.0f}$'.format(N),
                    marker='o', markersize=7,
                    fillstyle = fillstyles[i],
                    ls='',)
        
        if plot_fit:
            
            a = None
            
            if a is None:
                bounds = ([1e-10, 1e-10, 1e-3], 
                          [2, 2, 2])
                
                f_fit = lambda x, c, b, a: c + b * np.power(x, a)
                popt, pcov = curve_fit(f_fit, xy[:,0], xy[:,1], 
                                       bounds = bounds)
            else:
                f_fit = lambda x, c, b: c + b * np.power(a)
                popt, pcov = curve_fit(f_fit, xy[:,0], xy[:,1])
            
            
            perr = np.sqrt(np.diag(pcov))
            
            fmt = len(popt)*'{:.5g} '
            print ('N={}'.format(N))
            print (' popt:', fmt.format(*popt))
            print (' perr:', fmt.format(*perr))
        
        i += 1

    # end for N

    if plot_fit:
    
        x = np.logspace(log10(7.5e-7), log10(1.2e-3))
        y = f_fit(x, *popt)
        
        st_sub = '' if sub_indx_fit is None else '_{}'.format(sub_indx_fit)
        lable = '$' + 'c{0}+b{0}'.format(st_sub) + \
            r'\,\dot{\gamma}^{' + 'a{}'.format(st_sub) + '}$'
            
        ax.plot(x, y,
                label=lable,
                color='k', #ax.lines[-1].get_color(),,
                lw = 1,
                ls = '--',
                zorder = 4,
                )


    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\sigma_f$', fontsize=fntsize)

    if show_legend:
        ax.legend(loc='best', frameon=False, 
                  ncol=1, fontsize=13)

########################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    
    ax.set_xscale('log')
    plot(ax)








