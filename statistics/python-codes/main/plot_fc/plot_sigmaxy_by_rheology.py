import numpy as np
import os, sys
from math import log10

sys.path.insert(0, '../aux-codes')
from params import fntsize, fntsize_info, markers, \
    get_mycolors, fillstyles
from scipy.optimize import curve_fit

########################################################################
idir = 'DATA/data_f-sigma_phi=0.86'
fname = 'rheo-stat.dat'
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #

########################################################################

def get_data(idir):
    res = {}
    for N in N_arr:

        fpath = os.path.join(idir, 'N_{}'.format(N), fname)
        data = np.loadtxt(fpath)

        if N not in res: res[N] = {}

        for d in data: res[N][d[0]] = -d[3]
    # end for N

    return res
    #


########################################################################

def plot(ax,
         plot_fit=True,
         show_legend=True,
         plt_xlabel=False,
         plt_ylabel=True):

    print ('sigmaxy : ')


    res = get_data(idir)
    
    #colors = get_mycolors('tab10', None)   

    i = 0
    for N in res.keys():
        #xy = np.array(list(res[N].items()))
        xy = np.array([[k, res[N][k]] for k in res[N]])
        xy = xy[xy[:,0].argsort()]
        

        ax.plot(xy[:,0], xy[:,1], 
                    label=r'$N={}$'.format(N),
                    marker='o',  markersize=7,
                    fillstyle = fillstyles[i],
                    ls='', 
                    #color=colors[len(res.keys()) - i - 1],
                    )
        
        if plot_fit:
            f_fit = lambda x, a, b: a + b * np.sqrt(x)
            popt, pcov = curve_fit(f_fit, xy[:,0], xy[:,1])
            perr = np.sqrt(np.diag(pcov))
            print ('N={:.0f}, a={:.5g}, b={:.5g}'.\
                   format(N, *popt), 'err_a: {:.5g}, err_b={:.5g}'.\
                       format(*perr))
        
        i += 1

    # end for N
       
    

    if plot_fit:
        x = np.logspace(log10(7.5e-7), log10(1.2e-3))
        y = f_fit(x, *popt)
        ax.plot(x, y,
                label='$ a_1+b_1\,\dot{\gamma}^{0.5}$',
                color='k', #ax.lines[-1].get_color())
                lw = 1, 
                ls = '--',
                zorder = 4,
                )


    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\sigma_{xy}$', fontsize=fntsize)

    if show_legend:
        ax.legend(loc='best', frameon=False, 
                  ncol=2, fontsize=13)



########################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    plot(ax)








