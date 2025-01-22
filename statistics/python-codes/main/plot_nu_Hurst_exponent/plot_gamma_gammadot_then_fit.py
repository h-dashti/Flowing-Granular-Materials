import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from colors import shadecolor, hlinecolor, hline_linestyle, edgecolor
from colors import mainfacecolor
from catch_data import get_XY
from params import markers
from params import ls_fc, c_fc
sys.path.insert(0, '../../modules')
from fit import loglog_slope, loglog_slope2
from mstring import as_si
from scipy.optimize import curve_fit


markersize = 9
markercolor = 'black' #'royalblue'
fillstyle = 'none'
elinewidth = 0.5

########################################################################
idir_expo = '../DATA2'
fname_nu = 'nu__by-Ps.dat'
fname_chimax_expo = 'chimax_exponent.dat'


fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_chimax_expo = os.path.join(idir_expo, fname_chimax_expo)
#fpath_fc = os.path.join(idir_expo, fname_fc)

#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'
########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
epsilon = 1e-9
########################################################################
def sort_legends(ax, fntsize):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,
              loc='upper left', fontsize=fntsize, frameon=False)
########################################################################

def plot_gamma(ax,
            plt_xlabel=True,
            plt_ylabel=True,
            show_legend=True,
            fntsize=16,
            fntsize_info=14,
            ):


    print ('** plot nu ...')
    
    X = np.loadtxt(fpath_nu)
    Y = np.loadtxt(fpath_chimax_expo)
    
    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]

    not_same_gamma = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)

    if np.any(not_same_gamma):
        print('error in gamma_plot')
        sys.exit()


    x = gamma_arr1
    y = X[:,1] * Y[:,1]
    z = X[:,2] * Y[:,1] + X[:,1] * Y[:,2]

    yc, yc_err = 2.06, 0.08
    # ax.axhspan(yc - yc_err, yc + yc_err,
    #            facecolor = shadecolor,
    #            edgecolor = edgecolor) # alpha=0.2
    # hline = ax.axhline(y=yc,
    #                    color = hlinecolor,
    #                    ls = hline_linestyle,
    #                    label=r'$\nu_{\mathrm{RP}} $') #= 1.21 \pm 0.06

    ax.set_xscale('log')
    ax.errorbar(x, y, yerr=z,
                marker='o', ls='--', lw=1,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                label=r'$\mathrm{sim}$',
                )
    
    
    ax.plot([7e-7, 2e-5], [yc, yc], label=r'$\gamma={}$'.format(yc))
    
    if True:
        #fit_func = lambda x, a, c : c * x**a
        
        indxs = (x >= 1e-5-epsilon) & (x <= 1e-3+epsilon)
        
        x, y, z = x[indxs], np.abs(y[indxs] - yc), z[indxs]
        
        fitfunc = lambda x, popt : popt[1] * x ** popt[0]
        
        popt, perr = loglog_slope2(x, y, z)
        
        nstd = [-0.5, 0.5]
        popt_up = popt + nstd * perr
        popt_dw = popt - nstd * perr
        
        
     

        #print(*popt, *perr)
    
        xn = np.logspace(log10(x.min()), log10(1*x.max()))
        fit = fitfunc(xn, popt) + yc
        fit_up = fitfunc(xn, popt_up) + yc
        fit_dw = fitfunc(xn, popt_dw) + yc
        ax.plot(xn, fit,                 
                  label=r'$\gamma = {} + {:.0f}\;'.format(yc, popt[1]) + \
                      ' \dot{\gamma}^{' + '{:.2f}'.format(popt[0]) + '}$'
                  )
        

        ax.fill_between(xn, fit_up, fit_dw,
                        color='#EDBB99', alpha=0.5)
                    
    

    #ax.set_ylim(1, 2)

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\gamma$', fontsize=fntsize)

    if show_legend:
        ax.legend(loc='upper left', fontsize=fntsize_info, frameon=False, 
            #handles=[axplt, hline],
                  )
########################################################################


########################################################################
if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.family'] = 'serif'


    fig, ax = plt.subplots()
    ax.tick_params(which='both', direction='in', 
                   top=True,right=True)

    ax.set_facecolor(mainfacecolor)

    plot_gamma(ax)
    
    plt.savefig('gamma_gammadot_fit2.pdf',
            pad_inches=0.02, bbox_inches='tight')
    
    





