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


markersize = 6
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
def plot_nu(ax,
            plt_xlabel=True,
            plt_ylabel=True,
            show_legend=True,
            fntsize=16,
            fntsize_info=14,
            ):


    print ('** plot nu ...')

    X = np.loadtxt(fpath_nu)
    X = X[X[:, 0].argsort()]
    
    yc, yc_err = 1.21, 0.06
    
    x, y, z = X[:,0], X[:,1], X[:,2]
    y = y - yc
    
    # ax.axhspan(yc - yc_err, yc + yc_err,
    #            facecolor = shadecolor,
    #            edgecolor = edgecolor) # alpha=0.2
    # hline = ax.axhline(y=yc,
    #                    color = hlinecolor,
    #                    ls = hline_linestyle,
    #                    label=r'$\nu_{\mathrm{RP}} $') #= 1.21 \pm 0.06

    ax.set_xscale('log')
    ax.errorbar(x, y, yerr=z,
                marker='o', ls='', lw=1,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                #label=r'$\mathrm{sim}$',
                )
    
    
    ax.plot([7e-7, 1e-5], [0, 0], ls='--',
            color='k')
    
       
        
    indxs = (x >= 1e-5-epsilon) & (x <= 1e-3+epsilon)
     
    x, y, z = x[indxs], np.abs(y[indxs]), z[indxs]
     
    fitfunc = lambda x, popt : popt[1] * x ** popt[0]
     
    popt, perr = loglog_slope2(x, y, z)
     
     #fitfunc = lambda x, a, c : c * x**a
     # popt, pcov = curve_fit(fitfunc, x, y, 
     #                        sigma=z, absolute_sigma=True)
    # perr = np.sqrt(np.diag(pcov))
    
    print(*popt, *perr)
     
    nstd = [-0.5, 0.5]
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
     
        
     
    xn = np.logspace(log10(x.min()), log10(1*x.max()))
    fit = fitfunc(xn, popt)
     
    txt = r'$\quad a={:.0f}, b={:.2f}$'.format(popt[1], popt[0])
    ax.plot(xn, fit,                 
        label=r'$a \, \dot{\gamma}^b$' + txt,
        color='k',
        ls='-',
           #label=r'$\gamma = {} + {:.0f}\;'.format(yc, popt[1]) + \
           #    ' \dot{\gamma}^{' + '{:.2f}'.format(popt[0]) + '}$'
          )
     


     
    #print(*popt_up, *popt_dw)
    fit_up = fitfunc(xn, popt_up)
    fit_dw = fitfunc(xn, popt_dw)
    ax.fill_between(xn, fit_up, fit_dw,
                     color='#EDBB99', 
                     alpha=0.5
                     )
                    

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\nu - \nu_{\,\mathrm{RP}}$', fontsize=fntsize)

    if show_legend:
        ax.legend(loc='upper left', fontsize=fntsize_info, frameon=False, 
            #handles=[axplt, hline],
                  )
########################################################################

def plot_gamma(ax,
            plt_xlabel=True,
            plt_ylabel=True,
            show_legend=True,
            fntsize=16,
            fntsize_info=14,
            ):


    print ('** plot gamma ...')
    
    X = np.loadtxt(fpath_nu)
    X = X[X[:, 0].argsort()]
    Y = np.loadtxt(fpath_chimax_expo)
    Y = Y[Y[:, 0].argsort()]
    
    
    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]

    not_same_gamma = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)

    if np.any(not_same_gamma):
        print('error in gamma_plot')
        sys.exit()

    yc, yc_err = 2.06, 0.08
    x = gamma_arr1
    y = X[:,1] * Y[:,1]
    z = X[:,2] * Y[:,1] + X[:,1] * Y[:,2]
    y = y - yc
    
    print(y)
    
    ax.set_xscale('log')
    ax.errorbar(x, y, yerr=z,
                marker='o', ls='', lw=1,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                #label=r'$\mathrm{sim}$',
                )
    

    ax.plot([7e-7, 1e-5], [0, 0], ls='--',
            color='k')
    
    indxs = (x >= 1e-5-epsilon) & (x <= 1e-3+epsilon)
     
    x, y, z = x[indxs], np.abs(y[indxs]), z[indxs]
     
    fitfunc = lambda x, popt : popt[1] * x ** popt[0]
     
    popt, perr = loglog_slope2(x, y, z)
     
     #fitfunc = lambda x, a, c : c * x**a
     # popt, pcov = curve_fit(fitfunc, x, y, 
     #                        sigma=z, absolute_sigma=True)
    # perr = np.sqrt(np.diag(pcov))
    
    print(*popt, *perr)
     
    nstd = [-0.5, 0.5]
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
     
        
     
    xn = np.logspace(log10(x.min()), log10(1*x.max()))
    fit = fitfunc(xn, popt)
     
    txt = r'$\quad a^\prime={:.0f}, b^\prime={:.2f}$'.format(popt[1], popt[0])
    ax.plot(xn, fit,                 
        label=r"$a^\prime \, \dot{\gamma}^{b^\prime}$" + txt,
        color='k',
        ls='-',
           #label=r'$\gamma = {} + {:.0f}\;'.format(yc, popt[1]) + \
           #    ' \dot{\gamma}^{' + '{:.2f}'.format(popt[0]) + '}$'
          )
     
    #print(*popt_up, *popt_dw)
    fit_up = fitfunc(xn, popt_up)
    fit_dw = fitfunc(xn, popt_dw)
    ax.fill_between(xn, fit_up, fit_dw,
                     color='#EDBB99', 
                     alpha=0.5
                     )
                    

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\gamma - \gamma_{\,\mathrm{RP}}$', fontsize=fntsize)

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


    fig, axs = plt.subplots(nrows=2, ncols=1, 
                            sharex=True,
                            gridspec_kw={'hspace': 0})
    
    for ax in axs:
        ax.tick_params(which='both', direction='in', 
                   top=True,right=True)
        ax.set_facecolor(mainfacecolor)
    
    
    titles = ['a', 'b',]
    for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.11, y=0.9, fontsize=16)
    
    

    
    axs[0].set_ylim([-0.15, 1.6])
    plot_nu(axs[0], plt_xlabel=False)
    
    axs[1].set_ylim([-0.1, 1.8])
    plot_gamma(axs[1], plt_xlabel=True)
    
    plt.savefig('nu_gamma_gammadot.pdf',
            pad_inches=0.02, bbox_inches='tight')
    
    





