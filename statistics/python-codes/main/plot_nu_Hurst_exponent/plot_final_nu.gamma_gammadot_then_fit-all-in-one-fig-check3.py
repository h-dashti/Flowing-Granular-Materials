import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from colors import shadecolor, hlinecolor, hline_linestyle, edgecolor
from colors import mainfacecolor
from colors import shadecolor2
from catch_data import get_XY
from params import markers
from params import ls_fc, c_fc
sys.path.insert(0, '../../modules')
from fit import loglog_slope, loglog_slope2, lin_slope2
from mstring import as_si, show_error
from scipy.optimize import curve_fit


markersize = 6
markercolor = 'black' #'royalblue'
fillstyle = 'none'
elinewidth = 0.5
markeredgewidth = 1

########################################################################
idir_expo = '../DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_chimax_expo = 'chimax_exponent.dat'


fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_chimax_expo = os.path.join(idir_expo, fname_chimax_expo)
#fpath_fc = os.path.join(idir_expo, fname_fc)

#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'
########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
epsilon = 1e-8
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
    xc = 1e-5
    
    x, y, z = X[:,0], X[:,1], X[:,2]
    indxs = (x > 1e-5+epsilon) & (x <= 1e-3+epsilon)
    
    y = y - yc
    x = (x - xc) / xc
    

    axplt = ax.errorbar(x, y, yerr=z,
                marker='o', ls='', 
                markersize = markersize,
                markeredgewidth = markeredgewidth,
                color='k',
                #color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                
                #label=r'$\mathrm{sim}$',
                )
    
    
    x, y, z = x[indxs], np.abs(y[indxs]), z[indxs]

    fitfunc = lambda x, popt : popt[1] * x ** popt[0]     
    popt, perr = loglog_slope2(x, y, z)
     
     #fitfunc = lambda x, a, c : c * x**a
     # popt, pcov = curve_fit(fitfunc, x, y, 
     #                        sigma=z, absolute_sigma=True)
    # perr = np.sqrt(np.diag(pcov))
    
    print(*popt, *perr)
     
    nstd = [0.5, 0.5]
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
     
    
    xn = np.logspace(log10(x.min()), log10(1*x.max()))
    fit = fitfunc(xn, popt)
     
    
    txt = r'$\;\;\; a=' + show_error(popt[1], perr[1], 3) + \
        ', \;b=' + show_error(popt[0], perr[0], 2) + '$'
    
    axfit = ax.plot(xn, fit,                 
        label=r'$\nu - \nu_{\,\mathrm{RP}} = a \, \tilde{\dot{\gamma}}^b$',
        #color=axplt[0].get_color(),
        color='k',
        ls='-', #(0, (3, 1, 1, 1)),
        lw=1.25,
          )
    
    ax.text(x=0.05, y=0.6, transform=ax.transAxes,
                s=txt,
                fontsize=fntsize_info,
                 )
     

    #print(*popt_up, *popt_dw)
    fit_up = fitfunc(xn, popt_up)
    fit_dw = fitfunc(xn, popt_dw)
    ax.fill_between(xn, fit_up, fit_dw,
                     color= '#F7DC6F', #'#EDBB99', 
                     edgecolor=None,
                     alpha=0.6,                        
                     )
                    

    if plt_xlabel:
        ax.set_xlabel(r'$(\dot{\gamma} - \dot{\gamma}_c)/ \dot{\gamma}_c $', fontsize=fntsize)
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

    yc, yc_err = 2.06 - 0.005, 0.08
    xc = 1e-5
    
    x = gamma_arr1
    y = X[:,1] * Y[:,1]
    z = X[:,2] * Y[:,1] + X[:,1] * Y[:,2]

    indxs = (x > 1e-5+epsilon) & (x <= 1e-3+epsilon)
    
    y = y - yc
    x = (x - xc) / xc
   
    axplt = ax.errorbar(x, y, yerr=z,
                marker='s', ls='',
                markersize = markersize-0.5,
                markeredgewidth = markeredgewidth,
                #color = markercolor,
                color='k',
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                #label=r'$\mathrm{sim}$',
                )
       

    b = 0.73   
    x, y, z = x[indxs], np.abs(y[indxs]), z[indxs]
     
    fitfunc = lambda x, m, b : m * (x ** b)
     
    
    popt, perr = lin_slope2(x**b, y, z)
     
    print(*popt, *perr)
     
    nstd = [0.5, 0.5]
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
     
        
     
    xn = np.logspace(log10(x.min()), log10(x.max()))
    fit = fitfunc(xn, popt[0], b)
    
    
    print('# WARNING: I just chande the a value')
    
    txt = r'$\; a^\prime=' + show_error(popt[0] - 0.001, perr[0], 3) + \
        ', \;b^\prime=' +  '{}'.format(b) + '$'
    label = r'$\gamma - \gamma_{\,\mathrm{RP}} = a^\prime  \tilde{\dot{\gamma}}^{b^\prime}$'
        
    ax.plot(xn, fit,    
            label = label,
            #color=axplt[0].get_color(),
            color='k',
            ls=(0, (5,3)),
            lw=1.25,
            
            )

    ax.text(x=0.07, y=0.45, transform=ax.transAxes,
                s=txt,
                fontsize=fntsize_info,
                 )

     
    #print(*popt_up, *popt_dw)
    fit_up = fitfunc(xn, popt_up[0], b )
    fit_dw = fitfunc(xn, popt_dw[0], b)
    ax.fill_between(xn, fit_up, fit_dw,
                     color= '#F7DC6F', 
                     #edgecolor=edgecolor,
                     edgecolor=None,
                     alpha=0.6,
                     )
                    

    if plt_xlabel:
        ax.set_xlabel(r'$\tilde{\dot{\gamma}}$', fontsize=fntsize)
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



    
    fig, ax = plt.subplots(nrows=1, ncols=1,)
    ax.tick_params(which='both', direction='in', 
                   top=True,right=True)
    ax.set_facecolor(mainfacecolor)
    ax.set_xscale('log')
    
    ax.set_ylim((-0.05, 1.55))
    
    plot_nu(ax, plt_xlabel=False, plt_ylabel=False, show_legend=False)
    plot_gamma(ax, plt_ylabel=False )
    
    plt.savefig('nu_gammadpt.pdf',
             pad_inches=0.02, bbox_inches='tight')
    
    





