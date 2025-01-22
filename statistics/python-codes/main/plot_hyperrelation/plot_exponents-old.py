import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from catch_data import get_XY
from params import fntsize, fntsize_info, markers, get_mycolors 
from params import onecolor, msize1, ls_fc, c_fc, ls_dilute
sys.path.insert(0, '../modules')
from mstring import as_si

########################################################################
idir_expo = r'../DATA'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fname_pinf_expo = 'Pinffc_exponent.dat'
fname_chimax_expo = 'chimax_exponent.dat'
fname_df = 'Df_gammadot--phi=0.86.dat'
fname_eta_direct = 'eta-direct.dat'

fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_fc = os.path.join(idir_expo, fname_fc)
fpath_pinf_expo = os.path.join(idir_expo, fname_pinf_expo)
fpath_chimax_expo = os.path.join(idir_expo, fname_chimax_expo)
fpath_df = os.path.join(idir_expo, fname_df)
fpath_eta_direct = os.path.join(idir_expo, fname_eta_direct)

########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
########################################################################

########################################################################
def plot_gammaDIVnu_df(ax,
            plt_xlabel=True,
            show_legend=True ):

    
    print ('** plot beta/nu and d-df ...')

    X = np.loadtxt(fpath_chimax_expo)
    Y = np.loadtxt(fpath_df)
  

    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]
    not_same_gamma = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)
    if np.any(not_same_gamma):
        print('error in gamma_plot')
        sys.exit()
        
    
    x = gamma_arr1
    y, yerr = X[:,1], X[:,2]
    z, zerr = 2*Y[:,1] - 2, 2*Y[:,2]
    
        
    ax.set_xscale('log')
    ax.errorbar(x, y, yerr=yerr,
                marker='s', ls='--', lw=1,
                fillstyle='bottom',
                label=r'$\gamma/\nu$',
                #color=onecolor,
                )
    
    ax.errorbar(x, z, yerr=zerr,
                marker='o', ls='--', lw=1,
                fillstyle='right',
                label=r'$2d_f - d$'
                #color=onecolor,
                )

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)

        
    if show_legend:
        ax.legend(loc='best', fontsize=fntsize, frameon=False )

########################################################################

def plot_betaDIVnu_df(ax,
            plt_xlabel=True,
            show_legend=True ):

    
    print ('** plot beta/nu and d-df ...')

    X = np.loadtxt(fpath_pinf_expo)
    Y = np.loadtxt(fpath_df)

    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]
    not_same_gamma = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)
    if np.any(not_same_gamma):
        print('error in gamma_plot')
        sys.exit()
        
    
    x = gamma_arr1
    y, yerr = X[:,1], X[:,2]
    z, zerr = 2.0 - Y[:,1], Y[:,2]
    
    ax.set_xscale('log')
    ax.errorbar(x, y, yerr=yerr,
                marker='s', ls='--', lw=1,
                fillstyle='bottom',
                label=r'$\beta/\nu$',
                #color=onecolor,
                )
    ax.errorbar(x, z, yerr=zerr,
                marker='o', ls='--', lw=1,
                fillstyle='right',
                label=r'$d - d_f$'
                #color=onecolor,
                )
 
    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
        
    if show_legend:
        ax.legend(loc='best', fontsize=fntsize, frameon=False )

########################################################################
def plot_eta(ax,
             plt_xlabel=True,
             plt_ylabel = True,
             show_legend=True ):

    
    print ('** plot eta = 2 -gamma /nu and d-df ...')

    X = np.loadtxt(fpath_chimax_expo)
    Y = np.loadtxt(fpath_df)
    Z = np.loadtxt(fpath_eta_direct)
    
    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]
    gamma_arr3 = Z[:,0]
    not_same_gamma12 = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)
    not_same_gamma13 = (np.abs(gamma_arr1 - gamma_arr3) > 1e-5)
    if np.any(not_same_gamma12) or  np.any(not_same_gamma13):
        print('error in gamma_plot')
        sys.exit()
    
      
    x = X[:,0]
    y1, y1err = 2 - X[:,1], X[:,2]
    y2, y2err = 2 * (2 - Y[:,1]), 2*Y[:,2]
    y3, y3err = Z[:,1], Z[:,2]
    
        
    ax.set_xscale('log')
    ax.errorbar(x, y1, yerr=y1err,
                marker='s', ls='--', lw=1,
                fillstyle='bottom',
                label=r'$2-\gamma/\nu$',
                #color=onecolor,
                )
    
    ax.errorbar(x, y2, yerr=y2err,
                marker='o', ls='--', lw=1,
                fillstyle='right',
                label=r'$2(2 - d_f)$'
                #color=onecolor,
                )
    
    ax.errorbar(x, y3, yerr=y3err,
                marker='<', ls='--', lw=1,
                fillstyle='left',
                label=r'$\eta_{direct}$'
                #color=onecolor,
                )
    
    ax.axhline(y=5/24,  color=c_fc, ls=ls_fc, lw=1,
               label=r'$\eta_{percolation}$')

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
        
    if plt_ylabel:
        ax.set_ylabel(r'$\eta$', fontsize=fntsize)
       

        
    if show_legend:
        ax.legend(loc='best', fontsize=fntsize, frameon=False )
########################################################################

if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    #mpl.rcParams['font.family'] = 'serif'
    
    
    fig, ax = plt.subplots()
    
    
    
    

    plot_eta(ax)





