import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from catch_data import get_XY
from params import fntsize, fntsize_info, markers, get_mycolors
from colors import shadecolor, hlinecolor, hline_linestyle, edgecolor
from params import ls_fc, c_fc
sys.path.insert(0, '../../modules')
from mstring import as_si


########################################################################


idir_expo = r'../DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fname_pinf_expo = 'Pinffc_exponent.dat'
fname_chimax_expo = 'chimax_exponent.dat'
fname_df = 'Df_gammadot--phi=0.86.dat'
fname_eta_direct = 'eta-direct.dat'
fname_df_byM = 'Df_L--by-M.dat'

fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_fc = os.path.join(idir_expo, fname_fc)
fpath_pinf_expo = os.path.join(idir_expo, fname_pinf_expo)
fpath_chimax_expo = os.path.join(idir_expo, fname_chimax_expo)
fpath_df = os.path.join(idir_expo, fname_df)
fpath_eta_direct = os.path.join(idir_expo, fname_eta_direct)
fpath_df_byM = os.path.join(idir_expo, fname_df_byM)

########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
########################################################################

########################################################################
def plot_gammaDIVnu_df(ax,
            plt_xlabel=True,
            show_legend=True ):


    print ('** plot gamma/nu and d-df ...')

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


    #ax.set_xscale('log')
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

    #ax.set_xscale('log')
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
    
    if np.any(not_same_gamma13) or  np.any(not_same_gamma12) :
        print('error in gamma_plot')
        sys.exit()
    

    x = X[:,0]
    y1, y1err = 2 - X[:,1], X[:,2]
    y2, y2err = 2 * (2 - Y[:,1]), 2*Y[:,2]
    y3, y3err = Z[:,1], Z[:,2]
    


    #ax.set_xscale('log')
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

def plot_eta2(ax,
             plt_xlabel=True,
             plt_ylabel = True,
             show_legend=True ):



    # eta = 2 beta / nu
    # err_eta = 2 (nu* err_beta - beta * err_nu ) / nu^2

    yc = 0.30  # 2 * (0.18 / 1.21)
    yc_err = 0.02 # 2 * ( 1.21 * 0.02 - 0.18 * 0.06) / 1.21**2

    print ('** plot eta = 2 * beta /nu and eta_direct ...')

    X = np.loadtxt(fpath_pinf_expo)
    Z = np.loadtxt(fpath_eta_direct)
    W = np.loadtxt(fpath_df_byM)

    gamma_arr1 = X[:,0]
    gamma_arr3 = Z[:,0]
    gamma_arr4 = W[:, 0]  # eta = 2 + d - 2*df
    
    #not_same_gamma13 = (np.abs(gamma_arr1 - gamma_arr3) > 1e-5)
    #not_same_gamma14 = (np.abs(gamma_arr1 - gamma_arr4) > 1e-5)
    
    #if np.any(not_same_gamma13) or  np.any(not_same_gamma14) :
    #    print('error in gamma_plot')
    #    sys.exit()



    x1, x3, x4 = gamma_arr1, gamma_arr3, gamma_arr4
    y1, y1err = 2 * X[:,1], 2*X[:,2]
    y3, y3err = Z[:,1], Z[:,2]
    y4, y4err = 4 - 2 * W[:, 1], 2 * W[:,2]
    

    ax.axhspan(yc - yc_err, yc + yc_err,
               facecolor = shadecolor,
               edgecolor = edgecolor) # alpha=0.2
    hline = ax.axhline(y=yc,
                       color = hlinecolor,
                       ls = hline_linestyle,
                       label=r'$\eta_{\mathrm{RP}}$') # = 0.30 \pm 0.02


    axplt1 = ax.errorbar(x3, y3, yerr=y3err,
			             marker='o', ls='--', 
                         lw=0.5,
			             fillstyle='left',
			             label=r'$\eta_{\mathrm{direct}}$',
			             markersize= 8,
                         zorder = 4,
                         elinewidth=0.5,
			            #color=onecolor,
			            )


    
    axplt2 = ax.errorbar(x1, y1, yerr=y1err,
		                 marker='v', ls='--', 
                         lw=0.5,
		                 fillstyle='bottom',
		                 label=r'$2\beta/\nu$',
		                 markersize= 8,
                         zorder = 3,
                         elinewidth=0.5,
		                #color=onecolor,
		                )
    
    axplt3 = ax.errorbar(x4, y4, yerr=y4err,
		                 marker='s', ls='--', 
                         lw=0.5,
		                 fillstyle='right',
		                 label=r'$2(2-d_f)$',
		                 markersize= 7.,
                         zorder = 3,
                         elinewidth=0.5,
		                #color=onecolor,
		                )



    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)

    if plt_ylabel:
        ax.set_ylabel(r'$\eta$', fontsize=fntsize)



    if show_legend:
        #sort_legends(ax)
        ax.legend(handles=[axplt1, axplt2, axplt3, hline],
                  loc='upper right', ncol=2,
                  fontsize=fntsize_info, frameon=False )
########################################################################

if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    #mpl.rcParams['font.family'] = 'serif'


    fig, ax = plt.subplots()



    ax.set_xscale('log')

    plot_eta2(ax)





