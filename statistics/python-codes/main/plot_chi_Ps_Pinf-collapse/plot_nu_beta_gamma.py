import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from colors import shadecolor, hlinecolor, hline_linestyle, edgecolor
from colors import shadecolor2
from catch_data import get_XY
from params import markers
from params import ls_fc, c_fc
sys.path.insert(0, '../../modules')
from mstring import as_si


markersize = 9
markercolor = 'black' #'royalblue'
fillstyle = 'none'
elinewidth = 0.5

########################################################################
idir_expo = '../DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fname_pinf_expo = 'Pinffc_exponent.dat'
fname_chimax_expo = 'chimax_exponent.dat'

# df = 2 - beta/nu,  err_df = (err_beta*nu - err_nu*beta)nu^2
# df = 1.85 +- 0.01

fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_fc = os.path.join(idir_expo, fname_fc)
fpath_pinf_expo = os.path.join(idir_expo, fname_pinf_expo)
fpath_chimax_expo = os.path.join(idir_expo, fname_chimax_expo)

#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'
########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
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
    x, y, z = X[:,0], X[:,1], X[:,2]

    yc, yc_err = 1.21, 0.06
    ax.axhspan(yc - yc_err, yc + yc_err,
               facecolor = shadecolor,
               edgecolor = edgecolor
               ) # alpha=0.2
    hline = ax.axhline(y=yc,
                       color = hlinecolor,
                       ls = hline_linestyle,
                       label=r'$\nu_{\mathrm{RP}} $') #= 1.21 \pm 0.06

    ax.set_xscale('log')
    axplt = ax.errorbar(x, y, yerr=z,
                marker='o', ls='--', 
                lw=0.5,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                label=r'$\mathrm{sim}$',
                zorder=3
                )

    ax.set_ylim(1, 2)

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\nu$', fontsize=fntsize)

    if show_legend:
        ax.legend(handles=[axplt, hline],
                  loc='upper left', fontsize=fntsize_info, frameon=False )
########################################################################


def plot_beta(ax,
            plt_xlabel=True,
            plt_ylabel=True,
            show_legend=True,
            fntsize=16,
            fntsize_info=14,
            ):

    print ('** plot beta ...')

    X = np.loadtxt(fpath_nu)
    Y = np.loadtxt(fpath_pinf_expo)

    gamma_arr1 = X[:,0]
    gamma_arr2 = Y[:,0]
    not_same_gamma = (np.abs(gamma_arr1 - gamma_arr2) > 1e-5)
    if np.any(not_same_gamma):
        print('error in gamma_plot')
        sys.exit()


    x = gamma_arr1
    y = X[:,1] * Y[:,1]
    z = X[:,2] * Y[:,1] + X[:,1] * Y[:,2]

    if False:
        for d1, d2 in zip(X, Y):
            print ('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.\
                  format(d1[0], d1[1], d1[2], d2[1], d2[2],
                         d1[1]*d2[1],
                         d1[2]*d2[1]+d1[1]*d2[2]))

    yc, yc_err = 0.18, 0.02
    ax.axhspan(yc - yc_err, yc + yc_err,
               facecolor = shadecolor,
               edgecolor = edgecolor
               ) # alpha=0.2
    hline = ax.axhline(y=yc,
                       color = hlinecolor,
                       ls = hline_linestyle,
                       label=r'$\beta_{\mathrm{RP}}$') # = 0.18 \pm 0.02


    ax.set_xscale('log')
    axplt = ax.errorbar(x, y, yerr=z,
                marker='o', ls='--', 
                lw=0.5,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                #color = mainfacecolor,
                label=r'$\mathrm{sim}$',
                zorder=3
                )

    ax.set_ylim(0.15, 0.32)
    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\beta$', fontsize=fntsize)

    if show_legend:
        #sort_legends(ax)
        ax.legend(handles=[axplt, hline],
                  loc='upper left', fontsize=fntsize_info, frameon=False )

########################################################################
def plot_gamma(ax,
            plt_xlabel=True,
            plt_ylabel=True,
            show_legend=True,
            fntsize=16,
            fntsize_info=14,
            ):

    # gamma = nu * d - 2 * beta --> gamma = 2(nu - beta)
    yc = 2 * (1.21 - 0.18)
    yc_err = 2 * (0.06 - 0.02)

    print ('** plot gamma ...')

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

    ax.axhspan(yc - yc_err, yc + yc_err,
               facecolor = shadecolor,
               edgecolor = edgecolor
               ) # alpha=0.2
    hline = ax.axhline(y=yc,
                       color = hlinecolor,
                       ls = hline_linestyle,
                       label=r'$\gamma_{\mathrm{RP}}$') # = 2.06 \pm 0.08


    ax.set_xscale('log')
    axplt = ax.errorbar(x, y, yerr=z,
                marker='o', ls='--', 
                lw=0.5,
                markersize = markersize,
                color = markercolor,
                fillstyle = fillstyle,
                elinewidth = elinewidth,
                label=r'$\mathrm{sim}$',
                zorder=3
                )


    ax.set_ylim(1.7, 3.5)

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\gamma$', fontsize=fntsize)

    if show_legend:
        #sort_legends(ax)
        ax.legend(handles=[axplt, hline],
                  loc='upper left', fontsize=fntsize_info, frameon=False )

########################################################################
if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.family'] = 'serif'


    fig, ax = plt.subplots()



    plot_beta(ax)





