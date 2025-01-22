import numpy as np
import os, sys
from math import log10
sys.path.insert(0, '../aux-codes')
from colors import shadecolor, hlinecolor, hline_linestyle, edgecolor
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
idir_expo = '../DATA2'
fname_nu = 'nu__by-Ps.dat'


fpath_nu = os.path.join(idir_expo, fname_nu)
#fpath_fc = os.path.join(idir_expo, fname_fc)

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
    
    
    
    
    ax.plot([7e-7, 2e-5], [yc, yc], label='$y={}$'.format(yc))
    
    xn = np.logspace(log10(7e-6), log10(1e-3))
    yn = 738.2441258 * xn ** 0.94 + yc
    ax.plot(xn, yn, label='$y=738.2 x ^{0.94}$')
    
    yn = 859 * xn ** 1 + yc
    ax.plot(xn, yn, label='$up$')
    
    yn = 142 * xn ** 0.83 + yc
    ax.plot(xn, yn, label='$dw$')
    

    #ax.set_ylim(1, 2)

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\nu$', fontsize=fntsize)

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


    plot_nu(ax)
    
    plt.savefig('nu_gammadot_fit.pdf',
            pad_inches=0.02, bbox_inches='tight')
    
    





