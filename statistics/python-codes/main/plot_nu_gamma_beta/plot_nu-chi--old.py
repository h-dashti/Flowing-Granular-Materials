import numpy as np
import os, sys
from math import log10
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.patches import ArrowStyle

from params import ffmt, fntsize, markers, get_mycolors, onecolor
#from scipy.optimize import curve_fit
sys.path.insert(0, '../aux-codes')
from catch_data import get_XY

########################################################################
idir = 'DATA'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fpath_nu = os.path.join(idir, fname_nu)
fpath_fc = os.path.join(idir, fname_fc)

idir_chi = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\clus-stat\2021.02.10\clus_stat'
#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'
########################################################################
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
########################################################################

def plot_nu(ax,
            plt_xlabel=True,
            plt_ylabel=True):

    
    print ('** plot nu ...')

    data = np.loadtxt(fpath_nu)
    ax.set_xscale('log')
    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], 
                marker='s', ls='--', lw=1,
                fillstyle='bottom',
                #color=onecolor,
                )

    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$\nu$', fontsize=fntsize)
########################################################################

def plot_nu_chi(plt, ax):

    
    ax1 = plt.axes((0, 0, 1, 0.5))
    plot_nu(ax1)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(right=True, direction='in')
    #ax1.set_ylim(1.1, 2.12)

    
    
    
    ax2 = plt.axes((0, 0.5, 0.5, 0.5))
    plot_chi(ax2, show_legend=True)
    
    ax3 = plt.axes((0.5, 0.5, 0.5, 0.5))
    plot_chi(ax3, show_legend=False, rescale=True, plt_ylabel=False)
    #ax3.axes.get_yaxis().set_visible(False)
    ax3.set_yticklabels([])
    
    
    
    arrow_style = ArrowStyle("Fancy", 
                             head_length=.5, head_width=.6, tail_width=.3)
    
    ax1.annotate("",
            xy=(0.2,1), xycoords='axes fraction',
            xytext=(0.05, 0.1), textcoords='axes fraction',
            arrowprops=dict(arrowstyle=arrow_style,
                            color="0.5",
                            patchB=None,
                            shrinkB=4,
                            connectionstyle="arc3,rad=0.2",
                            ),
            )
    
    
     # gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)
    # ax1 = fig.add_subplot(gs1[:-1, :])
    # ax2 = fig.add_subplot(gs1[-1, :-1])
    # ax3 = fig.add_subplot(gs1[-1, -1])

    # axin1 = inset_axes(ax, width=2.2, height=1.5,
    #                    bbox_to_anchor=(.15, .3, .5, .5),
    #                    bbox_transform=ax.transAxes,
    #                    loc=3,)

    #plot_chi(ax2, show_legend=False)


    # axin2 = inset_axes(ax, width=2.2, height=1.5,
    #                    bbox_to_anchor=(.35, .8, .5, .5),
    #                    bbox_transform=ax.transAxes,
    #                    loc=3,)

    #plot_chi(ax3, show_legend=False, rescale=True)




########################################################################
if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    #print(matplotlib.rcParams['font.family'])
    
    #matplotlib.rcParams['text.usetex']=True
    mpl.rcParams['font.family'] = 'serif'
    #matplotlib.rcParams['font.serif'] = 'Times New Roman'
    
    #plt.rc( 'text', usetex=True ) 

    #with plt.style.context('Solarize_Light2'):

    #fig, ax = plt.subplots()
    fig = plt.figure()
    #fig.set_facecolor('#FCF3CF')


    plot_nu_chi(plt, None)

    plt.savefig('chi_nu.pdf',
        pad_inches=0.015, bbox_inches='tight',
        #facecolor=fig.get_facecolor()
        )







