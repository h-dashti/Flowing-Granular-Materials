import numpy as np
import os, sys
from math import log10
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.patches import ArrowStyle


#from scipy.optimize import curve_fit

#----adding modules folder-----
# import pathlib
# cwd = pathlib.Path(__file__).parent.absolute()
# extra_path = os.path.join(pathlib.Path(cwd).parent, 'modules')
# sys.path.insert(0, extra_path)
sys.path.insert(0, '../../modules')
from mstring import as_si


sys.path.insert(0, '../aux-codes')
from catch_data import get_XY
from params import markers, get_mycolors, fillstyles, markeredgewidth
from params import onecolor, msize1, ls_fc, c_fc


########################################################################
idir_expo = '../DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fname_pinf_expo = 'Pinffc_exponent.dat'
fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_fc = os.path.join(idir_expo, fname_fc)
fpath_pinf_exo = os.path.join(idir_expo, fname_pinf_expo)

ix, iy = 0, 4
########################################################################
N_arr = [2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86
########################################################################

def plot(idir_data,
         ax,
         plt_xlabel=True,
         plt_ylabel=True,
         show_legend = False,
         gammadot=1e-6,
         fc = None,
         fc_txt = False,
         rescale=False,
         show_info=True,
         title='',
         marker_size=4,
         line_style='',
         fntsize=16,
         fntsize_info=14,
         fntsize_expo=14):


    if rescale:
        fc_table = np.loadtxt(fpath_fc)
        nu_table = np.loadtxt(fpath_nu)
        pinffc_expo_table = np.loadtxt(fpath_pinf_exo)

        ig1 = np.where(np.abs(fc_table[:,0] - gammadot) < 1e-10)[0][0]
        ig2 = np.where(np.abs(nu_table[:,0] - gammadot) < 1e-10)[0][0]
        ig3 = np.where(
            np.abs(pinffc_expo_table[:,0] - gammadot) < 1e-10)[0][0]


    data_arr, params_arr = \
    get_XY(idir_data, 'stat_f_depend', ix, iy, N_arr, phi, gammadot)

    # plot #
    for i in range(len(data_arr)):

        x = data_arr[i][:,0]
        y = data_arr[i][:,1]
        N = params_arr[i][1]

        if rescale:
            
            x = (x - fc_table[ig1,1]) * N **(1/(2*nu_table[ig2,1]))
            y = y * N**(pinffc_expo_table[ig3,1] / 2)
        ##   

        if marker_size <= 2:
            dic_style = {'marker':'.', 'markersize':marker_size,
                         'linewidth':0.2, 'linestyle':line_style,
                        }
        else:
            dic_style = {'marker':'o', 'markersize':marker_size,
                         'fillstyle':fillstyles[i], 
                         'markeredgewidth': markeredgewidth,
                         #'markeredgecolor':'k', 
                         'linewidth':0.2, 'linestyle':line_style,
                        }
                        
        ax.plot(x, y, label='$N={:.0f}$'.format(N), \
			    **dic_style)

	# end for i
    
    if fc is not None:
        ax.axvline(x=fc, color=c_fc, ls=ls_fc, 
                   lw=0.75, zorder=1.1)

       
    
    if show_info:
        txt = r'$\dot{\gamma}=' + as_si(gammadot,0) + '$'
        ax.text(x=0.05, y=0.15, transform=ax.transAxes,
                s=txt,
                fontsize=fntsize_info,
                 )

    if rescale:
        txt = r'$\beta=0.20(4)$' # \quad \nu=1.21(1) r'$\beta=0.199 \pm 0.023$'
        ax.text(x=0.4, y=0.8, transform=ax.transAxes, s=txt,
                fontsize=fntsize_expo)
                

        
            #txt = r'$f_c=0.0108(3)$'
            #ax.text(x=0.6, y=0.7, transform=ax.transAxes, s=txt,
            #        fontsize=fntsize_info)
            
                

    if rescale:
        xlabel = r'$N^{1/2\nu} (f_t-f_c)$'
        ylabel = r'$N^{\beta/2\nu}  P_{\infty} $'
    else:
        xlabel = r'$f_t$'
        ylabel = r'$P_{\infty}$'


    if plt_xlabel:
        ax.set_xlabel(xlabel, fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(ylabel, fontsize=fntsize)


    if show_legend:
        ax.legend(loc='center left', frameon=False)
########################################################################


########################################################################
if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.family'] = 'serif'
    
    idir_data = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\1-final-paper\clus_stat'


    #with plt.style.context('Solarize_Light2'):

    fig, ax = plt.subplots()
    #fig = plt.figure()
    #fig.set_facecolor('#FCF3CF')


    #plot(idir_data, ax, fc=0.0108, line_style='--', fntsize = fntsize )


    #plt.savefig('chi.pdf',
    #    pad_inches=0.015, bbox_inches='tight',
        #facecolor=fig.get_facecolor()
    #    )







