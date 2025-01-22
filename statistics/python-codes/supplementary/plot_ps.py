import numpy as np
import os, sys
from math import log10, ceil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



sys.path.insert(0, '../modules')
from mstring import as_si

sys.path.insert(0, 'aux-codes')
from catch_data import get_XY
from params import markers, fillstyles, markeredgewidth
from params import ls_fc, c_fc #fntsize

########################################################################
idir_expo = '../main/DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_fc = 'fc__by-Ps.dat'
fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_fc = os.path.join(idir_expo, fname_fc)
ix, iy = 0, 15
########################################################################
idir = '/home/uqedasht/Dropbox/Dynamic/current_sims/Granular/Stats/fixed_N/DATA/2021.02-09--for-paper/clus_stat'
#idir = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\1-final-paper\clus_stat'
#r'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'    
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
phi = 0.86    
gammadot_arr = [1e-3,  #1e-1, 1e-2, 1e-3, 
                5e-4, 4e-4, 2e-4, 
                1e-4, 
                5e-5, 2e-5, 1e-5, 
                5e-6, 2e-6, 1e-6,
                ]
########################################################################

def plot(idir_data,
         ax,
         plt_xlabel=True,
         plt_ylabel=True,
         show_legend = False,
         gammadot=1e-6,
         rescale=False,
         show_info=False,
         show_expo = False,
         title=None,
         
         marker_size=4,
         line_style='--',
         fntsize=16,
         fntsize_info=13):


    fc_table = np.loadtxt(fpath_fc)
    nu_table = np.loadtxt(fpath_nu)

    ig1 = np.where(np.isclose(fc_table[:,0], gammadot))[0][0]
    ig2 = np.where(np.isclose(nu_table[:,0], gammadot))[0][0]
    fc = fc_table[ig1,1]
    nu = nu_table[ig2,1]
    
    data_arr, params_arr = \
    get_XY(idir_data, 'stat_f_depend', ix, iy, N_arr, phi, gammadot)

    # plot #
    for i in range(len(data_arr)):

        x = data_arr[i][:,0]
        y = data_arr[i][:,1]
        N = params_arr[i][1]

        if rescale:
            x = (x - fc) * N **(1/(2*nu))

        if marker_size <= 2:
            dic_style = {'marker':'.', 'markersize':marker_size,
                         'linewidth':0.2, 'linestyle':line_style,
                        }
        else:
            dic_style = {'marker':'o', 'markersize':marker_size,
                         'fillstyle':fillstyles[i], 
                         'markeredgewidth': markeredgewidth, 
                         'linewidth':0.2, 'linestyle':line_style,
                        }
                        
        ax.plot(x, y, label='$N={:.0f}$'.format(N), \
			    **dic_style)

	# end for i
    
    xfc = 0 if rescale else fc 
    ax.axvline(x=xfc, color=c_fc, ls=ls_fc, 
               lw=0.75, zorder=1.1)
    
    if not rescale:
        xfc2 = xfc - 0.002 if np.isclose(gammadot, 0.001) else xfc - 0.00075
        ax.annotate("$f_c$",
                    xy=(xfc, 0.01), xycoords='data',
                    xytext=(xfc2, 0.02), textcoords='data',
                    arrowprops=dict(arrowstyle="->", 
                                    connectionstyle="arc3",
                                    lw=0.5,),
                    )
           
    
    if show_info:
        txt = r'$\dot{\gamma}=' + as_si(gammadot,0) + '$'
        ax.text(x=0.05, y=0.15, transform=ax.transAxes,
                s=txt, fontsize=fntsize_info,)
    if show_expo:    
        txt = r'$\nu={:.2f}'.format(nu-0.001) + '$'
        ax.text(x=0.8, y=0.22, transform=ax.transAxes,
                s=txt, fontsize=fntsize_info,)
        

    if title is not None:
        ax.title(title)

    if rescale:
        xlabel = r'$N^{1/2\nu} (f_t-f_c)$'
    else:
        xlabel = r'$f_t$'


    if plt_xlabel:
        ax.set_xlabel(xlabel, fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$P_s$', fontsize=fntsize)


    if show_legend:
        ax.legend(loc='center left', frameon=False, fontsize='small')
########################################################################


########################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    ncols = 3
    nrows = ceil(len(gammadot_arr) / ncols)
    
    fig, axs = plt.subplots(nrows, ncols,
                           figsize=(6*ncols, 4*nrows),
                           gridspec_kw={'wspace': 0.15, 'hspace': 0.2},)
    axs = axs.flat
        
  
    
    ig = 0
    for gammadot in gammadot_arr:
        ax = axs[ig]
        ax.tick_params(direction='in', which='both')
        #if ig == 1 : ax.set_xlim(0.00825, 0.016)  
          
          
        plot(idir, ax, 
             show_info=True, show_legend=True, show_expo=True,
             gammadot=gammadot, fntsize = 16)
        
        ig += 1
    # end for g
    
   

    # now make inset ax
    if True:
        ig = 0
        for gammadot in gammadot_arr:
            axsize = [0.64, 0.525, 0.35, 0.45]
            #if ig == 1: axsize = [0.625+0.05, 0.5, 0.3, 0.45]          
                
            ax = axs[ig].inset_axes(axsize)
            ax.tick_params(direction='in', which='both')
            
            plot(idir, ax, gammadot=gammadot, 
                 rescale=True, 
                 fntsize = 12, marker_size=2, line_style='',)
            
            ig += 1
        #
    #
    
    
    titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for ax, title in zip(axs, titles):
        ax.set_title(title,
                     fontweight="bold", x=-0.10, y=0.9, 
                     fontsize=16)


    # remove extra axis
    for i in range(len(gammadot_arr), len(axs)):
        axs[i].remove()

   

    plt.savefig('ps.pdf',
                pad_inches=0.02, bbox_inches='tight')





