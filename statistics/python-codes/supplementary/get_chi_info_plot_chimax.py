
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import json
from math import log10
import scipy
from scipy.optimize import curve_fit


#import get_Ps_info_plot_Delta_N.get_data as get_data_ps

sys.path.insert(0, '../modules')
from mstring import as_si, show_error
from fit import loglog_slope, interp

sys.path.insert(0, 'aux-codes')
from catch_data import get_XY
from params import markers, fillstyles, markeredgewidth

#================================================================

def get_fcarray_vs_gammadot(fpath):
    
    data = np.loadtxt(fpath)
    
    info = {}
    for d in data:
        g = '{:.6g}'.format(d[0])
        info[g] = d[1]
    
    return info
#================================================================

def get_data(idir_all, idir_part, ix, iy, N_arr, gammadot_arr, phi):
    
    #fc_path = os.path.join(idir_part, 'fc__by-Ps.dat')
    #infofc = get_fcarray_vs_gammadot(fc_path)
        
    info = {}
    
    for N in N_arr:
        
        data_arr, params_arr = \
        get_XY(idir_all, 'stat_f_depend', ix, iy, N, phi, gammadot_arr)
        L = params_arr[0][0]
        
        info[N] = {}
    
        ig = 0
        for g in gammadot_arr:
            
            g_st = '{:.6g}'.format(g)
            
            x = data_arr[ig][:,0]
            y = data_arr[ig][:,1]
            
            imax = np.argmax(y)
            fc = x[imax] 
            chimax = y[imax]
            
            info[N][g_st] = {'L': L, 'fc': fc, 'chimax': chimax }
            
            ig += 1
        #end for g
        
    # end for N

    
    return info
        
    
#================================================================


def plot_qunatity_versus_N(ax, data, gammadot_arr, 
                           quantity = 'chimax',
                           xlabel = None,
                           ylabel = None,
                           slope = True,
                           show_label_of_curve = True,
                           show_legend = True,
                           ncol = 1,
                           markersize=5,
                           ls = '', lw = 0.5,
                           fntsize=16,
                           fntsize_info=14
                           ) :
            
    
    # now plot delta in terms of N for various gammadot
    
    ig = 0
    for g in gammadot_arr:
        g_st = '{:.6g}'.format(g)
        x, y = [], []
        x = [int(N) for N in data ]
        y = [data[N][g_st][quantity] for N in data ] 
        
        txt = ''
        if show_label_of_curve:
            txt = r'$\dot{\gamma}=' + as_si(g, 0) + '$'
       
        
        plt_func = ax.plot
            
        pl = plt_func(x, y, 
                    ls = ls,
                    lw = lw,
                    marker = markers[ig],
                    markersize = markersize,
                    label = txt,
                    fillstyle = 'none',
                    )
        color = pl[0].get_color()
        
        if True:
            expo, c, expo_err, c_err = loglog_slope(x, y)
            
            xn = np.logspace(log10(1.7e3), log10(0.8e5))
            yn = c * xn ** expo
          
            final_error = expo_err
            
            txt = '$N^{' + show_error(expo, final_error, 3) + '}$'
            ax.plot(xn, yn, 
                    label = txt,
                    lw = 0.75,
                    color = color,
                    )
            
        ig += 1
    # end for g
    
    if xlabel: ax.set_xlabel(xlabel, fontsize=fntsize)
    if ylabel: ax.set_ylabel(ylabel, fontsize=fntsize)
    if show_legend: 
        ax.legend(ncol=ncol, bbox_to_anchor=(1,1.05), frameon=False,
                  fontsize = fntsize_info, loc=2)


#================================================================
if __name__ == '__main__':
    
    idir_all = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\1-final-paper\clus_stat'
     #'/Users/ebi/Desktop/All/Sims/granular/1-final-paper/clus_stat'
    idir_part = '../main/DATA3'
   
    ix, iy = 0, 12
    
    N_arr = [2048, 4096, 8192, 16384, 32768, 65536]
    
    gammadot_arr = [1e-3,  #1e-1, 1e-2, 1e-3, 
                    5e-4, 4e-4, 2e-4, 1e-4, 
                    5e-5, 2e-5, 1e-5, 
                    5e-6, 2e-6, 1e-6,
                    ]
    phi = 0.86
    
    gammadot_arr.sort(reverse=True)
    N_arr.sort()
    
    info = get_data(idir_all, idir_part, ix, iy, N_arr, gammadot_arr, phi )
    
    
    #------------------------------------

    plt.figure(1)    
    fig, axs = plt.subplots(1, 1, 
                            #figsize=(6, 4)
                            #gridspec_kw={'wspace': 0.2},
                            )
                                # 'hspace': 0.15, 
    
    ax = axs
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plot_qunatity_versus_N(ax, info, gammadot_arr,
                            quantity = 'chimax',
                            slope = True,
                            show_label_of_curve = True,
                            show_legend= True,
                            ncol = 2,
                            xlabel = '$N$',
                            ylabel = r'$\chi_{max}$'
                            )
    
    
    plt.savefig('chimax.pdf', pad_inches=0.02, bbox_inches='tight')
    
    
    