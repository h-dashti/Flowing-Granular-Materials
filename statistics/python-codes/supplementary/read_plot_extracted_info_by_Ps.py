
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import json
from math import log10

sys.path.insert(0, '../modules')
from mstring import as_si, show_error
from fit import loglog_slope

sys.path.insert(0, 'aux-codes')
from catch_data import get_XY
from params import markers, fillstyles, markeredgewidth

#================================================================

def plot_qunatity_versus_N(ax, fpath, gammadot_arr, 
                           quantity = 'Delta',
                           xlabel = None,
                           ylabel = None,
                           slope = True,
                           show_label_of_curve = True,
                           show_legend = True,
                           ncol = 1,
                           markersize=7,
                           ls = '', lw = 0.5,
                           fntsize=16,
                           fntsize_info=13
                           ) :

    # Opening JSON file
    with open(fpath) as json_file:
        data = json.load(json_file)
            
    
    # now plot delta in terms of N for various gammadot
    
    ig = 0
    for g in gammadot_arr:
        g_st = '{:.4g}'.format(g)
        x, y = [], []
        x = [int(N) for N in data ]
        y = [data[N][g_st][quantity] for N in data ] 
        z = [data[N][g_st]['err_{}'.format(quantity)] for N in data ] 
        
        txt = ''
        if show_label_of_curve:
            txt = r'$\dot{\gamma}=' + as_si(g, 0) + '$'
       
            
        plt = ax.errorbar(x, y, yerr=z,
                    ls = ls,
                    lw = lw,
                    marker = markers[ig],
                    markersize = markersize,
                    label = txt,
                    fillstyle = 'none',
                    )
        color = plt[0].get_color()
        
        if slope:
            expo, c, expo_err, c_err = loglog_slope(x, y)
            
            xn = np.logspace(log10(1.7e3), log10(0.8e5))
            yn = c * xn ** expo
            
            txt = '$N^{' + show_error(expo, expo_err, 3) + '}$'
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
        ax.legend(ncol=ncol, bbox_to_anchor=(1,1), frameon=False, )


#================================================================
if __name__ == '__main__':
    
    N_arr = [2048, 4096, 8192, 16384, 32768, 65536]
    
    gammadot_arr = [1e-3,  #1e-1, 1e-2, 1e-3, 
                    5e-4, 4e-4, 2e-4, 1e-4, 
                    5e-5, 2e-5, 1e-5, 
                    5e-6, 2e-6, 1e-6,
                    ]
    phi = 0.86
    
    gammadot_arr.sort(reverse=True)
    N_arr.sort()
    idir = '../main/DATA3'
    fname = 'extracted_info_by_Ps.json'
    fpath = os.path.join(idir, fname)
    #------------------------------------

    plt.figure(1)    
    fig, axs = plt.subplots(1, 2, figsize=(7*2, 4),
                            gridspec_kw={'wspace': 0.2},)
                                # 'hspace': 0.15, 
    
    titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for ax, title in zip(axs, titles):
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.set_title(title,
                     fontweight="bold", x=-0.10, y=0.9, 
                     fontsize=16)
    #------------------------------------
    ax = axs[0]
    ax.set_xscale('log')
    #ax.set_yscale('log')
    
    plot_qunatity_versus_N(ax, fpath, gammadot_arr,
                           quantity = 'fc',
                           slope = False,
                           show_label_of_curve = True,
                           show_legend = False,
                           ls = '--',
                           xlabel = '$N$',
                           ylabel = r'$F_N$')
    #------------------------------------
    
    ax = axs[1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plot_qunatity_versus_N(ax, fpath, gammadot_arr,
                           quantity = 'Delta',
                           slope = True,
                           show_label_of_curve = True,
                           show_legend= True,
                           ncol = 2,
                           xlabel = '$N$',
                           ylabel = r'$\Delta_N$')
    #------------------------------------
    
    plt.savefig('Delta.pdf', pad_inches=0.02, bbox_inches='tight')
    
    
    