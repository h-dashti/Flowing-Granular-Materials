import numpy as np
import os, sys
from math import log10, ceil



sys.path.insert(0, '../modules')
from mstring import as_si, show_error
from fit import interp, loglog_slope


sys.path.insert(0, 'aux-codes')
from catch_data import get_XY
from params import markers, fillstyles, markeredgewidth
from params import ls_fc, c_fc
from colors import shadecolor,  edgecolor, mainfacecolor 
from colors import hlinecolor, hline_linestyle

fntsize = 16
fntsize_info = 14
elinewidth = 0.5
########################################################################
idir_expo = '../main/DATA3'
fname_fc = 'fc__by-Ps.dat'
fpath_fc = os.path.join(idir_expo, fname_fc)
ix, iy = 0, 4
########################################################################
idir = '/home/uqedasht/Dropbox/Dynamic/current_sims/Granular/Stats/fixed_N/DATA/2021.02-09--for-paper/clus_stat'
#idir = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\1-final-paper\clus_stat'
#'/Users/ebi/Desktop/All/Sims/granular/1-final-paper/clus_stat'
#r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\1-final-paper\clus_stat'
N_arr = [2048, 4096, 8192, 16384, 32768, 65536]

gammadot_arr = [1e-3,  #1e-1, 1e-2, 1e-3, 
                5e-4, 4e-4, 2e-4, 1e-4, 
                5e-5, 2e-5, 1e-5, 
                5e-6, 2e-6, 1e-6,
                ]
phi = 0.86
########################################################################

def getM(idir_data,
         gammadot=1e-6):


    fc_table = np.loadtxt(fpath_fc)
    ig1 = np.where(np.abs(fc_table[:,0] - gammadot) < 1e-10)[0][0]
    fcg = fc_table[ig1, 1]


    data_arr, params_arr = \
    get_XY(idir_data, 'stat_f_depend', ix, iy, N_arr, phi, gammadot)

    # plot #

    oset = []

    # we know tha  M = Pinf * N

    for i in range(len(data_arr)):
        val = [params_arr[i][0],
               params_arr[i][1],  
               interp(data_arr[i][:,0], data_arr[i][:,1], 
                      fcg) * params_arr[i][1]
               ]
        oset.append(val)

    return np.array(oset)

    #
    
#==================================================================

def plot_M_versus_L(ax, idir, 
                    markersize = 6):
    
    ig = 0
    leg_symbols = []
    leg_lines = []
    
    info = []
    
    for gammadot in gammadot_arr:
        oset = getM(idir, gammadot)
        x, y = oset[:,0], oset[:,2]  # L, M
        
  
        label = r'$\dot{\gamma}='+'{}'.format(gammadot) + '$'
        pl = ax.plot(x, y, 
                     #label = label,
                     marker = markers[ig], 
                     markersize = markersize,
                     fillstyle = 'none', 
                     linestyle ='', 
                     )
        
        leg_symbols.append(label)
        
        
        expo, c, expo_err, c_err = loglog_slope(x[1:],y[1:])
        info.append([gammadot, expo, expo_err])
        
        
        xn = np.logspace(log10(45), log10(350))
        yn = c * xn ** expo
        label = '$L^{' + show_error(expo, expo_err, 3) + '}$'
    
        pl = ax.plot(xn, yn, 
                #label = label,
                color = pl[0].get_color(),
                lw=1,)
        
        leg_lines.append(label)
    
        ig += 1
    # end fo g

    with open('Df_L--by-M.dat', 'w') as f:
            f.write('# gammadot\tdf\terr\n')
            for v in info:
                f.write('{}\t{}\t{}\n'.format(*v))
                
                
                
    
    lines = ax.get_lines()
    
    ax.set_xlabel(r'$L$', fontsize=fntsize)
    ax.set_ylabel(r'$M(f_t=f_c)$', fontsize=fntsize)
    #ax.legend(frameon=False, ncol=2, )
               #bbox_to_anchor=(1,1)) #
    leg1 = ax.legend([lines[i] for i in range(0, len(lines),2)], 
                       leg_symbols, 
                       loc='upper left',
                       frameon=False,
                       fontsize='small')
    
    leg2 = ax.legend([lines[i] for i in range(1, len(lines),2)], 
                       leg_lines, 
                       loc='lower right',
                       frameon=False,
                       fontsize='small')
    
    ax.add_artist(leg1)

#==================================================================

def plot_df_by_M(ax, fpath = 'Df_L--by-M.dat',
                 markersize = 9):
    
    info = np.loadtxt(fpath)
    
    x, y, yerr = info[:,0], info[:,1], info[:,2]
    
    
    yc, yc_err = 1.85, 0.01
    ax.axhspan(yc - yc_err, yc + yc_err,
               facecolor = shadecolor,
               edgecolor = edgecolor) # alpha=0.2
    hline = ax.axhline(y=yc,
                       color = hlinecolor,
                       ls = hline_linestyle,
                       label=r'$d_f^{\mathrm{RP}} $')
    
    axplt = ax.errorbar(x, y, yerr=yerr, 
                label=r'$\mathrm{sim}$',
                marker='o',
                fillstyle='none',
                linestyle = '--',
                markersize = markersize,
                elinewidth = elinewidth,
                color = 'k',
                lw = 1,)
    
    
    # df = 2 - beta/nu,  err_df = (err_beta*nu - err_nu*beta)nu^2
    # df = 1.85 +- 0.01
    
    ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    ax.set_ylabel(r'$d_f$', fontsize=fntsize)
    
    ax.legend(handles=[axplt, hline],
                  loc='lower right', fontsize=fntsize_info, frameon=False )
    


########################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    

    
    ncols, nrows = 2, 1
    fig, axs = plt.subplots(1, 2,
                           figsize=(6*ncols, 4*nrows),
                           gridspec_kw={'wspace': 0.2},)

        
    
    titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for ax, title in zip(axs, titles):
        ax.set_title(title,
                     fontweight="bold", x=-0.11, y=0.9, fontsize=16)
        
        
    # plot M versus L
    ax = axs[0]
    ax.tick_params(direction='in', which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plot_M_versus_L(ax,idir)
    


    
    # plot df versus gammadot
    ax = axs[1]
    ax.tick_params(direction='in', which='both', right=True)
    ax.set_xscale('log')
    ax.set_ylim([1.805, 1.885])
    #ax.set_facecolor(mainfacecolor)
    plot_df_by_M(ax)
    
    


    plt.savefig('M.pdf',
                pad_inches=0.02, bbox_inches='tight')
