import numpy as np
import os, sys
from math import log10, ceil, log
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit


sys.path.insert(0, '../modules')
from mstring import as_si
from fit import loglog_slope

sys.path.insert(0, 'aux-codes')
from catch_data import get_XY
from params import markers, fillstyles, markeredgewidth

########################################################################
idir_expo = '../main/DATA3'
fname_nu = 'nu__by-Ps.dat'
fname_eta_direct = 'eta-direct.dat'
fpath_nu = os.path.join(idir_expo, fname_nu)
fpath_eta_direct = os.path.join(idir_expo, fname_eta_direct)
########################################################################
idir = "/home/uqedasht/Dropbox/Dynamic/current_sims/Granular/Stats/fixed_N/DATA/v2021.03.15--clus_correlation/clus_correlation"
#idir = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\2021.03.15\clus_correlation'
N_arr = [ 2048, 4096, 8192, 16384, 32768, 65536] #
cutoff_arr = [10, 15, 25, 30, 40, 50]
phi = 0.86    
gammadot_arr = [0.001, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6,] 
########################################################################
def logspace_indxs(x, scale=1.1):

    x0, x1 = 1, len(x)
    space = np.logspace(log(x0, scale), log(x1, scale),
                        base=scale, dtype=int)
    return np.unique(space) - 1
########################################################################

def plot(idir,
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
         line_style='',
         fntsize=16,
         fntsize_info=13):


    
    oset = []
    
    nu_table = np.loadtxt(fpath_nu)
    ig1 = np.where(np.isclose(nu_table[:,0], gammadot))[0][0]
    nu = nu_table[ig1,1]
    
        
    
    data_arr, params_arr = \
    get_XY(idir, 'gr', 0, 1, N_arr, phi, gammadot, 
           st_samples = 'n_snapshots',
           ic = 2)
    
    # remeber that we should later normalize the data_arr
    i = 0
    for data, params in zip(data_arr, params_arr):
        x = data[:,0]
        y = data[:,1] / data[:,2]
        N = int(params[1])
        L = params[0]
        
        x, y = x[1:-1],  y[1:-1]
        #indxs = logspace_indxs(x, 1.00001)
        #x, y = x[indxs], y[indxs]
        #y *= (1.1**i)     # for more clearity
        
        
        
        dic_style = {'marker':'o', 'markersize':marker_size,
                     'fillstyle':fillstyles[i], 
                     'markeredgewidth': markeredgewidth, 
                     'linewidth':0.2, 'linestyle':line_style,
                    }
        
        x1, y1 = x, y
        if False:
            indxs = logspace_indxs(x, 1.01)
            x1, y1 = x1[indxs], y1[indxs]
            
        
        ax.plot(x1, y1, 
                label='$N={:.0f}$'.format(N),
                **dic_style
                )
        
        
        #indxs = (y > 0) & (x >= 1) & (x <= cutoff_arr[i])
        #x2, y2 = x[indxs], y[indxs]
        iend = int(6 + i*2)
        x2, y2 = x[:iend], y[:iend]
        
        expo, c, expo_err, c_err = loglog_slope(x2, y2)
        expo_err *= 2
        
        oset.append([N, L, -expo, expo_err])
        
        print(f'  expo={expo:.3f}, err={expo_err:.3f}')
        
            
        if show_expo and i == len(params_arr) - 1:
            xn = x2
            yn = c * np.power(xn, expo)

            clabel = r'$N={}, \eta={:.2f}\pm {:.2f}$'.\
                format(N, -expo, expo_err)

            ax.loglog(xn, yn, label=clabel, \
                    ls='-', color='k')
            
        i += 1
    # end for i
    
    
    if show_info:
        txt = r'$\dot{\gamma}=' + as_si(gammadot,0) + '$'
        ax.text(x=0.05, y=0.15, transform=ax.transAxes,
                s=txt, fontsize=fntsize_info,)
        

    if title is not None:
        ax.title(title)

    if rescale:
        xlabel = r'$r$'
        ylabel = r'$g\,(r)$'
    else:
        xlabel = r'$r$'
        ylabel = r'$g\,(r)$'


    if plt_xlabel:
        ax.set_xlabel(xlabel, fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(ylabel, fontsize=fntsize)

    if show_legend:
        ax.legend(loc='center left', 
                  frameon=False, fontsize='small',
                  #bbox_to_anchor = ()
                  )
        
    return oset

   
    
def plot_eta_set(ax, eta_exponents, 
                 markersize=6,
                 fntsize=16,
                 fntsize_info=13):
    

    print ('plot eta set')
    
    eta_gammadot_at_Ninfty = []
    
    ig = 0
    for gammadot in gammadot_arr:
        if gammadot not in eta_exponents: continue
        
        xy = np.array(eta_exponents[gammadot]) # [N, L, expo, err]
        x, y, z = 1.0/xy[:,0]**0.5, xy[:,2], xy[:,3]
        
        clabel = r'$\dot{\gamma}='+'{}'.format(gammadot) + '$'
        ax.errorbar(x, y, yerr=z,
                    label=clabel,
                    elinewidth = 0.75,
                    marker=markers[ig], 
                    fillstyle='none', 
                    markersize=markersize,
                    ls='', lw=0.2)
    
    
        #----------------#
        # part fit       #
        #----------------#
        xn, yn, zn = x[1:], y[1:], z[1:]
    
        a = 1
        if a is None:
            fit_func = lambda x, c, b : c + b * np.power(x, a)
            bounds = [[0.1, 1e-10, 1e-5], 
                      [0.3, np.infty, 2]]
            
            popt, pcov = curve_fit(fit_func, xn, yn, 
                               bounds = bounds,
                               sigma = 1.0/(zn**2),
                               )
        else:
            fit_func = lambda X, c, b : c + b * X ** a
            popt, pcov = curve_fit(fit_func, xn, yn,
                                   sigma = 1.0/(zn**2),
                                   )
            
        perr = np.sqrt(np.diag(pcov))
        
        eta_gammadot_at_Ninfty.append([gammadot, popt[0], perr[0]])
        
        fmt = len(popt) * '{:4g} '
        print ('gammadot={}'.format(gammadot))
        print (' popt:', fmt.format(*popt))
        print (' perr:', fmt.format(*perr))
        
        xn = np.linspace(0, max(x)+0.001)
        yn = fit_func(xn, *popt)
        ax.plot(xn, yn, 
                 color=ax.lines[-1].get_color(),
                 ls = '-', lw = 1,
                 )
        
        ig += 1
        
    # end for gammadot
    
    plt.xlim(0, None)
    ax.legend(frameon=False, ncol=2)
    ax.set_xlabel('$1/N^{0.5}$', fontsize=fntsize)
    ax.set_ylabel(r'$\eta\,(N)$', fontsize=fntsize)
    
    
    # print eta vs gammadot for N(infty)
    print (42*'-')
    print ('# eta at N(infty)')
    
    fmt = 3*'{}\t'
    for d in eta_gammadot_at_Ninfty:
        print(fmt.format(*d))
    


########################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    ncols = 3
    nrows = ceil(len(gammadot_arr) / ncols)
    
    fig, axs = plt.subplots(nrows, ncols,
                           figsize=(6*ncols, 4*nrows),
                           gridspec_kw={'wspace': 0.2, 'hspace': 0.2},)
    axs = axs.flat
    
    titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for ax, title in zip(axs, titles):
        ax.set_title(title,
                     fontweight="bold", x=-0.1, y=0.9, 
                     fontsize=16)
        
  
    
    ig = 0
    eta_exponents = {}
    for gammadot in gammadot_arr:
        ax = axs[ig]
        ax.tick_params(direction='in', which='both')
          
        ax.set_xscale('log')
        ax.set_yscale('log')
        oset = plot(idir, ax, 
                    show_info=True, show_legend=True, 
                    show_expo=False,
                    gammadot=gammadot,)
        
        eta_exponents[gammadot] = oset
        
        ig += 1
    # end for g
    


   
    print(42*'=')
    
    
    # now plot the last axis
    plot_eta_set(axs[-1], eta_exponents)
    
    
   

    plt.savefig('gr.pdf',
                pad_inches=0.02, bbox_inches='tight')





