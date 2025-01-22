import numpy as np
import os, sys
from math import log10

sys.path.insert(0, '../aux-codes')
from params import fntsize, fntsize_info, markers, \
    get_mycolors, fillstyles, msize1
from scipy.optimize import curve_fit

########################################################################
idir = 'DATA'
fname = 'N_gammadot_fc(phi=0.86)--by-Ps.dat'
fpath = os.path.join(idir, fname)
########################################################################

def get_data(fpath):
    data = np.loadtxt(fpath)

    res = {}

    for d in data:
        L, N, gammadot, fc, fc_err, delta, delta_err = d
        if N not in res:
            res[N] = {}

        res[N][gammadot] = (fc, fc_err)

    return res
########################################################################

def plot(ax,
         plot_fit=True,
         show_legend=True,
         plt_xlabel=False,
         plt_ylabel=True,
         sub_indx_fit = None,
         what_am_i_doing = 'plotting fc by Ps'
        ):
        
         


    print (what_am_i_doing)
    
    res = get_data(fpath)


    i = 0
    for N in res.keys():
        
        xy = np.array([[k, *res[N][k]] for k in res[N]])
        xy = xy[xy[:,0].argsort()]

        ax.plot(xy[:,0], xy[:,1], #xy[:, 2],
                    label=r'$N={:.0f}$'.format(N),
                    marker='o', markersize=7,
                    fillstyle = fillstyles[i],
                    ls='',)
        
        if plot_fit:
            # f_fit = lambda x, a, b, c: a + b * np.power(x, c)
            # popt, pcov = curve_fit(f_fit, xy[:,0], xy[:,1], 
            #                        bounds=([1e-10, 1e-10, 1e-3], [1, 1, 1])
            #                       )
            expo = 0.5
            f_fit = lambda x, a, b: a + b * np.sqrt(x)
            popt, pcov = curve_fit(f_fit, xy[:,0], xy[:,1])
            perr = np.sqrt(np.diag(pcov))
            print ('N={:.0f}, a={:.5g}, b={:.5g}'.\
                    format(N, *popt), 'err_a: {:.5g}, err_b={:.5g}'.\
                        format(*perr))
            #print ('N={:.0f}'.format(N), 'a:', *popt, 'err:', *perr)
        
        i += 1

    # end for N

    if plot_fit:
    
        x = np.logspace(log10(7.5e-7), log10(1.2e-3))
        y = f_fit(x, *popt)
        
        st_sub = '' if sub_indx_fit is None else '_{}'.format(sub_indx_fit)
        lable = '$' + 'a{0}+b{0}'.format(st_sub) + \
            r'\,\dot{\gamma}^{' + str(expo) + '}$'
            
        ax.plot(x, y,
                    label=lable,
                    color='k', #ax.lines[-1].get_color(),,
                    lw = 1,
                    ls = '--',
                    zorder = 4,
                    )


    if plt_xlabel:
        ax.set_xlabel(r'$\dot{\gamma}$', fontsize=fntsize)
    if plt_ylabel:
        ax.set_ylabel(r'$f_{c}$', fontsize=fntsize)

    if show_legend:
        ax.legend(loc='best', frameon=False, 
                  ncol=2, fontsize=13)


########################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    plot(ax)








