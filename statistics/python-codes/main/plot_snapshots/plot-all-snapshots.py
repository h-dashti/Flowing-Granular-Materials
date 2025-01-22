import matplotlib.pyplot as plt
import numpy as np
import sys, glob, os
from itertools import product

sys.path.insert(0, 'aux-codes')
import plot_one_snapshot

sys.path.insert(0, '../../modules')
from Regex import get_value
#====================================================================
N_arr = [2048]
phi_arr = [0.86]
gammadot_arr = [1e-6,] #1e-2, 1e-3, 1e-4, 5e-5,  1e-5, 1e-6
                #2e-5, 1e-5, 5e-6, 2e-6, 1e-6

idir_1st = r'DATA/snapshots_data'
#====================================================================
def get_path(idir, N, phi, gammadot):
    return os.path.join(idir, 'N_{}'.format(N),
                        'phi_{}'.format(phi),
                        'gammadot_{}'.format(gammadot))


#====================================================================

gammadot_arr.sort(reverse=True)
label_gammadot = ['1e-4', '1e-5', '1e-6']
#['10^{-4}', '10^{-5}', '10^{-6}']

ncol = len(gammadot_arr)
nrow = 1



for N, phi in product(N_arr, phi_arr):

    #plt.figure(1, figsize=(10*ncol, 10*nrow)) #
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                             figsize=(10*ncol, 10*nrow),
                             facecolor='black')
    if isinstance(axes, list):
        ax = axes.flatten()
    else:
        ax = [axes]


    ig = 0
    for gammadot in gammadot_arr:

        print ('# N={}, phi={}, gammdot={}'.format(N, phi, gammadot))


        idir_2nd = get_path(idir_1st, N, phi, gammadot)

        if not os.path.exists(idir_2nd):
            print ('--> No dir:', idir_2nd)
            continue

        force_dirs = glob.glob(os.path.join(idir_2nd, 'f=*'))
        fdir = force_dirs[0]
        f = get_value(fdir, 'f', float)
        print (' # f={}'.format(f), end=' ')


        plot_one_snapshot.plot(ax[ig], fdir, f, indx_config=0)

        #ax[ig].set_facecolor('black')
        ax[ig].set_axis_off()
        #ax[ig].patch.set_facecolor('black')


        #plt.axis('on')
        plt.subplots_adjust(wspace=0.02, hspace=0)
        #plt.margins(0.01)
        #plt.title(r'$\dot{\gamma}='+ '{}'.format(label_gammadot[ig])
        #          + '$',
                  #+ r'\;' + 'f_c={:.5f}'.format(f) + '$',
        #          fontsize=35)

        ig += 1
    # end for gammadot




    #fig.set_facecolor('darkblue')
    #plt.savefig('snapshots_{}_{}.pdf'.format(N, phi),
    #           pad_inches=0.05, bbox_inches='tight',
    #           facecolor=fig.get_facecolor())

# end for N, phi

####

