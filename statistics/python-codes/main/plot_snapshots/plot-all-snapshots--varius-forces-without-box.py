### THIS IS FOR PAPER PLOT ####
#####


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sys, glob, os
from itertools import product

import matplotlib.patches as patches

sys.path.insert(0, 'aux-codes')
import plot_one_snapshot

sys.path.insert(0, '../../modules')
from Regex import get_value

#mpl.rcParams['font.family'] = 'serif'
snapshot_facecolor = '#FEF9E7' ##  FCF3CF  FEF9E7
#====================================================================
N_arr = [2048]
phi_arr = [0.86]
gammadot_arr = [1e-6, ]  # 1e-6 #1e-2, 1e-3, 1e-4, 5e-5,  1e-5, 1e-6
                #2e-5, 1e-5, 5e-6, 2e-6, 1e-6

isample = 1  # the sample should be exported

idir_1st = r'DATA/snapshots_data'
#====================================================================
def get_path(idir, N, phi, gammadot):
    return os.path.join(idir, 'N_{}'.format(N),
                        'phi_{}'.format(phi),
                        'gammadot_{}'.format(gammadot))
#====================================================================
def get_desired_force_folder(idir) :
    force_dirs = glob.glob(os.path.join(idir, 'f=*'))
    f_arr = [ get_value(fdir, 'f', float) for fdir in force_dirs]

    sort_indx = sorted(range(len(f_arr)), key=lambda k: f_arr[k])

    return [force_dirs[i] for i in sort_indx], \
                [f_arr[i] for i in sort_indx]


#====================================================================
def draw_box(fig, ax, ):
    fancybox = patches.FancyBboxPatch(
            (-0.025, -0.025), 1.05, 1.05,
            linewidth=1.5,
            facecolor=snapshot_facecolor,
            edgecolor='gray',
            #boxstyle="round, pad=0.05", #, rounding_size=0.1
            boxstyle=patches.BoxStyle("round", pad=0.05),
            zorder=-1,
            transform=ax.transAxes,
            #figure=fig
            )

    fig.patches.extend([fancybox])

#====================================================================



gammadot_arr.sort(reverse=True)

indx_fig = 0
for N, phi, gammadot in product(N_arr, phi_arr, gammadot_arr):
    
    plt.figure(indx_fig)
    indx_fig += 1


    print ('# N={}, phi={}, gammdot={}'.format(N, phi, gammadot))


    idir_2nd = get_path(idir_1st, N, phi, gammadot)

    if not os.path.exists(idir_2nd):
        print ('--> No dir:', idir_2nd)
        continue


    xlabels = ['$f_t < f_c$', '$f_t = f_c$', '$f_t > f_c$']
    titles = ['A', 'B', 'C']

    f_dirs, f_arr = get_desired_force_folder(idir_2nd)
    nrow, ncol = 1, len(f_arr)

    fig, axes = plt.subplots(
        nrows=nrow, ncols=ncol,
        figsize=(11*ncol, 11*nrow),
        gridspec_kw={'wspace': 0.1, }, #'hspace': 0
        #facecolor=snapshot_facecolor,
        #edgecolor='gray',
        )

    #plt.subplots_adjust(wspace=0, hspace=0)




    i_force = 0
    for f, fdir, ax in zip(f_arr, f_dirs, axes):

        print (' # f={}'.format(f), end=' ')


        ax.set_facecolor('black') # 'black'  '##FEF9E7' 'FCF3CF' 'FEFCF1'
     
        plot_one_snapshot.plot(ax, idir_2nd, f, indx_config=isample)


        ax.set_aspect('equal')
        #ax.set_axis_off()
        ax.margins(-0.04)
        ax.set_xlabel(xlabels[i_force], fontsize=30)
        ax.set_title(titles[i_force],
                     x=-0.03, y=0.95,
                     #x=-0.01,
                     fontweight="bold", fontsize=32)

        i_force += 1

        #patch = patches.Circle((0, 0), radius=10, transform=ax.transAxes)
        #ax.set_clip_path(patch)

    # end for f
    
    print ('  indx_sample={}'.format(isample))
    plt.savefig('snapshots_{}_{}_{}_{}.pdf'.\
                format(N, phi, gammadot, isample),
                pad_inches=0.05,
                bbox_inches='tight',
                #facecolor=fig.get_facecolor(),
                #edgecolor=fig.get_edgecolor(),
                )


# end for N, phi, gammadot

####





