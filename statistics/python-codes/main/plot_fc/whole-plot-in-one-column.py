import matplotlib as mpl
import sys

import matplotlib.pyplot as plt

import plot_allf_ave_by_rheology
import plot_fbar_by_rheology
import plot_allf_var_by_rheology
import plot_fc_by_chimax
import plot_fc_by_Ps
import plot_sigmaxy_by_rheology

sys.path.insert(0, '../aux-codes')
from colors import mainfacecolor

########################################################################


#mpl.rcParams['font.family'] = 'serif'

fig, axs = plt.subplots(2,
                        figsize=(6,4*1.2),
                        sharex=True, #sharey=True,
                        gridspec_kw={'hspace': 0}
                        )


for ax in axs:
     ax.tick_params(direction='in', which='both',
                    right=True)
     ax.set_facecolor(mainfacecolor)
     ax.set_xscale('log')

axs[0].set_ylim(0.00675, 0.0125)     
axs[1].set_ylim(0.010, 0.0175)
     
#plot_sigmaxy_by_rheology.plot(axs[0])
plot_fbar_by_rheology.plot(axs[0], sub_indx_fit=1)

plot_fc_by_Ps.plot(axs[1], sub_indx_fit=2,
                   plt_xlabel=True)
#plot_fc_by_Ps.plot(axs[1], plt_xlabel=True)


titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.14, y=0.825, fontsize=16)
     
for ax in axs:
    # Hide x labels and tick labels for all but bottom plot.
    ax.label_outer()


plt.savefig('Fig2.pdf',
            pad_inches=0.02, bbox_inches='tight')





