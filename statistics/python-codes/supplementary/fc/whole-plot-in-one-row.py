import matplotlib as mpl
import sys

import matplotlib.pyplot as plt

import plot_allf_ave_by_rheology
import plot_allf_var_by_rheology
import plot_fc_by_Ps

sys.path.insert(0, '../aux-codes')
from colors import mainfacecolor
from params import fntsize

########################################################################


#mpl.rcParams['font.family'] = 'serif'


fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=(6*3,4),
                        gridspec_kw={'wspace': 0.225}
                        )


for ax in axs:
     ax.tick_params(direction='in', which='both',
                    right=True)
     ax.set_facecolor(mainfacecolor)
     ax.set_xscale('log')

#axs[0].set_ylim(0.00675, 0.0125)     
#axs[1].set_ylim(0.010, 0.0175)
     

plot_allf_ave_by_rheology.plot(axs[0], sub_indx_fit=1,
                               plt_xlabel=True)

plot_allf_var_by_rheology.plot(axs[1], sub_indx_fit=2,
                               plt_xlabel=True)

plot_fc_by_Ps.plot(axs[2], sub_indx_fit=3, plt_xlabel=True)


titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.145, y=0.9, 
                  fontsize=fntsize)
     


plt.savefig('fc.pdf',
            pad_inches=0.02, bbox_inches='tight')





