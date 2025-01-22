import matplotlib as mpl
import sys

import matplotlib.pyplot as plt

import plot_nu_beta_gamma

sys.path.insert(0, '../aux-codes')
from colors import mainfacecolor

########################################################################

# IT IS USED FOR PAPER


#mpl.rcParams['font.family'] = 'serif'

nrows, ncols = 1, 3

fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(ncols*6, 4),
                        #sharex=True,
                        #sharex=True, #sharey=True,
                        gridspec_kw={'wspace': 0.18}
                        )

#axs = axs.flat

for ax in axs:
     ax.tick_params(direction='in', which='both', 
                    right=True)
     ax.set_facecolor(mainfacecolor)
     #ax.grid(lw=0.2, axis='both')


axs[0].set_yticks([1, 1.2, 1.4, 1.6, 1.8])
axs[1].set_yticks([0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])
#axs[2].set_yticks([2, 2.4, 2.8, 3.2])

plot_nu_beta_gamma.plot_nu(axs[0], plt_xlabel=True)
plot_nu_beta_gamma.plot_beta(axs[1], plt_xlabel=True)
plot_nu_beta_gamma.plot_gamma(axs[2], plt_xlabel=True)



titles = ['A', 'B', 'C', 'd', 'e', 'f', 'g', 'h']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.11, y=0.9, fontsize=16)


for ax in axs:
     pass
     # Hide x labels and tick labels for all but bottom plot.
     #ax.label_outer()



plt.savefig('exponents-one-row.pdf',
            pad_inches=0.02, bbox_inches='tight')





