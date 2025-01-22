import matplotlib as mpl

import matplotlib.pyplot as plt

import plot_exponents

########################################################################

mpl.rcParams['font.family'] = 'serif'
########################################################################

nrows, ncols = 2, 1

fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(6, 6),
                        sharex='col',
                        #sharex=True, #sharey=True,
                        gridspec_kw={'hspace': 0}
                        )

axs = axs.flat

for ax in axs:
     ax.tick_params(direction='in', which='both')

plot_exponents.plot_betaDIVnu_df(axs[0], plt_xlabel=False)
plot_exponents.plot_gammaDIVnu_df(axs[1], plt_xlabel=True)


titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.11, y=0.9, fontsize=16)


for ax in axs:
     pass
     # Hide x labels and tick labels for all but bottom plot.
     #ax.label_outer()



plt.savefig('hyper-relation.pdf',
            pad_inches=0.015, bbox_inches='tight')





