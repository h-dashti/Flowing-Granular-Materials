import matplotlib as mpl

import matplotlib.pyplot as plt

import plot_ps, plot_chi, plot_pinf

########################################################################


mpl.rcParams['font.family'] = 'serif'

nrows, ncols = 3, 2

fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(12, 12),
                        sharex='col',
                        #sharex=True, #sharey=True,
                        gridspec_kw={'hspace': 0}
                        )

axs = axs.flat

for ax in axs:
     ax.tick_params(direction='in')

plot_ps.plot(axs[0], plt_xlabel=False)
plot_ps.plot(axs[1], plt_xlabel=False, rescale=True, show_legend=False)




plot_chi.plot(axs[2], plt_xlabel=False)
plot_chi.plot(axs[3], plt_xlabel=False, rescale=True, show_legend=False)


plot_pinf.plot(axs[4], )
plot_pinf.plot(axs[5], rescale=True, show_legend=False)

titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.1, y=0.93, fontsize=17)


for ax in axs:
     pass
     # Hide x labels and tick labels for all but bottom plot.
     #ax.label_outer()



plt.savefig('collapse.pdf',
            pad_inches=0.015, bbox_inches='tight')





