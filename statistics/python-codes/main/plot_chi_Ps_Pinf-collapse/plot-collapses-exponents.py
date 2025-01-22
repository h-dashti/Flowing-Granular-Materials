import matplotlib as mpl

import matplotlib.pyplot as plt

import plot_ps, plot_chi, plot_pinf
import plot_nu_beta_gamma

########################################################################

idir_data = r'/home/uqedasht/Dropbox/Dynamic/current_sims/Granular/Stats/fixed_N/DATA/v2021.02.10/clus_stat/'

#idir_data = r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\clus-stat\2021.02.10\clus_stat'
#r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\clus-stat\2021.02.10\clus_stat'
#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'

#mpl.rcParams['font.family'] = 'serif'
########################################################################

nrows, ncols = 3, 3

fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(19, 12),
                        sharex='col',
                        #sharex=True, #sharey=True,
                        gridspec_kw={'hspace': 0}
                        )

axs = axs.flat

for ax in axs:
     ax.tick_params(direction='in', which='both')

plot_ps.plot(idir_data, axs[0], plt_xlabel=False, fc=0.0108)
plot_ps.plot(idir_data, axs[1], plt_xlabel=False, fc=0,
             rescale=True, show_legend=False)

# plot_nu_beta_gamma.plot_nu(axs[2], plt_xlabel=False)

# plot_pinf.plot(idir_data, axs[3], plt_xlabel=False, fc=0.0108)
# plot_pinf.plot(idir_data, axs[4], plt_xlabel=False, fc=0,
#                rescale=True, show_legend=False)

# plot_nu_beta_gamma.plot_beta(axs[5], plt_xlabel=False)


# plot_chi.plot(idir_data, axs[6], fc=0.0108)
# plot_chi.plot(idir_data, axs[7], rescale=True, fc=0, fc_txt=False,
#               show_legend=False)

# plot_nu_beta_gamma.plot_gamma(axs[8], plt_xlabel=True)

#titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
titles = ['a', 'b', 'g', 'c', 'd', 'h', 'e', 'f', 'i', 'j', 'k']
for ax, title in zip(axs, titles):
     ax.set_title(title,
                  fontweight="bold", x=-0.11, y=0.9, fontsize=16)


for ax in axs:
     pass
     # Hide x labels and tick labels for all but bottom plot.
     #ax.label_outer()



plt.savefig('collapse-exponent.pdf',
            pad_inches=0.015, bbox_inches='tight')





