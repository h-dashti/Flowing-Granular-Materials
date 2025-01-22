import matplotlib as mpl

import matplotlib.pyplot as plt

import plot_ps, plot_chi, plot_pinf
#import plot_nu_beta_gamma

from colors import mainfacecolor, insetfacecolor
fntsize = 16

########################################################################
### this is the one we used in the paper

idir_data = '/Users/ebi/Library/CloudStorage/Dropbox/Dynamic/current_sims/Granular/Stats/fixed_N/DATA/v2021.02.10/clus_stat/'


#r'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'
#r'C:\Users\Ebi\Desktop\Sims\Granular\fixed_N\stat\clus-stat\2021.02.10\clus_stat'
#'/Users/ebi/Desktop/All/Sims/granular/v2021.02.10/clus_stat'

#mpl.rcParams['font.family'] = 'serif'

########################################################################



if True:

    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*6, 4),
                            #sharex=True, #sharey=True,
                            gridspec_kw={'wspace': 0.16},
                            #facecolor='#FCF3CF'
                            )

    #axs = axs.flat

    for ax in axs:
         ax.tick_params(direction='in', which='both')
         ax.set_facecolor(mainfacecolor)


    plot_ps.plot(idir_data, axs[0], fc=0.0108, 
                 line_style='--', fntsize = fntsize )
    plot_pinf.plot(idir_data, axs[1], fc=0.0108, 
                   line_style='--', fntsize = fntsize )
    plot_chi.plot(idir_data, axs[2], fc=0.0108, 
                  line_style='--', fntsize = fntsize )

    ## plot fc label
    i = 0
    for ax in axs:
        yfc = 0.02
        xfc = 0.275 if i < 2 else 0.435
       
        # ax.text(x=xfc, y=yfc,
        #         transform=ax.transAxes,
        #         s=r'$f_c$',
        #         fontsize=11)
        
        ax.annotate("$f_c$",
                    xy=(0.0108, 0.01), xycoords='data',
                    #va="center", ha="center",
                    xytext=(xfc, yfc), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", 
                                    connectionstyle="arc3",
                                    lw=0.5,),
                    )
        i += 1


    #####################
    # plot insets       #
    #####################
    inset_axes = []
    for ax in axs:
        axinset = ax.inset_axes([0.59, 0.485, 0.4, 0.5])
        axinset.tick_params(direction='in', which='both')
        inset_axes.append(axinset)
        axinset.set_facecolor(insetfacecolor) ##

    titles = ['A', 'B', 'C', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    for ax, title in zip(axs, titles):
         ax.set_title(title,
                      fontweight="bold", x=-0.11, y=0.9, 
                      fontsize=16)
         
         ax.legend(loc='center left', frameon=False, 
                   fontsize='small',
                   #bbox_to_anchor = (0, 0.4)
                   )

    plot_ps.plot(idir_data, inset_axes[0], fc=0.0,
                 rescale=True, show_legend=False, show_info=False,
                 marker_size=2)

    plot_pinf.plot(idir_data, inset_axes[1], fc=0.0,
                 rescale=True, show_legend=False, show_info=False,
                 marker_size=2)

    plot_chi.plot(idir_data, inset_axes[2], fc=0.0,
                 rescale=True, show_legend=False, show_info=False,
                 marker_size=2)





    plt.savefig('collapse-one-row.pdf',
                pad_inches=0.02, bbox_inches='tight')

##



