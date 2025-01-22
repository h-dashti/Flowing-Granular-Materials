import matplotlib as mpl
import sys

import matplotlib.pyplot as plt

import plot_exponents

sys.path.insert(0, '../aux-codes')
from colors import mainfacecolor
#from params import get_mycolors
########################################################################

#mpl.rcParams['font.family'] = 'serif'

########################################################################



fig, ax = plt.subplots()

#ax.set_prop_cycle(color=['b', 'r'])
ax.set_xscale('log')

ax.tick_params(direction='in', which='both',
               right=True)
ax.set_ylim(0.2, 0.45)
ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4])
ax.set_facecolor(mainfacecolor)
plot_exponents.plot_eta2(ax, plt_xlabel=True)

plt.savefig('eta-relation.pdf',
            pad_inches=0.02, bbox_inches='tight')





