#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, '../fixed-codes')
from params import fntsize, markers
from fit import loglog_slope

######################################################################

def plotXY(plt, XY_arr, lab_arr, xlabel, ylabel, tx='', oname='', \
		   plot_leg=True,do_extra=None):

	for i in range(len(XY_arr)):

		x = XY_arr[i][:,0]
		y = XY_arr[i][:,1]

		plt.plot(x, y, label=lab_arr[i], \
			   marker=markers[i], markersize=4, fillstyle='none', 
               lw=0.5, ls='--')

	# end for i


	if xlabel != '':
		plt.xlabel(xlabel, fontsize=fntsize)
	if ylabel != '':
		plt.ylabel(ylabel, fontsize=fntsize)

	if do_extra != None:
		do_extra

	if plot_leg:
		plt.legend(loc='best', ncol=1, frameon=True)

	if txt != '':
		plt.title(txt)

	if oname != '':
		plt.savefig('{}'.format(oname),pad_inches=0.01, bbox_inches='tight')

######################################################################

