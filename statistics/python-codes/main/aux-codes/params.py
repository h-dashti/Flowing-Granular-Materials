#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

#########################################################################

numeric_pattern = r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

ffmt = 'pdf'
fntsize = 16
fntsize_info = 16
fnstsize_expo = 14
markers = ('>', 'p', '^', 'd', 'v', 'o', '*', 'h', '<', '+','x', 's')
fillstyles = ( 'none', 'top', 'bottom', 'left', 'right', 'full')
msize1 = 4
markeredgewidth = 0.5
onecolor = '#FFBF00'



ls_fc =  (0, (5, 1)) #'--'#(0, (5,5))
ls_dilute = (0, (5, 5))
c_fc = 'gray'

#########################################################################

def get_mycolors(cmap_name='tab10', n_colors=None):

    cmap = cm.get_cmap(cmap_name, n_colors)
    
    colors= []
    
    for i in range(cmap.N):
        rgba = cmap(i)
        #hexc = matplotlib.colors.rgb2hex(rgba)
        #rgba = [round(v*255) for v in rgba[:-1]]
        #print (i, rgba, hexc)
        colors.append(rgba)
    return colors

#########################################################################
