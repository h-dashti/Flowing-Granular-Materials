#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

#########################################################################

numeric_pattern = r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

ffmt = 'pdf'
fntsize = 16
markers = ('o', '^', 'd', 'p', '<' ,'s', '*', '>', 'h', 'v', '+','h', 'x')
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
