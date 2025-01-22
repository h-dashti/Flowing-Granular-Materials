#import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
#import numpy as np
#import sys, os
from get_data import get_snapshot_data
from get_network import get_graph_from_scratch
from get_colors import get_color_from_int
#==========================================

def plot(ax, idir, f, indx_config=0, std_tab=''):

    config_list = get_snapshot_data(idir, 100)

    config = config_list[indx_config]

    G, Gcc, lx, ly = \
        get_graph_from_scratch(config, f, max_width=10)


    pos = nx.get_node_attributes(G, "pos")
    weights = list(nx.get_edge_attributes(G,'weight').values())

    nx.draw_networkx_nodes(G, pos=pos,
            node_size=0.1, node_color='lightgray',
            with_labels=False, ax=ax,
            )




    ######################
    # now plot components
    #####################

    if False:

        cc = sorted(nx.connected_components(Gcc),
                                      key=len, reverse=True)

        print ('{}n_clusters={}'.\
               format(std_tab, len(cc)))

        min_size = 1

        #i_target = next(i for i, Gi in enumerate(cc) if len(Gi) <= min_size)
        vmin, vmax = 0, 9

        icolor = 0
        for Gi in cc:

           if len(Gi) <= min_size: break

           if icolor == 0:
               color = 'gold'
           else:
               color = get_color_from_int((icolor-1)%(vmax+1),
                                          cm.tab10, vmin, vmax)

           gsub = G.subgraph(Gi)
           weights = list(nx.get_edge_attributes(gsub,'weight').values())
           nx.draw_networkx_edges(gsub,
                                  pos=pos,
                                  edge_color=color,
                                  width=weights,
                                  ax=ax,
                                  )

           icolor += 1
        #
    #

    #
#===========================================================================