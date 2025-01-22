import networkx as nx
import sys
import numpy as np

#==========================================

def over_boundary(pos1, pos2, lx, ly):
    return (abs(pos1[0] - pos2[0]) > lx//2) or \
           (abs(pos1[1] - pos2[1]) > ly//2)


#==========================================

def get_graph_from_scratch(data_arr, ft,
                           max_width=16, min_width=0.01,):

    Gmain= nx.Graph()
    Gcc = nx.Graph()


    data = data_arr[0]
    lx = int(np.max(data[:,1]) - np.min(data[:,1])) + 1
    ly = int(np.max(data[:,2]) - np.min(data[:,2])) + 1

    for d in data:
        Gmain.add_node(int(d[0]), pos=(d[1], d[2]))
        Gcc.add_node(int(d[0]))



    pos = nx.get_node_attributes(Gmain, "pos")

    fmax = np.max(data_arr[1][:,2])
    fmin = np.min(data_arr[1][:,2])




    for d in data_arr[1]:
        i1, i2, f = int(d[0]), int(d[1]), d[2], # d[3]

        if over_boundary(pos[i1], pos[i2], lx, ly):
            continue

        w = (max_width-min_width)/(fmax-fmin)*(f-fmin) + min_width

        Gmain.add_edge(i1, i2, weight=w)

        if f >= ft:
            Gcc.add_edge(i1, i2)

    ##

    return Gmain, Gcc, lx, ly




#==========================

if __name__ == '__main__':

    pass