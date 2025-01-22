import numpy as np
import sys
#==========================================
def relabel(labels):

    lmax = np.max(labels) + 1
    new_label = np.zeros(lmax, dtype=int)
    indexer = 0

    for i in range(len(labels)):

        l = labels[i]
        if new_label[l] == 0:
            indexer += 1
            new_label[l] = indexer

        labels[i] = new_label[l]
    ##
    return labels

#==========================================

def relabel_and_sort_clusters(data, min_size=0):


    # now we sort by size of clusters
    # bincounts = np.bincount(data[:,ic])
    # unic_labels = np.nonzero(bincounts)[0]
    # size_clus = bincounts[unic_labels]

    # sort_indxs = np.argsort(size_clus) #[::-1]

    # unic_labels = unic_labels[sort_indxs]
    # size_clus = size_clus[sort_indxs]
    
   
    clus_links = dict()
    label_links = dict()
    
    clusters = set()

    for d in data:
        i1, i2, c = d
        
        clusters.add(c)

        if c not in clus_links:
            clus_links[c] = []

        clus_links[c].append((i1, i2))
        label_links[(i1,i2)] = c
        label_links[(i2,i1)] = c
        
    # end for d

    ## sort the components
    
    print(clusters)
    
    clus_links = dict(sorted(clus_links.items(), key=lambda kv: len(kv[1])))
    return label_links


    ## get new components

    label_links = dict()
    #components_new = dict()
    #clus_sizes = []

    i = 0
    for c in clus_links.keys():

        if len(clus_links[c]) >= min_size:

            #clus_sizes.append(len(components[c]))

            #components_new[i] = []
            for xy in clus_links[c]:
                label_links[(xy[0], xy[1])] = i
                #components_new[i].append((xy[0], xy[1]))

            i += 1

    # now change

    return label_links #clus_sizes #, components_new



#==========================================

if __name__ == '__main__':
    data = [[0, 1, 10], [0, 3, 2], [0, 2, 2],
            [1,2, 4]]

    data = np.array(data, dtype=int)

    label_links, components = relabel_and_sort_clusters(data)

    print(label_links)
    print(components)


