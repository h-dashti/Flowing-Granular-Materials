import numpy as np
import os, sys
#==========================================
def get_snapshot_data(idir, max_n_files=100):

    base_names = ['nodes_info', 'force_chain']

    config_list = []


    ## node info
    for i in range(max_n_files):

        fnames = ['{}_{}.dat'.format(name, i) for name in base_names]
        fpaths = [os.path.join(idir, fname) for fname in fnames]

        for path in fpaths:
            if not os.path.exists(path):
                #print('# Error: no such path:', path)
                continue


        # part node info
        try:
            data_arr = [np.loadtxt(fpath) for fpath in fpaths]
        except:
            continue

        config_list.append(data_arr)


    return config_list


#==========================================