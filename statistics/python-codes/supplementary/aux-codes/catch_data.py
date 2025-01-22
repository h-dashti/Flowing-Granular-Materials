#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from itertools import product
import os, sys

sys.path.insert(0, '../../modules')
from Regex import get_value_file



###################################################################

def get_XY(idir, bname, ix, iy, N_arr, phi_arr, gammadot_arr, 
           st_samples = 'n_legal_snapshots', 
           ic = None, 
           catch_fc=False):


    if not os.path.exists(idir):
        print ('# Therse is not such dir:', idir)
        sys.exit()


    if np.isscalar(N_arr):          N_arr = [N_arr]
    if np.isscalar(phi_arr):        phi_arr = [phi_arr]
    if np.isscalar(gammadot_arr):   gammadot_arr = [gammadot_arr]



    data_arr = []
    params_arr = []
    fc_by_chimax_arr = []

    iN = 0
    for u, v, w in product(N_arr, phi_arr, gammadot_arr):


        print ('  # N={}, phi={}, gammadot={}'.format(u,v,w), end='')

        fname = '{}_{}_{}_{}.dat'.format(bname, u, v, w)
        fpath = os.path.join(idir, \
                          'N_{}'.format(u),'phi_{}'.format(v), \
                          'gammadot_{}'.format(w), fname)



        if not os.path.exists(fpath):
            print (' --> no')
            continue
      

        data = np.loadtxt(fpath)

        if np.size(data) == 0:
            print (' --> bad file:', os.path.basename(fpath))
            continue


        L = get_value_file(fpath, 'L', float, indx_line=0)
        n_samples = get_value_file(fpath, st_samples, int, indx_line=0)
        
        print (' L={}, n_samples={}'.format(L, n_samples))


        params_arr.append((L,u,v,w))
        
        if catch_fc:
            fc_by_chimax = get_value_file(fpath, 'FC_ave_all_samples', 
                                          float, indx_line=0)
            fc_by_chimax_arr.append(fc_by_chimax)
            
        if ic is None:
            data_arr.append(data[:,[ix,iy]])
        else:
            data_arr.append(data[:,[ix,iy,ic]])
        

        iN += 1

    # END FOR N


    if catch_fc:
        return data_arr, params_arr, fc_by_chimax_arr
    else:
        return data_arr, params_arr


###################################################################
def get_index_column(ch, st, path=None):

    if path is None:
        try:
            return st.lstrip('#').split().index(ch)
        except ValueError:
            return None
    else:
        with open(path, 'r') as f:
            for line in f:
                try:
                    return line.lstrip('#').split().index(ch) 
                except ValueError:
                    continue
                
        return None
        
###################################################################


def get_indx_keys_params(params_arr):

    items = {}
    #keys =[]

    i = 0
    for p in params_arr:
        #st = '{}_{}_{}_{}'.format(*p)
        #keys.append(st)

        items[(p[1],p[2],p[3])] = (i,p[0])

        i += 1

    return items#, keys

###################################################################
def get_dataHist_rowlines(idir, bname, ix, N_arr, phi_arr, gammadot_arr,
                          st_samples = 'n_legal_snapshots', 
                          ):

    if not os.path.exists(idir):
        print ('# Therse is not such dir:', idir)
        sys.exit()


    if np.isscalar(N_arr):  N_arr = [N_arr]
    if np.isscalar(phi_arr):        phi_arr = [phi_arr]
    if np.isscalar(gammadot_arr):   gammadot_arr = [gammadot_arr]



    data_arr = []
    params_arr = []


    for u, v, w in product(N_arr, phi_arr, gammadot_arr):


        print ('  # N={}, phi={}, gammadot={}'.format(u,v,w), end='')

        fname = '{}_{}_{}_{}.dat'.format(bname, u, v, w)
        fpath = os.path.join(idir, \
                          'N_{}'.format(u),'phi_{}'.format(v), \
                          'gammadot_{}'.format(w), fname)



        if not os.path.exists(fpath):
            print (' --> no')
            continue
        else:
            print (' -->> yes', end='\t')


        with open (fpath, 'r') as f: lines=f.readlines()


        L = get_value_file(fpath, 'L', float, indx_line=0)
        n_samples = get_value_file(fpath, st_samples, int, indx_line=0)

        print ('L={}, n_samples={}'.format(L, n_samples))

        lines = [line for line in lines if line[0] != '#' ]

        line = lines[ix].strip()
        Y = np.array([int(s) for s in line.split() ], dtype=int)
        X = np.arange(np.size(Y))

        params_arr.append((L,u,v,w))
        data_arr.append(np.column_stack((X,Y)))


    ###

    return np.array(data_arr), np.array(params_arr)

###################################################################


if __name__ == '__main__':

    pass

