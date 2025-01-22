#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np



numeric_pattern = r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

###################################################################
def find_info(line, types, keys):

    words = re.findall('='+numeric_pattern, line)
    numbers = [w[1:] for  w in words]

    #types = (int, int, int, int, int, int, float, float, int, int)
    #keys = ('L', 'W', 'ic', 'N', 'nc', 'bc','qmin', 'dq', 'nq', 'nr')
    info = {}

    for i in range(len(keys)):
        info[keys[i]] = types[i](numbers[i])


    return keys, info
###################################################################

def get_value(strings, target, dtype=None, assign_char='='):

    if np.isscalar(strings):
        strings = [strings]

    for st in strings:
        words = re.findall('{}{}{}'.\
                     format(target, assign_char, numeric_pattern), st)

        if len(words) == 0: continue

        value =  words[0][len(target)+1:]
        
        if dtype is None:
            return value

        try:
            return dtype(value)
        except:
            return None

    # end for


    return None

###################################################################

def get_value_file(fpath, target, dtype, indx_line=None):

    with open(fpath, 'r') as f:

        if indx_line == None:
            lines = f.readlines()
        else:
            for i in range(indx_line+1):
                lines = f.readline()


    return get_value(lines, target, dtype)
###################################################################


def get_numbers(st, dtype=None):

    words = re.findall(numeric_pattern, st)
    try:
        if dtype is None:
            return words
        else:
            return [dtype(w) for w in words]
    except:
        return None





###################################################################