# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief: pickle methods
"""

import pickle

def dump(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f) 
