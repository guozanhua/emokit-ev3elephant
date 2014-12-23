__author__ = 'igor'

import pandas as pd
import numpy as np
import os
import time

def get_raw_signals() :

    files_up = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_1') ]
    files_down = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_2') ]
    files_left = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_3') ]
    files_right = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_4') ]
    files_neutral = [ pd.read_csv('/data/' + file) for file in os.listdir('./data') if file.endswith('_0') ]

    csv_up = pd.concat(files_up)
    csv_down = pd.concat(files_down)
    csv_left = pd.concat(files_left)
    csv_right = pd.concat(files_right)
    csv_neutral = pd.concat(files_neutral)

    columns = 'AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4'.split(' ')

    data_sets = [csv_up, csv_down, csv_left, csv_right, csv_neutral]
    for i,data_set in enumerate(data_sets) :
        data_sets[i] = data_set[columns]

    return data_sets

# experimental
def dft(data) :
    return np.fft.fft(data)

def timed(func):
    """ Decorator for easy time measurement """

    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        tend = time.time()
        print("{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), tend - tstart))
        return result

    return timed