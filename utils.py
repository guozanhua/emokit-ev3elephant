__author__ = 'igor'

import pandas as pd
import numpy as np
import os

def get_raw_singals() :

    files_up = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_1') ]
    files_down = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_2') ]
    files_left = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_3') ]
    files_right = [ pd.read_csv('./data/' + file) for file in os.listdir('./data/') if file.endswith('_4') ]

    csv_up = pd.concat(files_up)
    csv_down = pd.concat(files_down)
    csv_left = pd.concat(files_left)
    csv_right = pd.concat(files_right)

    columns = 'AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4'.split(' ')

    data_sets = [csv_up, csv_down, csv_left, csv_right]
    for i,data_set in enumerate(data_sets) :
        data_sets[i] = data_set[columns]

    return data_sets

