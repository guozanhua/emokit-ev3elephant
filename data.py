__author__ = 'igor sieradzki'

from utils import *

@timed
def prepare_training_data() :

    data_sets = get_raw_signals()
    data_set = pd.concat(data_sets)

    return data_set