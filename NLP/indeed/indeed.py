import os
import sys

import math
import pandas as pd
import numpy as np

lib_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(lib_path)

from stats import BasicStat, NormalDistribution


def read_data(filename):
    if "csv" in filename.lower():
        return pd.read_table(filename, header=0, sep=sep)
    elif "xls" in filename.lower():
        return pd.read_excel(filename)
    elif "tsv" in filename.lower():
        return pd.read_table(filename, header=0, sep='\t')

s = read_data('train.tsv')
t = pd.read_csv('train.tsv', header=0, sep='\t')
print(t['tags'])
