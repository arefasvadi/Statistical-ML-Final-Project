import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl
import csv

# first we need to compute the sufficient statistics of each field

# reads input data from @parameter file_path
def read_input_data(file_path=""):
    if not file_path:
        raise Exception('File path should not be empty')
    with open(file_path) as f:
        hkdata = list(csv.reader(f, delimiter=',' ,quotechar='"'))
    # print(hkdata[0:3])
    # remove first line since it is header
    field_names = hkdata[0]
    field_names = field_names[1:]
    hkdata = np.array(hkdata[1:], dtype=np.int)

    return field_names, hkdata


field_names, hkdata = read_input_data("../dataset/hk.numeric.csv")
# print(hkdata)
# print(hkdata.shape)
