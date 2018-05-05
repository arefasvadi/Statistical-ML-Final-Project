import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl
import csv

# some setup variables

# general config variables
np.set_printoptions(precision = 5)

# whether to process all data or a portion
use_all_data = False

# number of rows to be used if use_all_data is set to True
rows_to_use = 2000

# division for training and test set
training_ratio = 0.8


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
    if not use_all_data:
        hkdata = hkdata[:rows_to_use,:]

    return field_names, hkdata

# compute frequencies associated with each column
def get_frequency_meta(data, num_features):

    feature_values = []
    feature_counts = []

    for i in range(num_features):
        unique_elements, count_elements = np.unique(data[:,i],return_counts=True)
        feature_values.append(np.array(unique_elements))
        feature_counts.append(np.array(count_elements))

    return feature_values, feature_counts

# we need to compute the sufficient statistics of each field
def compute_sufficient_stats(data, num_rows, num_features, f_vals, f_counts):

    needed_cols = 0
    for i in range(num_features):
        needed_cols = needed_cols + int(f_vals[i].shape[0])

    # print(needed_cols)
    suff_array = np.zeros((num_rows,needed_cols))
    for i in range(num_rows):
        row_data = np.zeros(needed_cols)
        col_ind = 0;
        for j in range(num_features):
            ind = np.where(f_vals[j] == data[i,j])
            bit_vect = np.zeros(int(f_vals[j].shape[0]))
            bit_vect[ind] = 1
            row_data[col_ind:col_ind + int(f_vals[j].shape[0])] = bit_vect
            col_ind = col_ind + int(f_vals[j].shape[0])
            # print(ind)
            # print(int(f_vals[j][ind][0]))
            # break
        suff_array[i,:] = row_data[:];
        # break

    return suff_array

# compute M and D matrices as mentioned in paper 
def get_M_D(sufficient_statstistics):
    
    return

field_names, hkdata = read_input_data("../dataset/hk.numeric.csv")
# print(hkdata)
# print(hkdata.shape)
num_features = int(hkdata.shape[1])
separator_row = int(training_ratio * hkdata.shape[0])
train_hkdata = hkdata[:separator_row, :]
test_hkdata = hkdata[separator_row:, :]

feature_values, feature_counts = get_frequency_meta(train_hkdata,num_features)
# print(feature_values[1].shape)
# print(feature_counts[1])
suff_array = compute_sufficient_stats(train_hkdata, separator_row, num_features, feature_values, feature_counts)
# print(suff_array.shape)

