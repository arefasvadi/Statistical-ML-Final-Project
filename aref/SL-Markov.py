from __future__ import division
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl
import csv
from scipy import io
# some setup variables

# general config variables
# np.set_printoptions(precision = 5)

#lambda values
lamb = 1e-03

# whether to process all data or a portion
use_all_data = True

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
def get_M_D(sufficient_statstistics, num_features, f_vals):

    needed_cols = 0
    for i in range(num_features):
        needed_cols = needed_cols + int(f_vals[i].shape[0])


    num_rows = int(sufficient_statstistics.shape[0])
    M = np.concatenate(((np.ones([num_rows, 1])), sufficient_statstistics), axis=1)
    # print (M[1:10,0])
    M = np.transpose(np.matrix(M))*np.matrix(M)/num_rows
    # print (M.shape)
    l0 = 0
    l1 = np.ones(needed_cols)
    D = np.hstack((l0,l1))
    D = np.diag(D)

    return M, D

def extract_block(Z, row_ind, col_ind, m, dim):
    new_row = 0
    Rdim = 0;
    for i in range(int(m.shape[1])):
        if row_ind < m[0,i]:
            new_row += dim[0,i]*row_ind
            Rdim = dim[0,i]
            break
        row_ind -= m[0,i]
        new_row += dim[0,i]*m[0,i]

    new_col = 0
    Cdim = 0
    for i in range(int(m.shape[1])):
        if col_ind < m[0,i]:
            new_col += dim[0,i]*col_ind
            Cdim = dim[0,i]
            break
        col_ind -= m[0,i]
        new_col += dim[0,i]*m[0,i]

    # print(new_row)
    # print(Rdim)
    # print(new_col)
    # print(Cdim)
    return Z[int(new_row):int(new_row+Rdim), int(new_col):int(new_col+Cdim)]

def Z_update(theta, u, eta, f_vals, num_features):
    row, col = theta.shape
    needed_cols = 0
    value_counts = np.zeros([1, num_features])
    for i in range(num_features):
        needed_cols = needed_cols + int(f_vals[i].shape[0])
        value_counts[0,i] = int(f_vals[i].shape[0])

    # dist_num = needed_cols
    dist_num = num_features
    # m = np.array([1, m1, m2, m3, m4])
    m = np.ones([1,num_features+1])
    # print(m)
    # dim = np.array([1, dim1, dim2, dim3, dim4])
    dim = np.concatenate((np.ones([1,1]), value_counts), axis=1)
    # print(dim)

    temp_matrix = theta + u
    
    B = [ [] for i in range(dist_num+1)]
   
    for i in range(dist_num + 1):
        for j in range(dist_num + 1):
            if i == 0 or j == 0 or i == j :
                B[i].append(extract_block(temp_matrix, i, j, m, dim)) 
            else:
                gamma = la.norm(extract_block(temp_matrix, i, j, m, dim), 'fro')
                if gamma > eta[i][j]:
                    B[i].append((1 - eta[i][j]/gamma)*extract_block(temp_matrix, i, j, m, dim))
                else:
                    B[i].append(np.zeros(extract_block(temp_matrix, i, j, m, dim).shape))

    Z_new = np.bmat(B)
    Z_new = (Z_new + np.transpose(Z_new))/2

    return Z_new

def ADMM(theta, Z, U, A, K, lamb, W, num_sam, f_vals, num_features):
    # k stands for number of iterations
    # lamb stands for the lambda, which is lasso parameter 
    # Dimension of theta, U, Z and A is (d+1) x (d+1)

    new_theta = theta
    prev_Z = Z
    
    new_U = U
    n = num_sam
    
    mat_dim = theta.shape[0]
    
    rho = 20 # define rho
    # print("n is: "+str(n))
    eta1 = rho/n # define eta for theta update
    # print("eta1 is: "+str(eta1))
    eta2 = (lamb*W)/rho; # define eta2 for Z update
    
    epsilon_abs = 1e-09
    
    epsilon_rel = 1e-08
    
    # Initialize R_k and S_k
    R_k = []
    S_k = []
    epsilon_pri = []
    epsilon_dual = []
    
    for k in range(K):
        print("Starting at iteration "+str(k+1))
        temp = np.multiply((Z-U),eta1) - A;
        temp = (temp + np.transpose(temp))/2
        lambmat, Q = la.eig(temp)
        # lambmat is eigenvalue
        # v is eigenvector 
        
        
        # Update theta
        # print("eta1 is "+str(eta1))
        new_theta = (1/(2*eta1))*Q*np.matrix((np.diag(lambmat) + np.sqrt(np.diag(np.power(lambmat,2))+4*eta1*np.identity(len(lambmat)))))*np.transpose(Q)
        
        theta = new_theta # re-assign the theta (correpsond to "theta_k")
        
        # Update Z
        new_Z = Z_update(new_theta, new_U, eta2, f_vals, num_features)
    
        # Calculate residual (stopping criteria 1)
        # calculate r^k
        R_k.append(la.norm(theta - new_Z, 'fro'))
        
        # calculate primal epsilon
        epsilon_pri.append( mat_dim * epsilon_abs  + epsilon_rel * max( la.norm(theta, 'fro'), la.norm(new_Z, 'fro') ) )
        
        if R_k[k] <= epsilon_pri[k]:
            sentence = 'Primal Residual True'
        
        # Calculate dual (stopping criteria 2)
        # calculate r^k
        S_k.append(rho * la.norm(new_Z-prev_Z, 'fro'))
        
        prev_Z = new_Z
        
        # Update U
        new_U = new_U + new_theta - new_Z
        
        # calculate dual epsilon
        epsilon_dual.append( mat_dim * epsilon_abs + epsilon_rel * rho * la.norm(new_U, 'fro'))
        
        if S_k[k] <= epsilon_dual[k]:
            sentence = 'Dual Residual True'
            print(sentence)
        
        if R_k[k] <= epsilon_pri[k] and S_k[k] <= epsilon_dual[k]:
            sentence = 'primal and dual both met'
            print(sentence)
            break
    
    sentence = 'iteration is over'
    print('stopping criteria 1')
    print(R_k[k-1], epsilon_pri[k-1])
    print()
    print('stopping criteria 2')
    print(S_k[k-1], epsilon_dual[k-1])
            
    print(k)
    print(sentence)
    
    return new_theta, new_Z, new_U, R_k, S_k, epsilon_pri, epsilon_dual

def createEmatrix(theta, f_vals, num_features):
    needed_cols = 0
    value_counts = np.zeros([1, num_features])
    for i in range(num_features):
        needed_cols = needed_cols + int(f_vals[i].shape[0])
        value_counts[0,i] = int(f_vals[i].shape[0])
    dist_num = num_features
    m = np.ones([1,num_features+1])
    dim = np.concatenate((np.ones([1,1]), value_counts), axis=1)

    E = np.zeros([dist_num+1,dist_num+1])
    B = [ [] for i in range(dist_num+1)]
    # m = np.array([1, m1, m2, m3, m4])
    # dim = np.array([1, dim1, dim2, dim3, dim4])
    temp_vec = []
    for i in range(dist_num+1):
        for j in range(dist_num+1):
            B[i].append(extract_block(theta, i, j, m, dim))
            temp = la.norm(extract_block(theta, i, j, m, dim), 'fro')
            temp_vec.append(temp)
            if np.absolute(temp) >= 1e-1:
                if i != j:
                    E[i,j] = 1
            else:
                E[i,j] = 0

    return E[1:dist_num+1,1:dist_num+1]

field_names, hkdata = read_input_data("../dataset/hk.numeric.csv")
# print(hkdata)
# print(hkdata.shape)
num_features = int(hkdata.shape[1])
separator_row = int(training_ratio * hkdata.shape[0])
train_hkdata = hkdata[:separator_row, :]
test_hkdata = hkdata[separator_row:, :]

feature_values, feature_counts = get_frequency_meta(train_hkdata,num_features)
# print(feature_values[1].shape)
# print(num_features)
required_cols = 0
val_counts = np.zeros([1, num_features])
for i in range(num_features):
    required_cols = required_cols + int(feature_values[i].shape[0])
    val_counts[0,i] = int(feature_values[i].shape[0])

# print(required_cols)
suff_array = compute_sufficient_stats(train_hkdata, separator_row, num_features, feature_values, feature_counts)

# print(suff_array.shape)
M, D = get_M_D(suff_array, num_features, feature_values)
# print(M.shape)
# print(D.shape)
A = M + D
# A = np.matrix(A) + np.identity(required_cols+1)
# A = (A+np.transpose(A))/2
theta = np.ones([required_cols+1, required_cols+1])*0.1
Z = np.ones([required_cols+1,required_cols+1])*0.2
U = np.ones([required_cols+1, required_cols+1])*0.1
K = 600
Mvec = np.zeros([required_cols+1,1])
Mvec[0] = 1
i = 0;
f = 0;
while True:
    # index i+1
    # print(int(feature_values[f].shape[0]))
    for j in range(int(feature_values[f].shape[0])):
        Mvec[i + j + 1] = int(feature_values[f].shape[0])
        # print("** i+j+1 :"+str(i+j+1))

    i = i + int(feature_values[f].shape[0])
    f = f + 1
    if f >= num_features:
            break

# print(Mvec)
W = np.sqrt(Mvec * np.transpose(Mvec))
# print(W.shape)

optimal_theta, optimal_Z, optimal_U, R_k, S_k, epsilon_pri, epsilon_dual = ADMM(theta, Z, U, A, K, lamb, W, int(train_hkdata.shape[0]), feature_values, num_features)
edges = createEmatrix(optimal_theta, feature_values, num_features)
print(edges)
io.savemat('../dataset/markov-edges.mat',{'edges':edges})
