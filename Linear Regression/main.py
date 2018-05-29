# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:02:54 2017

@author: smruti venkatesh
"""
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#Reading data
letor_input_data = np.genfromtxt('../proj2/Querylevelnorm_X.csv', delimiter=',') ;
letor_output_data= np.genfromtxt( '../proj2/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1]);
syn_input_data = np.loadtxt('../proj2/input.csv', delimiter=',') ;
syn_output_data = np.loadtxt( '../proj2/output.csv', delimiter=',').reshape([-1, 1]) ;

#Computing design matrix
def compute_design_matrix(X, centers, spreads): 
    # use broadcast 
    basis_func_outputs = np.exp( 
                                    np.sum( 
                                            np.matmul(X - centers, spreads) *(X - centers),
                                            axis=2 
                                          )/(-2)
                               ).T # insert ones to the 1st col 
    return np.insert(basis_func_outputs, 0, 1, axis=1)
#Computing closed form solution
def closed_form_sol(L2_lambda, design_matrix, output_data): 
    return np.linalg.solve( 
                            L2_lambda *np.identity(design_matrix.shape[1]) + 
                            np.matmul(design_matrix.T, design_matrix), 
                            np.matmul(design_matrix.T, output_data) 
                          ).flatten()
    
#computing gradient descent
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data):
    weights = np.zeros([1, 4]) ;
    for epoch in range(num_epochs): 
        for i in range(N // minibatch_size): 
            lower_bound= i* minibatch_size 
            upper_bound= min((i+1)*minibatch_size, N) 
            Phi = design_matrix[lower_bound: upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul( 
                                (np.matmul(Phi, weights.T)-t).T, 
                                Phi 
                           )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights -learning_rate *E 
        print (np.linalg.norm(E)) 
    return weights.flatten()

#shuffle the data
letor_input_data, letor_outputy_data = shuffle(letor_input_data, letor_output_data, random_state=0)
N, D = letor_input_data.shape 
Ns, Ds = syn_input_data.shape
#split the data for letor
x_train, x_test = train_test_split(letor_input_data, test_size=0.2)
x_test, x_val = train_test_split(x_test, test_size=0.5)
y_train, y_test = train_test_split(letor_output_data, test_size=0.2)
y_test, y_val = train_test_split(y_test, test_size=0.5)

#split data for synthetic data
syni_train, syni_test = train_test_split(syn_input_data, test_size=0.2)
syni_test, syni_val = train_test_split(syni_test, test_size=0.5)
syno_train, syno_test = train_test_split(syn_output_data, test_size=0.2)
syno_test, syno_val = train_test_split(syno_test, test_size=0.5)
 
#finding centers and spreads
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train);
labels = kmeans.labels_
centers = kmeans.cluster_centers_
centers = centers[:, np.newaxis, :] 
spreads = np.zeros((3,D,D))

for i in range(3):
    ds = x_train[np.where(labels==i)];
    cov = pd.DataFrame(ds).cov();
    cov = np.linalg.pinv(cov)
    spreads[i][:][:]=cov
#Find the design matrix for train,validation and test set
design_matrix_train =compute_design_matrix(x_train, centers, spreads)
design_matrix_val =compute_design_matrix(x_val, centers, spreads)
design_matrix_test =compute_design_matrix(x_test, centers, spreads)

#finding the closed form and gradient descent i.e wieghts
closed_form_train = pd.DataFrame(closed_form_sol(L2_lambda=0.1, design_matrix=design_matrix_train, output_data=y_train))
sgd_form_train = pd.DataFrame(SGD_sol(learning_rate=0.1, minibatch_size=N, num_epochs=100, L2_lambda=0.1, design_matrix=design_matrix_train, output_data=y_train))

#Predicting the output for validation set
#ans1 = np.matmul(design_matrix_val,closed_form_train)
ans1 = np.matmul(design_matrix_val,closed_form_train)
ans2 = np.matmul(design_matrix_val,sgd_form_train)
#predicting the output for test set
ans3 = np.matmul(design_matrix_test,closed_form_train)
ans4 = np.matmul(design_matrix_test,sgd_form_train)

sum1 = sum2 = sum3 = sum4 =0
j = 0

#Calculating the erms for validation set
for i in y_val:
    a = ans1[j][0] - y_val[j][0]
    b = ans2[j][0] - y_val[j][0]
    sum1 = sum1 + (a*a)
    sum2 = sum2 + (b*b)
    j = j+1
k = 0
#Calculating the erms for test set
for i in y_test:
    c = ans3[k][0] - y_test[k][0]
    d = ans4[k][0] - y_test[k][0]
    sum3 = sum3 + (c*c)
    sum4 = sum4 + (d*d)
error1 = (2*sum1)/N
error2 = (2*sum2)/N
error3 = (2*sum3)/N
error4 = (2*sum4)/N
Error1 = sqrt(error1);
Error2 = sqrt(error2);
Error3 = sqrt(error3);
Error4 = sqrt(error4);

print("--------------------------------------------------------")

#Finding the centres and spreads
kmeans_syn = KMeans(n_clusters=3, random_state=0).fit(syni_train);
labels_syn = kmeans_syn.labels_
centers_syn = kmeans_syn.cluster_centers_
centers_syn = centers_syn[:, np.newaxis, :] 
spreads_syn = np.zeros((3,Ds,Ds))

for i in range(3):
    ds = syni_train[np.where(labels_syn==i)];
    cov = pd.DataFrame(ds).cov();
    spreads_syn[i][:][:]=cov
    
#finding the design matrix for train test and validation set ofd synthetic data    
design_matrix_train_syn =compute_design_matrix(syni_train, centers_syn, spreads_syn)
design_matrix_val_syn =compute_design_matrix(syni_val, centers_syn, spreads_syn)
design_matrix_test_syn =compute_design_matrix(syni_test, centers_syn, spreads_syn)

#closed form and sgd of synthetic data
closed_form_train_syn = pd.DataFrame(closed_form_sol(L2_lambda=0.1, design_matrix=design_matrix_train_syn, output_data=syno_train))
sgd_form_train_syn = pd.DataFrame(SGD_sol(learning_rate=0.001, minibatch_size=Ns, num_epochs=1000, L2_lambda=0.1, design_matrix=design_matrix_train_syn, output_data=syno_train))

#predicting output of validation data for synthetic data
ans1_syn = np.matmul(design_matrix_val_syn,closed_form_train_syn)
ans2_syn = np.matmul(design_matrix_val_syn,sgd_form_train_syn)
#predicting output of test data for synthetic data
ans3_syn = np.matmul(design_matrix_test_syn,closed_form_train_syn)
ans4_syn = np.matmul(design_matrix_test_syn,sgd_form_train_syn)

sum1_syn = sum2_syn = sum3_syn = sum4_syn =0
j = 0

#Calculating the erms value of validation set of synthetic data
for i in syno_val:
    a_syn = ans1_syn[j][0] - syno_val[j][0]
    b_syn = ans2_syn[j][0] - syno_val[j][0]
    sum1_syn = sum1_syn + (a_syn*a_syn)
    sum2_syn = sum2_syn + (b_syn*b_syn)
    j = j+1
k = 0

#calculating the erms value of test set of synthetic data
for i in syno_test:
    c_syn = ans3_syn[k][0] - syno_test[k][0]
    d_syn = ans4_syn[k][0] - syno_test[k][0]
    sum3_syn = sum3_syn + (a*a)
    sum4_syn = sum4_syn + (b_syn*b_syn)
error1_syn = (2*sum1_syn)/Ns
error2_syn = (2*sum2_syn)/Ns
error3_syn = (2*sum3_syn)/Ns
error4_syn = (2*sum4_syn)/Ns
Error1_syn = sqrt(error1_syn);
Error2_syn = sqrt(error2_syn);
Error3_syn = sqrt(error3_syn);
Error4_syn = sqrt(error4_syn);

#Final results for synthetic data
print("the validation error for closed form for synthetic data is")
print(Error1_syn)
print("the validation error for sgd for synthetic data is")
print(Error2_syn)
print("the testing error for closed form for synthetic data is")
print(Error3_syn)
print("the testing error for sgd for synthetic data is")
print(Error4_syn)
print("The closed form for synthetic data is")
print(closed_form_train_syn)
print("The sgd_sol for synthetic data is")
print(sgd_form_train_syn)

print("------------------------------------------------------------")

#Final output
print("the validation error for letor dataset for closed form is")
print(Error1)
print("the validation error for letor dataset for sgd is")
print(Error2)
print("the testing error for letor dataset for closed form is")
print(Error3)
print("the testing error for letor dataset for sgd is")
print(Error4)
print("The closed form is")
print(closed_form_train)
print("The sgd_sol is")
print(sgd_form_train)