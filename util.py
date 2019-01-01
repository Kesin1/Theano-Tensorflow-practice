# utility package for logisitc_regression, ann, momentum and RMSprop with theano

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


def get_clouds():
    Nclass = 5000
    D = 2

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    return X, Y


def get_normalized_data():
    '''
    returns X, Y normalized,
    converts entries to np.float32

    '''
    print("Reading in and transforming data...")
    data0 = pd.read_csv("gestures_muscle/0.csv").values
    data1 = pd.read_csv("gestures_muscle/1.csv").values
    data2 = pd.read_csv("gestures_muscle/2.csv").values
    data3 = pd.read_csv("gestures_muscle/3.csv").values
    data = np.concatenate((data0, data1, data2, data3), axis=0)

    X, Y = data[:, :-1], data[:, -1]
    X = (X - np.mean(X, axis=1, keepdims=True)) / \
        np.std(X, axis=1, keepdims=True)
    # X - mean/ std_der

    X, Y = shuffle(X, Y)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return X, Y


def get_data_facial(balance_ones=True):
    '''
    returns X, Y, values between 0 and 1,
    if balance_ones=True, repeats X[Y==1] 9 times
    converts entries to np.float32
    '''
    print("Reading in and transforming data...")
    data = pd.read_csv("../gestures_muscle/fer2013.csv").values
    data_Y = data[:, 0]
    data_X = data[:, 1]

    X = []
    for i in xrange(data_X.shape[0]):
        temp = data_X[i].split()
        temp = [float(j) for j in temp]
        X.append(temp)

    X = np.array(X).astype(np.float)
    Y = np.array(data_Y).astype(np.float)

    if balance_ones:
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack((X0, X1))
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X/255.0, Y


def y2indicator(Y):
    '''
    return indicator matrix of type np.int32
    of vector Y which contains classes of N samples
    '''
    Y = Y.astype(np.int32)
    N = len(Y)
    K = len(set(Y))
    Y_ind = np.zeros((N, K))
    for i in xrange(N):
        Y_ind[i, Y[i]] = 1
    return Y_ind


def initialize_weight(M1, M2):
    ''' takes in '''
    # weights are normalized to mean 1
    W1 = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b1 = np.zeros(M2)
    return W1, b1


def error_rate(predicitons, Y):
    '''
    Y in original form (vector with multiple classes)
    predicitons after argmax of lastlayer output
    '''
    return np.mean(predicitons != Y)


def relu(Z):
    return Z*(Z > 0)


def softmax(A, W, b):
    exp_Z = np.exp(A.dot(W) + b)
    return exp_Z/(np.sum(exp_Z, axis=1, keepdims=True))


def main():
    get_normalized_data()


if __name__ == "__main__":
    main()
