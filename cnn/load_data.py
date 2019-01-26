import os
if os.getcwd() == "/home/keisn/Documents/Online Courses/Udemy courses/own_work/Tensorflow_and_Theano":
    os.chdir("cnn")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.utils import shuffle


def load_data_mnist():
    '''
    loads train and test images

    returns them as np.int32 and np.float32 and normalized
    return [num_of_samples, width, height, colors]
    '''
    test = loadmat(
        "../../Tensorflow_and_Theano/large_files/test_32x32.mat")
    train = loadmat(
        "../../Tensorflow_and_Theano/large_files/train_32x32.mat")

    Y_test = test['y'].astype(np.int32)
    Y_test = Y_test.squeeze()-1  # Indexing starting at 0
    X_test = test['X'].astype(np.float32)/255.
    X_test = np.transpose(X_test, (3, 0, 1, 2))

    Y_train = train['y'].astype(np.int32)
    Y_train = Y_train.squeeze()-1  # indexing starting at 0
    X_train = train['X'].astype(np.float32)/255.
    X_train = np.transpose(X_train, (3, 0, 1, 2))

    return X_train, Y_train, X_test, Y_test


def load_data_facial(test_perc, num_of_samples=None):
    '''
    returns X, Y, values between 0 and 1,
    if balance_ones=True, repeats X[Y==1] 9 times
    converts entries to np.float32
    '''
    data = pd.read_csv("../large_files/fer2013.csv").values
    data_Y = data[:, 0]
    data_X = data[:, 1]
    data_X, data_Y = shuffle(data_X, data_Y)
    if num_of_samples == None:
        num_of_samples = len(data_X)
    else:
        data_X, data_Y = data_X[:num_of_samples], data_Y[:num_of_samples]

    X = []
    for i in xrange(data_X.shape[0]):
        temp = data_X[i].split()
        temp = [float(j) for j in temp]
        X.append(temp)

    X = np.array(X).astype(np.float32)
    Y = np.array(data_Y).astype(np.int32)

    X0, Y0 = X[Y != 1, :], Y[Y != 1]
    X1 = X[Y == 1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack((X0, X1))
    Y = np.concatenate((Y0, [1]*len(X1)))
    X, Y = shuffle(X, Y)
    X = np.reshape(X, (X.shape[0], 48, 48, 1))
    X = X/255

    num_of_test_samples = int(num_of_samples * test_perc)
    X_train, Y_train = X[:-num_of_test_samples], Y[:-num_of_test_samples]
    X_test, Y_test = X[-num_of_test_samples:], Y[-num_of_test_samples:]

    return X_train, Y_train, X_test, Y_test
