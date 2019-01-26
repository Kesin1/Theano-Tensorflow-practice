import os
if os.getcwd() == "/home/keisn/Documents/Online Courses/Udemy courses/own_work/Tensorflow_and_Theano":
    os.chdir("cnn")

from hyperparams import hyperparams
from load_data import load_data_mnist, load_data_facial
from cnn_model import CNN

X_train, Y_train, X_test, Y_test = load_data_facial(0.05)
model = CNN(hyperparams["CNN"])
model.fit(X_train, Y_train, hyperparams["train"])
