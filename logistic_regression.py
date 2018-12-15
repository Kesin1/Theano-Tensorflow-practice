# implemenent logistic_regression with theano
# for multiclass problem of recognizing hand gesture by muscular acitivity

# it works for simple multiclass problems like get_clouds but not for muscular
# activity/ hand gestures_muscle

# theano implemenented functions that we use here are softmax, sigmoid, relu

import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from util import error_rate, y2indicator, initialize_weight_log_reg,\
    get_clouds, get_normalized_data
from sklearn.utils import shuffle


class log_reg_model(object):
    __doc__ = """Logistic Regression Class"""

    def __init__(self):
        pass

    def fit(self, X, Y, lr=1e-8, reg=1e-4, max_iter=20, batchsz=0,
            train_sz=0.8, prind_period=10, show_fig=False):
        N, D = X.shape
        K = len(set(Y))

        N_train = int(train_sz * N)
        X_train, Y_train = X[:N_train, :], Y[:N_train]
        Y_train_ind = y2indicator(Y_train)

        X_test, Y_test = X[N_train:, :], Y[N_train:]
        Y_test_ind = y2indicator(Y_test)

        W1_init, b1_init = initialize_weight_log_reg((D, K))

        X_th = T.matrix('X_th')  # will take in X values (train or test)
        T_th = T.matrix('T_th')  # will take in Y values

        W1 = theano.shared(W1_init, 'W1')
        b1 = theano.shared(b1_init, 'b1')

        pY = self.forward(X_th, W1, b1)

        cost = -(T_th * T.log(pY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum())
        predicitons = self.predict(pY)

        W1_updated = W1 - T.grad(cost, W1)
        b1_updated = b1 - T.grad(cost, b1)

        train_op = theano.function(inputs=[X_th, T_th], updates=[
                                   (W1, W1_updated), (b1, b1_updated)])
        pred_op = theano.function(
            inputs=[X_th, T_th], outputs=[cost, predicitons])

        if batchsz == 0:
            batchsz = N_train
            nbatches = 1
        nbatches = int(N_train/batchsz)

        costs = []
        for i in xrange(max_iter):
            for n in xrange(nbatches):
                X_batch, Y_batch_ind = X_train[n * batchsz: (
                    n+1)*batchsz, :], Y_train_ind[n * batchsz: (n+1)*batchsz]
                train_op(X_batch, Y_batch_ind)

                if n % prind_period == 0:
                    c, prediction = pred_op(X_test, Y_test_ind)
                    costs.append(c)
                    err = error_rate(prediction, Y_test)
                    print "Cost/predictions at %d batch %d: %.3f, %.3f" % (
                        i, n, c, err)

        if show_fig == True:
            plt.plot(costs)
            plt.show()

    def forward(self, X_th, W1, b1):
        return T.nnet.softmax(X_th.dot(W1) + b1)

    def predict(self, pY):
        return T.argmax(pY, axis=1)


def main():
    # basic structure:
    # initialize data
    # instantinate class
    # do class operations with the data (fit and print score)

    X, Y = get_clouds()
    X, Y = shuffle(X, Y)
    model = log_reg_model()
    # choose batch size before passing to fit

    model.fit(X, Y, batchsz=50, show_fig=True)


if __name__ == '__main__':
    main()
