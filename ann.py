# implemenent ANN with theano
# for multiclass problem of recognizing hand gesture by muscular acitivity

# theano implemenented functions that we use here are softmax, sigmoid, relu

import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from util import error_rate, y2indicator, initialize_weight,\
    get_clouds, get_normalized_data
from sklearn.utils import shuffle


class ANN(object):
    __doc__ = """Neural Network Class
    implemenented with relu; M variable; 1 hidden layer"""

    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, lr=1e-4, reg=1e-2, max_iter=20, batchsz=0,
            train_sz=0.9, prind_period=10, show_fig=False):
        N, D = X.shape
        K = len(set(Y))

        N_train = int(train_sz * N)
        X_train, Y_train = X[:N_train, :], Y[:N_train]
        Y_train_ind = y2indicator(Y_train)

        X_test, Y_test = X[N_train:, :], Y[N_train:]
        Y_test_ind = y2indicator(Y_test)

        W1_init, b1_init = initialize_weight((D, self.M))
        W2_init, b2_init = initialize_weight((self.M, K))

        X_th = T.matrix('X_th')  # will take in X values (train or test)
        T_th = T.matrix('T_th')  # will take in Y values

        W1 = theano.shared(W1_init, 'W1')
        b1 = theano.shared(b1_init, 'b1')
        W2 = theano.shared(W2_init, 'W2')
        b2 = theano.shared(b2_init, 'b2')

        pY = self.forward(X_th, W1, b1, W2, b2)

        cost = -(T_th * T.log(pY)).sum() + reg*((W1*W1).sum() +
                                                (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
        predicitons = self.predict(pY)
        W1_updated = W1 - lr*T.grad(cost, W1)
        b1_updated = b1 - lr*T.grad(cost, b1)
        W2_updated = W2 - lr*T.grad(cost, W2)
        b2_updated = b2 - lr*T.grad(cost, b2)

        train_op = theano.function(inputs=[X_th, T_th], updates=[
                                   (W1, W1_updated), (b1, b1_updated), (W2, W2_updated), (b2, b2_updated)])
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

    def forward(self, X_th, W1, b1, W2, b2):
        Z = T.nnet.relu(X_th.dot(W1) + b1)
        return T.nnet.softmax(Z.dot(W2) + b2)

    def predict(self, pY):
        return T.argmax(pY, axis=1)


def main():
    # basic structure:
    # initialize data
    # instantinate class
    # do class operations with the data (fit and print score)

    X, Y = get_normalized_data()
    X, Y = shuffle(X, Y)
    M = 300
    model = ANN(M)
    # choose batch size before passing to fit

    model.fit(X, Y, lr=0.0004, reg=0.01,
              max_iter=200, batchsz=500, show_fig=True)


if __name__ == '__main__':
    main()
