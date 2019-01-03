# Implementation of a ANN with RMSprop and Momentum

import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data, initialize_weight, get_data_facial
from sklearn.utils import shuffle


class Hiddenlayer(object):
    def __init__(self, M1, M2, hl_num, fun=T.nnet.relu):
        self.M1 = M1
        self.M2 = M2
        self.hl_num = hl_num
        self.fun = fun
        W_init, b_init = initialize_weight(M1, M2)
        self.W = theano.shared(W_init, name="W%d" % hl_num)
        self.b = theano.shared(b_init, name="b%d" % hl_num)
        self.params = [self.W, self.b]

    def forward(self, Z):
        return self.fun(Z.dot(self.W) + self.b)


class ANN_RMS_MOM(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def fit(self, X, Y, lr=1e-4, reg=0., mu=0.9, decay=0.9, eps=1e-9, batchsz=100, epochs=20, show_fig=False, print_period=20):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)

        N, D = X.shape
        K = len(set(Y))

        # initialize_layers
        self.Hiddenlayers = []
        M1 = D
        count = 0
        for M2 in self.layer_sizes:
            layer = Hiddenlayer(M1, M2, count)
            M1 = M2
            count += 1
            self.Hiddenlayers.append(layer)

        self.Hiddenlayers.append(Hiddenlayer(M1, K, count, fun=T.nnet.softmax))

        self.params = []
        for layer in self.Hiddenlayers:
            self.params += layer.params

        cache = [theano.shared(np.ones_like(p.get_value()))
                 for p in self.params]
        mom_vecs = [theano.shared(np.zeros_like(p.get_value()))
                    for p in self.params]

        Xth = T.matrix('X')
        Yth = T.ivector('Y')    # this is necessary to be an int vector

        pY = self.forward(Xth)
        reg_cost = reg * T.mean([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(Yth.shape[0]), Yth])) + reg_cost
        prediction = T.argmax(pY, axis=1)
        grads = T.grad(cost, self.params)

        updates = [
            (c, decay * c + (1-decay) * g**2) for c, g in zip(cache, grads)
        ] + [
            (w, w + mu*v - lr*g / T.sqrt(c+eps)) for w, g, c, v in zip(self.params, grads, cache, mom_vecs)
        ] + [
            (v, mu*v - lr*g/T.sqrt(c+eps)) for v, c, g in zip(mom_vecs, cache, grads)
        ]

        train = theano.function(
            inputs=[Xth, Yth], outputs=cost, updates=updates)
        self.predict_op = theano.function(inputs=[Xth], outputs=prediction)

        costs = []
        if batchsz == None:
            batchsz = N
        nbatches = N/batchsz
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            for j in xrange(nbatches):
                Xbatch, Ybatch = X[j*batchsz: (
                    j+1)*batchsz, :], Y[j*batchsz: (j+1)*batchsz]
                c = train(Xbatch, Ybatch)
                costs.append(c)
                if j % print_period == 0:
                    score = self.score(X, Y)
                    print "Iteration: %d, Train cost: %.3f, Train score: %.3f" % (
                        i, c, score)

                if np.round(score, 2) == 1.:
                    break

            if np.round(score, 2) == 1.:
                break
        if show_fig == True:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for layer in self.Hiddenlayers:
            Z = layer.forward(Z)
        return Z

    def predict(self, X):
        return self.predict_op(X)

    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)


if __name__ == "__main__":
    X, Y = get_normalized_data()
    X, Y = shuffle(X, Y)

    N = len(X)
    Ntrain = int(0.9 * N)
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    # implmenet cross_validation later
    model = ANN_RMS_MOM([500, 300])
    model.fit(Xtrain, Ytrain, show_fig=True)
    print "Test score:", model.score(Xtest, Ytest)
