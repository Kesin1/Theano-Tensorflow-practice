# Implementation of a ANN with RMSprop and Momentum

import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data, initialize_weight, get_data_facial
from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams


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
        self.rng = RandomStreams()

    def forward(self, Z, noise_variance_on_weights=0):
        if noise_variance_on_weights == 0:
            return self.fun(Z.dot(self.W) + self.b)
        else:
            return self.fun(Z.dot(self.W + self.rng.normal(size=self.W.shape, std=noise_variance_on_weights)) + self.b)


class ANN_RMS_MOM(object):
    def __init__(self, layer_sizes, p_keep):
        self.layer_sizes = layer_sizes
        self.p_keep = p_keep

    def fit(self, X, Y, lr=1e-4, reg=0., mu=0.9, decay=0.9, eps=1e-9, batchsz=100, epochs=20, show_fig=False, train_perc=0.95, print_period=20, noise_variance_on_x=0, noise_variance_on_weights=0):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        X, Y = shuffle(X, Y)

        N, D = X.shape
        K = len(set(Y))
        Ntrain = int(N*train_perc)

        if noise_variance_on_x == 0:
            Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
            Xvalid, Yvalid = X[Ntrain:, :], Y[Ntrain:]
        else:
            Xtrain, Ytrain = X[:Ntrain, :] + \
                np.random.normal(scale=noise_variance_on_x,
                                 size=(Ntrain, D)), Y[:Ntrain]
            Xvalid, Yvalid = X[Ntrain:, :], Y[Ntrain:]

        self.rng = RandomStreams()

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

        Xth = T.matrix('X')
        Yth = T.ivector('Y')    # this is necessary to be an int vector

        pY = self.forward_train(Xth)
        # reg_cost = reg * T.mean([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(Yth.shape[0]), Yth]))  # + reg_cost
        grads = T.grad(cost, self.params)

        cache = [theano.shared(np.ones_like(p.get_value()))
                 for p in self.params]
        mom_vecs = [theano.shared(np.zeros_like(p.get_value()))
                    for p in self.params]

        new_cache = [decay * c + (1-decay)*g*g
                     for c, g in zip(cache, grads)]
        new_mom_vecs = [mu*v - lr*g/T.sqrt(new_c+eps)
                        for v, new_c, g in zip(mom_vecs, new_cache, grads)]

        updates = [
            (c, new_c) for c, new_c in zip(cache, new_cache)
        ] + [
            (v, new_v) for v, new_v in zip(mom_vecs, new_mom_vecs)
        ] + [
            (w, w + new_v) for w, new_v in zip(self.params, new_mom_vecs)
        ]

        train = theano.function(
            inputs=[Xth, Yth], updates=updates)

        pY_predict = self.forward(Xth)
        cost_test = - \
            T.mean(
                T.log(pY_predict[T.arange(Yth.shape[0]), Yth]))  # + reg_cost
        prediction = self.predict(Xth, noise_variance_on_weights)
        cost_score_op = theano.function(
            inputs=[Xth, Yth], outputs=[cost_test, prediction])

        costs = []
        if batchsz == None:
            batchsz = Ntrain
        nbatches = Ntrain/batchsz
        for i in xrange(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in xrange(nbatches):
                Xbatch, Ybatch = Xtrain[j*batchsz:(
                    j+1)*batchsz, :], Ytrain[j*batchsz: (j+1)*batchsz]

                train(Xbatch, Ybatch)

                if j % print_period == 0:
                    c, p = cost_score_op(Xvalid, Yvalid)
                    costs.append(c)
                    print "i:", i, "j:", j, "nbatches:", nbatches, "cost %.3f:" % c, "score %.3f:" % np.mean(
                        p == Yvalid)

        if show_fig == True:
            plt.plot(costs)
            plt.show()

    def forward_train(self, X):
        Z = X
        for layer, p_keep in zip(self.Hiddenlayers, self.p_keep):
            mask = self.rng.binomial(n=1, p=p_keep, size=Z.shape)
            Z = layer.forward(Z*mask)
        return Z

    def forward(self, X, noise_variance_on_weights=0):
        Z = X
        for layer, p_keep in zip(self.Hiddenlayers, self.p_keep):
            Z = layer.forward(Z*p_keep, noise_variance_on_weights)
        return Z

    def predict(self, X, noise_variance_on_weights=0):
        pY = self.forward(X, noise_variance_on_weights)
        return T.argmax(pY, axis=1)

    def score(self, X, Y):
        p = self.predict(X)
        return T.mean(p == Y)


if __name__ == "__main__":
    X, Y = get_normalized_data()

    # implmenet cross_validation later
    model = ANN_RMS_MOM([500, 300], [0.8, 0.5, 0.5])
    model.fit(X, Y, show_fig=True)
    model.fit(X, Y, noise_variance_on_x=0.01)
    model.fit(X, Y, noise_variance_on_x=0.01, noise_variance_on_weights=0.01)
