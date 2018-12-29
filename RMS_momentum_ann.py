import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data, initialize_weight
from sklearn.utils import shuffle


class Hiddenlayer(object):
    def __init__(self, M1, M2, fun):
        self.M1 = M1
        self.M2 = M2
        self.func = fun
        W_init, b_init = initialize_weight(M1, M2)
        W = theano.shared(W_init, name="W")
        b = theano.shared(b_init, name="b")
        self.params = [self.W, self.b]

    def forward(self, Z):
        if self.fun == T.nnet.relu:
            return T.nnet.relu(Z.dot(self.W) + self.b)
        if self.fun == T.nnet.softmax:
            return T.nnet.softmax(Z.dot(self.W) + self.b)


class ANN_RMS_MOM(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def fit(self, Xtrain, Ytrain, lr=1e-4, reg=1e-2, mu=0., decay=0.99, eps=1e-8, batchsz=100, epochs=2000, show_fig=False, print_period=20):
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

        self.Hiddenlayers.append(Hiddenlayer(M1, K, count, func='softmax'))
        self.Hiddenlayers.append(layer)

        self.params = []
        for layer in self.Hiddenlayer:
            self.params += layer.params

        Xth = T.matrix('X')
        Yth = T.vector('Y')

        pY = self.forward(Xth)
        cost = -T.mean((T.log(pY[T.arange(thY.shape[0]), Yth])))
        prediction = T.argmax(pY, axis=1)
        grads = T.grad(self.params)

        updates = [
            (w, w - lr*g) for w, g in zip(self.params, grads)
        ]

        train__op = theano.function(
            input=[X, Y], outputs=cost, updates=updates)
        self.predict_op = theano.function(inputs=[X], output=predictions)

        costs = []
        if batchsz = None:
            batchsz = N
        nbatches = N/batchsz
        for i in xrange(epochs):
            for j in xrange(nbatches):
                Xbatch, Ybatch = Xtrain[j*nbatches: (
                    j+1)*nbatches, :], Ytrain[j*nbatches: (j+1)*nbatches]
                c = train(Xbatch, Ybatch)
                costs.append(c)
                if j % print_period == 0:
                    print "Train cost: %.3f, Train score: %.3f" % (
                        c, self.predict(Xtrain, Ytrain))

        if show_fig == True:
            plt.plot(c)
            plt.show

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

    Ntrain = 0.9 * N
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    # implmenet cross_validation later
    model = ann_prop_mom()
    model.fit(Xtrain, Ytrain, show_fig=True)
    print "Test score:", model.score(Xtest, Ytest)
