# implemenent logistic_regression
# for multiclass problem of recognizing hand gesture by muscular acitivity

# theano implemenented functions that we use here are softmax, sigmoid, relu

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import initialize_weight, get_clouds, y2indicator, get_normalized_data, get_data_facial
from sklearn.utils import shuffle


class hiddenlayer(object):
    def __init__(self, M1, M2, fun=tf.nn.relu, last_layer=False):
        self.M1 = M1
        self.M2 = M2
        self.fun = fun
        self.last_layer = last_layer
        W = np.random.randn(M1, M2) / np.sqrt(M1)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        if self.last_layer:
            return tf.matmul(X, self.W) + self.b
        else:
            return self.fun(tf.matmul(X, self.W) + self.b)


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, Y, lr=1e-3, reg=1-4, train_perc=0.9, eps=1e-9, batchsz=50, epochs=20, print_per=20, show_fig=False):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        X, Y = shuffle(X, Y)

        N, D = X.shape
        K = len(set(Y))
        Ntrain = int(train_perc * N)
        Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
        Ytrain_ind = y2indicator(Ytrain)
        Xvalid, Yvalid = X[Ntrain:], Y[Ntrain:]
        Yvalid_ind = y2indicator(Yvalid)

        self.hiddenlayers = []
        # self.hiddenlayers.append(hiddenlayer(D, 100))
        # self.hiddenlayers.append(hiddenlayer(100, 50))
        self.hiddenlayers.append(hiddenlayer(D, K, last_layer=True))

        self.params = []
        for layer in self.hiddenlayers:
            self.params += layer.params

        tfX = tf.placeholder(tf.float32, name='tfX')
        tfT = tf.placeholder(tf.int32, name='tfT')
        Yish = self.forward(tfX)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=Yish,
            labels=tfT))

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        predict_op = tf.argmax(tf.nn.softmax(Yish), axis=1)

        costs = []
        init = tf.global_variables_initializer()
        if batchsz == 0:
            nbatches = Ntrain
        nbatches = int(Ntrain / batchsz)

        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
                for j in range(nbatches):
                    Xbatch, Ybatch = Xtrain[j*batchsz:(j+1) *
                                            batchsz], Ytrain_ind[j*batchsz:(j+1)*batchsz]
                    session.run([train_op, cost], feed_dict={
                        tfX: Xbatch, tfT: Ybatch})
                    if j % print_per == 0:
                        test_cost = session.run(
                            cost, feed_dict={tfX: Xvalid, tfT: Yvalid_ind})
                        prediction = session.run(
                            predict_op, feed_dict={tfX: Xvalid})
                        error = np.mean(prediction != Yvalid)
                        print "Iteration i = %d, batchsz = %d, cost = %.3f, error = %.3f" % (
                            i, j, test_cost, error)

                        costs.append(test_cost)

        if show_fig == True:
            plt.plot(costs)
            plt.show()

    def forward(self, tfX):
        tfZ = tfX
        for layer in self.hiddenlayers:
            tfZ = layer.forward(tfZ)
        return tfZ

    def predict(self):
        pass

    def score(self):
        pass


if __name__ == "__main__":
    X, Y = get_normalized_data()
    model = LogisticRegression()
    model.fit(X, Y, epochs=3000, show_fig=True)
