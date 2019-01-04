# implemenent logistic_regression
# for multiclass problem of recognizing hand gesture by muscular acitivity

# theano implemenented functions that we use here are softmax, sigmoid, relu

import numpy as np
import tensorflow as tf
from util import initialize_weight, get_clouds
from sklearn.utils import shuffle


class hiddenlayer(object):
    def __init__(self, M1, M2, fun=tf.nn.relu, last_layer=False):
        self.M1 = M1
        self.M2 = M2
        self.fun = fun
        W_init, b_init = initialize_weight(self.M1, self.M2)
        W = tf.Variable(W_init)
        b = tf.Variable(b_init)

    def forward(self, X):
        if last_layer:
            return tf.matmul(X, W) + b
        else:
            return self.fun(tf.matmul(X, W) + b)


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, Y, lr=1e-4, reg=1-4, train_perc=0.9, eps=1e-9, batchsz=100, epochs=20, print_per=20, show_fig=False):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)

        N, D = X.shape
        Y = len(set(Y))
        Ntrain = int(train_perc * N)
        Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
        Xvalid, Yvalid = X[Ntrain:], Y[Ntrain:]

        self.hiddenlayers = []
        self.hiddenlayers += hiddenlayer(D, K, last_layer=True)

        tfX = tf.placeholder(tf.float32, name='X')
        tfT = tf.placeholder(tf.int32, name='T')
        logits = forward(tfX)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tfT, logits=logits))
        predict_op = tf.math.argmax(tf.nn.softmax(logits))

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

        if batchsz == 0:
            nbatches = Ntrain
        nbatches = int(Ntrain / batchsz)
        costs = []
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(nbatches):
                    Xbatch, Ybatch = Xtrain[j*batchsz:(j+1) *
                                            batchsz], Ytrain[j*batchsz:(j+1)*batchsz]
                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
                    if j % print_per == 0:
                        test_cost = session.run(
                            cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(test_cost)
                        prediction = session.run(
                            predict_op, feed_dict={tfX: Xvalid})
                        error = np.mean(prediction != Ytest)
                        print "Iterition i = %d, batchsz = %d, cost = %.3f, error = %.3f" % (
                            i, j, test_cost, error)

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
    X, Y = get_clouds
    model = LogisticRegression()
    model.fit()
