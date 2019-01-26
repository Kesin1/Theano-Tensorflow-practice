import os
if os.getcwd() == "/home/keisn/Documents/Online Courses/Udemy courses/own_work/Tensorflow_and_Theano":
    os.chdir("cnn")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cnn_layers import hidden_layer, conv_layer
from sklearn.utils import shuffle


class CNN(object):
    def __init__(self, CNN_hyperparameters):
        '''
        hyperparameters needs to be a dictionary with key "CNN"
        and "train"
        key "CNN" needs to include:
        list ot out_channels for expl. [20, 50]
        list of tuples witch are the filter shapes for expl. [(5, 5),
        (5, 5)]
        list of hidden_layers_sizes for expl. [500, 300]

        key "train" needs to include:
        "learning_rate": 1e-2,
        "decay": 0.9,
        "momentum": 0.9,
        "epsilon": 1e-10,
        "batch_size": 512,
        "#validation_batch_size": 10000,
        "validation_batch_size": 512,
        "epochs": 100,
        "print_period": 10

        batch_sizes must be given before training because of
        tensorflow_implementation; therefore must be the same
        '''
        self.out_channels = CNN_hyperparameters["out_channels"]
        self.filter_shapes = CNN_hyperparameters["filter_shapes"]
        self.hidden_layer_sizes = CNN_hyperparameters["hidden_layer_sizes"]

    def fit(self, X, Y, train_hyperparameters, show_fig=True, show_filter=True):
        '''
        X is given in shape [num_of_samples, width, height, color]
        Builds the Tensorflow tree for the Training with the given batch_size
        Trains the model with given Data
        '''
        X, Y = shuffle(X, Y)
        validation_batch_size = train_hyperparameters["validation_batch_size"]
        X, Y = X[:-validation_batch_size], Y[:-validation_batch_size]
        X_valid, Y_valid = X[-validation_batch_size:], Y[-validation_batch_size:]

        width, height, in_channel = X.shape[1:]
        K = len(set(Y))
        # build Tensorflow tree
        self.conv_layers, self.hidden_layers, self.params = self.build_tree(
            in_channel, width, height, K)

        # define cost_function
        batch_size = train_hyperparameters["batch_size"]
        X_tf = tf.placeholder(tf.float32, shape=(batch_size, width,
                                                 height, in_channel), name="X")
        T_tf = tf.placeholder(tf.int32, shape=(batch_size,), name="T")
        Yish = self.forward_train(X_tf)
        cost = tf.math.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=Yish,
                labels=T_tf,
            )
        )

        lr = train_hyperparameters["learning_rate"]
        decay = train_hyperparameters["decay"]
        momentum = train_hyperparameters["momentum"]
        eps = train_hyperparameters["epsilon"]
        train_op = tf.train.RMSPropOptimizer(
            lr, decay, momentum, eps).minimize(cost)
        predict_op = tf.math.argmax(Yish, axis=1)

        init = tf.global_variables_initializer()

        # train
        if batch_size == None:
            batch_size = X.shape[0]
        nbatches = X.shape[0]/batch_size
        epochs = train_hyperparameters["epochs"]
        print_period = train_hyperparameters["print_period"]
        cost_list = []
        error = []

        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(nbatches):
                    X_batch, Y_batch = X[j*batch_size:(j+1) *
                                         batch_size], Y[j*batch_size:(j+1)*batch_size]
                    session.run(train_op, feed_dict={
                                X_tf: X_batch, T_tf: Y_batch})

                    if j % print_period == 0:
                        c = session.run(cost, feed_dict={
                                        X_tf: X_valid, T_tf: Y_valid})
                        cost_list.append(c)

                        prediction = session.run(
                            predict_op, feed_dict={X_tf: X_valid})
                        err = np.mean(prediction != Y_valid)
                        error.append(err)
                        print "At epochs % d and batch % d the cost is %.3f and error_rate is % .3f" % (
                            i, j, c, err)

            if show_fig == True:
                plt.plot(error)
                plt.title("error")
                plt.show()
                plt.plot(cost_list)
                plt.title("cost")
                plt.show()

            if show_filter == True:
                self.filter_show()

    def forward_train(self, X_tf):
        Z = X_tf
        for layer in self.conv_layers:
            Z = layer.convolution_and_bias(Z)
            Z = layer.pooling(Z)

        Z = tf.layers.flatten(Z)
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        return Z

    def build_tree(self, in_channel, width, height, K):
        '''
        Builds the tree; needs initial input_channel and width/height
        of the image

        padding and convolution go in "same mode"
        '''
        count = 0
        conv_layers = []
        hidden_layers = []
        params = []
        for out_channel, filter_shape in zip(self.out_channels, self.filter_shapes):
            layer = conv_layer(in_channel, out_channel,
                               filter_shape[0], filter_shape[1], count)
            conv_layers.append(layer)
            params += layer.params
            count += 1
            in_channel = out_channel
            width, height = width/2, height/2

        M1 = in_channel * width * height
        count = 0
        for M2 in self.hidden_layer_sizes:
            layer = hidden_layer(M1, M2, count)
            hidden_layers.append(layer)
            count += 1
            M1 = M2
            params += layer.params

        layer = hidden_layer(M1, K, count, fun=None)
        hidden_layers.append(layer)
        params += layer.params

        return conv_layers, hidden_layers, params

    def filter_show(self):
        # 2 layers
        # layer 1 (20, 3, 5, 5)
        # 60 5 by 5 images
        # layer 2 (50, 20, 5, 5)
        count = 1
        for layer in self.conv_layers:
            filter_ = layer.filter_.eval()
            num_of_rows = int(
                np.sqrt(layer.out_channels * layer.in_channels)) + 1
            grid = np.zeros((num_of_rows * layer.filter_height,
                             num_of_rows*layer.filter_width))

            n = 0
            m = 0
            anker = 0
            for i in range(layer.in_channels):
                for j in range(layer.out_channels):
                    m = anker
                    filt = filter_[:, :, i, j]
                    grid[n:n+layer.filter_height,
                         m:m + layer.filter_width] = filt
                    n += layer.filter_height
                    if n == (num_of_rows * layer.filter_height):
                        n = 0
                        anker += layer.filter_width
            count += 1

            plt.imshow(grid, cmap='gray')
            plt.title("W%d" % count)
            plt.show()
