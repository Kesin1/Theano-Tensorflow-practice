import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from scipy.io import loadmat
from datetime import datetime

from sklearn.utils import shuffle


def error_rate(pred, Y):
    return np.mean(pred != Y)


class conv_layer():
    def __init__(self, input_channels, output_channels, filter_shape, conv_lay_no, mode='valid', ignore_border=True):
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.filter_rows = filter_shape[0]
        self.filter_columns = filter_shape[1]
        self.mode = mode
        self.ignore_border = ignore_border

        filter_init = np.random.randn(
            self.output_channels, self.input_channels, self.filter_rows, self.filter_columns)/np.sqrt(self.filter_rows + self.filter_columns)
        # one bias per feature map
        b_init = np.zeros(self.output_channels)

        self.filter_ = theano.shared(
            filter_init, name="filter of conv_lay: %d" % conv_lay_no)
        self.b = theano.shared(
            b_init, name="bias of conv_lay: %d" % conv_lay_no)
        self.params = [self.filter_, self.b]

    def forward(self, X):
        # do convolution and pooling
        Y = conv2d(X, self.filter_)
        Y = pool.pool_2d(
            Y, (2, 2), ignore_border=self.ignore_border, mode="max")
        return T.nnet.relu(Y + self.b.dimshuffle('x', 0, 'x', 'x'))

    # def print_layer(self, z):
    #     z = T.transpose(z, (0, 2, 3, 1))
    #     plt.imshow(z.eval()[0, :, :, :])
    #     plt.show()

    def shapes_after_conv(self, width, height):
        if self.mode == 'valid':
            width = width - self.filter_columns + 1
            height = height - self.filter_rows + 1
        elif self.mode == 'full':
            width = width + self.filter_columns - 1
            height = height + self.filter_rows - 1
        elif self.mode == 'same':
            pass

        if self.ignore_border == True:
            width = width/2
            height = height/2
        else:
            width = width/2 + (width % 2)
            height = height/2 + (height % 2)

        return width, height


class hidden_layer():
    def __init__(self, M1, M2, hid_lay_no, fun=T.nnet.relu):
        self.M1 = M1
        self.M2 = M2
        self.fun = fun
        # iniatialisation of update variables
        W_init = np.random.randn(M1, M2)/np.sqrt(M1 + M2)
        b_init = np.zeros(M2)
        self.W = theano.shared(W_init, name="W_hidden_num: %d" % hid_lay_no)
        self.b = theano.shared(b_init, name="b_hidden_num: %d" % hid_lay_no)

        self.params = [self.W, self.b]
        # batch normalisation later

    def forward(self, Z):
        # do one step in fully connected Neural Network
        return self.fun(Z.dot(self.W) + self.b)


class CNN():
    def __init__(self, output_channels, filter_shapes, hidden_layers_sizes):
        self.output_channels = output_channels
        self.filter_shapes = filter_shapes
        self.hidden_layers_sizes = hidden_layers_sizes

    def fit(self, X, Y, lr=1e-2, reg=1e-2, epochs=1, batchsz=None, print_period=10, show_fig=True, show_filter=True):
        '''
        Takes in the input in shape (num_of_samples, color_channel, width, height)
        '''
        X, Y = shuffle(X, Y)

        X_valid, Y_valid = X[-100:], Y[-100:]
        X, Y = X[:-100], Y[:-100]
        K = len(set(Y))

        # comments on changes of shapes
        # since we don't ingore the border inside the convolution
        # and we use valid convolution, we get:
        # new_width = width - filter_size_width + 1
        # pooled_width = new_width / 2 (+1 in case of odd new_width)

        # layer_configs
        input_channel = X.shape[1]
        width = X.shape[2]
        height = X.shape[3]

        count = 0
        self.conv_layers = []
        self.params = []
        for output_channel, filter_shape in zip(self.output_channels, self.filter_shapes):
            layer = conv_layer(input_channel, output_channel,
                               filter_shape, count)
            self.conv_layers.append(layer)
            self.params += layer.params
            count += 1
            input_channel = output_channel
            width, height = layer.shapes_after_conv(width, height)

        M1 = input_channel * width * height
        count = 0
        self.hidden_layers = []
        for M2 in self.hidden_layers_sizes:
            layer = hidden_layer(M1, M2, count)
            self.hidden_layers.append(layer)
            count += 1
            M1 = M2
            self.params += layer.params

        layer = hidden_layer(M1, K, count, T.nnet.softmax)
        self.hidden_layers.append(layer)
        self.params += layer.params

        # theano_tree
        th_X = T.tensor4('X')
        th_T = T.ivector('Y')
        p_Y = self.forward(th_X)
        # reg_cost = reg * T.mean([(p*p).sum() for p in self.params])
        cost = - \
            T.mean(T.log(p_Y[T.arange(th_T.shape[0]), th_T]))  # + reg_cost
        grads = T.grad(cost, self.params)
        prediction = self.predict(th_X)

        # update_configs
        updates = [
            (W, W - lr*g) for W, g in zip(self.params, grads)
        ]

        train_op = theano.function(inputs=[th_X, th_T], updates=updates)
        self.predict_op = theano.function(
            inputs=[th_X, th_T], outputs=[cost, prediction])

        # th_x = T.tensor4('x')
        # self.print_layer(th_x)
        # print_layer_theano = theano.function(inputs=[th_x])

        if batchsz == None:
            batchsz = X.shape[0]
        nbatches = X.shape[0]/batchsz
        cost = []
        error = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(nbatches):
                X_batch, Y_batch = X[j*batchsz:(j+1)
                                     * batchsz], Y[j*batchsz:(j+1)*batchsz]
                train_op(X_batch, Y_batch)
                if j % print_period == 0:
                    c, p = self.predict_op(X_valid, Y_valid)
                    err = error_rate(p, Y_valid)
                    cost.append(c)
                    error.append(err)
                    print "At epochs %d and batch %d the cost is %.3f and error_rate is %.2f" % (
                        i, j, c, err)
                    # print_layer_theano(X_valid[:2, :, :, :])

        if show_fig == True:
            plt.plot(error)
            plt.title("error")
            plt.show()
            plt.plot(cost)
            plt.title("cost")
            plt.show()

        if show_filter == True:
            self.filter_show()

    def forward(self, th_X):
        Z = th_X
        for layer in self.conv_layers:
            Z = layer.forward(Z)
        Z = T.flatten(Z, ndim=2)
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        return Z

    def predict(self, X):
        p_Y = self.forward(X)
        return T.argmax(p_Y, axis=1)

    def score(self, X, Y):
        _, pred = self.predict_op(X, Y)
        return np.mean(pred == Y)

    def filter_show(self):
        # 2 layers
        # layer 1 (20, 3, 5, 5)
        # 60 5 by 5 images
        # layer 2 (50, 20, 5, 5)
        count = 1
        for layer in self.conv_layers:
            filter_ = layer.filter_.get_value()
            num_of_rows = int(
                np.sqrt(layer.output_channels * layer.input_channels)) + 1
            grid = np.zeros((num_of_rows * layer.filter_rows,
                             num_of_rows*layer.filter_columns))

            n = 0
            m = 0
            anker = 0
            for i in range(self.input_channels):
                for j in range(self.output_channels):
                    m = anker
                    filt = filter_[i, j]
                    grid[n:n+layer.filter_rows, m:m +
                         layer.filter_columns] = filt
                    n += layer.filter_rows
                    if n == (num_of_rows * layer.filter_rows):
                        n = 0
                        anker += layer.filter_columns
            count += 1

            plt.imshow(grid, cmap='gray')
            plt.title("W%d" % count)
            plt.show()


def main():
    # load data
    test = loadmat(
        "../Tensorflow_and_Theano/large_files/test_32x32.mat")
    train = loadmat(
        "../Tensorflow_and_Theano/large_files/train_32x32.mat")

    Y_test = test['y'].astype(np.int32)
    Y_test = Y_test.squeeze()-1  # Indexing starting at 0
    X_test = test['X'].astype(np.float32)/255.
    X_test = np.transpose(X_test, (3, 2, 0, 1))

    Y_train = train['y'].astype(np.int32)
    Y_train = Y_train.squeeze()-1  # indexing starting at 0
    X_train = train['X'].astype(np.float32)/255.
    X_train = np.transpose(X_train, (3, 2, 0, 1))

    output_channels = [20, 50]      # output channels
    filter_shapes = [(5, 5), (5, 5)]
    hidden_layers_sizes = [500]

    model = CNN(output_channels, filter_shapes, hidden_layers_sizes)
    t0 = datetime.now()
    model.fit(X_train, Y_train, batchsz=500)
    print "Time to fit the model %.4f" % (datetime.now() - t0)

    t0 = datetime.now()
    score = model.score(X_test, Y_test)
    print "Time to test the model %.4f" % (datetime.now() - t0)
    print "Score: %.2f" % score


# def by():
#     filter_ = np.random.randint(0, high=255, size=(50, 20, 5, 5))
#     num_of_rows = int(np.sqrt(50*20)) + 1
#     grid = np.zeros((num_of_rows * 5, num_of_rows*5))

#     n = 0
#     m = 0
#     anker = 0
#     count = 1
#     for i in range(50):
#         for j in range(20):
#             m = anker
#             filt = filter_[i, j]
#             grid[n:n+5, m:m+5] = filt
#             n += 5
#             if n == (num_of_rows * 5):
#                 n = 0
#                 anker += 5

#     plt.imshow(grid, cmap='gray')
#     plt.title("W%d" % count)
#     plt.show()


# by()


if __name__ == "__main__":
    main()
