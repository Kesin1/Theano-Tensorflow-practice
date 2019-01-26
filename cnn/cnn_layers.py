import numpy as np
import tensorflow as tf


class conv_layer():
    '''
    Implementation of a convolutional layer with tensorflow
    '''

    def __init__(self, in_channels, out_channels, filter_height, filter_width, conv_lay_no):
        '''
        filter shape [filter_height, filter_width, in_channels, out_channels]
        sets all variables to class variables 

        creates tf.Variables
        conv_lay_no saved in tf.Variable names

        creates list of tf.Variables which is set as class variable
        '''

        # setting variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.conv_lay_no = conv_lay_no

        # initializing tf.Variables
        filter_init = np.random.randn(self.filter_height,
                                      self.filter_width, self.in_channels,
                                      self.out_channels)/np.sqrt(self.filter_height +
                                                                 self.filter_width).astype(np.float32)
        b_init = np.zeros(self.out_channels).astype(np.float32)

        self.filter_ = tf.Variable(filter_init.astype(np.float32), name="filter_%d" %
                                   self.conv_lay_no)
        self.b = tf.Variable(
            b_init.astype(np.float32), name="bias_%d" % self.conv_lay_no)

        # set params
        self.params = [self.filter_, self.b]

    def convolution_and_bias(self, X, strides=[1, 1, 1, 1], padding='SAME'):
        '''
        - does convolution with given stride (default = [1, 3, 3, 1])
        and padding (default = 'SAME') and default tf-settings of
        tf.nn.conv2d
        - shape of input X [batch, in_height, in_width, in_channels]
        - shape of filer [filter_height, filter_width, in_channels, out_channels]
        - shape of output [batch, in_height, in_width, out_channels]
        - strides by default 3 over feature maps
        - saves conv_lay_no in the tf.nn.conv2d name variable

        - performs bias_add afterwards
        '''
        Y = tf.nn.conv2d(X, self.filter_, strides=strides,
                         padding=padding, name="conv2d_%d" %
                         self.conv_lay_no)

        Y = tf.nn.bias_add(Y, self.b)
        return Y

    def pooling(self, X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        '''
        - does pooling with given stride and padding and default tf
        - saves conv_lay_no in the tf.nn.conv2d name variable
        - add bias term afterwards 
        '''
        return tf.nn.tanh(tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding))


class hidden_layer():
    '''
    Implementation of a simple hidden_layer in a fully connected
    Neural Network
    '''

    def __init__(self, M1, M2, hid_lay_no, fun=tf.nn.relu):
        '''
        Setting Variables 
        Initializing tf.Variables
        '''
        self.M1 = M1
        self.M2 = M2
        self.hid_lay_no = hid_lay_no
        self.fun = fun

        # iniatialisation of update variables
        W_init = np.random.randn(M1, M2)/np.sqrt(M1 + M2)
        b_init = np.zeros(M2)
        self.W = tf.Variable(W_init.astype(np.float32),
                             name="W_hidden_num_%d" % self.hid_lay_no)
        self.b = tf.Variable(b_init.astype(np.float32),
                             name="b_hidden_num_%d" % self.hid_lay_no)

        self.params = [self.W, self.b]
        # batch normalisation later

    def forward(self, Z):
        # do one step in fully connected Neural Network
        # only do linear Transformation for last layer
        if self.fun == None:
            return tf.matmul(Z, self.W) + self.b
        return self.fun(tf.matmul(Z, self.W) + self.b)
